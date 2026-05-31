"""
src/ingestion/motion_detector.py
─────────────────────────────────
Stage 1 of the Smart Storage Engine.

Implements two complementary strategies:
  1. Gaussian Background Subtraction (MOG2) – handles gradual lighting change
  2. Frame Differencing – fast absolute pixel delta fallback

Only when motion_score > motion_threshold does the system wake the heavy
YOLO/VSUMM pipeline, dramatically cutting GPU idle cycles.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MotionResult:
    """Return type from MotionDetector.detect()."""
    has_motion: bool
    motion_score: float
    motion_mask: Optional[np.ndarray]
    frame: np.ndarray
    contours: list = field(default_factory=list)


class MotionDetector:
    """
    Detects foreground motion using MOG2 background subtraction with an
    optional frame-differencing fallback.

    Parameters
    ----------
    motion_threshold : float
        Minimum foreground pixel fraction required to declare motion.
    diff_threshold : float
        Pixel intensity change needed to count as 'moved' in frame-diff mode.
    history : int
        MOG2 history length (frames used to model the background).
    min_blob_area : int
        Contours smaller than this (px²) are discarded as camera noise.
    use_bg_subtraction : bool
        True (default) → MOG2; False → raw frame differencing.
    """

    def __init__(
        self,
        motion_threshold: float = 0.008,
        diff_threshold: float = 25.0,
        history: int = 500,
        min_blob_area: int = 800,
        use_bg_subtraction: bool = True,
    ) -> None:
        self.motion_threshold = motion_threshold
        self.diff_threshold = diff_threshold
        self.min_blob_area = min_blob_area
        self._prev_gray: Optional[np.ndarray] = None
        self._frame_count = 0

        if use_bg_subtraction:
            self._bg = cv2.createBackgroundSubtractorMOG2(
                history=history, varThreshold=16, detectShadows=True
            )
        else:
            self._bg = None

        logger.info(
            "MotionDetector ready | mode=%s thresh=%.4f",
            "MOG2" if self._bg else "diff",
            motion_threshold,
        )

    def detect(self, frame: np.ndarray) -> MotionResult:
        self._frame_count += 1
        if self._bg is not None:
            return self._mog2_detect(frame)
        return self._diff_detect(frame)

    def reset(self) -> None:
        if self._bg is not None:
            self._bg = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=16, detectShadows=True
            )
        self._prev_gray = None
        self._frame_count = 0

    def _mog2_detect(self, frame: np.ndarray) -> MotionResult:
        fg_mask = self._bg.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        motion_score = float(np.count_nonzero(fg_mask)) / float(fg_mask.size)
        contours = self._get_significant_contours(fg_mask)
        has_motion = motion_score > self.motion_threshold and len(contours) > 0
        return MotionResult(
            has_motion=has_motion,
            motion_score=motion_score,
            motion_mask=fg_mask,
            frame=frame,
            contours=contours,
        )

    def _diff_detect(self, frame: np.ndarray) -> MotionResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self._prev_gray is None:
            self._prev_gray = gray
            return MotionResult(False, 0.0, None, frame)
        diff = cv2.absdiff(self._prev_gray, gray)
        _, thresh = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        motion_score = float(np.count_nonzero(thresh)) / float(thresh.size)
        contours = self._get_significant_contours(thresh)
        has_motion = motion_score > self.motion_threshold
        self._prev_gray = gray
        return MotionResult(
            has_motion=has_motion,
            motion_score=motion_score,
            motion_mask=thresh,
            frame=frame,
            contours=contours,
        )

    def _get_significant_contours(self, mask: np.ndarray) -> list:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= self.min_blob_area]


class VideoIngester:
    """
    Wraps cv2.VideoCapture with motion-gated frame yielding.
    Accepts RTSP URLs, file paths, and webcam indices.
    """

    def __init__(
        self,
        source: str,
        detector: MotionDetector,
        fps_target: int = 10,
    ) -> None:
        self.source = source
        self.detector = detector
        self.fps_target = fps_target

        try:
            src = int(source)
        except ValueError:
            src = source

        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source!r}")

        self._native_fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._frame_skip = max(1, int(self._native_fps / fps_target))
        logger.info(
            "VideoIngester source=%s native_fps=%.1f skip=%d",
            source, self._native_fps, self._frame_skip,
        )

    @property
    def frame_width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def frame_height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        return self._cap.read()

    def release(self) -> None:
        self._cap.release()
        logger.info("VideoIngester released for source=%s", self.source)

    def __iter__(self):
        frame_idx = 0
        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % self._frame_skip != 0:
                    continue
                result = self.detector.detect(frame)
                yield frame, result
        finally:
            self.release()
