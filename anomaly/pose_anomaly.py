"""
src/anomaly/pose_anomaly.py
────────────────────────────
Privacy-preserving shoplifting detection via pose-sequence anomaly scoring.

Architecture (Shopformer-lite):
  1. COCO17 keypoints (17 joints × 3: x, y, confidence) per frame
  2. Per-person sliding window of `window_size` frames
  3. Feature vector = normalised joint positions | wrist velocities |
                       concealment distances | elbow angles
  4. Rolling normal-behaviour statistics (mean + std per feature dim)
     updated from low-anomaly frames — continual unsupervised learning
  5. Anomaly score = avg z-score of top-25% most deviant features
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

WRIST_IDX    = [9, 10]
ELBOW_IDX    = [7, 8]
HIP_IDX      = [11, 12]
SHOULDER_IDX = [5, 6]

CONCEALMENT_JOINTS = [
    (9, 11), (10, 12),   # wrist → hip (pockets)
    (9, 6),  (10, 5),    # wrist → opposite shoulder (jacket)
]


@dataclass
class AnomalyState:
    person_id:      int
    frame_buffer:   deque = field(default_factory=lambda: deque(maxlen=60))
    feature_buffer: deque = field(default_factory=lambda: deque(maxlen=300))
    scores:         deque = field(default_factory=lambda: deque(maxlen=60))
    normal_mean:    Optional[np.ndarray] = None
    normal_std:     Optional[np.ndarray] = None
    normal_count:   int = 0
    last_score:     float = 0.0
    flagged_frames: int = 0


class PoseAnomalyDetector:
    """
    Per-person shoplifting detector using pose kinematic features.

    Self-calibrates during warm-up: first `warm_up_frames` frames of low-motion
    data build the 'normal' baseline. Subsequent frames are z-scored against it.
    Continual learning keeps the baseline current (adapts to store layout, seasons).
    """

    def __init__(
        self,
        window_size: int = 30,
        anomaly_threshold: float = 2.8,
        warm_up_frames: int = 50,
        adaptation_rate: float = 0.01,
    ) -> None:
        self.window_size       = window_size
        self.anomaly_threshold = anomaly_threshold
        self.warm_up_frames    = warm_up_frames
        self.adaptation_rate   = adaptation_rate
        self._states: dict[int, AnomalyState] = {}
        logger.info("PoseAnomalyDetector ready threshold=%.2f warm_up=%d",
                    anomaly_threshold, warm_up_frames)

    def update(self, person_id: int, keypoints: Optional[np.ndarray]) -> float:
        """
        Feed one frame's keypoints. Returns anomaly score in [0, 1].
        0 = normal, 1 = maximum anomaly (definite shoplifting signal).
        """
        if person_id not in self._states:
            self._states[person_id] = AnomalyState(person_id=person_id)

        state = self._states[person_id]

        if keypoints is None or not self._sufficient_confidence(keypoints):
            return state.last_score

        state.frame_buffer.append(keypoints.copy())

        if len(state.frame_buffer) < 3:
            return 0.0

        features = self._extract_features(list(state.frame_buffer))
        if features is None:
            return 0.0

        score = self._compute_anomaly_score(state, features)
        state.last_score = score
        state.scores.append(score)

        if score < 0.4:
            self._update_normal_stats(state, features)
        if score > 0.75:
            state.flagged_frames += 1

        return score

    def get_person_summary(self, person_id: int) -> dict:
        if person_id not in self._states:
            return {"person_id": person_id, "status": "not_tracked"}
        state = self._states[person_id]
        scores = list(state.scores)
        return {
            "person_id":     person_id,
            "last_score":    round(state.last_score, 4),
            "max_score":     round(max(scores, default=0.0), 4),
            "mean_score":    round(float(np.mean(scores)) if scores else 0.0, 4),
            "flagged_frames": state.flagged_frames,
            "normal_frames": state.normal_count,
            "is_suspicious": state.last_score > self.anomaly_threshold / 5.0,
        }

    def remove_person(self, person_id: int) -> None:
        self._states.pop(person_id, None)

    def active_persons(self) -> list[int]:
        return list(self._states.keys())

    # ── Feature engineering ───────────────────────────────────────────────────

    def _extract_features(self, frame_buffer: list[np.ndarray]) -> Optional[np.ndarray]:
        """
        52-dimensional feature vector per frame window:
          [A] Normalised joint positions   (34 dims)
          [B] Wrist velocity last frame    (4 dims)
          [C] Wrist velocity std over window (4 dims)
          [D] Concealment distances        (4 dims)
          [E] Concealment distance std     (4 dims)
          [F] Elbow angles                 (2 dims)
        """
        kp_seq = np.array(frame_buffer, dtype=np.float32)  # (T, 17, 3)
        T = kp_seq.shape[0]

        mid_shoulder = (kp_seq[:, 5, :2] + kp_seq[:, 6, :2]) / 2
        mid_hip      = (kp_seq[:, 11, :2] + kp_seq[:, 12, :2]) / 2
        torso_len    = np.linalg.norm(mid_shoulder - mid_hip, axis=1, keepdims=True) + 1e-6
        torso_centre = mid_hip

        positions = (kp_seq[:, :, :2] - torso_centre[:, np.newaxis, :]) / torso_len[:, np.newaxis, :]

        velocities = np.diff(positions, axis=0, prepend=positions[:1]) if T >= 2 \
                     else np.zeros_like(positions)
        wrist_vel = velocities[:, WRIST_IDX, :]

        concealment = np.stack(
            [np.linalg.norm(positions[:, wi, :] - positions[:, hi, :], axis=1)
             for wi, hi in CONCEALMENT_JOINTS],
            axis=1,
        )

        elbow_angles = self._compute_elbow_angles(positions)

        feats = np.concatenate([
            positions[-1].flatten(),           # 34
            wrist_vel[-1].flatten(),           # 4
            wrist_vel.std(axis=0).flatten(),   # 4
            concealment[-1],                   # 4
            concealment.std(axis=0),           # 4
            elbow_angles[-1],                  # 2
        ])  # total: 52
        return feats

    def _compute_elbow_angles(self, positions: np.ndarray) -> np.ndarray:
        T = positions.shape[0]
        angles = np.zeros((T, 2), dtype=np.float32)
        for i, (s, e, w) in enumerate([(5, 7, 9), (6, 8, 10)]):
            v1 = positions[:, s, :] - positions[:, e, :]
            v2 = positions[:, w, :] - positions[:, e, :]
            norm1 = np.linalg.norm(v1, axis=1, keepdims=True) + 1e-8
            norm2 = np.linalg.norm(v2, axis=1, keepdims=True) + 1e-8
            cos_a = np.clip(np.sum((v1 / norm1) * (v2 / norm2), axis=1), -1.0, 1.0)
            angles[:, i] = np.arccos(cos_a)
        return angles

    # ── Anomaly scoring ───────────────────────────────────────────────────────

    def _compute_anomaly_score(self, state: AnomalyState, features: np.ndarray) -> float:
        if state.normal_count < self.warm_up_frames:
            self._update_normal_stats(state, features)
            return 0.0

        z_scores = np.abs((features - state.normal_mean) / (state.normal_std + 1e-8))
        top_k = max(1, len(z_scores) // 4)
        raw_score = float(np.mean(np.sort(z_scores)[-top_k:]))
        normalised = 1.0 / (1.0 + np.exp(-0.8 * (raw_score - self.anomaly_threshold)))
        return float(np.clip(normalised, 0.0, 1.0))

    def _update_normal_stats(self, state: AnomalyState, features: np.ndarray) -> None:
        state.feature_buffer.append(features)
        if len(state.feature_buffer) >= 10:
            buf = np.array(list(state.feature_buffer))
            new_mean = buf.mean(axis=0)
            new_std  = buf.std(axis=0) + 1e-8
            if state.normal_mean is None:
                state.normal_mean = new_mean
                state.normal_std  = new_std
            else:
                a = self.adaptation_rate
                state.normal_mean = (1 - a) * state.normal_mean + a * new_mean
                state.normal_std  = (1 - a) * state.normal_std  + a * new_std
            state.normal_count += 1

    @staticmethod
    def _sufficient_confidence(keypoints: np.ndarray, min_conf: float = 0.3) -> bool:
        if keypoints.shape[0] < 17:
            return False
        return int(np.sum(keypoints[:, 2] >= min_conf)) >= 8
