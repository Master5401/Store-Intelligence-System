"""
pipeline/orchestrator.py
─────────────────────────
Central orchestrator that connects every module into a single processing loop.

Processing loop (per motion-triggered frame):
  1.  MotionDetector    → gate heavy processing
  2.  DetectorTracker   → YOLO + ByteTrack → TrackedObject list
  3.  PoseAnomalyDetector → per-person anomaly score (z-score / Shopformer-lite)
  4.  HeatmapGenerator  → accumulate foot-points
  5.  VSUMM             → extract keyframes from motion clip
  6.  VLMAnnotator      → describe keyframes → text summary
  7.  EventRepository   → persist event + summary + tracks → SQLite
  8.  KafkaEventProducer → publish tracking + alert payloads
  9.  AlertSystem       → Twilio SMS/call if anomaly_score > critical threshold
  10. ConnectionManager → broadcast live frame data to WebSocket clients
  11. Periodic Adaptation → every `adaptation_interval` seconds re-baseline
                            the anomaly detector on the buffered normal frames
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from config.settings import settings
from src.alerts.twilio_alert import AlertSystem, MockAlertSystem
from src.anomaly.pos_integration import POSIntegration
from src.anomaly.pose_anomaly import PoseAnomalyDetector
from src.ingestion.motion_detector import MotionDetector, VideoIngester
from src.storage.database import EventRepository, init_db, get_session_factory
from src.storage.vlm_annotator import get_annotator
from src.storage.vsumm import VSUMM, KeyframeStore
from src.streaming.kafka_producer import KafkaEventProducer
from src.vision.detector_tracker import DetectorTracker, TrackedObject
from src.vision.heatmap import HeatmapGenerator

logger = logging.getLogger(__name__)


class StoreIntelligenceOrchestrator:
    """
    Coordinates the full edge inference pipeline.

    Parameters
    ----------
    settings       : application settings singleton
    mock_alerts    : if True, AlertSystem is replaced with MockAlertSystem
                     (no real Twilio calls — for CI / local dev)
    broadcast_fn   : optional async callable(store_id, data) called every
                     processed frame to push data to WebSocket clients
    """

    def __init__(
        self,
        mock_alerts: bool = True,
        broadcast_fn=None,
    ) -> None:
        self.settings     = settings
        self.broadcast_fn = broadcast_fn

        # ── Component initialisation ──────────────────────────────────────────
        self.motion_detector = MotionDetector(
            motion_threshold=settings.motion_threshold,
            diff_threshold=settings.diff_threshold,
        )
        self.detector_tracker = DetectorTracker(
            model_path=settings.yolo_model,
            pose_model_path=settings.yolo_pose_model,
            confidence=settings.detection_confidence,
            tracker_config="config/bytetrack.yaml",
            draw_annotations=True,
        )
        self.pose_anomaly = PoseAnomalyDetector(
            window_size=settings.pose_window_size,
            anomaly_threshold=settings.anomaly_threshold,
        )
        self.heatmap = HeatmapGenerator(
            decay=settings.heatmap_decay,
        )
        self.vsumm        = VSUMM(n_clusters=settings.vsumm_clusters)
        self.keyframe_store = KeyframeStore(base_dir=settings.keyframes_dir)
        self.annotator    = get_annotator(prefer="auto")
        self.kafka        = KafkaEventProducer(settings)
        self.pos          = POSIntegration()

        self.alert_system: AlertSystem = (
            MockAlertSystem() if mock_alerts else AlertSystem(settings)
        )

        # ── State ──────────────────────────────────────────────────────────────
        self._running           = False
        self._frame_count       = 0
        self._motion_clip_frames: list[np.ndarray] = []
        self._motion_clip_start: Optional[datetime] = None
        self._last_adaptation   = time.time()
        self._alerted_ids: set[int] = set()   # track IDs already alerted this session

        # Stats
        self.stats = {
            "frames_processed":   0,
            "motion_events":      0,
            "alerts_fired":       0,
            "keyframes_saved":    0,
            "avg_inference_ms":   0.0,
        }

        logger.info(
            "Orchestrator ready | store=%s camera=%s mock_alerts=%s",
            settings.store_id, settings.camera_id, mock_alerts,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(self, source: Optional[str] = None) -> None:
        """
        Main async entry point. Opens the video source and runs forever
        (or until the stream ends).
        """
        src = source or settings.rtsp_url
        await init_db(settings.database_url)
        logger.info("Pipeline starting on source: %s", src)

        ingester = VideoIngester(
            source=src,
            detector=self.motion_detector,
            fps_target=settings.fps_target,
        )
        self.heatmap.frame_width  = ingester.frame_width  or 1280
        self.heatmap.frame_height = ingester.frame_height or 720
        self.heatmap._matrix = np.zeros(
            (self.heatmap.frame_height, self.heatmap.frame_width), dtype=np.float32
        )

        self._running = True

        # Run the blocking cv2 loop in a thread executor so the event loop
        # stays responsive for WebSocket broadcasts
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_loop, ingester)

        self.kafka.close()
        logger.info("Pipeline stopped. Stats: %s", self.stats)

    def process_single_frame(self, frame: np.ndarray) -> dict:
        """
        Process one frame synchronously.
        Useful for testing and for the demo endpoint.
        """
        motion_result = self.motion_detector.detect(frame)
        pf = self.detector_tracker.process_frame(frame)
        self._apply_anomaly_scores(pf.tracks)
        self.heatmap.update(pf.tracks)

        return {
            "person_count":  pf.person_count,
            "inference_ms":  round(pf.inference_ms, 2),
            "has_motion":    motion_result.has_motion,
            "tracks": [
                {
                    "id":            t.track_id,
                    "bbox":          list(t.bbox),
                    "anomaly_score": round(t.anomaly_score, 3),
                }
                for t in pf.tracks
            ],
        }

    # ── Sync processing loop (runs in thread) ─────────────────────────────────

    def _sync_loop(self, ingester: VideoIngester) -> None:
        for frame, motion_result in ingester:
            if not self._running:
                break

            self._frame_count += 1
            self.stats["frames_processed"] += 1

            # ── Step 1: accumulate motion clip ──────────────────────────────
            if motion_result.has_motion:
                if not self._motion_clip_frames:
                    self._motion_clip_start = datetime.utcnow()
                    self.stats["motion_events"] += 1
                self._motion_clip_frames.append(frame.copy())
            else:
                if len(self._motion_clip_frames) >= settings.vsumm_min_frames:
                    asyncio.run_coroutine_threadsafe(
                        self._process_clip(list(self._motion_clip_frames),
                                           self._motion_clip_start),
                        asyncio.get_event_loop(),
                    )
                self._motion_clip_frames.clear()

            # ── Step 2: detection + tracking ────────────────────────────────
            pf = self.detector_tracker.process_frame(frame)

            # ── Step 3: pose anomaly scoring ────────────────────────────────
            self._apply_anomaly_scores(pf.tracks)

            # ── Step 4: heatmap update ──────────────────────────────────────
            self.heatmap.update(pf.tracks)

            # ── Step 5: check for critical anomalies ────────────────────────
            self._check_and_fire_alerts(pf.tracks)

            # ── Step 6: Kafka tracking publish ──────────────────────────────
            self.kafka.publish_tracking_event(
                store_id=settings.store_id,
                camera_id=settings.camera_id,
                tracks=pf.tracks,
                heatmap_stats=self.heatmap.stats(),
            )

            # ── Step 7: broadcast to WebSocket clients ──────────────────────
            if self.broadcast_fn and pf.tracks:
                ws_payload = self._build_ws_payload(pf, motion_result)
                asyncio.run_coroutine_threadsafe(
                    self.broadcast_fn(settings.store_id, ws_payload),
                    asyncio.get_event_loop(),
                )

            # ── Step 8: periodic adaptation ─────────────────────────────────
            if time.time() - self._last_adaptation > settings.adaptation_interval:
                self._periodic_adaptation()
                self._last_adaptation = time.time()

            # Update rolling avg inference stat
            self.stats["avg_inference_ms"] = round(self.detector_tracker.avg_inference_ms, 2)

    # ── Async clip processor ──────────────────────────────────────────────────

    async def _process_clip(
        self,
        frames: list[np.ndarray],
        start_time: Optional[datetime],
    ) -> None:
        """VSUMM → keyframe save → VLM annotation → DB persist."""
        clip_id  = str(uuid.uuid4())[:12]
        end_time = datetime.utcnow()
        st       = start_time or end_time

        # VSUMM keyframe extraction
        _, keyframes = self.vsumm.extract_keyframes(frames)
        kf_paths = self.keyframe_store.save(
            keyframes, event_id=clip_id, store_id=settings.store_id
        )
        self.stats["keyframes_saved"] += len(kf_paths)

        # VLM annotation
        context = f"Store {settings.store_id}, camera {settings.camera_id}"
        summary_text = self.annotator.describe(keyframes, context=context)

        # DB persist
        sf = get_session_factory(settings.database_url)
        async with sf() as session:
            repo = EventRepository(session)
            event = await repo.create_event(
                event_id=str(uuid.uuid4()),
                store_id=settings.store_id,
                camera_id=settings.camera_id,
                event_type="motion",
                severity="info",
            )
            event.data = {"person_count": 0, "clip_id": clip_id}
            await session.commit()

            await repo.create_summary(
                event_id=event.id,
                store_id=settings.store_id,
                camera_id=settings.camera_id,
                start_time=st,
                end_time=end_time,
                summary_text=summary_text,
                keyframe_paths=str(kf_paths),
                keyframe_count=len(kf_paths),
            )

        logger.info(
            "Clip processed: id=%s keyframes=%d summary='%s...'",
            clip_id, len(kf_paths), summary_text[:80],
        )

        # Kafka heatmap snapshot every clip
        self.kafka.publish_heatmap_snapshot(
            store_id=settings.store_id,
            camera_id=settings.camera_id,
            matrix_payload=self.heatmap.matrix_to_json_payload(),
        )

    # ── Anomaly helpers ───────────────────────────────────────────────────────

    def _apply_anomaly_scores(self, tracks: list[TrackedObject]) -> None:
        for track in tracks:
            score = self.pose_anomaly.update(track.track_id, track.keypoints)
            track.anomaly_score = score

    def _check_and_fire_alerts(self, tracks: list[TrackedObject]) -> None:
        for track in tracks:
            if (
                track.anomaly_score > 0.75
                and track.track_id not in self._alerted_ids
            ):
                self._alerted_ids.add(track.track_id)
                self.stats["alerts_fired"] += 1
                logger.warning(
                    "SHOPLIFTING DETECTED: person_id=%d score=%.3f",
                    track.track_id, track.anomaly_score,
                )
                self.alert_system.send_alert(
                    store_id=settings.store_id,
                    person_id=track.track_id,
                    anomaly_score=track.anomaly_score,
                    alert_type="shoplifting",
                )
                self.kafka.publish_alert(
                    store_id=settings.store_id,
                    alert_type="shoplifting",
                    data={
                        "person_id":    track.track_id,
                        "anomaly_score": track.anomaly_score,
                        "bbox":          list(track.bbox),
                    },
                )
                asyncio.run_coroutine_threadsafe(
                    self._persist_alert(track),
                    asyncio.get_event_loop(),
                )

    async def _persist_alert(self, track: TrackedObject) -> None:
        sf = get_session_factory(settings.database_url)
        async with sf() as session:
            repo = EventRepository(session)
            event = await repo.create_event(
                event_id=str(uuid.uuid4()),
                store_id=settings.store_id,
                camera_id=settings.camera_id,
                event_type="alert",
                severity="critical",
            )
            event.data = {
                "person_id":     track.track_id,
                "anomaly_score": round(track.anomaly_score, 4),
            }
            await session.commit()
            await repo.create_alert(
                event_id=event.id,
                store_id=settings.store_id,
                alert_type="shoplifting",
                severity="critical",
                message=f"Shoplifting behaviour detected for person ID {track.track_id}",
                anomaly_score=track.anomaly_score,
            )

    def _periodic_adaptation(self) -> None:
        """Log the periodic adaptation tick. The z-score model self-updates inline."""
        active = self.pose_anomaly.active_persons()
        logger.info(
            "Periodic adaptation tick | active_persons=%d | "
            "frames=%d | alerts=%d",
            len(active),
            self.stats["frames_processed"],
            self.stats["alerts_fired"],
        )
        # Persist heatmap snapshot to DB (fire-and-forget via thread-safe call)
        asyncio.run_coroutine_threadsafe(
            self._save_heatmap_to_db(),
            asyncio.get_event_loop(),
        )

    async def _save_heatmap_to_db(self) -> None:
        sf = get_session_factory(settings.database_url)
        async with sf() as session:
            repo = EventRepository(session)
            snap = await repo.save_heatmap(
                store_id=settings.store_id,
                camera_id=settings.camera_id,
            )
            m = self.heatmap.get_matrix()
            snap.matrix = m
            await session.commit()

    # ── WebSocket payload builder ─────────────────────────────────────────────

    @staticmethod
    def _build_ws_payload(pf, motion_result) -> dict:
        return {
            "type":         "tracking_frame",
            "timestamp":    datetime.utcnow().isoformat(),
            "person_count": pf.person_count,
            "inference_ms": round(pf.inference_ms, 2),
            "has_motion":   motion_result.has_motion,
            "motion_score": round(motion_result.motion_score, 4),
            "tracks": [
                {
                    "id":            t.track_id,
                    "bbox":          [round(v, 1) for v in t.bbox],
                    "confidence":    round(t.confidence, 3),
                    "anomaly_score": round(t.anomaly_score, 3),
                    "suspicious":    t.anomaly_score > 0.5,
                }
                for t in pf.tracks
            ],
        }

    def stop(self) -> None:
        self._running = False
