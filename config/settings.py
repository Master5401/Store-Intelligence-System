"""
config/settings.py
──────────────────
Single source of truth for all configuration. Values are read from environment
variables (prefixed SIS_) or the .env file.  Every module imports `settings`
from here; nothing else touches os.environ directly.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Identity ────────────────────────────────────────────────────────────
    store_id: str = Field(default="STORE_001")
    camera_id: str = Field(default="CAM_001")

    # ── Video Source ─────────────────────────────────────────────────────────
    rtsp_url: str = Field(default="0", description="RTSP URL, video file path, or '0' for webcam")

    # ── Motion Detection ─────────────────────────────────────────────────────
    motion_threshold: float = Field(default=0.008)
    diff_threshold: float = Field(default=25.0)

    # ── VSUMM ────────────────────────────────────────────────────────────────
    vsumm_clusters: int = Field(default=5, ge=1, le=30)
    vsumm_min_frames: int = Field(default=8, ge=3)

    # ── YOLO ─────────────────────────────────────────────────────────────────
    yolo_model: str = Field(default="yolov8n.pt")
    yolo_pose_model: str = Field(default="yolov8n-pose.pt")
    detection_confidence: float = Field(default=0.25, ge=0.01, le=1.0)

    # ── ByteTrack ─────────────────────────────────────────────────────────────
    track_high_thresh: float = Field(default=0.25)
    track_low_thresh: float = Field(default=0.1)
    new_track_thresh: float = Field(default=0.25)
    track_buffer: int = Field(default=30)
    match_thresh: float = Field(default=0.8)

    # ── Anomaly ───────────────────────────────────────────────────────────────
    anomaly_threshold: float = Field(default=2.8, description="Z-score threshold for pose anomaly")
    pose_window_size: int = Field(default=30)
    adaptation_interval: int = Field(default=1800, description="Seconds between model adaptation cycles")

    # ── Storage ───────────────────────────────────────────────────────────────
    database_url: str = Field(default="sqlite+aiosqlite:///./store_intelligence.db")
    keyframes_dir: str = Field(default="./keyframes")

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    jwt_secret: str = Field(default="CHANGE_ME_IN_PRODUCTION")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expire_minutes: int = Field(default=60)

    # ── Kafka ─────────────────────────────────────────────────────────────────
    kafka_enabled: bool = Field(default=False)
    kafka_bootstrap_servers: str = Field(default="localhost:9092")

    # ── Twilio ────────────────────────────────────────────────────────────────
    twilio_account_sid: Optional[str] = Field(default=None)
    twilio_auth_token: Optional[str] = Field(default=None)
    twilio_from_number: Optional[str] = Field(default=None)
    alert_phone_number: Optional[str] = Field(default=None)

    # ── Processing ────────────────────────────────────────────────────────────
    fps_target: int = Field(default=10, ge=1, le=60)
    heatmap_decay: float = Field(default=0.995, ge=0.9, le=1.0)

    @field_validator("rtsp_url")
    @classmethod
    def coerce_webcam_index(cls, v: str) -> str:
        """Keep "0" as a string so VideoCapture handles it uniformly."""
        return v.strip()

    class Config:
        env_file = ".env"
        env_prefix = "SIS_"
        case_sensitive = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Module-level singleton ─────────────────────────────────────────────────────
settings = get_settings()
