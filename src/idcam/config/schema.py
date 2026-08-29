"""Configuration schema and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

# First-install production template values. These are always rejected so the
# service cannot look "ready" while still pointing at a placeholder endpoint.
PLACEHOLDER_DEVICE_IDS = frozenset({"change_me", "changeme", "replace_me", "todo"})
PLACEHOLDER_SERVER_HOSTS = frozenset({"unconfigured.invalid"})
EXAMPLE_SERVER_HOSTS = frozenset({"example.com", "api.example.com"})


class ServerConfig(BaseModel):
    url: HttpUrl
    path: str = "/api/document"
    connect_timeout_sec: float = Field(10.0, gt=0, le=120)
    read_timeout_sec: float = Field(60.0, gt=0, le=600)
    total_timeout_sec: float = Field(90.0, gt=0, le=900)
    verify_tls: bool = True
    max_upload_bytes: int = Field(50_000_000, gt=0)

    @field_validator("path")
    @classmethod
    def normalize_path(cls, value: str) -> str:
        if not value.startswith("/"):
            return f"/{value}"
        return value

    @field_validator("url")
    @classmethod
    def reject_placeholder_url(cls, value: HttpUrl) -> HttpUrl:
        host = (value.host or "").lower().rstrip(".")
        if host in PLACEHOLDER_SERVER_HOSTS:
            raise ValueError(
                "server.url is not configured. Set the real HTTPS API URL in "
                "/etc/idcam/config.yaml or IDCAM_SERVER_URL."
            )
        return value


class AuthConfig(BaseModel):
    type: Literal["bearer", "none"] = "none"
    token: str = ""

    def resolved_token(self) -> str | None:
        if self.type == "none" or not self.token:
            return None
        return self.token


class CameraRecoveryConfig(BaseModel):
    max_attempts: int = Field(3, ge=1, le=10)
    delay_seconds: float = Field(2.0, gt=0, le=60)


class CameraConfig(BaseModel):
    preview_width: int = Field(1920, ge=320, le=3840)
    preview_height: int = Field(1080, ge=240, le=2160)
    capture_width: int = Field(9152, ge=640)
    capture_height: int = Field(6944, ge=480)
    jpeg_quality: int = Field(95, ge=50, le=100)
    preview_fps: float = Field(5.0, gt=0, le=30)
    af_max_retries: int = Field(3, ge=1, le=10)
    af_timeout_sec: float = Field(10.0, gt=0, le=120)
    af_verify_max_retries: int = Field(2, ge=1, le=5)
    # Production AF search range (dioptres). Hardware calibration: sharp ~8.9.
    af_min_lens_position: float = Field(8.7, ge=0.0, le=32.0)
    af_max_lens_position: float = Field(9.2, ge=0.0, le=32.0)
    af_lens_search_step: float = Field(0.05, gt=0.0, le=1.0)
    capture_timeout_sec: float = Field(45.0, gt=0, le=300)
    recovery: CameraRecoveryConfig = CameraRecoveryConfig()

    @model_validator(mode="after")
    def _af_search_range_order(self) -> CameraConfig:
        if self.af_max_lens_position < self.af_min_lens_position:
            raise ValueError("af_max_lens_position must be >= af_min_lens_position")
        return self

    @property
    def recovery_max_attempts(self) -> int:
        return self.recovery.max_attempts

    @property
    def recovery_delay_sec(self) -> float:
        return self.recovery.delay_seconds


class DetectionConfig(BaseModel):
    roi_center_fraction: float = Field(0.7, gt=0.1, le=1.0)
    motion_threshold: float = Field(8.0, gt=0)
    edge_density_threshold: float = Field(0.02, gt=0, le=1.0)
    stable_frames_required: int = Field(8, ge=2, le=60)
    removal_frames_required: int = Field(10, ge=2, le=120)
    removal_motion_factor: float = Field(0.5, gt=0, le=1.0)
    removal_edge_factor: float = Field(0.5, gt=0, le=2.0)
    baseline_alpha: float = Field(0.05, gt=0, le=1.0)
    # Geometry detector (Phase 2B). Independent of motion_threshold.
    geometry_hold_frames: int = Field(3, ge=1, le=30)
    geometry_smooth_alpha: float = Field(0.40, gt=0, le=1.0)
    # Geometry stability gate (Phase 2C). Preview pixels, not motion_threshold.
    # Mean TL/TR/BR/BL corner travel between consecutive detect_geometry results.
    # At 1920×1080 / ~5 Hz, 12 px is a small page shift, not a place-down sweep.
    geometry_stability_threshold_px: float = Field(12.0, gt=0, le=200)
    # Consecutive low-movement geometry frames before auto-capture may start.
    # At preview_fps=5 this is 1.0 s after the first present observation.
    geometry_stable_frames_required: int = Field(5, ge=2, le=60)


class QualityConfig(BaseModel):
    min_sharpness: float = Field(80.0, gt=0)
    min_file_bytes: int = Field(100_000, gt=0)
    max_file_bytes: int = Field(50_000_000, gt=0)
    min_brightness: float = Field(30.0, ge=0, le=255)
    max_brightness: float = Field(225.0, ge=0, le=255)
    min_width: int = Field(640, gt=0)
    min_height: int = Field(480, gt=0)
    require_expected_dimensions: bool = True
    dimension_tolerance: int = Field(64, ge=0, le=512)


class StorageConfig(BaseModel):
    min_free_mb: int = Field(1024, ge=100, le=100_000)
    stale_temp_max_age_sec: float = Field(3600.0, gt=60, le=86400)
    capture_reserve_mb: int = Field(200, ge=50, le=5000)


class StateTimeoutsConfig(BaseModel):
    """Per-state timeout configuration (seconds). None-equivalent = very large."""

    boot: float = Field(10.0, gt=0, le=120)
    initializing: float = Field(60.0, gt=0, le=300)
    camera_initializing: float = Field(120.0, gt=0, le=600)
    document_detected: float = Field(5.0, gt=0, le=60)
    waiting_for_stability: float = Field(15.0, gt=0, le=120)
    focusing: float = Field(10.0, gt=0, le=120)
    focus_verification: float = Field(5.0, gt=0, le=60)
    capturing: float = Field(45.0, gt=0, le=300)
    image_validation: float = Field(10.0, gt=0, le=60)
    uploading: float = Field(60.0, gt=0, le=600)
    waiting_for_server: float = Field(60.0, gt=0, le=600)
    processing_response: float = Field(10.0, gt=0, le=60)
    success: float = Field(5.0, gt=0, le=30)
    recovery: float = Field(180.0, gt=0, le=600)
    shutdown: float = Field(30.0, gt=0, le=120)

    def for_state(self, state_name: str) -> float | None:
        mapping = {
            "BOOT": self.boot,
            "INITIALIZING": self.initializing,
            "CAMERA_INITIALIZING": self.camera_initializing,
            "DOCUMENT_DETECTED": self.document_detected,
            "WAITING_FOR_STABILITY": self.waiting_for_stability,
            "FOCUSING": self.focusing,
            "FOCUS_VERIFICATION": self.focus_verification,
            "CAPTURING": self.capturing,
            "IMAGE_VALIDATION": self.image_validation,
            "UPLOADING": self.uploading,
            "WAITING_FOR_SERVER": self.waiting_for_server,
            "PROCESSING_RESPONSE": self.processing_response,
            "SUCCESS": self.success,
            "RECOVERY": self.recovery,
            "SHUTDOWN": self.shutdown,
        }
        return mapping.get(state_name)


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: Path = Path("/var/log/idcam")
    max_bytes: int = Field(10_485_760, gt=0)
    backup_count: int = Field(5, ge=1, le=50)
    json_format: bool = True


class UIConfig(BaseModel):
    status_file: Path = Path("/run/idcam/status.json")
    admin_socket: Path = Path("/run/idcam/admin.sock")
    refresh_interval_sec: float = Field(0.5, gt=0, le=5)


class WatchdogConfig(BaseModel):
    check_interval_sec: float = Field(1.0, gt=0, le=10)
    state_timeout_sec: float = Field(300.0, gt=0, le=3600)


class DebugConfig(BaseModel):
    """Optional Live Debug Viewer. Disabled by default — not required for capture."""

    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = Field(8088, ge=1024, le=65535)
    fps: float = Field(5.0, gt=0, le=15)
    jpeg_quality: int = Field(70, ge=30, le=100)
    max_width: int = Field(1280, ge=320, le=3840)
    overlay: bool = True

    @field_validator("host")
    @classmethod
    def strip_host(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("debug.host cannot be empty")
        return stripped


class AppConfig(BaseModel):
    device_id: str = Field(..., min_length=1, max_length=128)
    temp_dir: Path = Path("/var/lib/idcam/tmp")
    server: ServerConfig
    auth: AuthConfig = AuthConfig()
    camera: CameraConfig = CameraConfig()
    detection: DetectionConfig = DetectionConfig()
    quality: QualityConfig = QualityConfig()
    storage: StorageConfig = StorageConfig()
    state_timeouts: StateTimeoutsConfig = StateTimeoutsConfig()
    logging: LoggingConfig = LoggingConfig()
    ui: UIConfig = UIConfig()
    watchdog: WatchdogConfig = WatchdogConfig()
    debug: DebugConfig = DebugConfig()

    @field_validator("device_id")
    @classmethod
    def strip_device_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("device_id cannot be empty")
        normalized = stripped.lower().replace("-", "_")
        if normalized in PLACEHOLDER_DEVICE_IDS:
            raise ValueError(
                "device_id is not configured. Set a unique device identifier in "
                "/etc/idcam/config.yaml or IDCAM_DEVICE_ID."
            )
        return stripped
