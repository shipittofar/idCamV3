"""Picamera2 camera manager with mock fallback for development."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from idcam.config.schema import CameraConfig
from idcam.exceptions import CameraError, CameraUnavailableError
from idcam.logging.setup import get_logger

logger = get_logger("camera.manager")

try:
    from picamera2 import Picamera2
    from libcamera import controls

    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    Picamera2 = None  # type: ignore
    controls = None  # type: ignore

# libcamera AfMode / AfTrigger integer values (stable across bindings).
_AF_MODE_MANUAL = 0
_AF_MODE_AUTO = 1
_AF_MODE_CONTINUOUS = 2
_AF_TRIGGER_START = 0
_AF_STATE_FOCUSED = 2
_AF_STATE_FAILED = 3
_AF_METERING_WINDOWS = 1
_AF_RANGE_FULL = 2

# Preview-pixel AfWindows source must be large enough to map to a real sensor window.
MIN_PREVIEW_AF_SIDE = 8

# Passed explicitly to every still encoder so the container is never inferred
# from the output filename.
STILL_ENCODER_FORMAT = "jpeg"


def describe_preview_frame(frame: np.ndarray) -> dict[str, Any]:
    """Return shape/dtype/channel statistics for diagnostics. Does not mutate ``frame``."""
    info: dict[str, Any] = {
        "shape": [int(x) for x in frame.shape],
        "dtype": str(frame.dtype),
        "ndim": int(frame.ndim),
    }
    if frame.ndim == 3 and frame.shape[-1] >= 3:
        means = frame.mean(axis=(0, 1))
        info["channel_means"] = {
            "c0": round(float(means[0]), 2),
            "c1": round(float(means[1]), 2),
            "c2": round(float(means[2]), 2),
        }
    elif frame.ndim == 2:
        info["channel_means"] = {"c0": round(float(frame.mean()), 2)}
    return info


def parse_xywh(value: Any) -> tuple[int, int, int, int] | None:
    """Normalize libcamera Rectangle / tuple to (x, y, width, height)."""
    if value is None:
        return None
    if hasattr(value, "x") and hasattr(value, "width"):
        return int(value.x), int(value.y), int(value.width), int(value.height)
    try:
        if len(value) >= 4:
            return int(value[0]), int(value[1]), int(value[2]), int(value[3])
    except TypeError:
        return None
    return None


def map_preview_roi_to_sensor(
    preview_roi: tuple[int, int, int, int],
    preview_size: tuple[int, int],
    scaler_crop: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Map a preview-pixel ROI into AfWindows / ScalerCrop coordinates.

    Preview is typically 1920x1080 (16:9). The sensor/ScalerCropMaximum is
    ~9152x6944 (≈4:3). Taking 70% of ScalerCropMaximum independently does
    **not** match 70% of the preview. Use the live ScalerCrop of the current
    stream so AF and Laplacian see the same document region.
    """
    px, py, pw, ph = (int(v) for v in preview_roi)
    fw, fh = int(preview_size[0]), int(preview_size[1])
    sx, sy, sw, sh = (int(v) for v in scaler_crop)
    if fw <= 0 or fh <= 0 or sw <= 0 or sh <= 0:
        return sx, sy, sw, sh
    return (
        sx + int(round(px * sw / fw)),
        sy + int(round(py * sh / fh)),
        max(1, int(round(pw * sw / fw))),
        max(1, int(round(ph * sh / fh))),
    )


def clip_preview_roi(
    roi: tuple[int, int, int, int],
    preview_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Clip an (x, y, w, h) ROI to preview pixel bounds (width, height)."""
    fw, fh = int(preview_size[0]), int(preview_size[1])
    x, y, rw, rh = (int(v) for v in roi)
    if fw <= 0 or fh <= 0:
        return 0, 0, 1, 1
    x = max(0, min(x, max(0, fw - 1)))
    y = max(0, min(y, max(0, fh - 1)))
    rw = max(1, min(rw, fw - x))
    rh = max(1, min(rh, fh - y))
    return x, y, rw, rh


def clip_xywh_to_bounds(
    roi: tuple[int, int, int, int],
    bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Clip ``roi`` so it lies entirely inside ``bounds`` (both xywh)."""
    bx, by, bw, bh = (int(v) for v in bounds)
    x, y, rw, rh = (int(v) for v in roi)
    x2 = min(x + rw, bx + bw)
    y2 = min(y + rh, by + bh)
    x = max(x, bx)
    y = max(y, by)
    rw = x2 - x
    rh = y2 - y
    if rw < 1 or rh < 1:
        raise ValueError("af_window_empty")
    return int(x), int(y), int(rw), int(rh)


def preview_roi_is_usable(
    roi: tuple[int, int, int, int] | None,
    preview_size: tuple[int, int],
    *,
    min_side: int = MIN_PREVIEW_AF_SIDE,
) -> bool:
    """True when ``roi`` is a finite rectangle that clips to a usable preview window."""
    if roi is None:
        return False
    try:
        x, y, rw, rh = (int(v) for v in roi)
    except (TypeError, ValueError):
        return False
    if rw <= 0 or rh <= 0:
        return False
    fw, fh = int(preview_size[0]), int(preview_size[1])
    if fw <= 0 or fh <= 0:
        return False
    clipped = clip_preview_roi((x, y, rw, rh), preview_size)
    return clipped[2] >= min_side and clipped[3] >= min_side


def preview_roi_to_af_window(
    preview_roi: tuple[int, int, int, int],
    preview_size: tuple[int, int],
    scaler_crop: tuple[int, int, int, int],
) -> dict[str, tuple[int, int, int, int]]:
    """Convert a preview-pixel document bbox to AfWindows (ScalerCrop) coordinates.

    Raises ValueError for empty, inverted, or unusable geometry. Never substitutes
    a centered search rectangle.
    """
    fw, fh = int(preview_size[0]), int(preview_size[1])
    if fw <= 0 or fh <= 0:
        raise ValueError("invalid_preview_size")
    if not preview_roi_is_usable(preview_roi, preview_size):
        raise ValueError("invalid_roi")
    clipped = clip_preview_roi(tuple(int(v) for v in preview_roi), preview_size)
    mapped = map_preview_roi_to_sensor(clipped, preview_size, scaler_crop)
    sensor = clip_xywh_to_bounds(mapped, scaler_crop)
    return {
        "preview_roi": clipped,
        "sensor_roi": sensor,
        "scaler_crop": tuple(int(v) for v in scaler_crop),
    }


def normalize_picamera_preview_frame(frame: np.ndarray) -> np.ndarray:
    """Convert a Picamera2 RGB888/XRGB8888 ndarray to RGB channel order.

    Picamera2 requests libcamera/DRM ``RGB888``, which is B,G,R in memory.
    Picamera2's own JPEG encoder documents this as ``FORMAT_TABLE['RGB888'] = 'BGR'``.
    idCam preview consumers (detector, sharpness, debug viewer) assume RGB.
    """
    if frame.ndim == 3 and frame.shape[-1] == 3:
        return np.ascontiguousarray(frame[..., ::-1])
    if frame.ndim == 3 and frame.shape[-1] == 4:
        return np.ascontiguousarray(frame[..., 2::-1])
    return frame


class CameraManager:
    """
    Manages Picamera2 lifecycle: preview stream and full-res capture.

  Thread-safe via lock. All public methods may be called from executor threads.
    """

    def __init__(self, config: CameraConfig, *, use_mock: bool = False) -> None:
        self.config = config
        self._use_mock = use_mock or not PICAMERA2_AVAILABLE
        self._lock = threading.RLock()
        self._picam2: Any = None
        self._started = False
        self._preview_config: Any = None
        self._still_config: Any = None
        self._logged_preview_format = False
        self._preview_af_mode: str | None = None

    @property
    def is_mock(self) -> bool:
        return self._use_mock

    @property
    def is_started(self) -> bool:
        return self._started

    def initialize(self) -> None:
        with self._lock:
            if self._started:
                return
            if self._use_mock:
                logger.warning("Using mock camera (Picamera2 not available)")
                self._started = True
                return

            try:
                self._picam2 = Picamera2()
                self._preview_config = self._picam2.create_preview_configuration(
                    main={"size": (self.config.preview_width, self.config.preview_height), "format": "RGB888"},
                    buffer_count=4,
                )
                # One buffer is enough for one-shot stills. At 9152x6944 RGB888 a
                # single buffer is ~182 MB of CMA, so a second one doubles the
                # allocation for no benefit and can fail outright on a Pi 5.
                self._still_config = self._picam2.create_still_configuration(
                    main={
                        "size": (self.config.capture_width, self.config.capture_height),
                        "format": "RGB888",
                    },
                    buffer_count=1,
                )
                self._picam2.configure(self._preview_config)
                self._picam2.start()
                self._started = True
                self._logged_preview_format = False
                logger.info(
                    "event=camera_initialized requested_preview=%dx%d requested_capture=%dx%d "
                    "configured=%s",
                    self.config.preview_width,
                    self.config.preview_height,
                    self.config.capture_width,
                    self.config.capture_height,
                    self._stream_geometry_unlocked(),
                )
                try:
                    self._enable_preview_autofocus(wait=False)
                except Exception as exc:
                    logger.warning("event=autofocus_preview_failed error=%s", exc)
            except Exception as exc:
                # Picamera2() may already hold the device even when configure()
                # or start() failed. Dropping the reference without close()
                # leaves /dev/media* busy, so every later restart fails with
                # "camera is already in use" and recovery can never succeed.
                self._release_device_unlocked()
                self._started = False
                raise CameraUnavailableError(f"Camera initialization failed: {exc}") from exc

    def _release_device_unlocked(self) -> None:
        """Stop and close the Picamera2 handle. Safe to call in any state."""
        picam2, self._picam2 = self._picam2, None
        if picam2 is None:
            return
        try:
            picam2.stop()
        except Exception as exc:
            logger.debug("event=camera_stop_failed error=%s", exc)
        try:
            picam2.close()
        except Exception as exc:
            logger.error("event=camera_close_failed error=%s", exc)

    def shutdown(self) -> None:
        with self._lock:
            # Release unconditionally: a failed initialize() can leave a live
            # handle behind while _started is False.
            had_device = self._picam2 is not None
            was_started = self._started
            self._release_device_unlocked()
            self._started = False
            self._logged_preview_format = False
            self._preview_af_mode = None
            logger.info(
                "event=camera_shutdown was_started=%s device_released=%s", was_started, had_device
            )

    def restart(self) -> None:
        """Atomic stop/start so no other caller observes a half-open camera."""
        with self._lock:
            logger.info("event=camera_restart_started")
            self.shutdown()
            # libcamera needs the media device to be fully released before it
            # can be reacquired; this is a device settle, not a retry padding.
            time.sleep(1.0)
            self.initialize()
            logger.info("event=camera_restart_completed")

    def capture_preview_array(self) -> np.ndarray:
        with self._lock:
            if not self._started:
                raise CameraError("Camera not started")
            if self._use_mock:
                return self._mock_frame()
            assert self._picam2 is not None
            native = self._picam2.capture_array("main")
            frame = normalize_picamera_preview_frame(native)
            if not self._logged_preview_format:
                self._log_preview_format(native, frame)
                self._logged_preview_format = True
            return frame

    def capture_still_to_file(self, path: Path) -> None:
        """Capture one full-resolution JPEG to ``path``.

        The encoder format is always passed explicitly, so ``path`` may carry
        any extension (including in-progress names such as ``*.partial.jpg``)
        without the container being guessed from the filename.
        """
        with self._lock:
            if not self._started:
                raise CameraError("Camera not started")
            if self._use_mock:
                self._mock_still(path)
                self._verify_still_written(path)
                return

            assert self._picam2 is not None
            lens_position = self._read_lens_position()
            try:
                self._picam2.stop()
                self._picam2.configure(self._still_config)
                self._picam2.start()
                if lens_position is not None:
                    self._set_controls_unlocked(
                        {
                            "AfMode": self._af_mode_manual(),
                            "LensPosition": float(lens_position),
                        }
                    )
                    logger.info(
                        "event=autofocus_result method=hold_lens_position lens_position=%s",
                        lens_position,
                    )
                # Block until one frame has actually come through the new still
                # configuration, so the JPEG is not encoded from the first
                # frame of a mode switch that has not settled yet.
                try:
                    self._picam2.capture_metadata()
                except Exception:
                    logger.debug("event=still_pre_capture_metadata_failed", exc_info=True)
                self._picam2.capture_file(str(path), format=STILL_ENCODER_FORMAT)
            finally:
                try:
                    self._picam2.stop()
                    self._picam2.configure(self._preview_config)
                    self._picam2.start()
                    try:
                        self._enable_preview_autofocus(wait=False)
                    except Exception as exc:
                        logger.warning("event=autofocus_preview_failed error=%s", exc)
                except Exception as exc:
                    logger.error("Failed to restore preview mode: %s", exc)
                    raise CameraError(f"Failed to restore preview after capture: {exc}") from exc
            self._verify_still_written(path)

    @staticmethod
    def _verify_still_written(path: Path) -> None:
        """Post-condition: the encoder produced a real, non-empty file."""
        if not path.exists():
            raise CameraError(f"Capture reported success but no file was written: {path}")
        size = path.stat().st_size
        if size == 0:
            raise CameraError(f"Capture wrote an empty file: {path}")
        logger.info("event=still_written path=%s bytes=%d", path, size)

    def get_controls(self) -> dict[str, Any]:
        """Return advertised control *capabilities* (typically min/max/default), not live values."""
        with self._lock:
            if self._use_mock:
                return {"AfMode": (0, 2, 0), "AfTrigger": (0, 1, 0), "LensPosition": (0.0, 10.0, 1.0)}
            assert self._picam2 is not None
            return dict(self._picam2.camera_controls)

    def capture_metadata(self) -> dict[str, Any]:
        """Return live per-frame metadata (AfState, LensPosition, etc.)."""
        with self._lock:
            if self._use_mock:
                return {"AfMode": 1, "AfState": 2, "AfTrigger": 0, "LensPosition": 1.0}
            assert self._picam2 is not None
            return dict(self._picam2.capture_metadata())

    def set_controls(self, control_dict: dict[str, Any]) -> None:
        with self._lock:
            self._set_controls_unlocked(control_dict)

    def _set_controls_unlocked(self, control_dict: dict[str, Any]) -> None:
        if self._use_mock:
            return
        assert self._picam2 is not None
        advertised = set(self._picam2.camera_controls.keys())
        filtered = {k: v for k, v in control_dict.items() if k in advertised}
        if not filtered:
            logger.warning("None of controls %s are advertised", list(control_dict.keys()))
            return
        self._picam2.set_controls(filtered)

    def stream_geometry(self) -> dict[str, Any]:
        """Actually configured preview/still geometry.

        libcamera aligns requested sizes to what the sensor and ISP can do, so
        the configured stream can differ from the requested one. Preview->sensor
        ROI mapping and capture dimension checks must use these numbers, not the
        values from config.
        """
        with self._lock:
            return self._stream_geometry_unlocked()

    def _stream_geometry_unlocked(self) -> dict[str, Any]:
        requested = {
            "preview": (int(self.config.preview_width), int(self.config.preview_height)),
            "still": (int(self.config.capture_width), int(self.config.capture_height)),
        }
        if self._use_mock or self._picam2 is None:
            return {
                "requested": requested,
                "source": "mock" if self._use_mock else "not_initialized",
                "preview": requested["preview"],
                "still": requested["still"],
            }

        def main_stream(cfg: Any) -> dict[str, Any]:
            try:
                main = cfg["main"] if cfg is not None else None
            except (TypeError, KeyError):
                main = None
            if not isinstance(main, dict):
                return {}
            size = main.get("size")
            return {
                "size": tuple(int(v) for v in size) if size else None,
                "format": main.get("format"),
                "stride": main.get("stride"),
            }

        preview = main_stream(self._preview_config)
        still = main_stream(self._still_config)
        return {
            "requested": requested,
            # "configured" means the values libcamera wrote back during
            # configure(); "requested" means we could not read them and are
            # echoing the request, which must not be mistaken for confirmation.
            "source": "configured" if preview.get("size") and still.get("size") else "requested",
            "preview": preview.get("size") or requested["preview"],
            "preview_format": preview.get("format"),
            "preview_stride": preview.get("stride"),
            "still": still.get("size") or requested["still"],
            "still_format": still.get("format"),
        }

    def current_scaler_crop(self) -> tuple[int, int, int, int]:
        """Live stream crop on the sensor, else ScalerCropMaximum, else preview size."""
        with self._lock:
            return self._current_scaler_crop_unlocked()

    def _current_scaler_crop_unlocked(self) -> tuple[int, int, int, int]:
        fallback = (0, 0, int(self.config.preview_width), int(self.config.preview_height))
        if self._use_mock or self._picam2 is None:
            return fallback
        try:
            meta = dict(self._picam2.capture_metadata())
        except Exception:
            meta = {}
        crop = parse_xywh(meta.get("ScalerCrop"))
        if crop is None:
            props = getattr(self._picam2, "camera_properties", None) or {}
            crop = parse_xywh(props.get("ScalerCropMaximum"))
        return crop or fallback

    def apply_document_focus_window(
        self,
        preview_roi: tuple[int, int, int, int],
        preview_size: tuple[int, int],
    ) -> dict[str, Any]:
        """Point hardware AF at a preview-pixel ROI mapped through live ScalerCrop."""
        with self._lock:
            fw, fh = int(preview_size[0]), int(preview_size[1])
            scaler = self._current_scaler_crop_unlocked()
            try:
                mapped = preview_roi_to_af_window(preview_roi, (fw, fh), scaler)
            except ValueError as exc:
                raise CameraError(f"Invalid document AF ROI: {exc}") from exc
            clipped = mapped["preview_roi"]
            sensor = mapped["sensor_roi"]
            if not self._use_mock and self._picam2 is not None:
                x, y, w, h = sensor
                self._set_controls_unlocked(
                    {
                        "AfMetering": self._af_metering_windows(),
                        "AfWindows": [(int(x), int(y), int(w), int(h))],
                        "AfRange": self._af_range_full(),
                    }
                )
            logger.info(
                "event=autofocus_roi preview=%d,%d,%d,%d preview_size=%dx%d "
                "scaler_crop=%d,%d,%d,%d sensor=%d,%d,%d,%d",
                clipped[0],
                clipped[1],
                clipped[2],
                clipped[3],
                preview_size[0],
                preview_size[1],
                *scaler,
                *sensor,
            )
            return {
                "preview_roi": clipped,
                "preview_size": (fw, fh),
                "scaler_crop": scaler,
                "sensor_roi": sensor,
            }

    def trigger_autofocus(self) -> None:
        """Start a one-shot Auto cycle. Do not wait on AfState (often unavailable)."""
        with self._lock:
            if self._use_mock:
                return
            assert self._picam2 is not None
            # Do not call picam2.autofocus_cycle(): with AfState unadvertised it
            # often returns immediately, and a blocking wait holds the camera lock.
            self._set_controls_unlocked(
                {
                    "AfMode": self._af_mode_auto(),
                    "AfSpeed": self._af_speed_fast(),
                    "AfRange": self._af_range_full(),
                    "AfTrigger": self._af_trigger_start(),
                }
            )
            logger.info("event=autofocus_triggered method=af_controls AfMode=Auto AfTrigger=Start wait=lens_settle")

    def wait_for_lens_settle(self, timeout_sec: float) -> float | None:
        """Wait until LensPosition stops changing. AfState is not required."""
        if self._use_mock:
            time.sleep(0.05)
            return 1.0
        deadline = time.monotonic() + max(0.4, float(timeout_sec))
        time.sleep(0.25)
        last: float | None = None
        stable = 0
        while time.monotonic() < deadline:
            meta = self.capture_metadata()
            raw = meta.get("LensPosition")
            try:
                lens = float(raw) if raw is not None else None
            except (TypeError, ValueError):
                lens = None
            if lens is None:
                time.sleep(0.1)
                continue
            if last is not None and abs(lens - last) < 0.05:
                stable += 1
                if stable >= 3:
                    logger.info("event=autofocus_result method=lens_settle lens_position=%s", lens)
                    return lens
            else:
                stable = 0
            last = lens
            time.sleep(0.1)
        logger.info("event=autofocus_result method=lens_settle_timeout lens_position=%s", last)
        return last

    def autofocus_cycle(self) -> bool:
        """Trigger AF and wait for the lens to settle (not AfState)."""
        self.trigger_autofocus()
        self.wait_for_lens_settle(min(float(self.config.af_timeout_sec), 3.0))
        return True

    def pause_autofocus(self) -> None:
        """Lock the current lens position so Continuous AF does not hunt during still capture."""
        with self._lock:
            if self._use_mock or self._picam2 is None:
                return
            advertised = self._picam2.camera_controls
            if "AfPause" in advertised:
                self._set_controls_unlocked({"AfPause": 0})
                logger.info("event=lens_lock method=af_pause")
            elif "AfMode" in advertised and "LensPosition" in advertised:
                lens = self._read_lens_position()
                if lens is not None:
                    self._set_controls_unlocked({"AfMode": self._af_mode_manual(), "LensPosition": lens})
                    logger.info("event=lens_lock method=hold_lens_position lens_position=%s", lens)
                else:
                    logger.warning("event=lens_lock success=0 reason=lens_position_unavailable")
            else:
                logger.warning("event=lens_lock success=0 reason=no_pause_or_manual_controls")

    def _af_enum(self, group: str, name: str, fallback: int) -> int:
        if controls is None:
            return fallback
        enum_group = getattr(controls, group, None)
        if enum_group is None:
            return fallback
        value = getattr(enum_group, name, None)
        return fallback if value is None else int(value)

    def _af_mode_manual(self) -> int:
        return self._af_enum("AfModeEnum", "Manual", _AF_MODE_MANUAL)

    def _af_mode_auto(self) -> int:
        return self._af_enum("AfModeEnum", "Auto", _AF_MODE_AUTO)

    def _af_mode_continuous(self) -> int:
        return self._af_enum("AfModeEnum", "Continuous", _AF_MODE_CONTINUOUS)

    def _af_trigger_start(self) -> int:
        return self._af_enum("AfTriggerEnum", "Start", _AF_TRIGGER_START)

    def _af_speed_fast(self) -> int:
        return self._af_enum("AfSpeedEnum", "Fast", 1)

    def _af_metering_windows(self) -> int:
        return self._af_enum("AfMeteringEnum", "Windows", _AF_METERING_WINDOWS)

    def _af_range_full(self) -> int:
        return self._af_enum("AfRangeEnum", "Full", _AF_RANGE_FULL)

    def _af_mode_max(self) -> int | None:
        if self._picam2 is None:
            return None
        info = self._picam2.camera_controls.get("AfMode")
        if info is None:
            return None
        try:
            return int(info[1])
        except (TypeError, IndexError, ValueError):
            return self._af_mode_continuous()

    def _read_lens_position(self) -> float | None:
        if self._picam2 is None:
            return None
        try:
            meta = dict(self._picam2.capture_metadata())
        except Exception:
            return None
        value = meta.get("LensPosition")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _enable_preview_autofocus(self, *, wait: bool = True) -> None:
        """Start real AF on the preview stream (Continuous, else one-shot Auto)."""
        if self._use_mock or self._picam2 is None:
            return
        advertised = self._picam2.camera_controls
        if "AfMode" not in advertised:
            logger.info("event=autofocus_preview_skipped reason=AfMode_not_advertised")
            return

        continuous = self._af_mode_continuous()
        max_mode = self._af_mode_max()
        if max_mode is not None and max_mode >= continuous:
            self.set_controls({"AfMode": continuous, "AfSpeed": self._af_speed_fast()})
            mode_name = "Continuous"
            logger.info("event=autofocus_triggered mode=%s speed=Fast", mode_name)
        else:
            self.set_controls(
                {
                    "AfMode": self._af_mode_auto(),
                    "AfTrigger": self._af_trigger_start(),
                    "AfSpeed": self._af_speed_fast(),
                }
            )
            mode_name = "Auto"
            logger.info("event=autofocus_triggered mode=%s trigger=Start", mode_name)
        self._preview_af_mode = mode_name
        if not wait:
            return

        deadline = time.monotonic() + min(float(self.config.af_timeout_sec), 3.0)
        last_state: Any = None
        last_lens: Any = None
        while time.monotonic() < deadline:
            try:
                meta = dict(self._picam2.capture_metadata())
            except Exception as exc:
                logger.debug("event=autofocus_metadata_unavailable error=%s", exc)
                break
            last_state = meta.get("AfState")
            last_lens = meta.get("LensPosition")
            if last_state is None:
                logger.info(
                    "event=autofocus_result mode=%s af_state=unavailable lens_position=%s",
                    mode_name,
                    last_lens,
                )
                return
            if int(last_state) in (_AF_STATE_FOCUSED, _AF_STATE_FAILED):
                logger.info(
                    "event=autofocus_result mode=%s af_state=%s lens_position=%s",
                    mode_name,
                    last_state,
                    last_lens,
                )
                return
            time.sleep(0.05)
        logger.warning(
            "event=autofocus_timeout mode=%s last_af_state=%s lens_position=%s",
            mode_name,
            last_state,
            last_lens,
        )

    def _log_preview_format(self, native: np.ndarray, rgb: np.ndarray) -> None:
        native_info = describe_preview_frame(native)
        rgb_info = describe_preview_frame(rgb)
        logger.info(
            "event=preview_frame_format requested=RGB888 "
            "native_order=BGR native_shape=%s native_dtype=%s native_means=%s "
            "rgb_shape=%s rgb_means=%s",
            native_info.get("shape"),
            native_info.get("dtype"),
            native_info.get("channel_means"),
            rgb_info.get("shape"),
            rgb_info.get("channel_means"),
        )

    def _mock_frame(self) -> np.ndarray:
        h, w = self.config.preview_height, self.config.preview_width
        frame = np.full((h, w, 3), 200, dtype=np.uint8)
        # Add some edges in center for detection tests
        ch, cw = h // 3, w // 3
        y0, x0 = h // 3, w // 3
        frame[y0 : y0 + ch, x0 : x0 + 10] = 50
        frame[y0 : y0 + 10, x0 : x0 + cw] = 50
        return frame

    def _mock_still(self, path: Path) -> None:
        from PIL import Image

        img = Image.new("RGB", (self.config.capture_width, self.config.capture_height), color=(220, 220, 220))
        # Draw document-like rectangle
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        margin = 200
        draw.rectangle(
            [margin, margin, self.config.capture_width - margin, self.config.capture_height - margin],
            outline=(30, 30, 30),
            width=20,
        )
        img.save(path, format="JPEG", quality=self.config.jpeg_quality)
