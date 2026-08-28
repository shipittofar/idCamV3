"""Camera autofocus workflow.

Hardware AF is aimed at a **preview-pixel** document ROI mapped through live
ScalerCrop. Sharpness is measured on that same preview ROI. Success is relative
improvement on the 1080p stream (typical focused Laplacian ~10–40), not the
full-resolution still-image floor of 48/80.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Protocol

from idcam.camera.manager import preview_roi_is_usable
from idcam.config.schema import CameraConfig
from idcam.exceptions import AutofocusError
from idcam.logging.setup import get_logger
from idcam.vision.sharpness import accept_preview_focus, clip_roi, laplacian_variance, roi_rect

logger = get_logger("camera.autofocus")

AF_STATE_IDLE = 0
AF_STATE_SCANNING = 1
AF_STATE_FOCUSED = 2
AF_STATE_FAILED = 3

# Preview Laplacian of a focused 1080p document is typically ~10–40.
# Full-res still validation keeps min_sharpness=80; do not reuse 0.6*80=48 here.
PREVIEW_ABSOLUTE_FLOOR = 28.0


class FocusCapableCamera(Protocol):
    def get_controls(self) -> dict[str, Any]: ...
    def set_controls(self, controls: dict[str, Any]) -> None: ...
    def autofocus_cycle(self) -> Any: ...
    def capture_preview_array(self) -> Any: ...
    def capture_metadata(self) -> dict[str, Any]: ...


@dataclass
class AutofocusCapabilities:
    af_mode: bool = False
    af_trigger: bool = False
    af_state: bool = False
    lens_position: bool = False
    autofocus_cycle: bool = False
    af_windows: bool = False
    af_range: bool = False
    af_metering: bool = False

    @property
    def hardware_af_available(self) -> bool:
        return self.af_mode or self.autofocus_cycle

    @property
    def degraded(self) -> bool:
        """True when only sharpness-based verification is available."""
        return not self.hardware_af_available

    def to_dict(self) -> dict[str, bool]:
        return {
            "AfMode": self.af_mode,
            "AfTrigger": self.af_trigger,
            "AfState": self.af_state,
            "LensPosition": self.lens_position,
            "AfWindows": self.af_windows,
            "AfRange": self.af_range,
            "AfMetering": self.af_metering,
            "autofocus_cycle": self.autofocus_cycle,
            "hardware_af_available": self.hardware_af_available,
            "degraded": self.degraded,
        }


@dataclass
class AutofocusResult:
    success: bool
    sharpness: float
    attempts: int
    af_state: int | None
    method: str
    min_required_sharpness: float = 0.0
    before_sharpness: float = 0.0
    improved: bool = False
    roi: tuple[int, int, int, int] = (0, 0, 0, 0)
    lens_before: float | None = None
    lens_after: float | None = None
    improvement_ratio: float = 0.0
    sensor_roi: tuple[int, int, int, int] = (0, 0, 0, 0)
    preview_size: tuple[int, int] = (0, 0)
    duration_ms: float = 0.0
    lens_locked: bool = False


@dataclass
class AutofocusDebug:
    running: bool = False
    method: str = ""
    attempt: int = 0
    focus_before: float = 0.0
    focus_after: float = 0.0
    focus_roi: str = ""
    preview_roi: str = ""
    sensor_roi: str = ""
    preview_roi_wh: str = ""
    lens_before: float | None = None
    lens_after: float | None = None
    improvement_ratio: float = 0.0
    scaler_crop: str = ""
    preview_size: str = ""


def _fmt_roi(roi: tuple[int, int, int, int] | None) -> str:
    if not roi:
        return ""
    return "{},{},{},{}".format(*(int(v) for v in roi))


class AutofocusController:
    """Manages autofocus with API probing and document-ROI sharpness verification."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.capabilities = AutofocusCapabilities()
        self.debug = AutofocusDebug()
        self._last_preview_roi: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._last_preview_shape: tuple[int, ...] = ()

    @property
    def hardware_af_available(self) -> bool:
        return self.capabilities.hardware_af_available

    @property
    def is_degraded(self) -> bool:
        return self.capabilities.degraded

    def probe_capabilities(self, camera: FocusCapableCamera) -> AutofocusCapabilities:
        controls = camera.get_controls()
        self.capabilities = AutofocusCapabilities(
            af_mode="AfMode" in controls,
            af_trigger="AfTrigger" in controls,
            af_state="AfState" in controls,
            lens_position="LensPosition" in controls,
            autofocus_cycle=hasattr(camera, "autofocus_cycle"),
            af_windows="AfWindows" in controls,
            af_range="AfRange" in controls,
            af_metering="AfMetering" in controls,
        )
        logger.info("Autofocus capabilities: %s", self.capabilities.to_dict())
        if self.capabilities.degraded:
            logger.warning(
                "Hardware autofocus unavailable; using sharpness-only verification (degraded mode)"
            )
        if not self.capabilities.af_state:
            logger.info(
                "event=autofocus_result af_state=unavailable "
                "reason=not_advertised using=lens_position_and_preview_roi_sharpness"
            )
        return self.capabilities

    def _center_roi(self, frame_shape: tuple[int, ...], roi_fraction: float) -> tuple[int, int, int, int]:
        return roi_rect(frame_shape, roi_fraction)

    def _resolve_preview_roi(
        self,
        frame: Any,
        roi_fraction: float,
        preview_roi: tuple[int, int, int, int] | None,
        *,
        require_document_roi: bool,
    ) -> tuple[int, int, int, int]:
        size = (int(frame.shape[1]), int(frame.shape[0]))
        if preview_roi is not None:
            clipped = clip_roi(preview_roi, frame.shape)
            if require_document_roi:
                if not preview_roi_is_usable(clipped, size):
                    raise AutofocusError("invalid_roi: document ROI is empty or too small")
            elif clipped[2] < 1 or clipped[3] < 1:
                raise AutofocusError("invalid_roi: document ROI is empty or too small")
            return clipped
        if require_document_roi:
            raise AutofocusError("invalid_roi: document ROI required")
        return clip_roi(self._center_roi(frame.shape, roi_fraction), frame.shape)

    async def run_autofocus(
        self,
        camera: FocusCapableCamera,
        *,
        sharpness_threshold: float,
        roi_fraction: float = 0.7,
        preview_roi: tuple[int, int, int, int] | None = None,
        require_document_roi: bool = False,
    ) -> AutofocusResult:
        last_after = 0.0
        last_before = 0.0
        last_af_state: int | None = None
        last_roi = (0, 0, 0, 0)
        last_sensor = (0, 0, 0, 0)
        last_lens_before: Any = None
        last_lens_after: Any = None
        last_ratio = 0.0
        last_method = "none"
        preview_size = (0, 0)
        started = time.monotonic()
        # Absolute still-image floor must not veto a real preview improvement.
        preview_floor = min(float(sharpness_threshold), PREVIEW_ABSOLUTE_FLOOR)

        for attempt in range(1, self.config.af_max_retries + 1):
            self.debug = AutofocusDebug(running=True, method="hardware_trigger", attempt=attempt)
            try:
                preview = await asyncio.to_thread(camera.capture_preview_array)
                if preview is None or getattr(preview, "size", 1) == 0:
                    raise AutofocusError("Invalid preview frame for sharpness check")
                h, w = int(preview.shape[0]), int(preview.shape[1])
                preview_size = (w, h)
                last_roi = self._resolve_preview_roi(
                    preview, roi_fraction, preview_roi, require_document_roi=require_document_roi
                )
                mapping = await self._apply_document_roi(camera, last_roi, preview_size)
                last_sensor = mapping.get("sensor_roi", last_roi)
                last_before = laplacian_variance(preview, roi=last_roi)
                meta_before = self._read_metadata(camera)
                last_lens_before = meta_before.get("LensPosition")
                last_af_state = self._af_state_from_meta(meta_before)

                self.debug.focus_before = last_before
                self.debug.lens_before = float(last_lens_before) if last_lens_before is not None else None
                self.debug.focus_roi = _fmt_roi(last_roi)
                self.debug.preview_roi = _fmt_roi(last_roi)
                self.debug.sensor_roi = _fmt_roi(last_sensor)
                self.debug.preview_roi_wh = f"{last_roi[2]}x{last_roi[3]}"
                self.debug.scaler_crop = _fmt_roi(mapping.get("scaler_crop"))
                self.debug.preview_size = f"{w}x{h}"
                self._last_preview_roi = last_roi
                self._last_preview_shape = preview.shape
                logger.info(
                    "event=autofocus_start attempt=%d/%d preview_roi=%s sensor_roi=%s "
                    "preview_size=%dx%d scaler_crop=%s before_sharpness=%.1f "
                    "preview_floor=%.1f still_threshold=%.1f lens=%s "
                    "degraded=%s af_state_advertised=%s",
                    attempt,
                    self.config.af_max_retries,
                    self.debug.preview_roi,
                    self.debug.sensor_roi,
                    w,
                    h,
                    self.debug.scaler_crop,
                    last_before,
                    preview_floor,
                    sharpness_threshold,
                    last_lens_before,
                    self.capabilities.degraded,
                    self.capabilities.af_state,
                )

                last_method, last_af_state = await self._run_hardware_af(camera, attempt)
                self.debug.method = last_method

                after_frame = await asyncio.to_thread(camera.capture_preview_array)
                if after_frame is None or getattr(after_frame, "size", 1) == 0:
                    raise AutofocusError("Invalid preview frame after autofocus")
                last_after = laplacian_variance(after_frame, roi=last_roi)
                meta_after = self._read_metadata(camera)
                last_lens_after = meta_after.get("LensPosition")
                measured_state = self._af_state_from_meta(meta_after)
                if last_af_state is None:
                    last_af_state = measured_state

                last_ratio = (last_after / last_before) if last_before > 1e-6 else (2.0 if last_after > 0 else 0.0)
                improved = last_after > last_before
                accepted = accept_preview_focus(
                    before=last_before,
                    after=last_after,
                    absolute_floor=preview_floor,
                )
                self.debug.focus_after = last_after
                self.debug.lens_after = float(last_lens_after) if last_lens_after is not None else None
                self.debug.improvement_ratio = round(last_ratio, 3)
                self.debug.running = False
                logger.info(
                    "event=autofocus_result method=%s success=%s "
                    "before_sharpness=%.1f after_sharpness=%.1f ratio=%.3f improved=%s "
                    "preview_floor=%.1f still_threshold=%.1f attempt=%d "
                    "lens_before=%s lens_after=%s af_state=%s preview_roi=%s sensor_roi=%s",
                    last_method,
                    accepted,
                    last_before,
                    last_after,
                    last_ratio,
                    improved,
                    preview_floor,
                    sharpness_threshold,
                    attempt,
                    last_lens_before,
                    last_lens_after,
                    last_af_state,
                    self.debug.preview_roi,
                    self.debug.sensor_roi,
                )
                if accepted:
                    pause = getattr(camera, "pause_autofocus", None)
                    locked = False
                    if callable(pause):
                        await asyncio.to_thread(pause)
                        locked = True
                    duration_ms = (time.monotonic() - started) * 1000.0
                    logger.info(
                        "event=autofocus_result method=%s success=1 duration_ms=%.0f "
                        "lens_locked=%s preview_roi=%s sensor_roi=%s "
                        "before_sharpness=%.1f after_sharpness=%.1f",
                        last_method,
                        duration_ms,
                        int(locked),
                        self.debug.preview_roi,
                        self.debug.sensor_roi,
                        last_before,
                        last_after,
                    )
                    return AutofocusResult(
                        success=True,
                        sharpness=last_after,
                        attempts=attempt,
                        af_state=last_af_state,
                        method=last_method,
                        min_required_sharpness=preview_floor,
                        before_sharpness=last_before,
                        improved=improved,
                        roi=last_roi,
                        lens_before=self.debug.lens_before,
                        lens_after=self.debug.lens_after,
                        improvement_ratio=last_ratio,
                        sensor_roi=last_sensor,
                        preview_size=preview_size,
                        duration_ms=duration_ms,
                        lens_locked=locked,
                    )
                logger.info(
                    "event=autofocus_retry attempt=%d reason=roi_sharpness_not_improved "
                    "before=%.1f after=%.1f ratio=%.3f",
                    attempt,
                    last_before,
                    last_after,
                    last_ratio,
                )
            except AutofocusError as exc:
                self.debug.running = False
                logger.error("event=autofocus_failure reason=%s", exc)
                raise
            except Exception as exc:
                self.debug.running = False
                logger.error("event=autofocus_failure attempt=%d error=%s", attempt, exc)
                raise AutofocusError(f"Autofocus failed: {exc}") from exc

        self.debug.running = False
        duration_ms = (time.monotonic() - started) * 1000.0
        logger.error(
            "event=autofocus_failure reason=roi_sharpness_not_improved duration_ms=%.0f "
            "before=%.1f after=%.1f preview_roi=%s",
            duration_ms,
            last_before,
            last_after,
            _fmt_roi(last_roi),
        )
        raise AutofocusError(
            f"Autofocus did not improve document ROI sharpness "
            f"(before={last_before:.1f} after={last_after:.1f} ratio={last_ratio:.2f}, "
            f"attempts={self.config.af_max_retries} method={last_method} "
            f"preview_roi={_fmt_roi(last_roi)} sensor_roi={_fmt_roi(last_sensor)} "
            f"lens={last_lens_before}->{last_lens_after})"
        )

    async def verify_focus(
        self,
        camera: FocusCapableCamera,
        *,
        sharpness_threshold: float,
        roi_fraction: float = 0.7,
        preview_roi: tuple[int, int, int, int] | None = None,
        require_document_roi: bool = False,
    ) -> AutofocusResult:
        preview = await asyncio.to_thread(camera.capture_preview_array)
        if preview is None or getattr(preview, "size", 1) == 0:
            raise AutofocusError("Invalid preview frame for focus verification")
        size = (int(preview.shape[1]), int(preview.shape[0]))
        candidate = preview_roi if preview_roi is not None else self._last_preview_roi
        if candidate is not None and candidate != (0, 0, 0, 0):
            roi = clip_roi(candidate, preview.shape)
            if not preview_roi_is_usable(roi, size):
                raise AutofocusError("invalid_roi: document ROI is empty or too small")
        elif require_document_roi:
            raise AutofocusError("invalid_roi: document ROI required")
        else:
            roi = clip_roi(roi_rect(preview.shape, roi_fraction), preview.shape)
        sharpness = laplacian_variance(preview, roi=roi)
        meta = self._read_metadata(camera)
        af_state = self._af_state_from_meta(meta)
        last_after = self.debug.focus_after
        last_before = self.debug.focus_before
        preview_floor = min(float(sharpness_threshold), PREVIEW_ABSOLUTE_FLOOR)
        held = last_after > 0 and sharpness >= last_after * 0.7
        relative = accept_preview_focus(
            before=last_before if last_before > 0 else sharpness,
            after=sharpness,
            absolute_floor=preview_floor,
        )
        ok = held or relative or sharpness >= float(sharpness_threshold)

        logger.info(
            "event=focus_verification score=%.1f last_after=%.1f min_still=%.1f "
            "preview_floor=%.1f held=%s relative=%s ok=%s af_state=%s "
            "lens_position=%s preview_roi=%s",
            sharpness,
            last_after,
            sharpness_threshold,
            preview_floor,
            held,
            relative,
            ok,
            af_state,
            meta.get("LensPosition"),
            _fmt_roi(roi),
        )

        if not ok:
            logger.error(
                "event=autofocus_failure reason=focus_verification preview_roi=%s score=%.1f",
                _fmt_roi(roi),
                sharpness,
            )
            raise AutofocusError(
                f"Focus verification failed: roi_sharpness={sharpness:.1f} "
                f"last={last_after:.1f} min_still={sharpness_threshold:.1f} "
                f"preview_roi={_fmt_roi(roi)}"
            )

        self.debug.focus_after = sharpness
        return AutofocusResult(
            success=True,
            sharpness=sharpness,
            attempts=0,
            af_state=af_state,
            method="verification",
            min_required_sharpness=preview_floor,
            before_sharpness=last_before,
            improved=sharpness >= last_before,
            roi=roi,
            lens_before=self.debug.lens_before,
            lens_after=float(meta["LensPosition"]) if meta.get("LensPosition") is not None else self.debug.lens_after,
            improvement_ratio=self.debug.improvement_ratio,
            sensor_roi=tuple(int(x) for x in self.debug.sensor_roi.split(","))
            if self.debug.sensor_roi.count(",") == 3
            else (0, 0, 0, 0),
        )

    async def _apply_document_roi(
        self,
        camera: FocusCapableCamera,
        preview_roi: tuple[int, int, int, int],
        preview_size: tuple[int, int],
    ) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "preview_roi": preview_roi,
            "preview_size": preview_size,
            "scaler_crop": (0, 0, preview_size[0], preview_size[1]),
            "sensor_roi": preview_roi,
        }
        applier = getattr(camera, "apply_document_focus_window", None)
        if not callable(applier):
            return mapping
        try:
            result = await asyncio.to_thread(applier, preview_roi, preview_size)
        except TypeError:
            result = await asyncio.to_thread(applier, 0.7)
        if isinstance(result, dict):
            mapping.update(result)
            if "sensor_roi" in result and result["sensor_roi"]:
                mapping["sensor_roi"] = tuple(int(v) for v in result["sensor_roi"])
            if "preview_roi" in result and result["preview_roi"]:
                mapping["preview_roi"] = tuple(int(v) for v in result["preview_roi"])
            return mapping
        if result and len(result) == 4:
            # Legacy API returned a sensor window; sharpness still uses preview_roi.
            mapping["sensor_roi"] = (int(result[0]), int(result[1]), int(result[2]), int(result[3]))
        return mapping

    def _read_metadata(self, camera: FocusCapableCamera) -> dict[str, Any]:
        reader = getattr(camera, "capture_metadata", None)
        if reader is None:
            return {}
        try:
            meta = reader()
        except Exception as exc:
            logger.debug("event=autofocus_metadata_unavailable error=%s", exc)
            return {}
        return dict(meta) if meta else {}

    def _af_state_from_meta(self, meta: dict[str, Any]) -> int | None:
        if "AfState" not in meta:
            return None
        try:
            return int(meta["AfState"])
        except (TypeError, ValueError):
            return None

    async def _run_hardware_af(self, camera: FocusCapableCamera, attempt: int) -> tuple[str, int | None]:
        """Trigger a real AF scan. Do not wait on AfState (often unadvertised)."""
        trigger = getattr(camera, "trigger_autofocus", None)
        settle = getattr(camera, "wait_for_lens_settle", None)
        if callable(trigger) and callable(settle):
            logger.info("event=autofocus_triggered method=af_trigger_lens_settle attempt=%d", attempt)
            await asyncio.to_thread(trigger)
            await asyncio.to_thread(settle, min(float(self.config.af_timeout_sec), 3.0))
            method = "hardware_trigger" if self.capabilities.hardware_af_available else "sharpness_only"
            return method, None

        if self.capabilities.autofocus_cycle:
            logger.info("event=autofocus_triggered method=autofocus_cycle attempt=%d", attempt)
            try:
                cycle_ok = await asyncio.wait_for(
                    asyncio.to_thread(camera.autofocus_cycle),
                    timeout=self.config.af_timeout_sec,
                )
            except TimeoutError:
                logger.warning(
                    "event=autofocus_timeout method=autofocus_cycle attempt=%d timeout_sec=%.1f",
                    attempt,
                    self.config.af_timeout_sec,
                )
                return "hardware_cycle", None
            if cycle_ok is False:
                logger.warning(
                    "event=autofocus_result success=false method=autofocus_cycle attempt=%d",
                    attempt,
                )
            method = "hardware_cycle" if self.capabilities.hardware_af_available else "sharpness_only"
            return method, None

        if self.capabilities.af_mode:
            logger.info(
                "event=autofocus_triggered method=af_controls AfMode=Auto AfTrigger=Start attempt=%d",
                attempt,
            )
            await self._trigger_af(camera)
            state = await self._wait_for_af(camera)
            return "hardware_controls", state

        logger.warning("event=autofocus_triggered method=sharpness_only attempt=%d", attempt)
        return "sharpness_only", None

    async def _trigger_af(self, camera: FocusCapableCamera) -> None:
        def _do() -> None:
            camera.set_controls({"AfMode": 1, "AfTrigger": 0, "AfSpeed": 1})

        await asyncio.to_thread(_do)

    async def _wait_for_af(self, camera: FocusCapableCamera) -> int | None:
        """Wait on LensPosition settle, or AfState only if the camera actually exposes it."""
        settle = getattr(camera, "wait_for_lens_settle", None)
        if callable(settle):
            await asyncio.to_thread(settle, min(float(self.config.af_timeout_sec), 3.0))
            return None

        deadline = time.monotonic() + self.config.af_timeout_sec
        last_state: int | None = None
        last_lens: Any = None
        stable = 0
        while time.monotonic() < deadline:
            meta = await asyncio.to_thread(self._read_metadata, camera)
            lens = meta.get("LensPosition")
            if "AfState" not in meta:
                if last_lens is not None and lens is not None:
                    try:
                        if abs(float(lens) - float(last_lens)) < 0.05:
                            stable += 1
                            if stable >= 3:
                                logger.info(
                                    "event=autofocus_result af_state=unavailable "
                                    "method=lens_settle lens_position=%s",
                                    lens,
                                )
                                return None
                        else:
                            stable = 0
                    except (TypeError, ValueError):
                        stable = 0
                last_lens = lens
                await asyncio.sleep(0.1)
                continue
            last_state = int(meta["AfState"])
            if last_state in (AF_STATE_FOCUSED, AF_STATE_FAILED):
                logger.info(
                    "event=autofocus_result af_state=%s lens_position=%s",
                    last_state,
                    meta.get("LensPosition"),
                )
                return last_state
            await asyncio.sleep(0.1)
        logger.warning(
            "event=autofocus_timeout after_sec=%.1f last_af_state=%s lens_position=%s",
            self.config.af_timeout_sec,
            last_state,
            last_lens,
        )
        return last_state
