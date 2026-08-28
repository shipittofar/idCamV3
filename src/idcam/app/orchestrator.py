"""Application orchestrator — single authority for state and transactions."""

from __future__ import annotations

import asyncio
import signal
import threading
import time
from collections import deque
from pathlib import Path

from idcam import __version__
from idcam.app.state_machine import AppState, StateMachine
from idcam.app.transaction import ActiveCaptureTransaction, TransactionPhase
from idcam.camera.autofocus import AutofocusController
from idcam.camera.capture import CaptureService
from idcam.camera.manager import CameraManager, describe_preview_frame, preview_roi_is_usable
from idcam.camera.quality import ImageValidator
from idcam.config.schema import AppConfig
from idcam.exceptions import (
    AutofocusError,
    CameraError,
    CameraUnavailableError,
    CaptureError,
    ImageValidationError,
    NetworkError,
    ProtocolError,
    RecoveryExhaustedError,
    ServerRejectedError,
    ServerError,
    ShutdownRequested,
)
from idcam.logging.setup import get_logger, set_log_context
from idcam.network.api_client import ApiClient
from idcam.network.protocol import UploadOutcome, UploadRequest
from idcam.preview.live_viewer import LiveDebugViewer
from idcam.storage.disk_monitor import DiskMonitor, StorageHealth
from idcam.storage.paths import ensure_writable_directory
from idcam.storage.temp_files import TempFileStore
from idcam.system.component_health import ComponentStatus, HealthLevel, SystemHealth
from idcam.system.watchdog import Watchdog
from idcam.ui.status import StatusPublisher
from idcam.vision.document_detector import DocumentDetector, DocumentGeometryResult
from idcam.vision.geometry import find_document_quad
from idcam.vision.geometry_stability import GeometryStability, mean_corner_displacement_px
from idcam.vision.sharpness import clip_roi

logger = get_logger("orchestrator")

# States that run a single long-lived action per state entry (no re-entry per loop tick).
ONE_SHOT_STATES = frozenset({
    AppState.DOCUMENT_DETECTED,
    AppState.FOCUSING,
    AppState.FOCUS_VERIFICATION,
    AppState.CAPTURING,
    AppState.IMAGE_VALIDATION,
    AppState.UPLOADING,
    AppState.WAITING_FOR_SERVER,
    AppState.PROCESSING_RESPONSE,
    AppState.SUCCESS,
})


class ApplicationOrchestrator:
    """
    Main application loop with explicit state machine.

    Exactly one capture transaction at a time. No queue.
    """

    def __init__(self, config: AppConfig, *, use_mock_camera: bool = False) -> None:
        self.config = config
        self.sm = StateMachine()
        self.sm.set_timeouts(config.state_timeouts)
        self.health = SystemHealth()
        self.status = StatusPublisher(config.ui.status_file, version=__version__)
        self.camera = CameraManager(config.camera, use_mock=use_mock_camera)
        self.autofocus = AutofocusController(config.camera)
        self.temp_store = TempFileStore(config.temp_dir)
        self.capture_service = CaptureService(config.camera, self.temp_store)
        self.validator = ImageValidator(config.quality, config.camera)
        self.detector = DocumentDetector(config.detection)
        self.geometry_stability = GeometryStability(
            threshold_px=config.detection.geometry_stability_threshold_px,
            frames_required=config.detection.geometry_stable_frames_required,
        )
        self.disk = DiskMonitor(config.storage, config.temp_dir)
        self.api = ApiClient(config)
        self.watchdog = Watchdog(config.watchdog, config.state_timeouts)
        self._shutdown_event = asyncio.Event()
        self._transaction: ActiveCaptureTransaction | None = None
        self._recovery_attempts = 0
        self._next_recovery_at: float = 0.0
        self._recovery_in_progress = False
        self._removal_count = 0
        self._state_tick_running = False
        self._state_action_claimed: AppState | None = None
        self._error_backoff_claimed = False
        self._capture_path: Path | None = None
        self._pending_response = None
        self._debug_viewer: LiveDebugViewer | None = None
        self._idle_reject_count = 0
        self._last_geometry_status = ""
        self._accepted_geometry: DocumentGeometryResult | None = None
        self._cmd_lock = threading.Lock()
        self._pending_command: str | None = None
        self._command_result: dict | None = None
        self._command_event = threading.Event()
        self._manual_phase = "idle"
        self._manual_message = ""
        self._resume_state = AppState.IDLE
        self._awaiting_clear_scene = False
        self._clear_scene_count = 0
        self._last_scene_empty = False
        self._last_transition_reason = ""
        self._capture_started = False
        self._upload_started = False
        self._upload_result = ""
        self._last_capture_duration_ms = 0.0
        self._manual_capture_requested = False
        self._logged_preview_rgb = False
        self._preview_tick_times: deque[float] = deque(maxlen=25)
        self._preview_tick_count = 0
        self._last_tick_ms = 0.0
        self._last_detect_hz = 0.0
        if config.debug.enabled:
            self._debug_viewer = LiveDebugViewer(config.debug)
            self._debug_viewer.set_capture_handler(self.request_manual_capture)

    @property
    def has_active_transaction(self) -> bool:
        """True while autofocus/capture/upload is running. Not WAITING_FOR_REMOVAL."""
        return self.sm.in_transaction()

    async def run(self) -> int:
        self._install_signal_handlers()
        try:
            await self._startup()
            await self._main_loop()
        except ShutdownRequested:
            logger.info("Shutdown requested")
        except RecoveryExhaustedError:
            return 1
        except Exception as exc:
            logger.exception("Fatal error: %s", exc)
            self._transition(AppState.ERROR, reason=str(exc))
            return 1
        finally:
            await self._shutdown()
        return 0

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._request_shutdown)
            except NotImplementedError:
                signal.signal(sig, lambda *_: self._request_shutdown())

    def _request_shutdown(self) -> None:
        logger.info("Signal received, requesting shutdown")
        self._shutdown_event.set()

    async def _startup(self) -> None:
        self._transition(AppState.INITIALIZING)
        if self._debug_viewer is not None:
            self._debug_viewer.start()
            self._publish_debug()
        ensure_writable_directory(self.config.temp_dir, "temp")
        removed = self.temp_store.cleanup_stale(self.config.storage.stale_temp_max_age_sec)
        if removed:
            logger.info("Startup: removed %d stale temp files", removed)

        storage_status = self.disk.check()
        self._update_storage_health(storage_status)

        await self.api.start()
        network_ok = await self.api.check_connectivity()
        self.health.network = ComponentStatus(
            level=HealthLevel.HEALTHY if network_ok else HealthLevel.DEGRADED,
            message=None if network_ok else "Network unavailable",
            details={"healthy": network_ok},
        )
        self.health.server = ComponentStatus(
            level=HealthLevel.HEALTHY if network_ok else HealthLevel.DEGRADED,
            details={"configured": True, "reachable": network_ok},
        )

        self._transition(AppState.CAMERA_INITIALIZING)
        try:
            await asyncio.to_thread(self.camera.initialize)
        except CameraUnavailableError as exc:
            logger.error("event=camera_init_failed %s", exc)
            self.health.camera = ComponentStatus(
                level=HealthLevel.UNAVAILABLE,
                message=str(exc),
                details={"healthy": False},
            )
            self._transition(AppState.RECOVERY, reason="camera_init_failed")
            return

        caps = self.autofocus.probe_capabilities(self.camera)
        self.health.autofocus_available = caps.hardware_af_available
        self.health.autofocus_degraded = caps.degraded
        self.health.camera = ComponentStatus(
            level=HealthLevel.HEALTHY,
            details={
                "healthy": True,
                "autofocus_available": caps.hardware_af_available,
                "autofocus_degraded": caps.degraded,
            },
        )
        self.health.application = ComponentStatus(level=HealthLevel.HEALTHY, details={"healthy": True})

        self._transition(AppState.CAMERA_READY)
        if storage_status.health != StorageHealth.HEALTHY:
            self._transition(AppState.ERROR, reason="storage_degraded")
            return
        self._transition(AppState.IDLE)
        logger.info("Application ready")

    async def _main_loop(self) -> None:
        frame_interval = 1.0 / self.config.camera.preview_fps

        while not self._shutdown_event.is_set():
            self._refresh_health()
            alert = self.watchdog.check(
                self.sm,
                capture_id=self._transaction.capture_id if self._transaction else None,
            )
            if alert is not None:
                await self._handle_watchdog(alert)
                await asyncio.sleep(frame_interval)
                continue

            if await self._handle_state_timeout():
                await asyncio.sleep(frame_interval)
                continue

            tick_started = time.monotonic()
            await self._dispatch_state_tick()
            self._note_preview_tick(time.monotonic() - tick_started)
            await asyncio.sleep(frame_interval)

    def _note_preview_tick(self, elapsed_sec: float) -> None:
        """Record application sample/detect rate. Not camera FPS and not MJPEG FPS."""
        self._last_tick_ms = elapsed_sec * 1000.0
        now = time.monotonic()
        self._preview_tick_times.append(now)
        if len(self._preview_tick_times) >= 2:
            span = self._preview_tick_times[-1] - self._preview_tick_times[0]
            if span > 0:
                self._last_detect_hz = (len(self._preview_tick_times) - 1) / span
        self._preview_tick_count += 1
        if self._preview_tick_count == 1 or self._preview_tick_count % 25 == 0:
            doc = self.detector.debug_info()
            logger.info(
                "event=preview_tick tick_ms=%.1f detect_hz=%.2f preview_wh=%dx%d document_roi=%s "
                "configured_preview_fps=%.1f note=detect_hz_is_app_sample_rate_not_camera_fps",
                self._last_tick_ms,
                self._last_detect_hz,
                doc.frame_width,
                doc.frame_height,
                "{},{},{},{}".format(*doc.document_roi) if doc.document_roi != (0, 0, 0, 0) else "",
                self.config.camera.preview_fps,
            )

    async def _handle_state_timeout(self) -> bool:
        if not self.sm.is_timed_out():
            return False
        state = self.sm.state
        logger.warning(
            "event=state_timeout state=%s elapsed=%.1fs capture_id=%s",
            state.value,
            self.sm.elapsed(),
            self._transaction.capture_id if self._transaction else None,
        )
        if state == AppState.DOCUMENT_DETECTED:
            self._abort_transaction("document_detected_timeout")
            self._clear_transaction()
            self._transition(AppState.IDLE, reason="document_detected_timeout")
            return True
        if state == AppState.WAITING_FOR_STABILITY:
            await self._fail_transaction_and_wait("stability_timeout", "Document not stable in time")
            return True
        if state in {AppState.FOCUSING, AppState.FOCUS_VERIFICATION}:
            await self._fail_transaction_and_wait("focus_failed", f"Focus timed out in {state.value}")
            return True
        if state in {AppState.UPLOADING, AppState.WAITING_FOR_SERVER, AppState.PROCESSING_RESPONSE}:
            await self._fail_transaction_and_wait("upload_failed", "Server did not respond in time")
            return True
        if state == AppState.SUCCESS:
            self._transition(AppState.WAITING_FOR_REMOVAL)
            return True
        if state == AppState.CAPTURING:
            await self._enter_recovery("capture_timeout")
            return True
        if state == AppState.IMAGE_VALIDATION:
            await self._fail_transaction_and_wait("image_invalid", "Image validation timed out")
            return True
        return False

    def _claim_state_action(self) -> bool:
        """Return True if this tick may run the state's action (one-shot per entry)."""
        if self.sm.state not in ONE_SHOT_STATES:
            return True
        if self._state_action_claimed == self.sm.state:
            return False
        self._state_action_claimed = self.sm.state
        return True

    async def _dispatch_state_tick(self) -> None:
        if self._state_tick_running:
            return
        self._state_tick_running = True
        try:
            handlers = {
                AppState.IDLE: self._idle_tick,
                AppState.DOCUMENT_DETECTED: self._document_detected_tick,
                AppState.WAITING_FOR_STABILITY: self._stability_tick,
                AppState.FOCUSING: self._focusing_tick,
                AppState.FOCUS_VERIFICATION: self._focus_verification_tick,
                AppState.CAPTURING: self._capturing_tick,
                AppState.IMAGE_VALIDATION: self._validation_tick,
                AppState.UPLOADING: self._uploading_tick,
                AppState.WAITING_FOR_SERVER: self._waiting_server_tick,
                AppState.PROCESSING_RESPONSE: self._processing_response_tick,
                AppState.SUCCESS: self._success_tick,
                AppState.WAITING_FOR_REMOVAL: self._removal_tick,
                AppState.RECOVERY: self._recovery_tick,
                AppState.ERROR: self._error_tick,
            }
            handler = handlers.get(self.sm.state)
            if handler:
                await handler()
        finally:
            self._state_tick_running = False

    async def _safe_preview(self):
        try:
            frame = await asyncio.to_thread(self.camera.capture_preview_array)
        except CameraError as exc:
            logger.error("event=camera_preview_failed %s", exc)
            self._publish_debug(camera_error=str(exc))
            await self._enter_recovery(str(exc))
            return None
        if not self._logged_preview_rgb:
            info = describe_preview_frame(frame)
            logger.info(
                "event=preview_rgb_contract shape=%s dtype=%s channel_means=%s assumed_order=RGB "
                "note=one_shot_detector_input_not_a_hardware_color_proof",
                info.get("shape"),
                info.get("dtype"),
                info.get("channel_means"),
            )
            self._logged_preview_rgb = True
        return frame

    def _snapshot_accepted_geometry(self, geom: DocumentGeometryResult | None) -> None:
        """Freeze the READY document location for AF, verify, and capture-time abort."""
        if geom is not None and geom.present and geom.bbox is not None and geom.corners is not None:
            self._accepted_geometry = geom
            return
        self._accepted_geometry = None

    def _accepted_preview_roi(self, frame=None, *, allow_search_fallback: bool = True) -> tuple[int, int, int, int] | None:
        geom = self._accepted_geometry
        if geom is not None:
            if geom.bbox is None:
                return None
            if frame is not None:
                roi = clip_roi(geom.bbox, frame.shape)
                size = (int(frame.shape[1]), int(frame.shape[0]))
                if not preview_roi_is_usable(roi, size):
                    return None
                return roi
            if geom.bbox[2] < 8 or geom.bbox[3] < 8:
                return None
            return geom.bbox
        if not allow_search_fallback:
            return None
        return self.detector.document_preview_roi(frame)

    def _observe_geometry(self, frame) -> tuple[DocumentGeometryResult, bool]:
        """Authoritative presence + the single geometry-stability gate."""
        geom = self.detector.detect_geometry(frame)
        ready = self.geometry_stability.update(geom)
        return geom, ready

    def _geometry_moved_readonly(self, frame) -> tuple[bool, float]:
        """Capture-time jitter vs accepted READY corners. Does not mutate detector state."""
        previous = None
        if self._accepted_geometry is not None and self._accepted_geometry.corners is not None:
            previous = self._accepted_geometry.corners_array()
        if previous is None:
            previous = self.geometry_stability.corners
        if previous is None:
            last = self.detector.last_geometry()
            if last.corners is None:
                return False, 0.0
            previous = last.corners_array()
        raw, _count = find_document_quad(frame)
        if raw is None:
            return True, float("inf")
        displacement = mean_corner_displacement_px(previous, raw.corners)
        return displacement > self.geometry_stability.threshold_px, displacement

    def _log_geometry_status(self, status: str, geom: DocumentGeometryResult) -> None:
        if status == self._last_geometry_status:
            return
        self._last_geometry_status = status
        stab = self.geometry_stability.debug_info()
        logger.info(
            "event=geometry_gate status=%s present=%s reason=%s confidence=%.3f "
            "bbox=%s corners=%s stable_count=%d/%d displacement_px=%.2f",
            status,
            int(geom.present),
            geom.reason,
            geom.confidence,
            geom.bbox,
            geom.corners,
            stab.stable_count,
            stab.stable_frames_required,
            stab.displacement_px,
        )

    async def _check_motion_abort(self) -> bool:
        """Return False if document moved and transaction was failed.

        Uses a read-only quad measurement. Must not call classify()/detect()
        (those update the empty-scene EMA via accumulateWeighted).
        """
        frame = await self._safe_preview()
        if frame is None:
            return False
        moving, displacement = self._geometry_moved_readonly(frame)
        self._publish_debug(frame)
        if moving:
            logger.warning(
                "event=capture_geometry_abort state=%s displacement_px=%.2f threshold_px=%.2f",
                self.sm.state.value,
                displacement,
                self.geometry_stability.threshold_px,
            )
            await self._fail_transaction_and_wait("stability_timeout", "Document moved during capture")
            return False
        logger.info(
            "event=capture_geometry_check passed=1 displacement_px=%.2f threshold_px=%.2f",
            displacement,
            self.geometry_stability.threshold_px,
        )
        return True

    async def _idle_tick(self) -> None:
        if self.has_active_transaction:
            return
        if not self.disk.can_capture(self.config.storage.capture_reserve_mb):
            self._update_storage_health(self.disk.check())
            return
        frame = await self._safe_preview()
        if frame is None:
            return
        if self._take_manual_request():
            logger.info("event=pipeline_stage stage=manual_capture_start state=IDLE")
            self.geometry_stability.reset()
            self._resume_state = AppState.IDLE
            self._awaiting_clear_scene = False
            self._manual_capture_requested = True
            self._begin_transaction(trigger="manual")
            self._set_manual_phase("focusing", "Manual capture started")
            self._publish_debug(frame)
            self._transition(AppState.DOCUMENT_DETECTED, reason="manual_capture")
            return
        geom, ready = self._observe_geometry(frame)
        present = geom.present
        self._last_scene_empty = not present
        self._publish_debug(frame)
        if self._awaiting_clear_scene and not present:
            self._clear_scene_count += 1
            self._log_geometry_status("geometry_absent", geom)
            if self._clear_scene_count >= self.config.detection.removal_frames_required:
                self._awaiting_clear_scene = False
                self._clear_scene_count = 0
                logger.info("event=auto_capture_armed reason=geometry_absent_after_failure")
            return
        if self._awaiting_clear_scene and present:
            self._clear_scene_count = 0
            self._log_geometry_status("auto_rearm_blocked", geom)
            return
        if present and ready:
            self._log_geometry_status("geometry_stable", geom)
            logger.info(
                "event=document_detected reason=%s confidence=%.3f bbox=%s",
                geom.reason,
                geom.confidence,
                geom.bbox,
            )
            logger.info("event=pipeline_stage stage=document_detected")
            self._begin_transaction(trigger="auto")
            self._transition(AppState.DOCUMENT_DETECTED, reason="document_detected")
            return
        if present:
            status = "geometry_held" if geom.reason == "geometry_held" else "geometry_unstable"
            self._log_geometry_status(status, geom)
            return
        self._idle_reject_count += 1
        self._log_geometry_status("geometry_absent", geom)

    def _begin_transaction(self, *, trigger: str = "auto") -> ActiveCaptureTransaction:
        if self._transaction is not None and not self._transaction.is_terminal:
            raise RuntimeError("Cannot begin transaction: one already active")
        if self.sm.state == AppState.WAITING_FOR_REMOVAL:
            self._resume_state = AppState.WAITING_FOR_REMOVAL
        else:
            self._resume_state = AppState.IDLE
        with self._cmd_lock:
            if self._pending_command == "manual_capture":
                self._pending_command = "manual_capture_running"
                trigger = "manual"
                self._manual_phase = "starting"
                self._manual_message = "Manual capture started"
        txn = ActiveCaptureTransaction(capture_id=ActiveCaptureTransaction.generate_id(), trigger=trigger)
        txn.mark_document_detected()
        self._transaction = txn
        self._capture_path = None
        self._idle_reject_count = 0
        self._capture_started = False
        self._upload_started = False
        self._upload_result = ""
        set_log_context(capture_id=txn.capture_id)
        logger.info("event=transaction_started capture_id=%s trigger=%s", txn.capture_id, trigger)
        return txn

    def request_manual_capture(self, timeout: float = 180.0) -> dict:
        """HTTP-thread entry: queue a capture through the existing state machine."""
        state = self.sm.state
        txn_active = self._transaction is not None and not self._transaction.is_terminal
        allowed = {AppState.IDLE, AppState.WAITING_FOR_REMOVAL}
        if state not in allowed or self.has_active_transaction or txn_active:
            logger.info("event=manual_capture_rejected status=busy state=%s", state.value)
            return {
                "ok": False,
                "status": "busy",
                "state": state.value,
                "message": f"Busy ({state.value})",
            }
        if not self.health.camera.healthy:
            return {
                "ok": False,
                "status": "busy",
                "state": state.value,
                "message": "Camera is not healthy",
            }
        with self._cmd_lock:
            if self._pending_command is not None:
                return {
                    "ok": False,
                    "status": "busy",
                    "state": state.value,
                    "message": "Capture already in progress",
                }
            self._pending_command = "manual_capture"
            self._command_result = None
            self._command_event.clear()
            self._manual_phase = "queued"
            self._manual_message = "Waiting for capture pipeline"
            self._manual_capture_requested = True
        logger.info("event=pipeline_stage stage=manual_capture_queued")
        if not self._command_event.wait(timeout):
            with self._cmd_lock:
                if self._pending_command == "manual_capture":
                    self._pending_command = None
                self._manual_phase = "timeout"
                self._manual_message = "Timed out waiting for capture"
            logger.warning("event=manual_capture_timeout")
            return {"ok": False, "status": "timeout", "message": "Timed out waiting for capture"}
        with self._cmd_lock:
            result = dict(self._command_result or {"ok": False, "status": "error", "message": "No result"})
        return result

    def _take_manual_request(self) -> bool:
        with self._cmd_lock:
            if self._pending_command == "manual_capture":
                self._pending_command = "manual_capture_running"
                self._manual_phase = "starting"
                return True
            return False

    def _set_manual_phase(self, phase: str, message: str = "") -> None:
        with self._cmd_lock:
            if self._pending_command in {"manual_capture", "manual_capture_running"}:
                self._manual_phase = phase
                if message:
                    self._manual_message = message

    def _finish_manual_request(self, result: dict) -> None:
        with self._cmd_lock:
            if self._pending_command not in {"manual_capture", "manual_capture_running"}:
                return
            self._command_result = result
            self._pending_command = None
            self._manual_phase = "success" if result.get("ok") else "failed"
            self._manual_message = str(result.get("message") or "")
            self._command_event.set()

    def _manual_status(self) -> tuple[bool, str, str]:
        with self._cmd_lock:
            txn_active = self._transaction is not None and not self._transaction.is_terminal
            available = (
                self.health.camera.healthy
                and self._pending_command is None
                and not self.has_active_transaction
                and not txn_active
                and self.sm.state in {AppState.IDLE, AppState.WAITING_FOR_REMOVAL}
            )
            return available, self._manual_phase, self._manual_message

    async def _document_detected_tick(self) -> None:
        if not self._claim_state_action():
            return
        if self._transaction and self._transaction.trigger == "manual":
            self._set_manual_phase("focusing", "Focusing...")
            self._snapshot_accepted_geometry(self.detector.last_geometry())
            self._transition(AppState.FOCUSING, reason="manual_capture_skip_stability")
            return
        self._transition(AppState.WAITING_FOR_STABILITY, reason="auto_wait_stability")

    async def _stability_tick(self) -> None:
        frame = await self._safe_preview()
        if frame is None:
            return
        geom, ready = self._observe_geometry(frame)
        self._publish_debug(frame)
        if not geom.present:
            self._log_geometry_status("geometry_absent", geom)
            self.sm.touch_progress()
            return
        if ready:
            self._log_geometry_status("geometry_stable", geom)
            logger.info(
                "event=document_ready capture_id=%s reason=%s confidence=%.3f bbox=%s "
                "stable_count=%d displacement_px=%.2f",
                self._transaction.capture_id if self._transaction else None,
                geom.reason,
                geom.confidence,
                geom.bbox,
                self.geometry_stability.stable_count,
                self.geometry_stability.last_displacement_px,
            )
            logger.info(
                "event=document_stable frames=%d displacement_px=%.2f threshold_px=%.2f",
                self.geometry_stability.stable_count,
                self.geometry_stability.last_displacement_px,
                self.geometry_stability.threshold_px,
            )
            assert self._transaction is not None
            self._transaction.phase = TransactionPhase.STABILIZING
            self._snapshot_accepted_geometry(geom)
            logger.info("event=pipeline_stage stage=focusing reason=document_stable")
            self._set_manual_phase("focusing", "Focusing...")
            self._transition(AppState.FOCUSING, reason="document_stable")
            return
        status = "geometry_held" if geom.reason == "geometry_held" else "geometry_unstable"
        self._log_geometry_status(status, geom)
        self.sm.touch_progress()

    async def _focusing_tick(self) -> None:
        if not self._claim_state_action():
            return
        assert self._transaction is not None
        if self._shutdown_event.is_set():
            return
        # Lens motion during AF must not be treated as the document moving.
        # ROI is the frozen READY geometry, not a live re-detect (avoids 70% search fallback).
        frame = await self._safe_preview()
        if frame is None:
            return
        auto = self._transaction.trigger == "auto"
        preview_roi = self._accepted_preview_roi(frame, allow_search_fallback=not auto)
        require_document_roi = auto or self._accepted_geometry is not None
        if require_document_roi and preview_roi is None:
            logger.error("event=autofocus_failure reason=invalid_roi capture_id=%s", self._transaction.capture_id)
            await self._fail_transaction_and_wait("focus_failed", "invalid_roi: no accepted document geometry")
            return
        self._publish_debug(frame)
        self._transaction.mark_focus_started()
        self._transaction.focus_attempts += 1
        self._set_manual_phase("focusing", "Focusing...")
        logger.info(
            "event=autofocus_start capture_id=%s preview_roi=%s accepted_bbox=%s",
            self._transaction.capture_id,
            preview_roi,
            None if self._accepted_geometry is None else self._accepted_geometry.bbox,
        )
        logger.info(
            "event=pipeline_stage stage=focusing capture_id=%s preview_roi=%s",
            self._transaction.capture_id,
            preview_roi,
        )
        min_sharp = self.config.quality.min_sharpness * 0.6
        try:
            result = await self.autofocus.run_autofocus(
                self.camera,
                sharpness_threshold=min_sharp,
                roi_fraction=self.config.detection.roi_center_fraction,
                preview_roi=preview_roi,
                require_document_roi=require_document_roi,
            )
            if self._accepted_geometry is not None:
                check_frame = await self._safe_preview()
                if check_frame is not None:
                    moved, displacement = self._geometry_moved_readonly(check_frame)
                    if moved:
                        raise AutofocusError(
                            f"document moved during autofocus displacement_px={displacement:.1f}"
                        )
            pause = getattr(self.camera, "pause_autofocus", None)
            if callable(pause) and not result.lens_locked:
                await asyncio.to_thread(pause)
            self._transaction.mark_focus_completed(result.sharpness, result.min_required_sharpness)
            self.sm.touch_progress()
            logger.info(
                "event=autofocus_result capture_id=%s success=1 method=%s sharpness=%.1f "
                "ratio=%.2f duration_ms=%.0f lens_locked=%s roi=%s",
                self._transaction.capture_id,
                result.method,
                result.sharpness,
                result.improvement_ratio,
                result.duration_ms,
                int(result.lens_locked),
                result.roi,
            )
            logger.info(
                "event=pipeline_stage stage=focus_verification method=%s sharpness=%.1f ratio=%.2f",
                result.method,
                result.sharpness,
                result.improvement_ratio,
            )
            self._transition(AppState.FOCUS_VERIFICATION, reason="autofocus_accepted")
        except AutofocusError as exc:
            logger.error(
                "event=autofocus_result capture_id=%s success=0 error=%s",
                self._transaction.capture_id,
                exc,
            )
            await self._fail_transaction_and_wait("focus_failed", str(exc))

    async def _focus_verification_tick(self) -> None:
        if not self._claim_state_action():
            return
        assert self._transaction is not None
        min_sharp = self.config.quality.min_sharpness * 0.6
        self._transaction.focus_verify_attempts += 1
        preview_roi = self._accepted_preview_roi(
            allow_search_fallback=self._transaction.trigger != "auto"
        )
        require_document_roi = self._transaction.trigger == "auto" or self._accepted_geometry is not None
        if require_document_roi and preview_roi is None:
            await self._fail_transaction_and_wait("focus_failed", "invalid_roi: no accepted document geometry")
            return
        try:
            result = await self.autofocus.verify_focus(
                self.camera,
                sharpness_threshold=min_sharp,
                roi_fraction=self.config.detection.roi_center_fraction,
                preview_roi=preview_roi,
                require_document_roi=require_document_roi,
            )
            self._transaction.mark_focus_completed(result.sharpness, result.min_required_sharpness)
            self.sm.touch_progress()
            self._transition(AppState.CAPTURING, reason="focus_verified")
        except AutofocusError:
            if self._transaction.focus_verify_attempts < self.config.camera.af_verify_max_retries:
                self._transition(AppState.FOCUSING, reason="refocus")
            else:
                await self._fail_transaction_and_wait("focus_failed", "Focus verification failed")

    async def _capturing_tick(self) -> None:
        if not self._claim_state_action():
            return
        assert self._transaction is not None
        if self._transaction.focus_completed_at is None:
            logger.error(
                "event=capture_result success=0 reason=focus_not_verified capture_id=%s",
                self._transaction.capture_id,
            )
            await self._fail_transaction_and_wait("focus_failed", "capture blocked: focus not verified")
            return
        if self._transaction.trigger == "auto" and self._accepted_geometry is None:
            logger.error(
                "event=capture_result success=0 reason=no_accepted_geometry capture_id=%s",
                self._transaction.capture_id,
            )
            await self._fail_transaction_and_wait("focus_failed", "capture blocked: no accepted document geometry")
            return
        if not await self._check_motion_abort():
            return
        if not self.disk.can_capture(self.config.storage.capture_reserve_mb):
            await self._fail_transaction_and_wait("storage_full", "Insufficient disk space")
            return
        output_path = self.temp_store.create_temp_path()
        self._transaction.mark_capture_started(str(output_path))
        self._capture_started = True
        self._set_manual_phase("capturing", "Capturing...")
        logger.info(
            "event=capture_start capture_id=%s path=%s accepted_bbox=%s",
            self._transaction.capture_id,
            output_path,
            None if self._accepted_geometry is None else self._accepted_geometry.bbox,
        )
        logger.info("event=pipeline_stage stage=capturing path=%s", output_path)
        started = time.monotonic()
        try:
            self._capture_path = await self.capture_service.capture(self.camera, output_path)
            self._last_capture_duration_ms = (time.monotonic() - started) * 1000.0
            self.sm.touch_progress()
            logger.info(
                "event=capture_encoded capture_id=%s path=%s duration_ms=%.0f",
                self._transaction.capture_id,
                self._capture_path,
                self._last_capture_duration_ms,
            )
            self._transition(AppState.IMAGE_VALIDATION)
        except CaptureError as exc:
            self._last_capture_duration_ms = (time.monotonic() - started) * 1000.0
            logger.error(
                "event=capture_result capture_id=%s success=0 reason=%s duration_ms=%.0f",
                self._transaction.capture_id,
                exc,
                self._last_capture_duration_ms,
            )
            self.temp_store.cleanup(output_path)
            await self._enter_recovery(str(exc))

    async def _validation_tick(self) -> None:
        if not self._claim_state_action():
            return
        assert self._transaction is not None
        assert self._capture_path is not None
        self._set_manual_phase("validating", "Validating image...")
        logger.info("event=pipeline_stage stage=image_validation path=%s", self._capture_path)
        try:
            result = self.validator.validate(self._capture_path)
            self._transaction.mark_validated(
                size=result.file_bytes,
                width=result.width,
                height=result.height,
                sha256=result.sha256,
                sharpness=result.sharpness,
            )
            logger.info(
                "event=capture_result capture_id=%s success=1 path=%s width=%d height=%d "
                "size_bytes=%d duration_ms=%.0f validation=ok sharpness=%.1f",
                self._transaction.capture_id,
                self._capture_path,
                result.width,
                result.height,
                result.file_bytes,
                self._last_capture_duration_ms,
                result.sharpness,
            )
            logger.info(
                "event=image_validated sha256=%s sharpness=%.1f",
                result.sha256[:16],
                result.sharpness,
            )
            self.sm.touch_progress()
            self._transition(AppState.UPLOADING)
        except ImageValidationError as exc:
            logger.error(
                "event=capture_result capture_id=%s success=0 reason=validation_failure detail=%s",
                self._transaction.capture_id,
                exc,
            )
            logger.error("event=image_validation_failed %s", exc)
            self.temp_store.cleanup(self._capture_path)
            await self._fail_transaction_and_wait("image_invalid", str(exc))

    async def _uploading_tick(self) -> None:
        if not self._claim_state_action():
            return
        assert self._transaction is not None
        assert self._capture_path is not None
        self._transaction.mark_upload_started()
        self._upload_started = True
        self._upload_result = "started"
        self._set_manual_phase("uploading", "Uploading...")
        logger.info(
            "event=pipeline_stage stage=uploading capture_id=%s path=%s",
            self._transaction.capture_id,
            self._capture_path,
        )
        request = UploadRequest(
            capture_id=self._transaction.capture_id,
            device_id=self.config.device_id,
            image_path=str(self._capture_path),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        try:
            self._pending_response = await self.api.upload_document(request)
            self._transaction.mark_upload_completed(self._pending_response.http_status)
            self._upload_result = f"http_{self._pending_response.http_status}"
            self.sm.touch_progress()
            self._transition(AppState.WAITING_FOR_SERVER, reason="upload_sent")
        except (ServerRejectedError, ServerError, ProtocolError) as exc:
            self.temp_store.cleanup(self._capture_path)
            key = "server_rejected" if isinstance(exc, ServerRejectedError) else "upload_failed"
            self._upload_result = key
            await self._fail_transaction_and_wait(key, str(exc))
        except NetworkError as exc:
            self.temp_store.cleanup(self._capture_path)
            self._upload_result = "upload_failed"
            await self._fail_transaction_and_wait("upload_failed", str(exc))

    async def _waiting_server_tick(self) -> None:
        if not self._claim_state_action():
            return
        self._transition(AppState.PROCESSING_RESPONSE)

    async def _processing_response_tick(self) -> None:
        if not self._claim_state_action():
            return
        assert self._transaction is not None
        response = self._pending_response
        logger.info(
            "event=pipeline_stage stage=server_response success=%s outcome=%s http_status=%s capture_id=%s",
            bool(response and response.success),
            response.outcome.value if response else "none",
            response.http_status if response else None,
            self._transaction.capture_id,
        )
        if response and response.success and response.outcome == UploadOutcome.ACCEPTED:
            self.health.record_success()
            self.health.server = ComponentStatus(level=HealthLevel.HEALTHY, details={"healthy": True, "reachable": True})
            self._transaction.mark_completed()
            self.temp_store.cleanup(self._capture_path)
            logger.info("event=pipeline_stage stage=success capture_id=%s", self._transaction.capture_id)
            self._finish_manual_request(
                {
                    "ok": True,
                    "status": "success",
                    "state": AppState.SUCCESS.value,
                    "capture_id": self._transaction.capture_id,
                    "message": "Upload accepted",
                }
            )
            self._upload_result = "accepted"
            self._transition(AppState.SUCCESS, reason="upload_accepted")
        else:
            msg = response.message if response else "Upload failed"
            self.health.record_failure(msg)
            self.temp_store.cleanup(self._capture_path)
            failure_key = "upload_failed"
            if response and response.outcome == UploadOutcome.REJECTED:
                failure_key = "server_rejected"
            self._upload_result = failure_key
            await self._fail_transaction_and_wait(failure_key, msg or "Upload failed")

    async def _success_tick(self) -> None:
        if not self._claim_state_action():
            return
        await asyncio.sleep(2.0)
        self._removal_count = 0
        self._last_scene_empty = False
        self._transition(AppState.WAITING_FOR_REMOVAL, reason="capture_success")

    async def _removal_tick(self) -> None:
        frame = await self._safe_preview()
        if frame is None:
            return
        if self._take_manual_request():
            logger.info("event=pipeline_stage stage=manual_capture_start state=WAITING_FOR_REMOVAL")
            self.geometry_stability.reset()
            self._begin_transaction(trigger="manual")
            self._manual_capture_requested = True
            self._set_manual_phase("focusing", "Manual capture started")
            self._publish_debug(frame)
            self._transition(AppState.DOCUMENT_DETECTED, reason="manual_capture")
            return
        geom, _ready = self._observe_geometry(frame)
        empty = not geom.present
        self._last_scene_empty = empty
        if empty:
            self._removal_count += 1
            self._log_geometry_status("geometry_absent", geom)
        else:
            self._removal_count = 0
            self._log_geometry_status("geometry_present", geom)
            self.sm.touch_progress()
        self._publish_debug(frame)

        if self._removal_count >= self.config.detection.removal_frames_required:
            logger.info(
                "event=document_removed reason=geometry_absent consecutive_frames=%d bbox=%s",
                self._removal_count,
                geom.bbox,
            )
            self._log_geometry_status("removal_confirmed", geom)
            self._clear_transaction()
            self.detector.reset_baseline()
            self.geometry_stability.reset()
            self._awaiting_clear_scene = False
            self._transition(AppState.IDLE, reason="document_removed")

    async def _recovery_tick(self) -> None:
        if self._recovery_in_progress:
            return
        if time.monotonic() < self._next_recovery_at:
            return
        if not self._claim_state_action():
            return

        self._recovery_in_progress = True
        max_attempts = self.config.camera.recovery_max_attempts
        try:
            if self._recovery_attempts >= max_attempts:
                logger.error("Recovery exhausted after %d attempts", max_attempts)
                self.health.record_failure("recovery_exhausted")
                self.health.camera = ComponentStatus(level=HealthLevel.UNAVAILABLE, message="Camera recovery exhausted")
                self._transition(AppState.ERROR)
                raise RecoveryExhaustedError("Recovery attempts exhausted")

            self._recovery_attempts += 1
            logger.info("event=recovery_started attempt=%d/%d", self._recovery_attempts, max_attempts)
            self.health.record_recovery()
            self._abort_transaction("recovery")

            await asyncio.sleep(self.config.camera.recovery_delay_sec)
            await asyncio.to_thread(self.camera.restart)
            self.health.camera = ComponentStatus(level=HealthLevel.HEALTHY, details={"healthy": True})
            self._recovery_attempts = 0
            self._next_recovery_at = 0.0
            if self.disk.check().healthy:
                self._transition(AppState.IDLE)
            else:
                self._transition(AppState.ERROR, reason="storage_still_degraded")
        except CameraError as exc:
            logger.error("event=recovery_failed %s", exc)
            self.health.camera = ComponentStatus(level=HealthLevel.UNAVAILABLE, message=str(exc))
            self._next_recovery_at = time.monotonic() + self.config.camera.recovery_delay_sec
            self._state_action_claimed = None
        finally:
            self._recovery_in_progress = False

    async def _error_tick(self) -> None:
        storage = self.disk.check()
        self._update_storage_health(storage)
        if not storage.healthy or self._shutdown_event.is_set():
            return
        if self._error_backoff_claimed:
            return
        self._error_backoff_claimed = True
        try:
            await asyncio.sleep(10.0)
            if not self._shutdown_event.is_set():
                self._transition(AppState.RECOVERY, reason="auto_recovery_from_error")
        finally:
            self._error_backoff_claimed = False

    async def _handle_watchdog(self, alert) -> None:
        await self._enter_recovery(f"watchdog:{alert.reason}")

    async def _enter_recovery(self, reason: str) -> None:
        logger.warning(
            "event=enter_recovery reason=%s capture_id=%s",
            reason,
            getattr(self._transaction, "capture_id", None),
        )
        self._cleanup_transaction_files()
        if self._transaction and not self._transaction.is_terminal:
            self._transaction.mark_failed(reason)
        self._finish_manual_request(
            {
                "ok": False,
                "status": "failed",
                "state": "RECOVERY",
                "capture_id": getattr(self._transaction, "capture_id", None),
                "message": reason,
            }
        )
        self._next_recovery_at = 0.0
        self._state_action_claimed = None
        self._transition(AppState.RECOVERY, reason=reason)

    def _abort_transaction(self, reason: str) -> None:
        if self._transaction and not self._transaction.is_terminal:
            self._transaction.mark_failed(reason)
        self._finish_manual_request(
            {
                "ok": False,
                "status": "failed",
                "state": self.sm.state.value,
                "capture_id": self._transaction.capture_id if self._transaction else None,
                "message": reason,
            }
        )
        self._cleanup_transaction_files()

    async def _fail_transaction_and_wait(self, failure_key: str, detail: str) -> None:
        if self._transaction:
            self._transaction.mark_failed(detail)
        self.health.record_failure(detail)
        self._cleanup_transaction_files()
        dest = (
            AppState.WAITING_FOR_REMOVAL
            if self._resume_state == AppState.WAITING_FOR_REMOVAL
            else AppState.IDLE
        )
        self.status.update(
            dest,
            capture_id=self._transaction.capture_id if self._transaction else None,
            failure_key=failure_key,
            last_error=detail,
            **self._health_status_kwargs(),
        )
        logger.info(
            "event=pipeline_stage stage=failed reason=%s detail=%s capture_id=%s next_state=%s",
            failure_key,
            detail,
            self._transaction.capture_id if self._transaction else None,
            dest.value,
        )
        self._finish_manual_request(
            {
                "ok": False,
                "status": "failed",
                "state": dest.value,
                "capture_id": self._transaction.capture_id if self._transaction else None,
                "message": f"{failure_key}: {detail}",
            }
        )
        self.geometry_stability.reset()
        self._accepted_geometry = None
        if dest == AppState.IDLE:
            self._awaiting_clear_scene = True
            self._clear_scene_count = 0
            self._clear_transaction()
            self._transition(AppState.IDLE, reason=f"{failure_key}")
        else:
            self._transition(AppState.WAITING_FOR_REMOVAL, reason=f"{failure_key}")

    def _cleanup_transaction_files(self) -> None:
        if self._capture_path:
            self.temp_store.cleanup(self._capture_path)
            self._capture_path = None

    def _clear_transaction(self) -> None:
        capture_id = self._transaction.capture_id if self._transaction else None
        logger.info("event=transaction_cleared capture_id=%s", capture_id)
        self._transaction = None
        self._capture_path = None
        self._pending_response = None
        self._accepted_geometry = None
        set_log_context(capture_id=None)

    def _refresh_health(self) -> None:
        storage = self.disk.check()
        self._update_storage_health(storage)

    def _update_storage_health(self, storage_status) -> None:
        level = HealthLevel.HEALTHY
        if storage_status.health.value == "degraded":
            level = HealthLevel.DEGRADED
        elif storage_status.health.value == "unavailable":
            level = HealthLevel.UNAVAILABLE
        self.health.storage = ComponentStatus(
            level=level,
            message=storage_status.message,
            details=storage_status.to_dict(),
        )

    def _health_status_kwargs(self) -> dict:
        return {
            "camera": self.health.camera.to_dict(),
            "network": self.health.network.to_dict(),
            "server": self.health.server.to_dict(),
            "storage": self.health.storage.to_dict(),
            "application": self.health.application.to_dict(),
            "uptime_seconds": self.health.uptime_sec(),
        }

    def _transition(self, target: AppState, *, reason: str = "") -> None:
        if self._shutdown_event.is_set() and target not in {AppState.SHUTDOWN, AppState.RECOVERY}:
            target = AppState.SHUTDOWN
        prev = self.sm.state
        self._last_transition_reason = reason or f"{prev.value}->{target.value}"
        self.sm.transition(target, reason=reason)
        self._state_action_claimed = None
        if target != AppState.ERROR:
            self._error_backoff_claimed = False
        set_log_context(state=target.value)
        geom = self.detector.last_geometry()
        stab = self.geometry_stability.debug_info()
        logger.info(
            "event=state_transition from=%s to=%s reason=%s capture_id=%s "
            "document_detected=%s geometry_reason=%s confidence=%.3f bbox=%s "
            "stable=%s stable_count=%d displacement_px=%.2f removal_detected=%s removal_count=%d",
            prev.value,
            target.value,
            reason,
            self._transaction.capture_id if self._transaction else None,
            int(geom.present),
            geom.reason,
            geom.confidence,
            geom.bbox,
            stab.stable,
            stab.stable_count,
            stab.displacement_px,
            self._last_scene_empty,
            self._removal_count,
        )
        ready = target == AppState.IDLE and self.health.overall_ready
        message_state = target
        if target == AppState.IDLE and not ready:
            message_state = AppState.ERROR if not self.health.storage.healthy else target

        self.status.update(
            message_state,
            capture_id=self._transaction.capture_id if self._transaction else None,
            last_success_at=self.health.last_success_at,
            last_error=self.health.last_error,
            **self._health_status_kwargs(),
        )
        self._publish_debug()

    def _publish_debug(self, frame=None, *, camera_error: str | None = None) -> None:
        """Push a debug snapshot. Never mutates ``frame``; the viewer copies it."""
        viewer = self._debug_viewer
        if viewer is None:
            return
        geom = self.detector.last_geometry()
        doc = self.detector.debug_info()
        stab = self.geometry_stability.debug_info()
        camera_healthy = self.health.camera.healthy and not camera_error
        error_msg = camera_error
        if not camera_healthy and error_msg is None:
            error_msg = self.health.camera.message
        available, phase, message = self._manual_status()
        af = self.autofocus.debug
        bbox = geom.bbox if geom.present and geom.bbox is not None else (0, 0, 0, 0)
        viewer.publish(
            frame,
            state=self.sm.state.value,
            document_detected=geom.present,
            motion=doc.motion,
            edge_density=doc.edge_density,
            motion_threshold=doc.motion_threshold,
            edge_density_threshold=doc.edge_density_threshold,
            stable_count=stab.stable_count,
            stable_required=stab.stable_frames_required,
            stable=stab.stable,
            camera_healthy=camera_healthy,
            autofocus_available=self.health.autofocus_available,
            camera_error=error_msg,
            roi_center_fraction=self.config.detection.roi_center_fraction,
            reject_reason=geom.reason,
            roi_width=bbox[2],
            roi_height=bbox[3],
            gray_std=doc.gray_std,
            gray_mean=doc.gray_mean,
            frame_width=doc.frame_width,
            frame_height=doc.frame_height,
            frame_motion=doc.frame_motion,
            manual_capture_available=available,
            manual_phase=phase,
            manual_message=message,
            removal_detected=self._last_scene_empty,
            removal_count=self._removal_count,
            autofocus_running=af.running,
            autofocus_method=af.method,
            autofocus_attempt=af.attempt,
            focus_before=af.focus_before,
            focus_after=af.focus_after,
            focus_roi=af.preview_roi or af.focus_roi,
            lens_before=af.lens_before,
            lens_after=af.lens_after,
            improvement_ratio=af.improvement_ratio,
            preview_focus_roi=af.preview_roi or af.focus_roi,
            sensor_focus_roi=af.sensor_roi,
            preview_roi_wh=af.preview_roi_wh,
            document_roi="{},{},{},{}".format(*bbox) if bbox != (0, 0, 0, 0) else "",
            stability_motion=stab.displacement_px,
            stability_threshold=stab.threshold_px,
            state_reason=self._last_transition_reason,
            manual_capture_requested=self._manual_capture_requested,
            capture_started=self._capture_started,
            upload_started=self._upload_started,
            upload_result=self._upload_result,
            detect_hz=self._last_detect_hz,
            tick_ms=self._last_tick_ms,
        )

    async def _shutdown(self) -> None:
        self._transition(AppState.SHUTDOWN)
        self._cleanup_transaction_files()
        if self._transaction and not self._transaction.is_terminal:
            self._transaction.mark_failed("shutdown")
        self._finish_manual_request(
            {
                "ok": False,
                "status": "failed",
                "state": "SHUTDOWN",
                "message": "Service shutting down",
            }
        )
        self._clear_transaction()
        if self._debug_viewer is not None:
            self._debug_viewer.stop()
        await self.api.stop()
        await asyncio.to_thread(self.camera.shutdown)
        logger.info("Shutdown complete")
