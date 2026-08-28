"""End-to-end capture state-machine behaviour (the Pi bugs unit tests missed)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from idcam.app.orchestrator import ApplicationOrchestrator
from idcam.app.state_machine import AppState
from idcam.app.transaction import ActiveCaptureTransaction
from idcam.camera.autofocus import AutofocusResult
from idcam.config.schema import AppConfig, CameraConfig, DetectionConfig, QualityConfig, ServerConfig, UIConfig
from idcam.exceptions import AutofocusError, CaptureError, ImageValidationError
from idcam.network.protocol import ServerResponse, UploadOutcome
from idcam.system.component_health import ComponentStatus, HealthLevel
from idcam.vision.document_detector import DocumentDetector, DocumentGeometryResult


@pytest.fixture
def app_config(tmp_path):
    return AppConfig(
        device_id="lifecycle",
        temp_dir=tmp_path / "tmp",
        server=ServerConfig(url="https://api.example.com"),
        quality=QualityConfig(
            min_sharpness=10.0,
            min_file_bytes=100,
            min_width=64,
            min_height=64,
            require_expected_dimensions=False,
        ),
        detection=DetectionConfig(
            motion_threshold=5.0,
            edge_density_threshold=0.01,
            stable_frames_required=2,
            removal_frames_required=2,
            geometry_stable_frames_required=2,
            geometry_stability_threshold_px=12.0,
        ),
        camera=CameraConfig(preview_fps=30.0, af_timeout_sec=1.0, af_max_retries=1),
        ui=UIConfig(
            status_file=tmp_path / "run" / "status.json",
            admin_socket=tmp_path / "run" / "admin.sock",
        ),
    )


def _orch(app_config):
    orch = ApplicationOrchestrator(app_config, use_mock_camera=True)
    orch.health.camera = ComponentStatus(level=HealthLevel.HEALTHY, details={"healthy": True})
    orch.disk.can_capture = MagicMock(return_value=True)
    return orch


def _empty_frame():
    return _wood_table((480, 640))


def _wood_table(shape: tuple[int, int], *, brightness: int = 0) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices((h, w))
    base = (85 + (xx * 3 + yy * 5) % 45).astype(np.int16) + brightness
    base = np.clip(base, 0, 255).astype(np.uint8)
    g = (base.astype(np.int16) * 70 // 100).clip(0, 255).astype(np.uint8)
    b = (base.astype(np.int16) * 48 // 100).clip(0, 255).astype(np.uint8)
    return np.stack([base, g, b], axis=-1)


def _document_frame(*, box: tuple[int, int, int, int] = (80, 60, 480, 360)) -> np.ndarray:
    x, y, pw, ph = box
    img = _wood_table((480, 640))
    img[y : y + ph, x : x + pw] = (242, 238, 230)
    img[y : y + 3, x : x + pw] = 35
    img[y + ph - 3 : y + ph, x : x + pw] = 35
    img[y : y + ph, x : x + 3] = 35
    img[y : y + ph, x + pw - 3 : x + pw] = 35
    return img


async def _idle_until_auto_detected(orch, *, ticks: int = 12) -> None:
    for _ in range(ticks):
        await orch._idle_tick()
        if orch.sm.state == AppState.DOCUMENT_DETECTED:
            return
    raise AssertionError(f"auto transaction did not start; state={orch.sm.state}")


class TestStartupGoesIdle:
    @pytest.mark.asyncio
    async def test_startup_idle_not_waiting_for_removal(self, app_config):
        orch = _orch(app_config)
        with patch.object(orch.api, "start", AsyncMock()), patch.object(
            orch.api, "check_connectivity", AsyncMock(return_value=True)
        ):
            await orch._startup()
        assert orch.sm.state == AppState.IDLE
        assert orch.sm.state != AppState.WAITING_FOR_REMOVAL
        assert not orch.has_active_transaction
        available, _phase, _msg = orch._manual_status()
        assert available is True


class TestAutoCapturePath:
    @pytest.mark.asyncio
    async def test_idle_document_starts_detected(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        await _idle_until_auto_detected(orch)
        assert orch.sm.state == AppState.DOCUMENT_DETECTED
        assert orch._transaction is not None
        assert orch._transaction.trigger == "auto"

    @pytest.mark.asyncio
    async def test_detected_to_stability(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.DOCUMENT_DETECTED
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1")
        await orch._document_detected_tick()
        assert orch.sm.state == AppState.WAITING_FOR_STABILITY

    @pytest.mark.asyncio
    async def test_stability_to_focusing(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.WAITING_FOR_STABILITY
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1")
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        for _ in range(8):
            await orch._stability_tick()
            if orch.sm.state == AppState.FOCUSING:
                break
        assert orch.sm.state == AppState.FOCUSING

    @pytest.mark.asyncio
    async def test_focusing_to_capturing(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.FOCUSING
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1")
        page = _document_frame()
        orch._snapshot_accepted_geometry(orch.detector.detect_geometry(page))
        orch._safe_preview = AsyncMock(return_value=page)
        orch._check_motion_abort = AsyncMock(return_value=True)
        orch.autofocus.run_autofocus = AsyncMock(
            return_value=AutofocusResult(True, 90.0, 1, None, "hardware_cycle", 10.0)
        )
        orch.autofocus.verify_focus = AsyncMock(
            return_value=AutofocusResult(True, 90.0, 0, None, "verification", 10.0)
        )
        await orch._focusing_tick()
        assert orch.sm.state == AppState.FOCUS_VERIFICATION
        await orch._focus_verification_tick()
        assert orch.sm.state == AppState.CAPTURING

    @pytest.mark.asyncio
    async def test_capturing_to_upload(self, app_config, tmp_path):
        orch = _orch(app_config)
        orch.sm.state = AppState.CAPTURING
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1")
        orch._transaction.mark_focus_completed(90.0, 10.0)
        page = _document_frame()
        orch._snapshot_accepted_geometry(orch.detector.detect_geometry(page))
        orch._check_motion_abort = AsyncMock(return_value=True)

        async def fake_capture(_camera, path):
            path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 500)
            return path

        orch.capture_service.capture = fake_capture
        orch.validator.validate = MagicMock(
            return_value=MagicMock(file_bytes=500, width=100, height=80, sha256="ab", sharpness=90.0)
        )
        await orch._capturing_tick()
        assert orch.sm.state == AppState.IMAGE_VALIDATION
        await orch._validation_tick()
        assert orch.sm.state == AppState.UPLOADING

    @pytest.mark.asyncio
    async def test_success_to_waiting_for_removal(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.SUCCESS
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-ok")
        orch._transaction.mark_completed()
        with patch("idcam.app.orchestrator.asyncio.sleep", AsyncMock()):
            await orch._success_tick()
        assert orch.sm.state == AppState.WAITING_FOR_REMOVAL


class TestIdleDoesNotStayWhenReady:
    @pytest.mark.asyncio
    async def test_awaiting_clear_scene_does_not_retry_same_document(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        orch._awaiting_clear_scene = True
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        for _ in range(6):
            await orch._idle_tick()
        assert orch.sm.state == AppState.IDLE
        assert orch._awaiting_clear_scene is True
        assert orch._transaction is None


class TestAutoDetectStableFocusCapture:
    @pytest.mark.asyncio
    async def test_detect_stable_focus_capture(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        orch.autofocus.run_autofocus = AsyncMock(
            return_value=AutofocusResult(True, 35.0, 1, None, "hardware_trigger", 28.0, 10.0, True)
        )
        orch.autofocus.verify_focus = AsyncMock(
            return_value=AutofocusResult(True, 35.0, 0, None, "verification", 28.0)
        )

        async def fake_capture(_camera, path):
            path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 500)
            return path

        orch.capture_service.capture = fake_capture
        await _idle_until_auto_detected(orch)
        assert orch.sm.state == AppState.DOCUMENT_DETECTED
        await orch._document_detected_tick()
        assert orch.sm.state == AppState.WAITING_FOR_STABILITY
        for _ in range(8):
            await orch._stability_tick()
            if orch.sm.state == AppState.FOCUSING:
                break
        assert orch.sm.state == AppState.FOCUSING
        await orch._focusing_tick()
        assert orch.sm.state == AppState.FOCUS_VERIFICATION
        await orch._focus_verification_tick()
        assert orch.sm.state == AppState.CAPTURING
        await orch._capturing_tick()
        assert orch.sm.state == AppState.IMAGE_VALIDATION

    @pytest.mark.asyncio
    async def test_validation_failure_does_not_upload(self, app_config, tmp_path):
        orch = _orch(app_config)
        orch.sm.state = AppState.IMAGE_VALIDATION
        orch._resume_state = AppState.IDLE
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-blur")
        path = tmp_path / "cap.jpg"
        path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 200)
        orch._capture_path = path
        orch.validator.validate = MagicMock(side_effect=ImageValidationError("Image too blurry"))
        orch.api.upload_document = AsyncMock()
        await orch._validation_tick()
        orch.api.upload_document.assert_not_called()
        assert orch.sm.state != AppState.UPLOADING
        assert orch.sm.state != AppState.SUCCESS


class TestRemovalHold:
    @pytest.mark.asyncio
    async def test_document_still_present_stays_in_removal(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.WAITING_FOR_REMOVAL
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-ok")
        orch._transaction.mark_completed()
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        await orch._removal_tick()
        await orch._removal_tick()
        assert orch.sm.state == AppState.WAITING_FOR_REMOVAL
        available, _phase, _msg = orch._manual_status()
        assert available is True

    @pytest.mark.asyncio
    async def test_document_removed_returns_idle(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.WAITING_FOR_REMOVAL
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-ok")
        orch._transaction.mark_completed()
        empty = _empty_frame()
        orch._safe_preview = AsyncMock(return_value=empty)
        await orch._removal_tick()
        await orch._removal_tick()
        assert orch.sm.state == AppState.IDLE
        assert orch._transaction is None


class TestManualCapturePolicy:
    def test_manual_from_idle(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        available, _phase, _msg = orch._manual_status()
        assert available is True

    def test_manual_from_waiting_for_removal(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.WAITING_FOR_REMOVAL
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-ok")
        orch._transaction.mark_completed()
        available, _phase, _msg = orch._manual_status()
        assert available is True
        result = orch.request_manual_capture(timeout=0.05)
        assert result["status"] == "timeout" or result.get("ok") is False
        assert orch._pending_command in {"manual_capture", None}

    def test_manual_busy_while_focusing(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.FOCUSING
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-busy")
        available, _phase, _msg = orch._manual_status()
        assert available is False
        result = orch.request_manual_capture(timeout=0.1)
        assert result["status"] == "busy"
        assert result["state"] == "FOCUSING"

    def test_manual_busy_while_uploading(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.UPLOADING
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-up")
        available, _phase, _msg = orch._manual_status()
        assert available is False
        result = orch.request_manual_capture(timeout=0.1)
        assert result["status"] == "busy"

    @pytest.mark.asyncio
    async def test_removal_tick_starts_manual_pipeline(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.WAITING_FOR_REMOVAL
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-old")
        orch._transaction.mark_completed()
        orch._pending_command = "manual_capture"
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        await orch._removal_tick()
        assert orch.sm.state == AppState.DOCUMENT_DETECTED
        assert orch._transaction.trigger == "manual"
        assert orch._resume_state == AppState.WAITING_FOR_REMOVAL

    @pytest.mark.asyncio
    async def test_failure_from_idle_capture_returns_idle_not_removal(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.FOCUSING
        orch._resume_state = AppState.IDLE
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-fail")
        await orch._fail_transaction_and_wait("focus_failed", "soft")
        assert orch.sm.state == AppState.IDLE
        assert orch._awaiting_clear_scene is True
        available, _phase, _msg = orch._manual_status()
        assert available is True


class TestPreviewTickObservability:
    def test_note_preview_tick_sets_detect_hz_not_camera_fps(self, app_config):
        orch = ApplicationOrchestrator(app_config, use_mock_camera=True)
        orch._note_preview_tick(0.04)
        orch._note_preview_tick(0.04)
        assert orch._last_tick_ms == pytest.approx(40.0, abs=0.1)
        assert orch._last_detect_hz > 0
        assert orch._preview_tick_count == 2


class TestPhase2CGeometryPolicy:
    @pytest.mark.asyncio
    async def test_boot_with_page_geometry_present_without_baseline(self, app_config):
        orch = _orch(app_config)
        page = _document_frame()
        geom = orch.detector.detect_geometry(page)
        assert geom.present is True
        assert orch.detector._baseline is None
        orch.sm.state = AppState.IDLE
        orch._safe_preview = AsyncMock(return_value=page)
        await orch._idle_tick()
        assert orch.detector.last_geometry().present is True
        assert orch._transaction is None
        assert orch.sm.state == AppState.IDLE

    @pytest.mark.asyncio
    async def test_one_geometry_frame_does_not_start_transaction(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        await orch._idle_tick()
        assert orch.sm.state == AppState.IDLE
        assert orch._transaction is None

    @pytest.mark.asyncio
    async def test_stable_geometry_eventually_arms(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        await _idle_until_auto_detected(orch)
        assert orch.geometry_stability.stable_count >= app_config.detection.geometry_stable_frames_required
        assert orch._transaction.trigger == "auto"

    @pytest.mark.asyncio
    async def test_moving_document_does_not_become_stable(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        boxes = [(40, 40, 480, 360), (90, 50, 480, 360), (140, 70, 480, 360), (190, 90, 400, 300)]
        frames = [_document_frame(box=b) for b in boxes]
        idx = {"i": 0}

        async def next_frame():
            frame = frames[idx["i"] % len(frames)]
            idx["i"] += 1
            return frame

        orch._safe_preview = AsyncMock(side_effect=next_frame)
        for _ in range(8):
            await orch._idle_tick()
        assert orch.sm.state == AppState.IDLE
        assert orch._transaction is None
        assert orch.geometry_stability.stable_count < app_config.detection.geometry_stable_frames_required

    @pytest.mark.asyncio
    async def test_lighting_change_on_empty_scene_is_not_a_document(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        frames = [_wood_table((480, 640), brightness=d) for d in (0, 25, 50, 80)]
        idx = {"i": 0}

        async def next_frame():
            frame = frames[min(idx["i"], len(frames) - 1)]
            idx["i"] += 1
            return frame

        orch._safe_preview = AsyncMock(side_effect=next_frame)
        for _ in range(6):
            await orch._idle_tick()
        assert orch.sm.state == AppState.IDLE
        assert orch._transaction is None
        assert orch.detector.last_geometry().present is False

        orch.sm.state = AppState.WAITING_FOR_REMOVAL
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-ok")
        orch._transaction.mark_completed()
        orch._removal_count = 0
        bright = _wood_table((480, 640), brightness=80)
        orch._safe_preview = AsyncMock(return_value=bright)
        await orch._removal_tick()
        await orch._removal_tick()
        assert orch.sm.state == AppState.IDLE

    @pytest.mark.asyncio
    async def test_geometry_absence_removes_document(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.WAITING_FOR_REMOVAL
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-ok")
        orch._transaction.mark_completed()
        orch._safe_preview = AsyncMock(return_value=_empty_frame())
        await orch._removal_tick()
        assert orch.sm.state == AppState.WAITING_FOR_REMOVAL
        await orch._removal_tick()
        assert orch.sm.state == AppState.IDLE

    @pytest.mark.asyncio
    async def test_failed_auto_capture_does_not_retry_same_document(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.FOCUSING
        orch._resume_state = AppState.IDLE
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-fail")
        await orch._fail_transaction_and_wait("focus_failed", "soft")
        assert orch._awaiting_clear_scene is True
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        for _ in range(8):
            await orch._idle_tick()
        assert orch.sm.state == AppState.IDLE
        assert orch._transaction is None
        assert orch._awaiting_clear_scene is True

        orch._safe_preview = AsyncMock(return_value=_empty_frame())
        for _ in range(12):
            await orch._idle_tick()
            if orch._awaiting_clear_scene is False:
                break
        assert orch._awaiting_clear_scene is False

        orch._safe_preview = AsyncMock(return_value=_document_frame())
        await _idle_until_auto_detected(orch)
        assert orch._transaction.capture_id != "cap-fail"

    @pytest.mark.asyncio
    async def test_capture_time_check_does_not_mutate_baseline(self, app_config):
        orch = _orch(app_config)
        page = _document_frame()
        geom = orch.detector.detect_geometry(page)
        orch.geometry_stability.update(geom)
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1")
        orch.sm.state = AppState.CAPTURING
        orch._safe_preview = AsyncMock(return_value=page)
        with patch("idcam.vision.document_detector.cv2.accumulateWeighted") as acc, patch.object(
            orch.detector, "classify", wraps=orch.detector.classify
        ) as classify:
            ok = await orch._check_motion_abort()
        assert ok is True
        acc.assert_not_called()
        classify.assert_not_called()
        assert orch.detector._baseline is None

    @pytest.mark.asyncio
    async def test_geometry_is_single_roi_source(self, app_config):
        orch = _orch(app_config)
        page = _document_frame()
        geom = orch.detector.detect_geometry(page)
        assert geom.present and geom.bbox is not None
        roi = orch.detector.document_preview_roi(page)
        assert roi == geom.bbox
        viewer = MagicMock()
        orch._debug_viewer = viewer
        orch._publish_debug(page)
        kwargs = viewer.publish.call_args.kwargs
        assert kwargs["document_detected"] is True
        assert kwargs["document_roi"] == "{},{},{},{}".format(*geom.bbox)
        assert kwargs["reject_reason"] == geom.reason

    @pytest.mark.asyncio
    async def test_legacy_motion_edge_is_not_authoritative(self, app_config):
        orch = _orch(app_config)
        page = _document_frame()
        legacy = DocumentDetector(app_config.detection)
        assert legacy.detect(page) is False
        assert DocumentDetector(app_config.detection).detect_geometry(page).present is True
        orch.detector.detect = MagicMock(side_effect=AssertionError("legacy detect must not be used"))
        orch.sm.state = AppState.IDLE
        orch._safe_preview = AsyncMock(return_value=page)
        await _idle_until_auto_detected(orch)
        orch.detector.detect.assert_not_called()
        assert orch._transaction is not None


class TestPhase2DFocusCaptureHandoff:
    @pytest.mark.asyncio
    async def test_ready_triggers_autofocus_exactly_once(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        orch.autofocus.run_autofocus = AsyncMock(
            return_value=AutofocusResult(True, 35.0, 1, None, "hardware_trigger", 28.0, 10.0, True)
        )
        await _idle_until_auto_detected(orch)
        await orch._document_detected_tick()
        for _ in range(8):
            await orch._stability_tick()
            if orch.sm.state == AppState.FOCUSING:
                break
        assert orch.sm.state == AppState.FOCUSING
        await orch._focusing_tick()
        assert orch.autofocus.run_autofocus.await_count == 1
        orch.sm.state = AppState.FOCUSING
        orch._state_action_claimed = AppState.FOCUSING
        await orch._focusing_tick()
        assert orch.autofocus.run_autofocus.await_count == 1
        roi = orch.autofocus.run_autofocus.await_args.kwargs["preview_roi"]
        assert roi == orch._accepted_geometry.bbox

    @pytest.mark.asyncio
    async def test_autofocus_failure_is_not_capture_success(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.FOCUSING
        orch._resume_state = AppState.IDLE
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-af-fail")
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        orch.autofocus.run_autofocus = AsyncMock(side_effect=AutofocusError("lens failed"))
        await orch._focusing_tick()
        assert orch.sm.state == AppState.IDLE
        assert orch.sm.state != AppState.CAPTURING
        assert orch.sm.state != AppState.SUCCESS
        assert orch._awaiting_clear_scene is True
        assert orch._capture_path is None

    @pytest.mark.asyncio
    async def test_successful_autofocus_proceeds_to_capture(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.FOCUSING
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1")
        page = _document_frame()
        orch._snapshot_accepted_geometry(orch.detector.detect_geometry(page))
        orch._safe_preview = AsyncMock(return_value=page)
        orch.autofocus.run_autofocus = AsyncMock(
            return_value=AutofocusResult(True, 90.0, 1, None, "hardware_cycle", 10.0)
        )
        orch.autofocus.verify_focus = AsyncMock(
            return_value=AutofocusResult(True, 90.0, 0, None, "verification", 10.0)
        )
        await orch._focusing_tick()
        assert orch.sm.state == AppState.FOCUS_VERIFICATION
        await orch._focus_verification_tick()
        assert orch.sm.state == AppState.CAPTURING

    @pytest.mark.asyncio
    async def test_capture_failure_does_not_become_success(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.CAPTURING
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1")
        orch._transaction.mark_focus_completed(90.0, 10.0)
        page = _document_frame()
        orch._snapshot_accepted_geometry(orch.detector.detect_geometry(page))
        orch._safe_preview = AsyncMock(return_value=page)
        orch.capture_service.capture = AsyncMock(side_effect=CaptureError("encoder failed"))
        await orch._capturing_tick()
        assert orch.sm.state != AppState.IMAGE_VALIDATION
        assert orch.sm.state != AppState.SUCCESS
        assert orch.sm.state == AppState.RECOVERY

    @pytest.mark.asyncio
    async def test_capture_time_geometry_movement_aborts(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.CAPTURING
        orch._resume_state = AppState.IDLE
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1")
        orch._transaction.mark_focus_completed(90.0, 10.0)
        still = _document_frame(box=(80, 60, 480, 360))
        moved = _document_frame(box=(180, 120, 400, 300))
        orch._snapshot_accepted_geometry(orch.detector.detect_geometry(still))
        orch._safe_preview = AsyncMock(return_value=moved)
        capture = AsyncMock()
        orch.capture_service.capture = capture
        await orch._capturing_tick()
        capture.assert_not_called()
        assert orch.sm.state != AppState.IMAGE_VALIDATION
        assert orch.sm.state != AppState.SUCCESS
        assert orch._awaiting_clear_scene is True

    @pytest.mark.asyncio
    async def test_stable_geometry_allows_capture(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.CAPTURING
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1")
        orch._transaction.mark_focus_completed(90.0, 10.0)
        page = _document_frame()
        orch._snapshot_accepted_geometry(orch.detector.detect_geometry(page))
        orch._safe_preview = AsyncMock(return_value=page)

        async def fake_capture(_camera, path):
            path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 500)
            return path

        orch.capture_service.capture = fake_capture
        await orch._capturing_tick()
        assert orch.sm.state == AppState.IMAGE_VALIDATION
        assert orch._capture_path is not None

    @pytest.mark.asyncio
    async def test_accepted_roi_not_replaced_by_search_rect(self, app_config):
        orch = _orch(app_config)
        page = _document_frame()
        geom = orch.detector.detect_geometry(page)
        orch._snapshot_accepted_geometry(geom)
        search = orch.detector.roi_rect(page.shape)
        assert geom.bbox != search
        assert orch._accepted_preview_roi(page) == geom.bbox
        empty = _empty_frame()
        assert orch._accepted_preview_roi(empty) == geom.bbox

    @pytest.mark.asyncio
    async def test_auto_focusing_without_geometry_does_not_center_af(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.FOCUSING
        orch._resume_state = AppState.IDLE
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1", trigger="auto")
        orch._safe_preview = AsyncMock(return_value=_empty_frame())
        orch.autofocus.run_autofocus = AsyncMock()
        await orch._focusing_tick()
        orch.autofocus.run_autofocus.assert_not_called()
        assert orch.sm.state != AppState.FOCUS_VERIFICATION
        assert orch._awaiting_clear_scene is True

    @pytest.mark.asyncio
    async def test_invalid_accepted_bbox_does_not_fall_back_to_search(self, app_config):
        orch = _orch(app_config)
        page = _document_frame()
        search = orch.detector.roi_rect(page.shape)
        orch._accepted_geometry = DocumentGeometryResult(
            present=True,
            confidence=0.9,
            corners=((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)),
            bbox=(0, 0, 2, 2),
            reason="quad",
        )
        assert orch._accepted_preview_roi(page, allow_search_fallback=False) is None
        assert orch._accepted_preview_roi(page, allow_search_fallback=False) != search

    @pytest.mark.asyncio
    async def test_document_moved_after_af_is_not_accepted(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.FOCUSING
        orch._resume_state = AppState.IDLE
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1")
        still = _document_frame(box=(80, 60, 480, 360))
        moved = _document_frame(box=(180, 120, 400, 300))
        orch._snapshot_accepted_geometry(orch.detector.detect_geometry(still))
        orch._safe_preview = AsyncMock(side_effect=[still, moved])
        orch.autofocus.run_autofocus = AsyncMock(
            return_value=AutofocusResult(True, 90.0, 1, None, "hardware_cycle", 10.0)
        )
        await orch._focusing_tick()
        assert orch.sm.state != AppState.FOCUS_VERIFICATION
        assert orch._awaiting_clear_scene is True


class TestPhase4CaptureGate:
    @pytest.mark.asyncio
    async def test_capture_blocked_without_verified_focus(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.CAPTURING
        orch._resume_state = AppState.IDLE
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1")
        page = _document_frame()
        orch._snapshot_accepted_geometry(orch.detector.detect_geometry(page))
        orch._safe_preview = AsyncMock(return_value=page)
        orch.capture_service.capture = AsyncMock()
        await orch._capturing_tick()
        orch.capture_service.capture.assert_not_called()
        assert orch.sm.state != AppState.IMAGE_VALIDATION

    @pytest.mark.asyncio
    async def test_auto_capture_blocked_without_accepted_geometry(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.CAPTURING
        orch._resume_state = AppState.IDLE
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1", trigger="auto")
        orch._transaction.mark_focus_completed(90.0, 10.0)
        orch.capture_service.capture = AsyncMock()
        await orch._capturing_tick()
        orch.capture_service.capture.assert_not_called()

    @pytest.mark.asyncio
    async def test_one_shot_capturing_does_not_double_capture(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.CAPTURING
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1")
        orch._transaction.mark_focus_completed(90.0, 10.0)
        page = _document_frame()
        orch._snapshot_accepted_geometry(orch.detector.detect_geometry(page))
        orch._safe_preview = AsyncMock(return_value=page)
        orch.capture_service.capture = AsyncMock()
        orch._state_action_claimed = AppState.CAPTURING
        await orch._capturing_tick()
        orch.capture_service.capture.assert_not_called()



