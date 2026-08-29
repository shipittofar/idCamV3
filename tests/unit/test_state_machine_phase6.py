"""Phase 6 state-machine behaviour: hysteresis, one pipeline, bounded failure."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from idcam.app.orchestrator import ApplicationOrchestrator
from idcam.app.state_machine import AppState
from idcam.app.transaction import ActiveCaptureTransaction
from idcam.camera.autofocus import AutofocusResult
from idcam.config.schema import AppConfig, CameraConfig, DetectionConfig, QualityConfig, ServerConfig, UIConfig
from idcam.exceptions import AutofocusError, CameraError, CaptureError, ImageValidationError, NetworkError, RecoveryExhaustedError, StateMachineError
from idcam.network.protocol import ServerResponse, UploadOutcome
from idcam.system.component_health import ComponentStatus, HealthLevel


@pytest.fixture
def app_config(tmp_path):
    return AppConfig(
        device_id="phase6",
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
        camera=CameraConfig(preview_fps=30.0, af_timeout_sec=1.0, af_max_retries=1, af_verify_max_retries=2),
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


def _wood_table(shape: tuple[int, int], *, brightness: int = 0) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices((h, w))
    base = (85 + (xx * 3 + yy * 5) % 45).astype(np.int16) + brightness
    base = np.clip(base, 0, 255).astype(np.uint8)
    g = (base.astype(np.int16) * 70 // 100).clip(0, 255).astype(np.uint8)
    b = (base.astype(np.int16) * 48 // 100).clip(0, 255).astype(np.uint8)
    return np.stack([base, g, b], axis=-1)


def _empty_frame():
    return _wood_table((480, 640))


def _document_frame(*, box: tuple[int, int, int, int] = (80, 60, 480, 360), brightness: int = 0) -> np.ndarray:
    x, y, pw, ph = box
    img = _wood_table((480, 640), brightness=brightness)
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


def _accepted_upload(capture_id: str) -> ServerResponse:
    return ServerResponse(
        success=True,
        capture_id=capture_id,
        result="ok",
        message=None,
        http_status=200,
        outcome=UploadOutcome.ACCEPTED,
    )


class TestEmptyScene:
    @pytest.mark.asyncio
    async def test_idle_empty_scene_does_not_start_pipeline(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        orch._safe_preview = AsyncMock(return_value=_empty_frame())
        orch.autofocus.run_autofocus = AsyncMock()
        orch.capture_service.capture = AsyncMock()
        orch.api.upload_document = AsyncMock()
        for _ in range(8):
            await orch._idle_tick()
        assert orch.sm.state == AppState.IDLE
        assert orch._transaction is None
        orch.autofocus.run_autofocus.assert_not_called()
        orch.capture_service.capture.assert_not_called()
        orch.api.upload_document.assert_not_called()


class TestDocumentLifecycle:
    @pytest.mark.asyncio
    async def test_document_appears_then_moves_then_stabilizes(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        still = _document_frame(box=(80, 60, 480, 360))
        moved = _document_frame(box=(180, 120, 400, 300))
        orch._safe_preview = AsyncMock(return_value=still)
        await _idle_until_auto_detected(orch)
        await orch._document_detected_tick()
        assert orch.sm.state == AppState.WAITING_FOR_STABILITY
        orch._safe_preview = AsyncMock(side_effect=[moved, still, still, still, still, still, still])
        await orch._stability_tick()
        assert orch.sm.state == AppState.WAITING_FOR_STABILITY
        for _ in range(8):
            if orch.sm.state == AppState.FOCUSING:
                break
            await orch._stability_tick()
        assert orch.sm.state == AppState.FOCUSING

    @pytest.mark.asyncio
    async def test_stable_document_triggers_af_once(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.FOCUSING
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1", trigger="auto")
        page = _document_frame()
        orch._snapshot_accepted_geometry(orch.detector.detect_geometry(page))
        orch._safe_preview = AsyncMock(return_value=page)
        async def slow_af(*_a, **_k):
            await asyncio.sleep(0.05)
            return AutofocusResult(True, 90.0, 1, None, "hardware_cycle", 10.0)

        orch.autofocus.run_autofocus = AsyncMock(side_effect=slow_af)
        first = asyncio.create_task(orch._focusing_tick())
        await asyncio.sleep(0.01)
        await orch._focusing_tick()
        await first
        assert orch.autofocus.run_autofocus.await_count == 1
        assert orch.sm.state == AppState.FOCUS_VERIFICATION

    @pytest.mark.asyncio
    async def test_successful_pipeline_to_upload(self, app_config, tmp_path):
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
        orch.api.upload_document = AsyncMock(return_value=_accepted_upload("cap-1"))
        await orch._capturing_tick()
        assert orch.sm.state == AppState.IMAGE_VALIDATION
        await orch._validation_tick()
        assert orch.sm.state == AppState.UPLOADING
        await orch._uploading_tick()
        assert orch.sm.state == AppState.WAITING_FOR_SERVER
        await orch._waiting_server_tick()
        await orch._processing_response_tick()
        assert orch.sm.state == AppState.SUCCESS
        orch.api.upload_document.assert_awaited_once()


class TestFailures:
    @pytest.mark.asyncio
    async def test_af_failure_is_bounded_and_does_not_loop(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.FOCUSING
        orch._resume_state = AppState.IDLE
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-af", trigger="auto")
        page = _document_frame()
        orch._snapshot_accepted_geometry(orch.detector.detect_geometry(page))
        orch._safe_preview = AsyncMock(return_value=page)
        orch.autofocus.run_autofocus = AsyncMock(
            side_effect=AutofocusError("no lock", reason="autofocus_timeout")
        )
        orch.camera.restart = MagicMock()
        await orch._focusing_tick()
        orch.camera.restart.assert_not_called()
        assert orch.sm.state == AppState.IDLE
        assert orch._awaiting_clear_scene is True
        orch.autofocus.run_autofocus.assert_awaited_once()
        orch._safe_preview = AsyncMock(return_value=page)
        for _ in range(6):
            await orch._idle_tick()
        assert orch.sm.state == AppState.IDLE
        assert orch._transaction is None

    @pytest.mark.asyncio
    async def test_focus_verify_retries_then_fails(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.FOCUS_VERIFICATION
        orch._resume_state = AppState.IDLE
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-vf", trigger="auto")
        page = _document_frame()
        orch._snapshot_accepted_geometry(orch.detector.detect_geometry(page))
        orch.autofocus.verify_focus = AsyncMock(side_effect=AutofocusError("verify", reason="focus_verification_failed"))
        await orch._focus_verification_tick()
        assert orch.sm.state == AppState.FOCUSING
        orch.sm.state = AppState.FOCUS_VERIFICATION
        orch._state_action_claimed = None
        await orch._focus_verification_tick()
        assert orch.sm.state == AppState.IDLE
        assert orch.autofocus.verify_focus.await_count == 2

    @pytest.mark.asyncio
    async def test_capture_failure_does_not_succeed(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.CAPTURING
        orch._resume_state = AppState.IDLE
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1")
        orch._transaction.mark_focus_completed(90.0, 10.0)
        page = _document_frame()
        orch._snapshot_accepted_geometry(orch.detector.detect_geometry(page))
        orch._check_motion_abort = AsyncMock(return_value=True)
        orch.capture_service.capture = AsyncMock(side_effect=CaptureError("encoder failed"))
        orch.camera.restart = MagicMock()
        await orch._capturing_tick()
        orch.camera.restart.assert_not_called()
        assert orch.sm.state == AppState.IDLE
        assert orch.sm.state != AppState.IMAGE_VALIDATION
        assert orch.sm.state != AppState.SUCCESS
        assert orch.sm.state != AppState.RECOVERY

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
        orch.camera.restart = MagicMock()
        await orch._validation_tick()
        orch.api.upload_document.assert_not_called()
        orch.camera.restart.assert_not_called()
        assert orch.sm.state == AppState.IDLE
        assert orch._awaiting_clear_scene is True

    @pytest.mark.asyncio
    async def test_upload_failure_does_not_succeed(self, app_config, tmp_path):
        orch = _orch(app_config)
        orch.sm.state = AppState.UPLOADING
        orch._resume_state = AppState.IDLE
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-up")
        path = tmp_path / "cap.jpg"
        path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 200)
        orch._capture_path = path
        orch.api.upload_document = AsyncMock(side_effect=NetworkError("timeout"))
        orch.camera.restart = MagicMock()
        await orch._uploading_tick()
        orch.camera.restart.assert_not_called()
        assert orch.sm.state == AppState.IDLE
        assert orch.sm.state != AppState.SUCCESS
        assert orch._awaiting_clear_scene is True


class TestSameDocumentAndRemoval:
    @pytest.mark.asyncio
    async def test_same_document_does_not_recapture_after_success(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.WAITING_FOR_REMOVAL
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-ok")
        orch._transaction.mark_completed()
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        orch.autofocus.run_autofocus = AsyncMock()
        for _ in range(6):
            await orch._removal_tick()
        assert orch.sm.state == AppState.WAITING_FOR_REMOVAL
        orch.autofocus.run_autofocus.assert_not_called()

    @pytest.mark.asyncio
    async def test_document_removal_returns_idle_then_replacement_starts(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.WAITING_FOR_REMOVAL
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-ok")
        orch._transaction.mark_completed()
        orch._safe_preview = AsyncMock(return_value=_empty_frame())
        await orch._removal_tick()
        await orch._removal_tick()
        assert orch.sm.state == AppState.IDLE
        assert orch._transaction is None
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        await _idle_until_auto_detected(orch)
        assert orch._transaction is not None
        assert orch._transaction.capture_id != "cap-ok"

    @pytest.mark.asyncio
    async def test_single_bad_frame_does_not_abandon_stability(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.WAITING_FOR_STABILITY
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1")
        orch._safe_preview = AsyncMock(side_effect=[_empty_frame(), _document_frame(), _document_frame(), _document_frame()])
        await orch._stability_tick()
        assert orch.sm.state == AppState.WAITING_FOR_STABILITY
        assert orch._transaction is not None
        for _ in range(6):
            if orch.sm.state == AppState.FOCUSING:
                break
            await orch._stability_tick()
        assert orch.sm.state == AppState.FOCUSING

    @pytest.mark.asyncio
    async def test_consecutive_absence_abandons_stability(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.WAITING_FOR_STABILITY
        orch._resume_state = AppState.IDLE
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1")
        orch._safe_preview = AsyncMock(return_value=_empty_frame())
        await orch._stability_tick()
        assert orch.sm.state == AppState.WAITING_FOR_STABILITY
        await orch._stability_tick()
        assert orch.sm.state == AppState.IDLE
        assert orch._awaiting_clear_scene is True

    @pytest.mark.asyncio
    async def test_small_geometry_jitter_does_not_leave_waiting(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.WAITING_FOR_STABILITY
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1")
        a = _document_frame(box=(80, 60, 480, 360), brightness=0)
        b = _document_frame(box=(82, 61, 478, 358), brightness=8)
        orch._safe_preview = AsyncMock(side_effect=[a, b, b, b, b, b, b, b])
        states = []
        for _ in range(8):
            await orch._stability_tick()
            states.append(orch.sm.state)
            if orch.sm.state == AppState.FOCUSING:
                break
        assert AppState.IDLE not in states
        assert AppState.WAITING_FOR_REMOVAL not in states
        assert orch.sm.state in {AppState.WAITING_FOR_STABILITY, AppState.FOCUSING}


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_long_upload_does_not_duplicate(self, app_config, tmp_path):
        orch = _orch(app_config)
        orch.sm.state = AppState.UPLOADING
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-slow")
        path = tmp_path / "cap.jpg"
        path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 200)
        orch._capture_path = path
        calls = []

        async def slow_upload(request):
            calls.append(request.capture_id)
            await asyncio.sleep(0.05)
            return _accepted_upload(request.capture_id)

        orch.api.upload_document = slow_upload
        first = asyncio.create_task(orch._uploading_tick())
        await asyncio.sleep(0.01)
        await orch._uploading_tick()
        idle = orch._idle_tick()
        await idle
        result = orch.request_manual_capture(timeout=0.05)
        assert result["status"] == "busy"
        await first
        assert calls == ["cap-slow"]
        assert orch.sm.state == AppState.WAITING_FOR_SERVER

    @pytest.mark.asyncio
    async def test_stale_validation_cannot_advance_newer_pipeline(self, app_config, tmp_path):
        orch = _orch(app_config)
        orch.sm.state = AppState.IMAGE_VALIDATION
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-old")
        path = tmp_path / "cap.jpg"
        path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 200)
        orch._capture_path = path

        def validate(*_a, **_k):
            orch._transaction = ActiveCaptureTransaction(capture_id="cap-new")
            orch.sm.state = AppState.FOCUSING
            orch._state_action_claimed = AppState.FOCUSING
            return MagicMock(file_bytes=500, width=100, height=80, sha256="ab", sharpness=90.0)

        orch.validator.validate = validate
        await orch._validation_tick()
        assert orch.sm.state == AppState.FOCUSING
        assert orch._transaction.capture_id == "cap-new"

    @pytest.mark.asyncio
    async def test_stale_upload_cannot_advance_newer_pipeline(self, app_config, tmp_path):
        orch = _orch(app_config)
        orch.sm.state = AppState.UPLOADING
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-old")
        path = tmp_path / "cap.jpg"
        path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 200)
        orch._capture_path = path

        async def upload(_request):
            orch._transaction = ActiveCaptureTransaction(capture_id="cap-new")
            orch.sm.state = AppState.FOCUSING
            return _accepted_upload("cap-old")

        orch.api.upload_document = upload
        await orch._uploading_tick()
        assert orch.sm.state == AppState.FOCUSING
        assert orch._transaction.capture_id == "cap-new"

    def test_manual_capture_rejected_during_automatic_pipeline(self, app_config):
        orch = _orch(app_config)
        for state in (
            AppState.FOCUSING,
            AppState.FOCUS_VERIFICATION,
            AppState.CAPTURING,
            AppState.IMAGE_VALIDATION,
            AppState.UPLOADING,
            AppState.WAITING_FOR_SERVER,
            AppState.PROCESSING_RESPONSE,
        ):
            orch.sm.state = state
            orch._transaction = ActiveCaptureTransaction(capture_id="cap-auto", trigger="auto")
            result = orch.request_manual_capture(timeout=0.05)
            assert result["ok"] is False
            assert result["status"] == "busy"

    @pytest.mark.asyncio
    async def test_recovery_exhausts_to_error(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.RECOVERY
        orch.camera.restart = MagicMock(side_effect=CameraError("no device"))
        with patch("idcam.app.orchestrator.asyncio.sleep", AsyncMock()):
            with pytest.raises(RecoveryExhaustedError):
                for _ in range(orch.config.camera.recovery_max_attempts + 2):
                    orch._next_recovery_at = 0.0
                    await orch._recovery_tick()
        assert orch.sm.state == AppState.ERROR


class TestRejectedTransitionLog:
    def test_invalid_transition_logs_and_raises(self, app_config, caplog):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        with pytest.raises(StateMachineError):
            orch._transition(AppState.SUCCESS, reason="illegal")
        assert orch.sm.state == AppState.IDLE
        assert "state_transition_rejected" in caplog.text
        assert "from_state=IDLE" in caplog.text
        assert "to_state=SUCCESS" in caplog.text


class TestBootAndIdle:
    @pytest.mark.asyncio
    async def test_startup_reaches_idle(self, app_config):
        orch = _orch(app_config)
        with patch.object(orch.api, "start", AsyncMock()), patch.object(
            orch.api, "check_connectivity", AsyncMock(return_value=True)
        ):
            await orch._startup()
        assert orch.sm.state == AppState.IDLE
        assert orch._transaction is None


class TestStabilityGate:
    @pytest.mark.asyncio
    async def test_unstable_geometry_does_not_trigger_af(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.WAITING_FOR_STABILITY
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1")
        still = _document_frame(box=(80, 60, 480, 360))
        moved = _document_frame(box=(200, 160, 360, 260))
        orch._safe_preview = AsyncMock(side_effect=[still, moved, still, moved, still, moved])
        orch.autofocus.run_autofocus = AsyncMock()
        for _ in range(6):
            await orch._stability_tick()
        assert orch.sm.state == AppState.WAITING_FOR_STABILITY
        orch.autofocus.run_autofocus.assert_not_called()


class TestSuccessHold:
    @pytest.mark.asyncio
    async def test_success_reaches_waiting_for_removal_and_holds(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.SUCCESS
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-ok")
        orch._transaction.mark_completed()
        with patch("idcam.app.orchestrator.asyncio.sleep", AsyncMock()):
            await orch._success_tick()
        assert orch.sm.state == AppState.WAITING_FOR_REMOVAL
        orch.autofocus.run_autofocus = AsyncMock()
        orch.capture_service.capture = AsyncMock()
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        for _ in range(5):
            await orch._removal_tick()
        assert orch.sm.state == AppState.WAITING_FOR_REMOVAL
        orch.autofocus.run_autofocus.assert_not_called()
        orch.capture_service.capture.assert_not_called()


class TestInfrastructureVsTransaction:
    @pytest.mark.asyncio
    async def test_server_reject_does_not_restart_camera(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.PROCESSING_RESPONSE
        orch._resume_state = AppState.IDLE
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-srv")
        orch._pending_response = ServerResponse(
            success=False,
            capture_id="cap-srv",
            result=None,
            message="rejected",
            http_status=400,
            outcome=UploadOutcome.REJECTED,
        )
        orch.camera.restart = MagicMock()
        await orch._processing_response_tick()
        assert orch.sm.state == AppState.IDLE
        assert orch.sm.state != AppState.RECOVERY
        orch.camera.restart.assert_not_called()

    @pytest.mark.asyncio
    async def test_camera_preview_failure_enters_recovery(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        orch.camera.capture_preview_array = MagicMock(side_effect=CameraError("disconnected"))
        await orch._idle_tick()
        assert orch.sm.state == AppState.RECOVERY


class TestStaleAsyncOps:
    @pytest.mark.asyncio
    async def test_stale_af_result_ignored(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.FOCUSING
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-old", trigger="auto")
        page = _document_frame()
        orch._snapshot_accepted_geometry(orch.detector.detect_geometry(page))
        orch._safe_preview = AsyncMock(return_value=page)

        async def af(*_a, **_k):
            orch._transaction = ActiveCaptureTransaction(capture_id="cap-new")
            orch.sm.state = AppState.IDLE
            return AutofocusResult(True, 90.0, 1, None, "hardware_cycle", 10.0)

        orch.autofocus.run_autofocus = AsyncMock(side_effect=af)
        await orch._focusing_tick()
        assert orch.sm.state == AppState.IDLE
        assert orch._transaction.capture_id == "cap-new"

    @pytest.mark.asyncio
    async def test_stale_capture_result_ignored(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.CAPTURING
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-old")
        orch._transaction.mark_focus_completed(90.0, 10.0)
        page = _document_frame()
        orch._snapshot_accepted_geometry(orch.detector.detect_geometry(page))
        orch._check_motion_abort = AsyncMock(return_value=True)

        async def fake_capture(_camera, path):
            path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 500)
            orch._transaction = ActiveCaptureTransaction(capture_id="cap-new")
            orch.sm.state = AppState.IDLE
            return path

        orch.capture_service.capture = fake_capture
        await orch._capturing_tick()
        assert orch.sm.state == AppState.IDLE
        assert orch.sm.state != AppState.IMAGE_VALIDATION
        assert orch._transaction.capture_id == "cap-new"

    @pytest.mark.asyncio
    async def test_stale_server_result_ignored(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.PROCESSING_RESPONSE
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-new")
        orch._pending_response = _accepted_upload("cap-old")
        await orch._processing_response_tick()
        assert orch.sm.state == AppState.PROCESSING_RESPONSE
        assert orch._transaction.capture_id == "cap-new"
        assert orch._transaction.phase.value != "completed"

    def test_consumed_async_result_is_ignored(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IMAGE_VALIDATION
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1")
        assert orch._stale_async_completion("cap-1", expected_state=AppState.IMAGE_VALIDATION, stage="image_validation") is False
        assert orch._stale_async_completion("cap-1", expected_state=AppState.IMAGE_VALIDATION, stage="image_validation") is True


class TestSerialization:
    @pytest.mark.asyncio
    async def test_concurrent_tick_is_rejected(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        orch._state_tick_running = True
        orch._idle_tick = AsyncMock()
        await orch._dispatch_state_tick()
        orch._idle_tick.assert_not_called()

    def test_second_transaction_is_rejected(self, app_config, caplog):
        orch = _orch(app_config)
        orch.sm.state = AppState.FOCUSING
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-live")
        with pytest.raises(RuntimeError):
            orch._begin_transaction()
        assert orch._transaction.capture_id == "cap-live"
        assert "capture_rejected" in caplog.text


class TestManualIdle:
    @pytest.mark.asyncio
    async def test_manual_capture_from_idle_starts_transaction(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        orch._pending_command = "manual_capture"
        orch._safe_preview = AsyncMock(return_value=_empty_frame())
        await orch._idle_tick()
        assert orch.sm.state == AppState.DOCUMENT_DETECTED
        assert orch._transaction is not None
        assert orch._transaction.trigger == "manual"

