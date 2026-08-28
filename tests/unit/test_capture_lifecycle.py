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
from idcam.network.protocol import ServerResponse, UploadOutcome
from idcam.system.component_health import ComponentStatus, HealthLevel


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
    return np.full((120, 160, 3), 200, dtype=np.uint8)


def _document_frame():
    frame = _empty_frame()
    frame[20:100, 20:140] = 40
    frame[20:28, 20:140] = 0
    frame[92:100, 20:140] = 0
    frame[20:100, 20:28] = 0
    frame[20:100, 132:140] = 0
    return frame


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
        orch.detector.update_baseline(_empty_frame())
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        await orch._idle_tick()
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
        orch.detector.update_baseline(_empty_frame())
        frame = _document_frame()
        orch._safe_preview = AsyncMock(return_value=frame)
        for _ in range(6):
            await orch._stability_tick()
            if orch.sm.state == AppState.FOCUSING:
                break
        assert orch.sm.state == AppState.FOCUSING

    @pytest.mark.asyncio
    async def test_focusing_to_capturing(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.FOCUSING
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1")
        orch._safe_preview = AsyncMock(return_value=_document_frame())
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
    async def test_awaiting_clear_scene_retries_when_document_still_present(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        orch._awaiting_clear_scene = True
        orch.detector.update_baseline(_empty_frame())
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        await orch._idle_tick()
        assert orch.sm.state == AppState.DOCUMENT_DETECTED
        assert orch._awaiting_clear_scene is False
        assert orch._transaction is not None
        assert orch._transaction.trigger == "auto"


class TestAutoDetectStableFocusCapture:
    @pytest.mark.asyncio
    async def test_detect_stable_focus_capture(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        orch.detector.update_baseline(_empty_frame())
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
        await orch._idle_tick()
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


class TestRemovalHold:
    @pytest.mark.asyncio
    async def test_document_still_present_stays_in_removal(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.WAITING_FOR_REMOVAL
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-ok")
        orch._transaction.mark_completed()
        orch.detector.update_baseline(_empty_frame())
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
        orch.detector.update_baseline(empty)
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

