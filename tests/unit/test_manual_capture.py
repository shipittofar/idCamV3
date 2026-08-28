"""Manual capture command goes through the existing state machine."""

from __future__ import annotations

import threading
import time
from unittest.mock import AsyncMock, MagicMock

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
        device_id="manual-test",
        temp_dir=tmp_path / "tmp",
        server=ServerConfig(url="https://api.example.com"),
        quality=QualityConfig(
            min_sharpness=10.0,
            min_file_bytes=100,
            min_width=64,
            min_height=64,
            require_expected_dimensions=False,
        ),
        detection=DetectionConfig(stable_frames_required=2, removal_frames_required=2),
        camera=CameraConfig(preview_fps=30.0, af_timeout_sec=1.0, af_max_retries=1),
        ui=UIConfig(
            status_file=tmp_path / "run" / "status.json",
            admin_socket=tmp_path / "run" / "admin.sock",
        ),
    )


def _make_orch(app_config):
    orch = ApplicationOrchestrator(app_config, use_mock_camera=True)
    orch.sm.state = AppState.IDLE
    orch.health.camera = ComponentStatus(level=HealthLevel.HEALTHY, details={"healthy": True})
    return orch


class TestManualCaptureCommand:
    def test_busy_when_not_idle(self, app_config):
        orch = _make_orch(app_config)
        orch.sm.state = AppState.FOCUSING
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-busy")
        result = orch.request_manual_capture(timeout=0.2)
        assert result["ok"] is False
        assert result["status"] == "busy"
        assert "FOCUSING" in result["message"]

    def test_second_request_busy_while_queued(self, app_config):
        orch = _make_orch(app_config)
        with orch._cmd_lock:
            orch._pending_command = "manual_capture"
        result = orch.request_manual_capture(timeout=0.2)
        assert result["status"] == "busy"

    @pytest.mark.asyncio
    async def test_idle_tick_starts_existing_pipeline(self, app_config):
        orch = _make_orch(app_config)
        orch._pending_command = "manual_capture"
        orch._safe_preview = AsyncMock(return_value=np.zeros((40, 40, 3), dtype=np.uint8))
        await orch._idle_tick()
        assert orch.sm.state == AppState.DOCUMENT_DETECTED
        assert orch._transaction is not None
        assert orch._transaction.trigger == "manual"
        assert orch._pending_command == "manual_capture_running"

    @pytest.mark.asyncio
    async def test_manual_skips_stability_and_detection(self, app_config):
        orch = _make_orch(app_config)
        orch._pending_command = "manual_capture"
        orch.detector.detect_geometry = MagicMock()
        orch._safe_preview = AsyncMock(return_value=np.zeros((40, 40, 3), dtype=np.uint8))
        await orch._idle_tick()
        orch.detector.detect_geometry.assert_not_called()
        assert orch.sm.state == AppState.DOCUMENT_DETECTED
        assert orch._transaction.trigger == "manual"
        await orch._document_detected_tick()
        assert orch.sm.state == AppState.FOCUSING

    def test_busy_when_non_terminal_transaction_exists(self, app_config):
        orch = _make_orch(app_config)
        orch.sm.state = AppState.IDLE
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-x")
        result = orch.request_manual_capture(timeout=0.2)
        assert result["status"] == "busy"

    def test_begin_transaction_attaches_queued_manual(self, app_config):
        orch = _make_orch(app_config)
        orch._pending_command = "manual_capture"
        txn = orch._begin_transaction(trigger="auto")
        assert txn.trigger == "manual"
        assert orch._pending_command == "manual_capture_running"

    @pytest.mark.asyncio
    async def test_success_signals_waiter(self, app_config, tmp_path):
        orch = _make_orch(app_config)
        img = tmp_path / "cap.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 500)
        orch.sm.state = AppState.PROCESSING_RESPONSE
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-ok", trigger="manual")
        orch._capture_path = img
        orch._pending_command = "manual_capture_running"
        orch._pending_response = ServerResponse(
            success=True,
            capture_id="cap-ok",
            result="ok",
            message="accepted",
            http_status=200,
            outcome=UploadOutcome.ACCEPTED,
        )
        orch._command_event.clear()

        await orch._processing_response_tick()
        assert orch._command_event.is_set()
        assert orch._command_result is not None
        assert orch._command_result["ok"] is True
        assert orch._command_result["capture_id"] == "cap-ok"
        assert "token" not in orch._command_result

    @pytest.mark.asyncio
    async def test_upload_failure_signals_waiter(self, app_config):
        orch = _make_orch(app_config)
        orch.sm.state = AppState.WAITING_FOR_STABILITY
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-fail", trigger="manual")
        orch._pending_command = "manual_capture_running"
        orch._command_event.clear()
        await orch._fail_transaction_and_wait("upload_failed", "connection refused")
        assert orch._command_event.is_set()
        assert orch._command_result["ok"] is False
        assert "upload_failed" in orch._command_result["message"]
        assert orch.sm.state == AppState.IDLE

    @pytest.mark.asyncio
    async def test_end_to_end_manual_capture_upload(self, app_config, tmp_path):
        orch = _make_orch(app_config)
        orch.camera.initialize()
        orch.autofocus.run_autofocus = AsyncMock(
            return_value=AutofocusResult(True, 100.0, 1, 2, "already_sharp", 10.0)
        )
        orch.autofocus.verify_focus = AsyncMock(
            return_value=AutofocusResult(True, 100.0, 0, 2, "verification", 10.0)
        )
        captured = tmp_path / "out.jpg"
        captured.write_bytes(b"\xff\xd8\xff" + b"\x00" * 2000)

        async def fake_capture(_camera, path):
            path.write_bytes(captured.read_bytes())
            return path

        orch.capture_service.capture = fake_capture
        orch.validator.validate = MagicMock(
            return_value=MagicMock(
                file_bytes=2000, width=100, height=80, sha256="abc", sharpness=120.0
            )
        )
        async def fake_upload(request):
            return ServerResponse(
                success=True,
                capture_id=request.capture_id,
                result="ok",
                message="accepted",
                http_status=200,
                outcome=UploadOutcome.ACCEPTED,
            )

        orch.api.upload_document = fake_upload
        called = {"n": 0}

        orig = fake_upload

        async def counting_upload(request):
            called["n"] += 1
            return await orig(request)

        orch.api.upload_document = counting_upload
        orch.disk.can_capture = MagicMock(return_value=True)
        orch._check_motion_abort = AsyncMock(return_value=True)

        results: list[dict] = []

        def waiter() -> None:
            results.append(orch.request_manual_capture(timeout=8.0))

        thread = threading.Thread(target=waiter)
        thread.start()
        deadline = time.monotonic() + 8.0
        while thread.is_alive() and time.monotonic() < deadline:
            await orch._dispatch_state_tick()
            await __import__("asyncio").sleep(0.02)
        thread.join(timeout=2.0)
        assert results
        assert results[0]["ok"] is True
        assert results[0]["status"] == "success"
        assert called["n"] == 1
        orch.camera.shutdown()

    def test_button_enabled_during_waiting_for_removal(self, app_config):
        orch = _make_orch(app_config)
        orch.sm.state = AppState.WAITING_FOR_REMOVAL
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-ok")
        orch._transaction.mark_completed()
        available, _phase, _msg = orch._manual_status()
        assert available is True

    def test_button_disabled_while_focusing(self, app_config):
        orch = _make_orch(app_config)
        orch.sm.state = AppState.FOCUSING
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-x")
        available, _phase, _msg = orch._manual_status()
        assert available is False
