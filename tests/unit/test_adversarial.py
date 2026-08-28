"""Adversarial QA tests — hostile scenarios against the orchestrator."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
import pytest
import respx

from idcam.app.orchestrator import ApplicationOrchestrator, ONE_SHOT_STATES
from idcam.app.state_machine import AppState
from idcam.app.transaction import ActiveCaptureTransaction, TransactionPhase
from idcam.camera.autofocus import AutofocusResult
from idcam.config.schema import (
    AppConfig,
    CameraConfig,
    CameraRecoveryConfig,
    DetectionConfig,
    QualityConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from idcam.exceptions import AutofocusError, CameraError, CaptureError
from idcam.network.protocol import ServerResponse, UploadOutcome, UploadRequest


@pytest.fixture
def app_config(tmp_path):
    return AppConfig(
        device_id="adv-test",
        temp_dir=tmp_path / "tmp",
        server=ServerConfig(url="https://api.example.com"),
        storage=StorageConfig(min_free_mb=100),
        quality=QualityConfig(min_sharpness=10.0, min_file_bytes=100, min_width=64, min_height=64, require_expected_dimensions=False),
        detection=DetectionConfig(stable_frames_required=2, removal_frames_required=2),
        camera=CameraConfig(recovery=CameraRecoveryConfig(max_attempts=3, delay_seconds=0.01)),
        ui=UIConfig(
            status_file=tmp_path / "run" / "status.json",
            admin_socket=tmp_path / "run" / "admin.sock",
        ),
    )


def _make_orch(app_config):
    return ApplicationOrchestrator(app_config, use_mock_camera=True)


class TestRecoveryRateLimit:
    @pytest.mark.asyncio
    async def test_recovery_not_called_every_tick(self, app_config):
        orch = _make_orch(app_config)
        orch.sm.state = AppState.RECOVERY
        orch.camera.restart = MagicMock(side_effect=CameraError("fail"))

        for _ in range(20):
            try:
                await orch._recovery_tick()
            except Exception:
                break
            await asyncio.sleep(0.001)

        assert orch._recovery_attempts <= 3


class TestTransactionOrphan:
    @pytest.mark.asyncio
    async def test_document_detected_timeout_clears_transaction(self, app_config):
        orch = _make_orch(app_config)
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-orphan")
        orch.sm.state = AppState.DOCUMENT_DETECTED
        orch.sm.entered_at = time.monotonic() - 999

        handled = await orch._handle_state_timeout()
        assert handled
        assert orch.sm.state == AppState.IDLE
        assert orch._transaction is None


class TestOneShotStates:
    def test_one_shot_covers_operational_states(self):
        assert AppState.FOCUSING in ONE_SHOT_STATES
        assert AppState.UPLOADING in ONE_SHOT_STATES
        assert AppState.RECOVERY not in ONE_SHOT_STATES

    @pytest.mark.asyncio
    async def test_focusing_not_reentered_same_state(self, app_config):
        orch = _make_orch(app_config)
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1")
        orch.sm.state = AppState.FOCUSING
        call_count = 0

        async def counting_af(*_a, **_k):
            nonlocal call_count
            call_count += 1
            return AutofocusResult(True, 100.0, 1, 2, "test", 10.0)

        orch.autofocus.run_autofocus = counting_af
        orch._safe_preview = AsyncMock(return_value=np.zeros((40, 40, 3), dtype=np.uint8))
        orch._check_motion_abort = AsyncMock(return_value=True)

        orch._state_action_claimed = AppState.FOCUSING
        await orch._focusing_tick()
        assert call_count == 0

        orch._state_action_claimed = None
        await orch._focusing_tick()
        assert call_count == 1


class TestActiveTransactionGuard:
    @pytest.mark.asyncio
    async def test_idle_blocked_during_transaction(self, app_config):
        orch = _make_orch(app_config)
        orch.sm.state = AppState.WAITING_FOR_STABILITY
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-active")
        orch.detector.detect_geometry = MagicMock(return_value=True)
        await orch._idle_tick()
        orch.detector.detect_geometry.assert_not_called()

    def test_has_active_transaction_follows_state_machine(self, app_config):
        orch = _make_orch(app_config)
        orch.sm.state = AppState.FOCUSING
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-x")
        assert orch.has_active_transaction


class TestMotionDuringCapture:
    @pytest.mark.asyncio
    async def test_lens_motion_during_focus_does_not_abort(self, app_config):
        orch = _make_orch(app_config)
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-move")
        orch.sm.state = AppState.FOCUSING
        orch.detector.classify = MagicMock(return_value=__import__(
            "idcam.vision.document_detector", fromlist=["DocumentPresence"]
        ).DocumentPresence.DOCUMENT_MOVING)
        orch.autofocus.run_autofocus = AsyncMock(
            return_value=AutofocusResult(True, 30.0, 1, None, "hardware_trigger", 28.0)
        )
        orch._safe_preview = AsyncMock(return_value=np.zeros((10, 10, 3), dtype=np.uint8))

        with patch.object(orch, "_fail_transaction_and_wait", AsyncMock()) as fail_mock:
            await orch._focusing_tick()
            fail_mock.assert_not_called()
        assert orch.sm.state == AppState.FOCUS_VERIFICATION


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_marks_transaction_failed(self, app_config):
        orch = _make_orch(app_config)
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-shutdown")
        orch._transaction.phase = TransactionPhase.CAPTURING
        with patch.object(orch.api, "stop", AsyncMock()), patch.object(orch.camera, "shutdown", MagicMock()):
            await orch._shutdown()
        assert orch._transaction is None
        assert orch.sm.state == AppState.SHUTDOWN


class TestNetworkAdversarial:
    @pytest.mark.asyncio
    @respx.mock
    async def test_tls_failure_returns_never_sent(self, app_config, tmp_path):
        img = tmp_path / "i.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 5000)
        respx.post("https://api.example.com/api/document").mock(
            side_effect=httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED]")
        )
        from idcam.network.api_client import ApiClient

        client = ApiClient(app_config)
        await client.start()
        resp = await client.upload_document(UploadRequest("cap-tls", "adv-test", str(img), "now"))
        await client.stop()
        assert resp.outcome == UploadOutcome.NEVER_SENT

    @pytest.mark.asyncio
    async def test_ambiguous_response_fails_transaction(self, app_config, tmp_path):
        orch = _make_orch(app_config)
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-amb")
        orch._capture_path = tmp_path / "x.jpg"
        orch._capture_path.write_bytes(b"x")
        orch.sm.state = AppState.PROCESSING_RESPONSE
        orch._pending_response = ServerResponse(
            success=False,
            capture_id="cap-amb",
            result=None,
            message="timeout",
            http_status=0,
            outcome=UploadOutcome.SENT_UNKNOWN,
        )
        with patch.object(orch, "_fail_transaction_and_wait", AsyncMock()) as fail:
            await orch._processing_response_tick()
            fail.assert_called_once()
            assert fail.call_args[0][0] == "upload_failed"


class TestStartup:
    @pytest.mark.asyncio
    async def test_camera_init_failure_enters_recovery(self, app_config):
        orch = _make_orch(app_config)
        orch.camera.initialize = MagicMock(side_effect=__import__(
            "idcam.exceptions", fromlist=["CameraUnavailableError"]
        ).CameraUnavailableError("no camera"))
        with patch.object(orch.api, "start", AsyncMock()), patch.object(
            orch.api, "check_connectivity", AsyncMock(return_value=True)
        ):
            await orch._startup()
        assert orch.sm.state == AppState.RECOVERY


class TestRepeatedCycles:
    @pytest.mark.asyncio
    async def test_rapid_state_claim_resets_on_transition(self, app_config):
        orch = _make_orch(app_config)
        orch.sm.state = AppState.FOCUSING
        assert orch._claim_state_action()
        assert not orch._claim_state_action()
        orch._transition(AppState.FOCUS_VERIFICATION)
        assert orch._claim_state_action()
