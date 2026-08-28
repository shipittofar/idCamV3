"""Regression tests for orchestrator lifecycle edge cases."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
import pytest
import respx

from idcam.app.orchestrator import ApplicationOrchestrator
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
from idcam.network.protocol import ServerResponse, UploadOutcome, UploadRequest


@pytest.fixture
def app_config(tmp_path):
    return AppConfig(
        device_id="reg-test",
        temp_dir=tmp_path / "tmp",
        server=ServerConfig(url="https://api.example.com"),
        storage=StorageConfig(min_free_mb=100),
        quality=QualityConfig(
            min_sharpness=10.0,
            min_file_bytes=100,
            min_width=64,
            min_height=64,
            require_expected_dimensions=False,
        ),
        detection=DetectionConfig(
            stable_frames_required=2,
            removal_frames_required=2,
            geometry_stable_frames_required=2,
        ),
        camera=CameraConfig(recovery=CameraRecoveryConfig(max_attempts=3, delay_seconds=0.05)),
        ui=UIConfig(
            status_file=tmp_path / "run" / "status.json",
            admin_socket=tmp_path / "run" / "admin.sock",
        ),
    )


def _make_orch(app_config):
    return ApplicationOrchestrator(app_config, use_mock_camera=True)


class TestTransactionLifecycleRegression:
    def test_begin_transaction_rejects_non_terminal_active(self, app_config):
        orch = _make_orch(app_config)
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-active")
        orch._transaction.phase = TransactionPhase.CAPTURING
        with pytest.raises(RuntimeError, match="already active"):
            orch._begin_transaction()

    def test_begin_transaction_replaces_terminal_transaction(self, app_config):
        orch = _make_orch(app_config)
        old = ActiveCaptureTransaction(capture_id="cap-old")
        old.mark_failed("prior_failure")
        orch._transaction = old
        new = orch._begin_transaction()
        assert orch._transaction is new
        assert new.capture_id != "cap-old"
        assert new.capture_id.startswith("cap-")

    def test_has_active_transaction_in_waiting_for_removal(self, app_config):
        orch = _make_orch(app_config)
        orch.sm.state = AppState.WAITING_FOR_REMOVAL
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-rem")
        orch._transaction.mark_completed()
        assert not orch.has_active_transaction
        assert not orch.sm.in_transaction()


class TestOneShotRegression:
    @pytest.mark.asyncio
    async def test_focus_refocus_allows_second_autofocus_run(self, app_config):
        orch = _make_orch(app_config)
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-refocus", trigger="manual")
        orch.sm.state = AppState.FOCUS_VERIFICATION
        call_count = 0

        async def counting_verify(*_a, **_k):
            nonlocal call_count
            call_count += 1
            raise __import__("idcam.exceptions", fromlist=["AutofocusError"]).AutofocusError("soft")

        orch.autofocus.verify_focus = counting_verify
        orch._check_motion_abort = AsyncMock(return_value=True)
        orch.config.camera.af_verify_max_retries = 2

        await orch._focus_verification_tick()
        assert orch.sm.state == AppState.FOCUSING
        assert call_count == 1

        async def counting_af(*_a, **_k):
            nonlocal call_count
            call_count += 1
            return AutofocusResult(True, 100.0, 1, 2, "test", 10.0)

        orch.autofocus.run_autofocus = counting_af
        orch._safe_preview = AsyncMock(return_value=np.zeros((40, 40, 3), dtype=np.uint8))
        await orch._focusing_tick()
        assert call_count == 2


class TestNetworkRegression:
    @pytest.mark.asyncio
    async def test_sent_unknown_upload_tick_fails_to_removal(self, app_config, tmp_path):
        orch = _make_orch(app_config)
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-net")
        orch._capture_path = tmp_path / "img.jpg"
        orch._capture_path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 500)
        orch.sm.state = AppState.UPLOADING

        ambiguous = ServerResponse(
            success=False,
            capture_id="cap-net",
            result=None,
            message="read timeout",
            http_status=0,
            outcome=UploadOutcome.SENT_UNKNOWN,
        )
        orch.api.upload_document = AsyncMock(return_value=ambiguous)

        await orch._uploading_tick()
        assert orch.sm.state == AppState.WAITING_FOR_SERVER

        await orch._waiting_server_tick()
        await orch._processing_response_tick()
        assert orch.sm.state == AppState.IDLE
        assert orch._transaction is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_read_timeout_single_post_attempt(self, app_config, tmp_path):
        img = tmp_path / "i.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 5000)
        route = respx.post("https://api.example.com/api/document").mock(
            side_effect=httpx.ReadTimeout("slow")
        )
        from idcam.network.api_client import ApiClient

        client = ApiClient(app_config)
        await client.start()
        resp = await client.upload_document(UploadRequest("cap-1", "reg-test", str(img), "now"))
        await client.stop()
        assert resp.outcome == UploadOutcome.SENT_UNKNOWN
        assert len(route.calls) == 1


class TestShutdownRegression:
    @pytest.mark.asyncio
    async def test_shutdown_marks_failed_before_clear(self, app_config):
        orch = _make_orch(app_config)
        txn = ActiveCaptureTransaction(capture_id="cap-shut")
        txn.phase = TransactionPhase.UPLOADING
        orch._transaction = txn
        failed_reasons: list[str] = []

        with patch.object(orch.api, "stop", AsyncMock()), patch.object(
            orch.camera, "shutdown", MagicMock()
        ), patch.object(txn, "mark_failed", side_effect=lambda r: failed_reasons.append(r)):
            await orch._shutdown()

        assert failed_reasons == ["shutdown"]
        assert orch._transaction is None
        assert orch.sm.state == AppState.SHUTDOWN


class TestIdleAfterEarlyAbortRegression:
    @pytest.mark.asyncio
    async def test_idle_can_start_after_document_detected_timeout(self, app_config):
        orch = _make_orch(app_config)
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-early")
        orch.sm.state = AppState.DOCUMENT_DETECTED
        orch.sm.entered_at = time.monotonic() - 999

        handled = await orch._handle_state_timeout()
        assert handled
        assert orch.sm.state == AppState.IDLE
        assert not orch.has_active_transaction

        page = np.zeros((480, 640, 3), dtype=np.uint8)
        page[60:420, 80:560] = (242, 238, 230)
        page[60:63, 80:560] = 35
        page[417:420, 80:560] = 35
        page[60:420, 80:83] = 35
        page[60:420, 557:560] = 35
        orch._safe_preview = AsyncMock(return_value=page)
        for _ in range(8):
            await orch._idle_tick()
            if orch._transaction is not None and orch._transaction.capture_id != "cap-early":
                break
        assert orch._transaction is not None
        assert orch._transaction.capture_id != "cap-early"
