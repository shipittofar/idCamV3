"""Production hardening tests."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
import pytest
import respx
from PIL import Image

from idcam.app.orchestrator import ApplicationOrchestrator
from idcam.app.state_machine import AppState, StateMachine
from idcam.app.transaction import ActiveCaptureTransaction
from idcam.camera.quality import ImageValidator
from idcam.config.schema import (
    AppConfig,
    CameraConfig,
    DetectionConfig,
    QualityConfig,
    ServerConfig,
    StateTimeoutsConfig,
    StorageConfig,
    UIConfig,
)
from idcam.exceptions import ProtocolError
from idcam.network.protocol import ServerResponse
from idcam.storage.disk_monitor import DiskMonitor
from idcam.storage.temp_files import TempFileStore
from idcam.system.watchdog import Watchdog
from idcam.ui.status import StatusPublisher
from idcam.vision.document_detector import DocumentDetector, DocumentPresence


@pytest.fixture
def app_config(tmp_path):
    return AppConfig(
        device_id="test-device",
        temp_dir=tmp_path / "tmp",
        server=ServerConfig(url="https://api.example.com"),
        storage=StorageConfig(min_free_mb=100),
        quality=QualityConfig(min_sharpness=10.0, min_file_bytes=500, min_width=100, min_height=100, require_expected_dimensions=False),
        detection=DetectionConfig(stable_frames_required=3, removal_frames_required=3),
        ui=UIConfig(
            status_file=tmp_path / "run" / "status.json",
            admin_socket=tmp_path / "run" / "admin.sock",
        ),
    )


class TestTransactionContext:
    def test_single_transaction_id_format(self):
        cid = ActiveCaptureTransaction.generate_id()
        assert cid.startswith("cap-")
        assert len(cid) > 20

    def test_transaction_metadata(self):
        txn = ActiveCaptureTransaction(capture_id="cap-test")
        txn.mark_validated(size=1000, width=640, height=480, sha256="abc", sharpness=50.0)
        assert txn.image_sha256 == "abc"
        assert txn.phase.value == "validated"


class TestSameDocumentProtection:
    def test_removal_required_before_idle(self):
        sm = StateMachine(state=AppState.WAITING_FOR_REMOVAL)
        assert not sm.in_transaction()
        assert sm.can_transition(AppState.DOCUMENT_DETECTED)
        assert sm.can_transition(AppState.IDLE)
        assert not sm.can_transition(AppState.CAPTURING)

    def test_removal_timeout_can_elapse(self, app_config):
        sm = StateMachine(state=AppState.WAITING_FOR_REMOVAL)
        sm.set_timeouts(app_config.state_timeouts)
        sm.entered_at = time.monotonic() - 9999
        assert sm.is_timed_out()

    def test_scene_empty_detection(self, app_config):
        detector = DocumentDetector(app_config.detection)
        baseline = np.full((480, 640, 3), 200, dtype=np.uint8)
        for _ in range(20):
            detector.update_baseline(baseline)
        assert detector.is_scene_empty(baseline)


class TestDocumentDetection:
    def test_presence_states(self, app_config):
        detector = DocumentDetector(app_config.detection)
        empty = np.full((480, 640, 3), 180, dtype=np.uint8)
        detector.classify(empty)
        assert detector.classify(empty) == DocumentPresence.NO_DOCUMENT

    def test_no_immediate_capture_on_single_frame(self, app_config):
        detector = DocumentDetector(app_config.detection)
        frame = np.full((480, 640, 3), 180, dtype=np.uint8)
        detector.classify(frame)
        assert detector.classify(frame) != DocumentPresence.DOCUMENT_READY


class TestImageValidation:
    def test_sha256_computed(self, tmp_path, app_config):
        path = tmp_path / "img.jpg"
        arr = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        Image.fromarray(arr).save(path, quality=90)
        result = ImageValidator(app_config.quality).validate(path)
        assert len(result.sha256) == 64

    def test_truncated_jpeg_rejected(self, tmp_path, app_config):
        path = tmp_path / "trunc.jpg"
        path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)
        with pytest.raises(Exception):
            ImageValidator(app_config.quality).validate(path)

    def test_insufficient_file_size(self, tmp_path, app_config):
        path = tmp_path / "tiny.jpg"
        path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)
        with pytest.raises(Exception):
            ImageValidator(app_config.quality).validate(path)


class TestTempFiles:
    def test_stale_cleanup_only_own_files(self, tmp_path):
        store = TempFileStore(tmp_path)
        own = store.create_temp_path()
        own.write_bytes(b"x")
        other = tmp_path / "not-our-file.txt"
        other.write_text("keep")
        import os

        os.utime(own, (time.time() - 7200, time.time() - 7200))
        removed = store.cleanup_stale(max_age_sec=3600)
        assert removed == 1
        assert other.exists()

    def test_atomic_finalize(self, tmp_path):
        store = TempFileStore(tmp_path)
        final = tmp_path / "final.jpg"
        partial = store.create_partial_path(final)
        partial.write_bytes(b"jpeg-data")
        store.finalize_capture(partial, final)
        assert final.exists()
        assert not partial.exists()


class TestStorage:
    def test_disk_monitor_healthy(self, tmp_path, app_config):
        monitor = DiskMonitor(app_config.storage, tmp_path)
        status = monitor.check()
        assert status.free_mb > 0

    @patch("idcam.storage.disk_monitor.DiskMonitor._free_bytes")
    def test_disk_below_threshold(self, mock_free, tmp_path, app_config):
        mock_free.return_value = 50 * 1024 * 1024  # 50 MB
        monitor = DiskMonitor(app_config.storage, tmp_path)
        status = monitor.check()
        assert not status.healthy


class TestProtocol:
    def test_wrong_capture_id_rejected(self):
        with pytest.raises(ProtocolError):
            ServerResponse.from_http(
                200,
                {"success": True, "capture_id": "cap-wrong"},
                "cap-expected",
            )


class TestWatchdog:
    def test_does_not_interrupt_valid_long_capture_with_progress(self, app_config):
        timeouts = StateTimeoutsConfig(capturing=45.0)
        wd = Watchdog(app_config.watchdog, timeouts)
        sm = StateMachine(state=AppState.CAPTURING)
        sm.set_timeouts(timeouts)
        sm.entered_at = time.monotonic() - 30
        sm.last_progress_at = time.monotonic() - 1
        assert wd.check(sm) is None

    def test_triggers_without_progress(self, app_config):
        timeouts = StateTimeoutsConfig(capturing=10.0)
        wd = Watchdog(app_config.watchdog, timeouts)
        sm = StateMachine(state=AppState.CAPTURING)
        sm.set_timeouts(timeouts)
        sm.last_progress_at = time.monotonic() - 20
        assert wd.check(sm, capture_id="cap-1") is not None


class TestStatusAtomic:
    def test_status_written_atomically(self, tmp_path):
        pub = StatusPublisher(tmp_path / "status.json")
        pub.update(AppState.IDLE, camera={"healthy": True})
        assert (tmp_path / "status.json").exists()
        data = json.loads((tmp_path / "status.json").read_text())
        assert data["state"] == "IDLE"
        assert "camera" in data


class TestOrchestratorGuards:
    @pytest.mark.asyncio
    async def test_no_second_transaction_while_active(self, app_config):
        orch = ApplicationOrchestrator(app_config, use_mock_camera=True)
        orch.sm.state = AppState.CAPTURING
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1")
        assert orch.has_active_transaction
        assert orch.sm.in_transaction()
        assert orch.sm.in_transaction()

    @pytest.mark.asyncio
    async def test_startup_cleans_stale_temps(self, app_config, tmp_path):
        stale = app_config.temp_dir / "capture-deadbeef.jpg"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"old")
        import os

        os.utime(stale, (time.time() - 99999, time.time() - 99999))
        orch = ApplicationOrchestrator(app_config, use_mock_camera=True)
        with patch.object(orch.api, "start", AsyncMock()), patch.object(
            orch.api, "check_connectivity", AsyncMock(return_value=True)
        ), patch.object(orch.camera, "initialize", MagicMock()):
            await orch._startup()
        assert not stale.exists()

    @pytest.mark.asyncio
    async def test_shutdown_during_idle(self, app_config):
        orch = ApplicationOrchestrator(app_config, use_mock_camera=True)
        orch._shutdown_event.set()
        with patch.object(orch.api, "stop", AsyncMock()), patch.object(
            orch.camera, "shutdown", MagicMock()
        ):
            await orch._shutdown()
        assert orch.sm.state == AppState.SHUTDOWN


class TestIdempotency:
    @pytest.mark.asyncio
    @respx.mock
    async def test_idempotency_key_matches_capture_id(self, app_config, tmp_path):
        img = tmp_path / "i.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 5000)
        from idcam.network.api_client import ApiClient
        from idcam.network.protocol import UploadRequest

        route = respx.post("https://api.example.com/api/document").mock(
            return_value=httpx.Response(200, json={"success": True, "capture_id": "cap-xyz"})
        )
        client = ApiClient(app_config)
        await client.start()
        await client.upload_document(
            UploadRequest("cap-xyz", "test-device", str(img), "now")
        )
        await client.stop()
        req = route.calls[0].request
        assert req.headers["Idempotency-Key"] == "cap-xyz"
        assert req.headers["X-Capture-ID"] == "cap-xyz"


class TestHttpErrors:
    @pytest.mark.asyncio
    @respx.mock
    async def test_http_401(self, app_config, tmp_path):
        img = tmp_path / "i.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 5000)
        respx.post("https://api.example.com/api/document").mock(return_value=httpx.Response(401))
        from idcam.network.api_client import ApiClient
        from idcam.network.protocol import UploadRequest
        from idcam.exceptions import ServerRejectedError

        client = ApiClient(app_config)
        await client.start()
        with pytest.raises(ServerRejectedError):
            await client.upload_document(UploadRequest("cap-1", "d", str(img), "now"))
        await client.stop()

    @pytest.mark.asyncio
    @respx.mock
    async def test_http_403(self, app_config, tmp_path):
        img = tmp_path / "i.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 5000)
        respx.post("https://api.example.com/api/document").mock(return_value=httpx.Response(403))
        from idcam.network.api_client import ApiClient
        from idcam.network.protocol import UploadRequest
        from idcam.exceptions import ServerRejectedError

        client = ApiClient(app_config)
        await client.start()
        with pytest.raises(ServerRejectedError):
            await client.upload_document(UploadRequest("cap-1", "d", str(img), "now"))
        await client.stop()

    @pytest.mark.asyncio
    @respx.mock
    async def test_malformed_json_on_error(self, app_config, tmp_path):
        img = tmp_path / "i.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 5000)
        respx.post("https://api.example.com/api/document").mock(
            return_value=httpx.Response(500, content=b"not json", headers={"content-type": "application/json"})
        )
        from idcam.network.api_client import ApiClient
        from idcam.network.protocol import UploadRequest
        from idcam.exceptions import ServerError

        client = ApiClient(app_config)
        await client.start()
        with pytest.raises(ServerError):
            await client.upload_document(UploadRequest("cap-1", "d", str(img), "now"))
        await client.stop()

    @pytest.mark.asyncio
    @respx.mock
    async def test_connection_failure_before_send(self, app_config, tmp_path):
        img = tmp_path / "i.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 5000)
        respx.post("https://api.example.com/api/document").mock(side_effect=httpx.ConnectError("refused"))
        from idcam.network.api_client import ApiClient
        from idcam.network.protocol import UploadRequest, UploadOutcome

        client = ApiClient(app_config)
        await client.start()
        resp = await client.upload_document(UploadRequest("cap-1", "d", str(img), "now"))
        assert resp.outcome == UploadOutcome.NEVER_SENT
        await client.stop()

    @pytest.mark.asyncio
    @respx.mock
    async def test_read_timeout_ambiguous(self, app_config, tmp_path):
        img = tmp_path / "i.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 5000)
        respx.post("https://api.example.com/api/document").mock(side_effect=httpx.ReadTimeout("slow"))
        from idcam.network.api_client import ApiClient
        from idcam.network.protocol import UploadRequest, UploadOutcome

        client = ApiClient(app_config)
        await client.start()
        resp = await client.upload_document(UploadRequest("cap-1", "d", str(img), "now"))
        assert resp.outcome == UploadOutcome.SENT_UNKNOWN
        await client.stop()


class TestUIConsistency:
    def test_status_derived_from_state(self, tmp_path):
        from idcam.ui.messages import USER_MESSAGES

        pub = StatusPublisher(tmp_path / "s.json")
        pub.update(AppState.CAPTURING)
        assert pub.snapshot.message == USER_MESSAGES[AppState.CAPTURING]
        assert pub.snapshot.state == AppState.CAPTURING.value
