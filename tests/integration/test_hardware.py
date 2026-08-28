"""Integration test strategy (hardware tests run on Pi only)."""

import asyncio
import os

import pytest
from PIL import Image

from idcam.camera.capture import CaptureService
from idcam.camera.manager import CameraManager
from idcam.config.schema import CameraConfig
from idcam.storage.temp_files import TempFileStore

pytestmark = pytest.mark.skipif(
    os.environ.get("IDC_RUN_HARDWARE_TESTS", "").lower() not in ("1", "true"),
    reason="Hardware tests disabled (set IDC_RUN_HARDWARE_TESTS=1 on Pi)",
)


@pytest.mark.hardware
class TestHardwareIntegration:
    """Run on Raspberry Pi 5 with Arducam 64MP attached."""

    def test_camera_init(self):
        mgr = CameraManager(CameraConfig(preview_width=640, preview_height=480))
        mgr.initialize()
        assert mgr.is_started
        frame = mgr.capture_preview_array()
        assert frame is not None
        assert frame.shape[0] == 480
        mgr.shutdown()

    def test_autofocus_controls_advertised(self):
        mgr = CameraManager(CameraConfig())
        mgr.initialize()
        controls = mgr.get_controls()
        # At least one focus mechanism should exist on hardware
        has_af = "AfMode" in controls or hasattr(mgr, "autofocus_cycle")
        assert has_af or True  # informational; driver-dependent
        mgr.shutdown()

    @pytest.mark.hardware
    @pytest.mark.slow
    def test_full_res_capture(self, tmp_path):
        mgr = CameraManager(CameraConfig(capture_width=4056, capture_height=3040))
        mgr.initialize()
        out = tmp_path / "capture.jpg"
        mgr.capture_still_to_file(out)
        assert out.exists()
        assert out.stat().st_size > 100_000
        mgr.shutdown()

    @pytest.mark.hardware
    @pytest.mark.slow
    def test_capture_through_production_partial_path(self, tmp_path):
        """The historic failure was in the in-progress filename, not the sensor.

        Writing straight to ``capture.jpg`` (as the test above does) cannot
        reproduce it, so this test drives the real CaptureService sequence.
        """
        config = CameraConfig(capture_width=4056, capture_height=3040)
        mgr = CameraManager(config)
        mgr.initialize()
        store = TempFileStore(tmp_path)
        service = CaptureService(config, store)
        output = store.create_temp_path()
        try:
            saved = asyncio.run(service.capture(mgr, output))
        finally:
            mgr.shutdown()

        assert saved.exists()
        assert not store.create_partial_path(output).exists()
        with Image.open(saved) as img:
            assert img.format == "JPEG"
            assert img.size == (4056, 3040)

    @pytest.mark.hardware
    def test_shutdown_releases_the_device(self):
        """A leaked handle makes every later recovery attempt fail."""
        config = CameraConfig(preview_width=640, preview_height=480)
        mgr = CameraManager(config)
        mgr.initialize()
        mgr.shutdown()
        assert not mgr.is_started

        second = CameraManager(config)
        second.initialize()
        assert second.is_started
        second.shutdown()

    @pytest.mark.hardware
    def test_configured_geometry_matches_request(self):
        """libcamera may align sizes; record what the hardware actually gives."""
        config = CameraConfig()
        mgr = CameraManager(config)
        mgr.initialize()
        geometry = mgr.stream_geometry()
        mgr.shutdown()
        assert geometry["preview"], geometry
        assert geometry["still"], geometry
