"""Camera manager preview format and mock-camera tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from idcam.camera.manager import (
    CameraManager,
    describe_preview_frame,
    map_preview_roi_to_sensor,
    normalize_picamera_preview_frame,
    preview_roi_to_af_window,
)
from idcam.config.schema import CameraConfig
from idcam.exceptions import CameraError, CameraUnavailableError


class TestPicameraRgb888Normalize:
    def test_bgr_memory_becomes_rgb(self):
        # Picamera2 RGB888 ndarray: channel 0 is Blue.
        native = np.zeros((8, 12, 3), dtype=np.uint8)
        native[:, :] = (10, 20, 200)  # B, G, R
        rgb = normalize_picamera_preview_frame(native)
        assert rgb.shape == (8, 12, 3)
        assert rgb.dtype == np.uint8
        assert rgb[0, 0, 0] == 200  # R
        assert rgb[0, 0, 1] == 20  # G
        assert rgb[0, 0, 2] == 10  # B
        assert rgb.flags["C_CONTIGUOUS"]

    def test_gray_passthrough(self):
        gray = np.arange(16, dtype=np.uint8).reshape(4, 4)
        out = normalize_picamera_preview_frame(gray)
        assert np.array_equal(out, gray)

    def test_xrgb_four_channel_drops_padding(self):
        native = np.zeros((2, 2, 4), dtype=np.uint8)
        native[:, :] = (10, 20, 200, 255)  # B, G, R, X
        rgb = normalize_picamera_preview_frame(native)
        assert rgb.shape == (2, 2, 3)
        assert rgb[0, 0, 0] == 200
        assert rgb[0, 0, 2] == 10

    def test_describe_reports_channel_means(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[..., 0] = 10
        frame[..., 1] = 20
        frame[..., 2] = 30
        info = describe_preview_frame(frame)
        assert info["shape"] == [4, 4, 3]
        assert info["dtype"] == "uint8"
        assert info["channel_means"]["c0"] == 10.0
        assert info["channel_means"]["c2"] == 30.0


class TestMockCamera:
    def test_mock_preview_is_rgb_shaped(self):
        mgr = CameraManager(CameraConfig(preview_width=320, preview_height=240), use_mock=True)
        mgr.initialize()
        frame = mgr.capture_preview_array()
        assert frame.shape == (240, 320, 3)
        meta = mgr.capture_metadata()
        assert meta["AfState"] == 2
        advertised = mgr.get_controls()
        assert "AfMode" in advertised
        assert "AfState" not in advertised
        mapping = mgr.apply_document_focus_window((288, 162, 1344, 756), (1920, 1080))
        assert mapping["preview_roi"] == (288, 162, 1344, 756)
        assert mapping["preview_size"] == (1920, 1080)
        assert mapping["sensor_roi"][2] > 0
        mgr.shutdown()


class TestPreviewToSensorMap:
    def test_16_by_9_preview_not_70_percent_of_4_by_3_sensor(self):
        preview_roi = (288, 162, 1344, 756)
        mapped = map_preview_roi_to_sensor(preview_roi, (1920, 1080), (0, 898, 9152, 5148))
        assert mapped != (1936, 1960, 5376, 3024)
        assert mapped[1] >= 898


class TestAfWindowConversion:
    def test_off_center_preview_roi_is_not_sensor_center(self):
        scaler = (0, 898, 9152, 5148)
        left = preview_roi_to_af_window((40, 200, 480, 640), (1920, 1080), scaler)
        mid = preview_roi_to_af_window((720, 270, 480, 540), (1920, 1080), scaler)
        assert left["sensor_roi"][0] < mid["sensor_roi"][0]

    def test_mock_apply_rejects_empty_roi(self):
        from idcam.exceptions import CameraError

        mgr = CameraManager(CameraConfig(preview_width=320, preview_height=240), use_mock=True)
        mgr.initialize()
        with pytest.raises(CameraError, match="Invalid document AF ROI"):
            mgr.apply_document_focus_window((0, 0, 0, 0), (320, 240))
        mgr.shutdown()


class FakePicamera2:
    """Minimal Picamera2 stand-in that reproduces the real encoder contract.

    ``capture_file`` mirrors Pillow: with no explicit ``format`` the container
    is inferred from the filename extension and an unknown extension raises.
    This is what made every capture fail on hardware while CI stayed green.
    """

    instances: list["FakePicamera2"] = []

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.closed = False
        self.started = False
        self.configured: object = None
        self.still_config: dict | None = None
        self.preview_config: dict | None = None
        self.capture_file_calls: list[tuple[str, str | None]] = []
        self.camera_controls = {
            "AfMode": (0, 2, 0),
            "AfTrigger": (0, 1, 0),
            "AfSpeed": (0, 1, 0),
            "AfRange": (0, 2, 0),
            "AfMetering": (0, 1, 0),
            "AfWindows": (0, 0, 0),
            "LensPosition": (0.0, 10.0, 1.0),
        }
        self.camera_properties = {"ScalerCropMaximum": (0, 0, 9152, 6944)}
        FakePicamera2.instances.append(self)

    def create_preview_configuration(self, main=None, buffer_count=1, **_):
        self.preview_config = {"main": dict(main or {}), "buffer_count": buffer_count}
        return self.preview_config

    def create_still_configuration(self, main=None, buffer_count=1, **_):
        self.still_config = {"main": dict(main or {}), "buffer_count": buffer_count}
        return self.still_config

    def configure(self, config):
        if self.fail_on == "configure":
            raise RuntimeError("configure exploded")
        self.configured = config

    def start(self):
        if self.fail_on == "start":
            raise RuntimeError("start exploded")
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True

    def set_controls(self, controls):
        return None

    def capture_metadata(self):
        return {"LensPosition": 3.5, "ScalerCrop": (0, 898, 9152, 5148)}

    def capture_array(self, name="main"):
        size = (self.configured or {}).get("main", {}).get("size", (64, 48))
        w, h = size
        return np.full((h, w, 3), 128, dtype=np.uint8)

    def capture_file(self, file_output, name="main", format=None, **_):
        self.capture_file_calls.append((str(file_output), format))
        resolved = format or Path(file_output).suffix.lstrip(".")
        if resolved.lower() not in ("jpg", "jpeg"):
            raise ValueError(f"unknown file extension: {Path(file_output).suffix}")
        size = (self.still_config or {}).get("main", {}).get("size", (64, 48))
        Image.new("RGB", size, (180, 170, 160)).save(file_output, format="JPEG")


@pytest.fixture
def fake_camera(monkeypatch):
    FakePicamera2.instances.clear()
    monkeypatch.setattr("idcam.camera.manager.PICAMERA2_AVAILABLE", True)
    monkeypatch.setattr("idcam.camera.manager.Picamera2", FakePicamera2)
    return FakePicamera2


def _manager(**overrides) -> CameraManager:
    config = CameraConfig(
        preview_width=640, preview_height=480, capture_width=1280, capture_height=960, **overrides
    )
    return CameraManager(config)


class TestStillEncoderFormat:
    """Regression coverage for the .partial capture failure."""

    def test_capture_file_receives_explicit_jpeg_format(self, fake_camera, tmp_path):
        mgr = _manager()
        mgr.initialize()
        target = tmp_path / "capture-abc.partial.jpg"
        mgr.capture_still_to_file(target)
        mgr.shutdown()

        (called_path, called_format) = fake_camera.instances[0].capture_file_calls[0]
        assert called_format is not None, "format must not be left to filename inference"
        assert called_format.lower() in ("jpg", "jpeg")
        assert called_path == str(target)
        assert target.exists()

    def test_capture_succeeds_for_a_partial_named_path(self, fake_camera, tmp_path):
        mgr = _manager()
        mgr.initialize()
        target = tmp_path / "capture-abc.partial.jpg"
        mgr.capture_still_to_file(target)
        mgr.shutdown()
        with Image.open(target) as img:
            assert img.format == "JPEG"
            assert img.size == (1280, 960)

    def test_fake_encoder_rejects_unknown_extension_without_format(self, tmp_path):
        """Proves the fake models the real failure the old code hit."""
        cam = FakePicamera2()
        cam.create_still_configuration(main={"size": (32, 24)})
        with pytest.raises(ValueError, match="unknown file extension"):
            cam.capture_file(str(tmp_path / "capture-abc.jpg.partial"))

    def test_missing_output_file_is_reported_not_swallowed(self, fake_camera, tmp_path):
        mgr = _manager()
        mgr.initialize()
        target = tmp_path / "capture.partial.jpg"
        fake_camera.instances[0].capture_file = lambda *a, **k: None
        with pytest.raises(CameraError, match="no file was written"):
            mgr.capture_still_to_file(target)
        mgr.shutdown()


class TestCameraLifecycle:
    def test_still_stream_uses_single_buffer(self, fake_camera):
        mgr = _manager()
        mgr.initialize()
        assert fake_camera.instances[0].still_config["buffer_count"] == 1
        mgr.shutdown()

    def test_failed_start_closes_the_device(self, fake_camera, monkeypatch):
        monkeypatch.setattr(
            "idcam.camera.manager.Picamera2", lambda *a, **k: FakePicamera2(fail_on="start")
        )
        mgr = _manager()
        with pytest.raises(CameraUnavailableError):
            mgr.initialize()
        assert fake_camera.instances[0].closed, "a failed init must not leave the device open"
        assert not mgr.is_started

    def test_failed_configure_closes_the_device(self, fake_camera, monkeypatch):
        monkeypatch.setattr(
            "idcam.camera.manager.Picamera2", lambda *a, **k: FakePicamera2(fail_on="configure")
        )
        mgr = _manager()
        with pytest.raises(CameraUnavailableError):
            mgr.initialize()
        assert fake_camera.instances[0].closed

    def test_camera_can_be_reacquired_after_failed_init(self, fake_camera, monkeypatch):
        monkeypatch.setattr(
            "idcam.camera.manager.Picamera2", lambda *a, **k: FakePicamera2(fail_on="start")
        )
        mgr = _manager()
        with pytest.raises(CameraUnavailableError):
            mgr.initialize()

        monkeypatch.setattr("idcam.camera.manager.Picamera2", FakePicamera2)
        mgr.initialize()
        assert mgr.is_started
        mgr.shutdown()
        assert fake_camera.instances[-1].closed

    def test_shutdown_is_idempotent(self, fake_camera):
        mgr = _manager()
        mgr.initialize()
        mgr.shutdown()
        mgr.shutdown()
        assert not mgr.is_started

    def test_stream_geometry_reports_configured_sizes(self, fake_camera):
        mgr = _manager()
        mgr.initialize()
        geometry = mgr.stream_geometry()
        assert geometry["preview"] == (640, 480)
        assert geometry["still"] == (1280, 960)
        assert geometry["still_format"] == "RGB888"
        assert geometry["requested"]["still"] == (1280, 960)
        mgr.shutdown()

    def test_stream_geometry_in_mock_mode(self):
        mgr = CameraManager(CameraConfig(preview_width=320, preview_height=240), use_mock=True)
        mgr.initialize()
        assert mgr.stream_geometry()["preview"] == (320, 240)
        mgr.shutdown()


class TestProductionCapturePath:
    """End-to-end through CaptureService, which is where the partial name is made."""

    async def test_capture_service_produces_a_decodable_jpeg(self, fake_camera, tmp_path):
        from idcam.camera.capture import CaptureService
        from idcam.storage.temp_files import TempFileStore

        mgr = _manager()
        mgr.initialize()
        store = TempFileStore(tmp_path)
        service = CaptureService(mgr.config, store)

        output = store.create_temp_path()
        saved = await service.capture(mgr, output)
        mgr.shutdown()

        assert saved == output
        assert saved.exists()
        assert not store.create_partial_path(output).exists()
        with Image.open(saved) as img:
            assert img.format == "JPEG"
            assert img.size == (1280, 960)

    async def test_capture_service_path_given_to_encoder_ends_in_jpg(self, fake_camera, tmp_path):
        from idcam.camera.capture import CaptureService
        from idcam.storage.temp_files import TempFileStore

        mgr = _manager()
        mgr.initialize()
        store = TempFileStore(tmp_path)
        service = CaptureService(mgr.config, store)
        await service.capture(mgr, store.create_temp_path())
        mgr.shutdown()

        encoder_path, encoder_format = fake_camera.instances[0].capture_file_calls[0]
        assert Path(encoder_path).suffix == ".jpg"
        assert encoder_format is not None

    async def test_existing_output_file_is_not_treated_as_new_capture(self, fake_camera, tmp_path):
        from idcam.camera.capture import CaptureService
        from idcam.exceptions import CaptureError
        from idcam.storage.temp_files import TempFileStore

        mgr = _manager()
        mgr.initialize()
        store = TempFileStore(tmp_path)
        service = CaptureService(mgr.config, store)
        output = store.create_temp_path()
        old = b"\xff\xd8\xffOLD-CAPTURE"
        output.write_bytes(old)
        with pytest.raises(CaptureError, match="already exists"):
            await service.capture(mgr, output)
        assert output.read_bytes() == old
        mgr.shutdown()

    async def test_stale_partial_is_not_finalized_as_success(self, fake_camera, tmp_path):
        from idcam.camera.capture import CaptureService
        from idcam.exceptions import CaptureError
        from idcam.storage.temp_files import TempFileStore

        mgr = _manager()
        mgr.initialize()
        store = TempFileStore(tmp_path)
        service = CaptureService(mgr.config, store)
        output = store.create_temp_path()
        partial = store.create_partial_path(output)
        partial.write_bytes(b"\xff\xd8\xffSTALE")
        fake_camera.instances[0].capture_file = lambda *a, **k: None
        with pytest.raises(CaptureError):
            await service.capture(mgr, output)
        assert not output.exists()
        mgr.shutdown()

    async def test_truncated_jpeg_is_not_success(self, fake_camera, tmp_path):
        from idcam.camera.capture import CaptureService
        from idcam.exceptions import CaptureError
        from idcam.storage.temp_files import TempFileStore

        mgr = _manager()
        mgr.initialize()
        store = TempFileStore(tmp_path)
        service = CaptureService(mgr.config, store)
        output = store.create_temp_path()

        def truncated(path, format=None, **_):
            fake_camera.instances[0].capture_file_calls.append((str(path), format))
            Path(path).write_bytes(b"\xff\xd8\xff" + b"\x00" * 32)

        fake_camera.instances[0].capture_file = truncated
        with pytest.raises(CaptureError, match="JPEG|truncated|validation"):
            await service.capture(mgr, output)
        assert not output.exists()
        mgr.shutdown()

    async def test_wrong_still_dimensions_are_rejected(self, fake_camera, tmp_path):
        from idcam.camera.capture import CaptureService
        from idcam.exceptions import CaptureError
        from idcam.storage.temp_files import TempFileStore

        mgr = _manager()
        mgr.initialize()
        store = TempFileStore(tmp_path)
        service = CaptureService(mgr.config, store)
        output = store.create_temp_path()

        def tiny_jpeg(path, format=None, **_):
            fake_camera.instances[0].capture_file_calls.append((str(path), format))
            Image.new("RGB", (32, 24), color=(80, 80, 80)).save(path, format="JPEG")

        fake_camera.instances[0].capture_file = tiny_jpeg
        with pytest.raises(CaptureError, match="Invalid dimensions"):
            await service.capture(mgr, output)
        assert not output.exists()
        mgr.shutdown()

    async def test_finalize_write_failure_is_capture_error(self, fake_camera, tmp_path):
        from idcam.camera.capture import CaptureService
        from idcam.exceptions import CaptureError
        from idcam.storage.temp_files import TempFileStore

        mgr = _manager()
        mgr.initialize()
        store = TempFileStore(tmp_path)
        service = CaptureService(mgr.config, store)
        output = store.create_temp_path()
        with patch.object(store, "finalize_capture", side_effect=OSError("disk full")):
            with pytest.raises(CaptureError, match="write failed"):
                await service.capture(mgr, output)
        mgr.shutdown()
