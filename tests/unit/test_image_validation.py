"""Image validation tests."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from idcam.camera.quality import ImageValidator
from idcam.config.schema import QualityConfig
from idcam.exceptions import ImageValidationError


@pytest.fixture
def quality_config():
    return QualityConfig(min_sharpness=10.0, min_file_bytes=1000, min_width=100, min_height=100)


@pytest.fixture
def valid_image(tmp_path):
    path = tmp_path / "valid.jpg"
    # Create image with edges for sharpness
    arr = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(path, format="JPEG", quality=90)
    return path


class TestImageValidator:
    def test_valid_image(self, quality_config, valid_image):
        validator = ImageValidator(quality_config)
        result = validator.validate(valid_image)
        assert result.width == 640
        assert result.height == 480

    def test_missing_file(self, quality_config, tmp_path):
        validator = ImageValidator(quality_config)
        with pytest.raises(ImageValidationError, match="not found"):
            validator.validate(tmp_path / "missing.jpg")

    def test_empty_file(self, quality_config, tmp_path):
        path = tmp_path / "empty.jpg"
        path.write_bytes(b"")
        validator = ImageValidator(quality_config)
        with pytest.raises(ImageValidationError):
            validator.validate(path)

    def test_corrupt_file(self, quality_config, tmp_path):
        path = tmp_path / "corrupt.jpg"
        path.write_bytes(b"not a jpeg at all" * 100)
        validator = ImageValidator(quality_config)
        with pytest.raises(ImageValidationError):
            validator.validate(path)

    def test_too_small_dimensions(self, quality_config, tmp_path):
        path = tmp_path / "tiny.jpg"
        img = Image.new("RGB", (10, 10), color=(128, 128, 128))
        img.save(path)
        validator = ImageValidator(quality_config)
        with pytest.raises(ImageValidationError, match="too small"):
            validator.validate(path)

    def test_uniform_image_rejected(self, quality_config, tmp_path):
        path = tmp_path / "uniform.jpg"
        img = Image.new("RGB", (640, 480), color=(128, 128, 128))
        img.save(path, quality=90)
        validator = ImageValidator(quality_config)
        with pytest.raises(ImageValidationError):
            validator.validate(path)


def _save_rgb(path: Path, arr: np.ndarray) -> Path:
    Image.fromarray(arr).save(path, format="JPEG", quality=95)
    return path


def _checkerboard(h: int, w: int) -> np.ndarray:
    yy, xx = np.indices((h, w))
    mask = ((xx // 6) + (yy // 6)) % 2
    plane = np.where(mask, 255, 0).astype(np.uint8)
    return np.stack([plane, plane, plane], axis=-1)


class TestStillDocumentValidation:
    def test_full_resolution_sharp_document_passes(self, tmp_path):
        still_w, still_h = 400, 300
        preview = (200, 150)
        roi = (40, 30, 80, 60)
        arr = np.full((still_h, still_w, 3), 80, dtype=np.uint8)
        arr[60:180, 80:240] = _checkerboard(120, 160)
        path = _save_rgb(tmp_path / "still.jpg", arr)
        validator = ImageValidator(
            QualityConfig(
                min_sharpness=80.0,
                min_file_bytes=500,
                min_width=100,
                min_height=100,
                require_expected_dimensions=False,
            )
        )
        result = validator.validate(
            path,
            preview_roi=roi,
            preview_size=preview,
            scaler_crop=(0, 0, still_w, still_h),
            preview_sharpness=10.7,
            require_document_roi=True,
        )
        assert result.width == still_w
        assert result.height == still_h
        assert result.sharpness >= result.sharpness_threshold
        assert result.sharpness_method == "laplacian_variance_preview_equivalent"
        assert result.sharpness_threshold < 80.0

    def test_wrong_dimensions_fail(self, tmp_path):
        from idcam.config.schema import CameraConfig

        path = _save_rgb(tmp_path / "small.jpg", _checkerboard(120, 160))
        validator = ImageValidator(
            QualityConfig(
                min_sharpness=80.0,
                min_file_bytes=200,
                min_width=64,
                min_height=64,
                require_expected_dimensions=True,
            ),
            CameraConfig(capture_width=9152, capture_height=6944),
        )
        with pytest.raises(ImageValidationError, match="do not match expected"):
            validator.validate(path, preview_size=(1920, 1080))

    def test_empty_roi_fails(self, tmp_path):
        path = _save_rgb(tmp_path / "doc.jpg", _checkerboard(240, 320))
        validator = ImageValidator(
            QualityConfig(
                min_sharpness=80.0,
                min_file_bytes=200,
                min_width=64,
                min_height=64,
                require_expected_dimensions=False,
            )
        )
        with pytest.raises(ImageValidationError, match="invalid_roi"):
            validator.validate(
                path,
                preview_roi=(0, 0, 0, 0),
                preview_size=(320, 240),
                scaler_crop=(0, 0, 320, 240),
                require_document_roi=True,
            )

    def test_preview_roi_is_scaled_not_used_raw_on_still(self, tmp_path):
        still_w, still_h = 400, 300
        preview = (100, 75)
        preview_roi = (10, 10, 20, 15)
        arr = np.full((still_h, still_w, 3), 128, dtype=np.uint8)
        arr[40:100, 40:120] = _checkerboard(60, 80)
        path = _save_rgb(tmp_path / "mapped.jpg", arr)
        validator = ImageValidator(
            QualityConfig(
                min_sharpness=80.0,
                min_file_bytes=200,
                min_width=64,
                min_height=64,
                require_expected_dimensions=False,
            )
        )
        result = validator.validate(
            path,
            preview_roi=preview_roi,
            preview_size=preview,
            scaler_crop=(0, 0, still_w, still_h),
            preview_sharpness=12.0,
            require_document_roi=True,
        )
        assert result.validation_roi[0] == 10
        assert result.validation_roi[2] == 20
        assert result.sharpness > 6.0

    def test_genuinely_blurry_image_fails(self, tmp_path):
        import cv2

        sharp = _checkerboard(240, 320)
        blur = cv2.GaussianBlur(sharp, (51, 51), 12)
        path = _save_rgb(tmp_path / "blur.jpg", blur)
        validator = ImageValidator(
            QualityConfig(
                min_sharpness=80.0,
                min_file_bytes=200,
                min_width=64,
                min_height=64,
                require_expected_dimensions=False,
            )
        )
        with pytest.raises(ImageValidationError, match="too blurry"):
            validator.validate(
                path,
                preview_roi=(20, 20, 200, 160),
                preview_size=(320, 240),
                scaler_crop=(0, 0, 320, 240),
                preview_sharpness=10.7,
                require_document_roi=True,
            )

    def test_preview_equivalent_scale_does_not_use_min_80(self, tmp_path):
        path = _save_rgb(tmp_path / "ok.jpg", _checkerboard(240, 320))
        cfg = QualityConfig(
            min_sharpness=80.0,
            min_file_bytes=200,
            min_width=64,
            min_height=64,
            require_expected_dimensions=False,
        )
        validator = ImageValidator(cfg)
        native = validator.validate(path)
        assert native.sharpness_threshold == 80.0
        equivalent = validator.validate(
            path,
            preview_roi=(16, 16, 200, 160),
            preview_size=(320, 240),
            scaler_crop=(0, 0, 320, 240),
            preview_sharpness=10.7,
            require_document_roi=True,
        )
        assert equivalent.sharpness_threshold == pytest.approx(max(6.0, 0.5 * 10.7))
        assert equivalent.sharpness >= equivalent.sharpness_threshold
        assert equivalent.sharpness_threshold < 80.0

    def test_center70_misses_offcenter_document_roi_hits(self, tmp_path):
        arr = np.full((300, 400, 3), 128, dtype=np.uint8)
        arr[0:40, 0:50] = _checkerboard(40, 50)
        path = _save_rgb(tmp_path / "corner.jpg", arr)
        cfg = QualityConfig(
            min_sharpness=80.0,
            min_file_bytes=200,
            min_width=64,
            min_height=64,
            require_expected_dimensions=False,
        )
        validator = ImageValidator(cfg)
        with pytest.raises(ImageValidationError, match="too blurry"):
            validator.validate(path)
        result = validator.validate(
            path,
            preview_roi=(0, 0, 50, 40),
            preview_size=(400, 300),
            scaler_crop=(0, 0, 400, 300),
            preview_sharpness=11.0,
            require_document_roi=True,
        )
        assert result.sharpness >= result.sharpness_threshold


class TestValidationPerformance:
    def test_resize_input_is_document_crop_not_full_still(self, tmp_path, monkeypatch):
        import cv2

        seen: list[tuple[int, int]] = []
        orig = cv2.resize

        def spy(src, dsize, *args, **kwargs):
            seen.append((int(src.shape[1]), int(src.shape[0])))
            return orig(src, dsize, *args, **kwargs)

        monkeypatch.setattr("idcam.vision.sharpness.cv2.resize", spy)
        still_w, still_h = 400, 300
        preview = (200, 150)
        roi = (40, 30, 80, 60)
        arr = np.full((still_h, still_w, 3), 80, dtype=np.uint8)
        arr[60:180, 80:240] = _checkerboard(120, 160)
        path = _save_rgb(tmp_path / "perf.jpg", arr)
        validator = ImageValidator(
            QualityConfig(
                min_sharpness=80.0,
                min_file_bytes=200,
                min_width=64,
                min_height=64,
                require_expected_dimensions=False,
            )
        )
        result = validator.validate(
            path,
            preview_roi=roi,
            preview_size=preview,
            scaler_crop=(0, 0, still_w, still_h),
            preview_sharpness=10.7,
            require_document_roi=True,
        )
        assert result.sharpness >= result.sharpness_threshold
        assert seen
        src_w, src_h = seen[0]
        assert src_w * src_h < still_w * still_h
        assert src_w <= 162
        assert src_h <= 122
        assert "resize" in result.timing_ms
        assert "jpeg_decode" in result.timing_ms

    def test_one_validation_call_records_single_decode(self, tmp_path, caplog):
        import logging

        caplog.set_level(logging.INFO)
        path = _save_rgb(tmp_path / "once.jpg", _checkerboard(240, 320))
        validator = ImageValidator(
            QualityConfig(
                min_sharpness=80.0,
                min_file_bytes=200,
                min_width=64,
                min_height=64,
                require_expected_dimensions=False,
            )
        )
        validator.validate(
            path,
            preview_roi=(16, 16, 200, 160),
            preview_size=(320, 240),
            scaler_crop=(0, 0, 320, 240),
            preview_sharpness=10.0,
            require_document_roi=True,
        )
        completes = [r.message for r in caplog.records if "stage=validation_complete" in r.message]
        assert len(completes) == 1
        assert "decode_ops=1" in completes[0]
        assert "resize_ops=1" in completes[0]
