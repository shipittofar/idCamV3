"""Post-capture image validation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from idcam.camera.manager import preview_roi_is_usable, preview_roi_to_still_roi
from idcam.config.schema import CameraConfig, QualityConfig
from idcam.exceptions import ImageValidationError
from idcam.logging.setup import get_logger
from idcam.vision.sharpness import (
    clip_roi,
    laplacian_variance,
    mean_brightness,
    reconstruct_preview_view,
    roi_rect,
)

logger = get_logger("camera.quality")

# Preview AF Laplacian on a focused 1080p document is typically ~10–40.
# QualityConfig.min_sharpness (80) assumed native 64MP scores would be much
# higher. Hardware showed ~9 on the same operator — do not compare a
# preview-equivalent still score to 80.
PREVIEW_EQUIVALENT_FLOOR = 6.0
STILL_SHARPNESS_HOLD_RATIO = 0.50


@dataclass
class ValidationResult:
    width: int
    height: int
    file_bytes: int
    sharpness: float
    brightness: float
    sha256: str
    validation_roi: tuple[int, int, int, int] = (0, 0, 0, 0)
    sharpness_method: str = ""
    sharpness_threshold: float = 0.0


class ImageValidator:
    """Validates captured JPEG files before upload."""

    def __init__(self, quality_config: QualityConfig, camera_config: CameraConfig | None = None) -> None:
        self.config = quality_config
        self.camera_config = camera_config

    def validate(
        self,
        path: Path,
        *,
        preview_roi: tuple[int, int, int, int] | None = None,
        preview_size: tuple[int, int] | None = None,
        scaler_crop: tuple[int, int, int, int] | None = None,
        preview_sharpness: float | None = None,
        require_document_roi: bool = False,
    ) -> ValidationResult:
        if not path.exists():
            raise ImageValidationError(f"Image file not found: {path}")
        if not path.is_file():
            raise ImageValidationError(f"Not a regular file: {path}")

        file_bytes = path.stat().st_size
        if file_bytes < self.config.min_file_bytes:
            raise ImageValidationError(f"File too small: {file_bytes} bytes")
        if file_bytes > self.config.max_file_bytes:
            raise ImageValidationError(f"File too large: {file_bytes} bytes")

        sha256 = self._compute_sha256(path)

        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as exc:
            raise ImageValidationError(f"JPEG corrupt or undecodable: {exc}") from exc

        try:
            with Image.open(path) as img:
                img.load()
                width, height = img.size
                rgb = np.asarray(img.convert("RGB"))
        except Exception as exc:
            raise ImageValidationError(f"JPEG truncated or incomplete: {exc}") from exc

        cv_height, cv_width = rgb.shape[:2]
        channels = int(rgb.shape[2]) if rgb.ndim == 3 else 1
        if cv_width != width or cv_height != height:
            raise ImageValidationError("Image dimension mismatch between PIL size and decoded array")

        if width < self.config.min_width or height < self.config.min_height:
            raise ImageValidationError(f"Dimensions too small: {width}x{height}")

        if self.config.require_expected_dimensions and self.camera_config is not None:
            tol = self.config.dimension_tolerance
            ew, eh = self.camera_config.capture_width, self.camera_config.capture_height
            if abs(width - ew) > tol or abs(height - eh) > tol:
                raise ImageValidationError(
                    f"Dimensions {width}x{height} do not match expected {ew}x{eh} (tol={tol})"
                )

        still_size = (width, height)
        original_roi = preview_roi
        scaled_roi: tuple[int, int, int, int] | None = None
        validation_roi: tuple[int, int, int, int]
        sharpness_method: str
        resize_dimensions = ""

        if require_document_roi and (
            preview_roi is None or not preview_roi_is_usable(preview_roi, preview_size or still_size)
        ):
            raise ImageValidationError("invalid_roi: document ROI is empty or unusable")

        if preview_size is not None:
            crop = scaler_crop if scaler_crop is not None else (0, 0, width, height)
            try:
                preview_view = reconstruct_preview_view(rgb, crop, preview_size)
            except ValueError as exc:
                raise ImageValidationError(f"invalid_roi: {exc}") from exc
            resize_dimensions = f"{preview_size[0]}x{preview_size[1]}"
            if preview_roi is not None:
                if not preview_roi_is_usable(preview_roi, preview_size):
                    raise ImageValidationError("invalid_roi: document ROI is empty or unusable")
                validation_roi = clip_roi(preview_roi, preview_view.shape)
                try:
                    scaled_roi = preview_roi_to_still_roi(preview_roi, preview_size, crop, still_size)
                except ValueError as exc:
                    raise ImageValidationError(f"invalid_roi: {exc}") from exc
                sharpness = laplacian_variance(preview_view, roi=validation_roi)
                sharpness_method = "laplacian_variance_preview_equivalent"
            else:
                validation_roi = roi_rect(preview_view.shape)
                sharpness = laplacian_variance(preview_view, roi=validation_roi)
                sharpness_method = "laplacian_variance_preview_equivalent_center70"
            hold = (
                STILL_SHARPNESS_HOLD_RATIO * float(preview_sharpness)
                if preview_sharpness is not None and preview_sharpness > 0
                else 0.0
            )
            threshold = max(PREVIEW_EQUIVALENT_FLOOR, hold)
            brightness_frame = preview_view
        else:
            sharpness = laplacian_variance(rgb)
            validation_roi = roi_rect(rgb.shape)
            sharpness_method = "laplacian_variance_center70_native"
            threshold = float(self.config.min_sharpness)
            brightness_frame = rgb

        roi_w, roi_h = validation_roi[2], validation_roi[3]
        logger.info(
            "event=image_validation event_alias=validation_start path=%s file_size=%d "
            "decoded_width=%d decoded_height=%d decoded_channels=%d image_dtype=%s "
            "original_roi=%s scaled_roi=%s validation_roi=%s roi_width=%d roi_height=%d "
            "sharpness_method=%s sharpness_value=%.2f configured_threshold=%.2f "
            "min_sharpness_config=%.1f resize_dimensions=%s image_size=%dx%d",
            path,
            file_bytes,
            width,
            height,
            channels,
            rgb.dtype,
            original_roi,
            scaled_roi,
            validation_roi,
            roi_w,
            roi_h,
            sharpness_method,
            sharpness,
            threshold,
            self.config.min_sharpness,
            resize_dimensions or "none",
            width,
            height,
        )

        if sharpness < threshold:
            logger.info(
                "event=image_validation result=invalid reason=blurry sharpness=%.1f threshold=%.1f",
                sharpness,
                threshold,
            )
            raise ImageValidationError(
                f"Image too blurry: sharpness={sharpness:.1f} min={threshold:.1f}"
            )

        brightness = mean_brightness(brightness_frame)
        if brightness < self.config.min_brightness:
            raise ImageValidationError(f"Image too dark: brightness={brightness:.1f}")
        if brightness > self.config.max_brightness:
            raise ImageValidationError(f"Image too bright: brightness={brightness:.1f}")

        if float(np.std(rgb)) < 5.0:
            raise ImageValidationError("Image appears uniform/corrupted")

        result = ValidationResult(
            width=width,
            height=height,
            file_bytes=file_bytes,
            sharpness=sharpness,
            brightness=brightness,
            sha256=sha256,
            validation_roi=validation_roi,
            sharpness_method=sharpness_method,
            sharpness_threshold=threshold,
        )
        logger.info(
            "event=image_validation result=valid path=%s image_size=%dx%d "
            "roi_source=%s roi=%s roi_size=%dx%d sharpness_method=%s sharpness=%.1f "
            "threshold=%.1f",
            path,
            width,
            height,
            "document_preview" if preview_roi is not None else "center70",
            validation_roi,
            roi_w,
            roi_h,
            sharpness_method,
            sharpness,
            threshold,
        )
        logger.info(
            "Image validated: %dx%d, %d bytes, sharpness=%.1f, sha256=%s",
            width,
            height,
            file_bytes,
            sharpness,
            sha256[:16],
        )
        return result

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
