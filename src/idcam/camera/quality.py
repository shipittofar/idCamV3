"""Post-capture image validation."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from idcam.camera.manager import preview_roi_is_usable, preview_roi_to_still_roi
from idcam.config.schema import CameraConfig, QualityConfig
from idcam.exceptions import ImageValidationError
from idcam.logging.setup import get_logger
from idcam.vision.sharpness import (
    laplacian_variance,
    mean_brightness,
    resize_patch_to_preview,
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
    timing_ms: dict[str, float] = field(default_factory=dict)


def _log_timing(stage: str, duration_ms: float, cpu_ms: float | None = None, **fields: Any) -> None:
    extra = "".join(f" {key}={value}" for key, value in fields.items() if value is not None)
    if cpu_ms is None:
        logger.info("event=image_validation_timing stage=%s duration_ms=%.1f%s", stage, duration_ms, extra)
        return
    logger.info(
        "event=image_validation_timing stage=%s duration_ms=%.1f cpu_ms=%.1f%s",
        stage,
        duration_ms,
        cpu_ms,
        extra,
    )


class _Stage:
    def __init__(self, name: str, **fields: Any) -> None:
        self.name = name
        self.fields = fields
        self.wall0 = 0.0
        self.cpu0 = 0.0
        self.duration_ms = 0.0

    def __enter__(self) -> _Stage:
        self.wall0 = time.perf_counter()
        self.cpu0 = time.process_time()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.duration_ms = (time.perf_counter() - self.wall0) * 1000.0
        cpu_ms = (time.process_time() - self.cpu0) * 1000.0
        _log_timing(self.name, self.duration_ms, cpu_ms, **self.fields)


class ImageValidator:
    """Validates captured JPEG files before upload."""

    def __init__(self, quality_config: QualityConfig, camera_config: CameraConfig | None = None) -> None:
        self.config = quality_config
        self.camera_config = camera_config
        self._validation_calls = 0

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
        self._validation_calls += 1
        retry_count = max(0, self._validation_calls - 1)
        wall0 = time.perf_counter()
        cpu0 = time.process_time()
        decode_ops = 0
        resize_ops = 0
        timing_ms: dict[str, float] = {}

        if not path.exists():
            raise ImageValidationError(f"Image file not found: {path}")
        if not path.is_file():
            raise ImageValidationError(f"Not a regular file: {path}")

        with _Stage("file_stat") as st:
            file_bytes = path.stat().st_size
        timing_ms["file_stat"] = st.duration_ms
        _log_timing("file_size_available", 0.0, file_size=file_bytes)
        if file_bytes < self.config.min_file_bytes:
            raise ImageValidationError(f"File too small: {file_bytes} bytes")
        if file_bytes > self.config.max_file_bytes:
            raise ImageValidationError(f"File too large: {file_bytes} bytes")

        with _Stage("sha256") as st:
            sha256 = self._compute_sha256(path)
        timing_ms["sha256"] = st.duration_ms

        original_roi = preview_roi
        scaled_roi: tuple[int, int, int, int] | None = None
        validation_roi: tuple[int, int, int, int]
        sharpness_method: str
        resize_source = ""
        resize_target = ""
        brightness_frame: np.ndarray
        width = 0
        height = 0

        try:
            with Image.open(path) as img:
                with _Stage("jpeg_header") as st:
                    width, height = img.size
                    fmt = (img.format or "").upper()
                timing_ms["jpeg_header"] = st.duration_ms
                _log_timing("image_dimensions", 0.0, decoded_width=width, decoded_height=height, format=fmt)
                if fmt not in {"JPEG", "JPG", ""}:
                    raise ImageValidationError(f"Capture is not JPEG: format={fmt!r}")
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
                crop = scaler_crop if scaler_crop is not None else (0, 0, width, height)

                if require_document_roi and (
                    preview_roi is None or not preview_roi_is_usable(preview_roi, preview_size or still_size)
                ):
                    raise ImageValidationError("invalid_roi: document ROI is empty or unusable")

                if preview_size is not None:
                    measure_roi = preview_roi
                    if measure_roi is None:
                        measure_roi = roi_rect((int(preview_size[1]), int(preview_size[0])))
                        sharpness_method = "laplacian_variance_preview_equivalent_center70"
                    else:
                        if not preview_roi_is_usable(measure_roi, preview_size):
                            raise ImageValidationError("invalid_roi: document ROI is empty or unusable")
                        sharpness_method = "laplacian_variance_preview_equivalent"
                    validation_roi = tuple(int(v) for v in measure_roi)
                    with _Stage("roi_mapping") as st:
                        try:
                            scaled_roi = preview_roi_to_still_roi(
                                measure_roi, preview_size, crop, still_size
                            )
                        except ValueError as exc:
                            raise ImageValidationError(f"invalid_roi: {exc}") from exc
                    timing_ms["roi_mapping"] = st.duration_ms

                    with _Stage("jpeg_decode") as st:
                        img.load()
                        decode_ops += 1
                    timing_ms["jpeg_decode"] = st.duration_ms
                    x, y, rw, rh = scaled_roi
                    with _Stage("still_crop", crop=f"{rw}x{rh}") as st:
                        patch = img.crop((x, y, x + rw, y + rh))
                        if patch.mode != "RGB":
                            patch = patch.convert("RGB")
                        patch_rgb = np.asarray(patch)
                    timing_ms["still_crop"] = st.duration_ms
                    resize_source = f"{rw}x{rh}"
                    resize_target = f"{validation_roi[2]}x{validation_roi[3]}"
                    with _Stage(
                        "resize",
                        source_dimensions=resize_source,
                        target_dimensions=resize_target,
                    ) as st:
                        preview_patch = resize_patch_to_preview(
                            patch_rgb, (validation_roi[2], validation_roi[3])
                        )
                        resize_ops += 1
                    timing_ms["resize"] = st.duration_ms
                    brightness_frame = preview_patch
                    with _Stage("grayscale") as st:
                        if preview_patch.ndim == 3:
                            gray = cv2.cvtColor(preview_patch, cv2.COLOR_RGB2GRAY)
                        else:
                            gray = preview_patch
                    timing_ms["grayscale"] = st.duration_ms
                    with _Stage("laplacian", roi_width=gray.shape[1], roi_height=gray.shape[0]) as st:
                        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    timing_ms["laplacian"] = st.duration_ms
                else:
                    sharpness_method = "laplacian_variance_center70_native"
                    with _Stage("jpeg_decode") as st:
                        img.load()
                        decode_ops += 1
                        rgb = np.asarray(img.convert("RGB"))
                    timing_ms["jpeg_decode"] = st.duration_ms
                    validation_roi = roi_rect(rgb.shape)
                    brightness_frame = rgb
                    with _Stage("grayscale") as st:
                        sharpness = laplacian_variance(rgb)
                    timing_ms["grayscale"] = st.duration_ms
                    timing_ms["laplacian"] = timing_ms["grayscale"]
        except ImageValidationError:
            raise
        except Exception as exc:
            raise ImageValidationError(f"JPEG truncated or incomplete: {exc}") from exc

        roi_w, roi_h = validation_roi[2], validation_roi[3]
        hold = (
            STILL_SHARPNESS_HOLD_RATIO * float(preview_sharpness)
            if preview_size is not None and preview_sharpness is not None and preview_sharpness > 0
            else 0.0
        )
        threshold = (
            max(PREVIEW_EQUIVALENT_FLOOR, hold)
            if preview_size is not None
            else float(self.config.min_sharpness)
        )

        logger.info(
            "event=image_validation event_alias=validation_start path=%s file_size=%d "
            "decoded_width=%d decoded_height=%d decoded_channels=3 image_dtype=uint8 "
            "original_roi=%s scaled_roi=%s validation_roi=%s roi_width=%d roi_height=%d "
            "sharpness_method=%s sharpness_value=%.2f configured_threshold=%.2f "
            "min_sharpness_config=%.1f resize_dimensions=%s image_size=%dx%d "
            "decode_ops=%d resize_ops=%d validation_retry_count=%d",
            path,
            file_bytes,
            width,
            height,
            original_roi,
            scaled_roi,
            validation_roi,
            roi_w,
            roi_h,
            sharpness_method,
            sharpness,
            threshold,
            self.config.min_sharpness,
            f"{resize_source}->{resize_target}" if resize_target else "none",
            width,
            height,
            decode_ops,
            resize_ops,
            retry_count,
        )
        _log_timing(
            "sharpness_result",
            0.0,
            sharpness=f"{sharpness:.2f}",
            threshold=f"{threshold:.2f}",
            method=sharpness_method,
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

        with _Stage("brightness_std") as st:
            brightness = mean_brightness(brightness_frame)
            if brightness < self.config.min_brightness:
                raise ImageValidationError(f"Image too dark: brightness={brightness:.1f}")
            if brightness > self.config.max_brightness:
                raise ImageValidationError(f"Image too bright: brightness={brightness:.1f}")
            if float(np.std(brightness_frame)) < 5.0:
                raise ImageValidationError("Image appears uniform/corrupted")
        timing_ms["brightness_std"] = st.duration_ms

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
            timing_ms=timing_ms,
        )
        total_ms = (time.perf_counter() - wall0) * 1000.0
        total_cpu = (time.process_time() - cpu0) * 1000.0
        timing_ms["total"] = total_ms
        _log_timing(
            "validation_complete",
            total_ms,
            total_cpu,
            decode_ops=decode_ops,
            resize_ops=resize_ops,
            validation_retry_count=retry_count,
            file_size=file_bytes,
            image_size=f"{width}x{height}",
            roi_size=f"{roi_w}x{roi_h}",
            resize_source=resize_source or "none",
            resize_target=resize_target or "none",
        )
        logger.info(
            "event=image_validation result=valid path=%s image_size=%dx%d "
            "roi_source=%s roi=%s roi_size=%dx%d sharpness_method=%s sharpness=%.1f "
            "threshold=%.1f duration_ms=%.0f",
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
            total_ms,
        )
        return result

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
