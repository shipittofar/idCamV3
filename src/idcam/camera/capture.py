"""Full-resolution image capture."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Protocol

from PIL import Image

from idcam.config.schema import CameraConfig
from idcam.exceptions import CaptureError
from idcam.logging.setup import get_logger
from idcam.storage.temp_files import TempFileStore

logger = get_logger("camera.capture")

# Same default as QualityConfig.dimension_tolerance. libcamera may align still size.
_STILL_DIMENSION_TOLERANCE = 64


class CaptureCapableCamera(Protocol):
    def capture_still_to_file(self, path: Path) -> None: ...


class CaptureService:
    """Transactional still capture to disk with atomic finalize."""

    def __init__(self, config: CameraConfig, temp_store: TempFileStore) -> None:
        self.config = config
        self.temp_store = temp_store

    def _expected_still_size(self, camera: CaptureCapableCamera) -> tuple[int, int]:
        geo = getattr(camera, "stream_geometry", None)
        if callable(geo):
            try:
                still = geo().get("still")
                if still and len(still) == 2 and int(still[0]) > 0 and int(still[1]) > 0:
                    return int(still[0]), int(still[1])
            except Exception:
                pass
        return int(self.config.capture_width), int(self.config.capture_height)

    def _assert_valid_still_jpeg(self, path: Path, expected: tuple[int, int]) -> tuple[int, int, int]:
        """Decode JPEG and check production still dimensions. Never treats existence as success."""
        if not path.exists() or not path.is_file():
            raise CaptureError(f"Capture final file missing: {path}")
        size = path.stat().st_size
        if size <= 0:
            raise CaptureError("Capture file is empty")
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as exc:
            raise CaptureError(f"JPEG validation failed: {exc}") from exc
        try:
            with Image.open(path) as img:
                img.load()
                fmt = (img.format or "").upper()
                width, height = img.size
        except Exception as exc:
            raise CaptureError(f"JPEG truncated or incomplete: {exc}") from exc
        if fmt not in {"JPEG", "JPG"}:
            raise CaptureError(f"Capture is not JPEG: format={fmt!r}")
        ew, eh = expected
        if abs(width - ew) > _STILL_DIMENSION_TOLERANCE or abs(height - eh) > _STILL_DIMENSION_TOLERANCE:
            raise CaptureError(
                f"Invalid dimensions: {width}x{height} expected {ew}x{eh} "
                f"(tol={_STILL_DIMENSION_TOLERANCE})"
            )
        return width, height, size

    async def capture(self, camera: CaptureCapableCamera, output_path: Path) -> Path:
        if output_path.exists():
            raise CaptureError(f"Capture target already exists: {output_path}")
        partial_path = self.temp_store.create_partial_path(output_path)
        if partial_path.exists():
            logger.warning("event=capture_stale_partial_removed path=%s", partial_path)
            self.temp_store.cleanup(partial_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        logger.info("event=capture_start path=%s", output_path)

        try:
            await asyncio.wait_for(
                asyncio.to_thread(camera.capture_still_to_file, partial_path),
                timeout=self.config.capture_timeout_sec,
            )
        except asyncio.TimeoutError as exc:
            self.temp_store.cleanup(partial_path)
            raise CaptureError(f"Capture timed out after {self.config.capture_timeout_sec}s") from exc
        except CaptureError:
            self.temp_store.cleanup(partial_path)
            raise
        except Exception as exc:
            self.temp_store.cleanup(partial_path)
            raise CaptureError(f"Capture encode failed: {exc}") from exc

        if not partial_path.exists():
            raise CaptureError(f"Capture file not created: {partial_path}")

        size = partial_path.stat().st_size
        if size == 0:
            self.temp_store.cleanup(partial_path)
            raise CaptureError("Capture file is empty")

        try:
            final = self.temp_store.finalize_capture(partial_path, output_path)
        except Exception as exc:
            self.temp_store.cleanup(partial_path)
            self.temp_store.cleanup(output_path)
            raise CaptureError(f"Capture write failed: {exc}") from exc

        try:
            width, height, file_bytes = self._assert_valid_still_jpeg(
                final, self._expected_still_size(camera)
            )
        except CaptureError:
            self.temp_store.cleanup(final)
            raise

        duration_ms = (time.monotonic() - started) * 1000.0
        logger.info(
            "event=capture_encoded path=%s width=%d height=%d size_bytes=%d duration_ms=%.0f",
            final,
            width,
            height,
            file_bytes,
            duration_ms,
        )
        return final
