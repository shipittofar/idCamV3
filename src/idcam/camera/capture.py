"""Full-resolution image capture."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

from idcam.config.schema import CameraConfig
from idcam.exceptions import CaptureError
from idcam.logging.setup import get_logger
from idcam.storage.temp_files import TempFileStore

logger = get_logger("camera.capture")


class CaptureCapableCamera(Protocol):
    def capture_still_to_file(self, path: Path) -> None: ...


class CaptureService:
    """Transactional still capture to disk with atomic finalize."""

    def __init__(self, config: CameraConfig, temp_store: TempFileStore) -> None:
        self.config = config
        self.temp_store = temp_store

    async def capture(self, camera: CaptureCapableCamera, output_path: Path) -> Path:
        if output_path.exists():
            raise CaptureError(f"Capture target already exists: {output_path}")
        partial_path = self.temp_store.create_partial_path(output_path)
        if partial_path.exists():
            logger.warning("event=capture_stale_partial_removed path=%s", partial_path)
            self.temp_store.cleanup(partial_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("event=capture_started path=%s", output_path)

        try:
            await asyncio.wait_for(
                asyncio.to_thread(camera.capture_still_to_file, partial_path),
                timeout=self.config.capture_timeout_sec,
            )
        except asyncio.TimeoutError as exc:
            self.temp_store.cleanup(partial_path)
            raise CaptureError(f"Capture timed out after {self.config.capture_timeout_sec}s") from exc
        except Exception as exc:
            self.temp_store.cleanup(partial_path)
            raise CaptureError(f"Capture failed: {exc}") from exc

        if not partial_path.exists():
            raise CaptureError(f"Capture file not created: {partial_path}")

        size = partial_path.stat().st_size
        if size == 0:
            self.temp_store.cleanup(partial_path)
            raise CaptureError("Capture file is empty")

        final = self.temp_store.finalize_capture(partial_path, output_path)
        logger.info("event=capture_completed path=%s size=%d", final, size)
        return final
