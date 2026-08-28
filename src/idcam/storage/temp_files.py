"""Temporary file management with atomic writes and safe permissions."""

from __future__ import annotations

import os
import stat
import time
import uuid
from pathlib import Path

from idcam.logging.setup import get_logger
from idcam.storage.paths import ensure_writable_directory

logger = get_logger("storage.temp_files")

CAPTURE_PREFIX = "capture-"
PARTIAL_SUFFIX = ".partial"


class TempFileStore:
    """Manages temporary capture files with atomic rename and safe permissions."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        ensure_writable_directory(self.base_dir, "temp")
        self._apply_dir_permissions()

    def _apply_dir_permissions(self) -> None:
        """Restrict temp directory to owner/group only (no world-writable)."""
        try:
            self.base_dir.chmod(stat.S_IRWXU | stat.S_IRWXG)
        except OSError:
            pass

    def create_temp_path(self, suffix: str = ".jpg") -> Path:
        name = f"{CAPTURE_PREFIX}{uuid.uuid4().hex}{suffix}"
        return self.base_dir / name

    def create_partial_path(self, final_path: Path) -> Path:
        """In-progress path for ``final_path``, keeping the image extension last.

        The marker is infixed (``capture-x.partial.jpg``), never appended
        (``capture-x.jpg.partial``). Encoders such as Picamera2's
        ``capture_file()`` and ``PIL.Image.save()`` infer the container format
        from the final extension, and an unknown trailing suffix makes them
        raise instead of writing a JPEG.
        """
        suffix = final_path.suffix
        if not suffix:
            return final_path.with_name(final_path.name + PARTIAL_SUFFIX)
        return final_path.with_name(f"{final_path.stem}{PARTIAL_SUFFIX}{suffix}")

    @staticmethod
    def is_partial_path(path: Path) -> bool:
        return PARTIAL_SUFFIX in Path(path).name

    def finalize_capture(self, partial_path: Path, final_path: Path) -> Path:
        """Fsync partial file, validate exists, then atomically rename."""
        if not partial_path.exists():
            raise FileNotFoundError(f"Partial capture not found: {partial_path}")
        self._fsync_file(partial_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial_path, final_path)
        self._restrict_file_permissions(final_path)
        return final_path

    def _fsync_file(self, path: Path) -> None:
        try:
            with path.open("rb") as fh:
                os.fsync(fh.fileno())
        except OSError as exc:
            logger.warning("fsync failed for %s: %s", path, exc)

    def _restrict_file_permissions(self, path: Path) -> None:
        try:
            path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP)
        except OSError:
            pass

    def atomic_write_finalize(self, temp_path: Path, final_path: Path) -> Path:
        return self.finalize_capture(temp_path, final_path)

    def cleanup(self, path: Path | None) -> None:
        if path is None:
            return
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            logger.warning("Failed to cleanup %s: %s", path, exc)

    def cleanup_stale(self, max_age_sec: float = 3600) -> int:
        """Remove abandoned capture files in our directory only."""
        removed = 0
        now = time.time()
        for entry in self.base_dir.iterdir():
            if not entry.is_file():
                continue
            if not (entry.name.startswith(CAPTURE_PREFIX) or self.is_partial_path(entry)):
                continue
            try:
                if now - entry.stat().st_mtime > max_age_sec:
                    entry.unlink()
                    removed += 1
                    logger.info("Removed stale temp file: %s", entry.name)
            except OSError:
                continue
        return removed

    def ensure_space(self, required_bytes: int) -> None:
        """Raise OSError if insufficient disk space."""
        statvfs = os.statvfs(self.base_dir)
        free = statvfs.f_bavail * statvfs.f_frsize
        if free < required_bytes:
            raise OSError(f"Insufficient disk space: {free} bytes free, need {required_bytes}")
