"""Temp file storage tests."""

from pathlib import Path

from idcam.storage.temp_files import TempFileStore


class TestTempFileStore:
    def test_create_temp_path(self, tmp_path):
        store = TempFileStore(tmp_path)
        path = store.create_temp_path()
        assert path.parent == tmp_path
        assert path.suffix == ".jpg"

    def test_atomic_write_finalize(self, tmp_path):
        store = TempFileStore(tmp_path)
        temp = store.create_temp_path()
        temp.write_bytes(b"test data")
        final = tmp_path / "final.jpg"
        store.atomic_write_finalize(temp, final)
        assert final.exists()
        assert not temp.exists()
        assert final.read_bytes() == b"test data"

    def test_cleanup(self, tmp_path):
        store = TempFileStore(tmp_path)
        path = store.create_temp_path()
        path.write_bytes(b"x")
        store.cleanup(path)
        assert not path.exists()

    def test_partial_path_keeps_image_extension_last(self, tmp_path):
        """Regression: encoders infer the container from the final extension.

        ``capture-x.jpg.partial`` made Picamera2/PIL see an unknown ``.partial``
        container and refuse to encode, so every real capture failed.
        """
        store = TempFileStore(tmp_path)
        final = store.create_temp_path(".jpg")
        partial = store.create_partial_path(final)

        assert partial.suffix == ".jpg"
        assert not partial.name.endswith(".partial")
        assert ".partial" in partial.name
        assert partial != final
        assert partial.parent == final.parent

    def test_partial_path_without_extension(self, tmp_path):
        store = TempFileStore(tmp_path)
        partial = store.create_partial_path(tmp_path / "capture-noext")
        assert partial.name == "capture-noext.partial"

    def test_partial_path_is_recognised_and_finalizes(self, tmp_path):
        store = TempFileStore(tmp_path)
        final = tmp_path / "final.jpg"
        partial = store.create_partial_path(final)
        assert store.is_partial_path(partial)
        assert not store.is_partial_path(final)

        partial.write_bytes(b"jpeg-data")
        store.finalize_capture(partial, final)
        assert final.exists()
        assert not partial.exists()

    def test_cleanup_stale_removes_infixed_partials(self, tmp_path):
        store = TempFileStore(tmp_path)
        partial = store.create_partial_path(tmp_path / "orphan.jpg")
        partial.write_bytes(b"x")
        import os
        import time

        os.utime(partial, (time.time() - 7200, time.time() - 7200))
        assert store.cleanup_stale(max_age_sec=3600) == 1
        assert not partial.exists()

    def test_cleanup_stale(self, tmp_path):
        store = TempFileStore(tmp_path)
        old = store.create_temp_path()
        old.write_bytes(b"old")
        import os
        import time

        os.utime(old, (time.time() - 7200, time.time() - 7200))
        removed = store.cleanup_stale(max_age_sec=3600)
        assert removed == 1
        assert not old.exists()
