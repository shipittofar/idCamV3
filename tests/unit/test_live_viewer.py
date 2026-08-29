"""Live Debug Viewer tests. The viewer is an observer, not a camera producer."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from http.client import HTTPResponse

import cv2
import numpy as np
import pytest

from idcam.app.orchestrator import ApplicationOrchestrator
from idcam.config.schema import AppConfig, DebugConfig, DetectionConfig, ServerConfig, UIConfig
from idcam.preview.live_viewer import (
    DebugSnapshot,
    LiveDebugViewer,
    LiveFrameBuffer,
    compose_debug_frame,
    draw_overlay,
    encode_jpeg,
    make_placeholder,
    resize_debug_frame,
)
from idcam.vision.document_detector import DocumentDetector
from idcam.vision.stability_detector import StabilityDetector


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    if port < 1024:
        pytest.skip("could not allocate an ephemeral port >= 1024")
    return port


def _wait_jpeg(viewer: LiveDebugViewer, timeout: float = 4.0) -> bytes:
    deadline = time.monotonic() + timeout
    last = -1
    while time.monotonic() < deadline:
        seq, jpeg = viewer.wait_jpeg(last, timeout=0.2)
        if jpeg:
            return bytes(jpeg)
        last = seq
    raise TimeoutError("no JPEG produced")


def _read_mjpeg_part(resp: HTTPResponse, limit: int = 2_000_000) -> bytes:
    data = b""
    while len(data) < limit:
        chunk = resp.read(2048)
        if not chunk:
            break
        data += chunk
        if b"\xff\xd8" in data and b"\xff\xd9" in data:
            break
    return data


@pytest.fixture
def app_config(tmp_path):
    return AppConfig(
        device_id="test-device",
        temp_dir=tmp_path / "tmp",
        server=ServerConfig(url="https://api.example.com"),
        ui=UIConfig(
            status_file=tmp_path / "run" / "status.json",
            admin_socket=tmp_path / "run" / "admin.sock",
        ),
    )


@pytest.fixture
def debug_viewer():
    port = _free_port()
    viewer = LiveDebugViewer(
        DebugConfig(enabled=True, host="127.0.0.1", port=port, fps=5.0, jpeg_quality=70)
    )
    viewer.start()
    try:
        if not viewer.is_running:
            pytest.skip("debug HTTP server failed to bind")
        yield viewer
    finally:
        viewer.stop()


class TestDebugDisabled:
    def test_orchestrator_does_not_start_viewer(self, app_config):
        assert app_config.debug.enabled is False
        orch = ApplicationOrchestrator(app_config, use_mock_camera=True)
        assert orch._debug_viewer is None


class TestLiveFrameBuffer:
    def test_latest_frame_not_a_queue(self):
        buf = LiveFrameBuffer()
        buf.publish(b"one")
        buf.publish(b"two")
        buf.publish(b"three")
        seq, payload = buf.latest()
        assert payload == b"three"
        assert seq == 3

    def test_slow_consumer_gets_latest(self):
        buf = LiveFrameBuffer()
        buf.publish(b"a")
        buf.publish(b"b")
        seq, payload = buf.wait_latest(0, timeout=0.2)
        assert payload == b"b"
        assert seq == 2

    def test_thread_safe_publish_and_read(self):
        buf = LiveFrameBuffer()
        errors: list[BaseException] = []

        def writer() -> None:
            try:
                for i in range(50):
                    buf.publish(i)
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        def reader() -> None:
            try:
                last = -1
                for _ in range(50):
                    last, _payload = buf.wait_latest(last, timeout=0.2)
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3.0)
        assert not errors


class TestOverlayDoesNotMutate:
    def test_original_frame_unchanged(self):
        frame = np.full((240, 320, 3), 40, dtype=np.uint8)
        frame[10, 10] = (1, 2, 3)
        original = frame.copy()
        snap = DebugSnapshot(
            state="IDLE",
            document_detected=True,
            motion=12.5,
            edge_density=0.04,
            motion_threshold=8.0,
            edge_density_threshold=0.02,
            stable_count=3,
            stable_required=8,
            camera_healthy=True,
        )
        drawn = draw_overlay(frame, snap, fps=5.0)
        assert np.array_equal(frame, original)
        assert drawn is not frame
        assert not np.array_equal(drawn, original)

    def test_publish_copies_frame(self, debug_viewer):
        frame = np.full((120, 160, 3), 90, dtype=np.uint8)
        original = frame.copy()
        debug_viewer.publish(
            frame,
            state="IDLE",
            document_detected=False,
            motion=0.0,
            edge_density=0.0,
            motion_threshold=8.0,
            edge_density_threshold=0.02,
            stable_count=0,
            stable_required=8,
            stable=False,
            camera_healthy=True,
            autofocus_available=True,
        )
        frame[:] = 0
        stored = debug_viewer.latest_frame_copy()
        assert stored is not None
        assert np.array_equal(stored, original)
        assert not np.array_equal(stored, frame)


class TestOverlayResizeUsesPreviewCoordinates:
    """Regression: document_roi is preview pixels; MJPEG may shrink 1920→1280."""

    _SRC_W, _SRC_H = 1920, 1080
    _MAX_W = 1280
    _ROI = (400, 200, 1120, 680)

    def _snapshot(self) -> DebugSnapshot:
        x, y, w, h = self._ROI
        return DebugSnapshot(
            state="IDLE",
            document_detected=True,
            camera_healthy=True,
            roi_center_fraction=0.7,
            document_roi=f"{x},{y},{w},{h}",
            frame_width=self._SRC_W,
            frame_height=self._SRC_H,
        )

    def _window_mean(self, image: np.ndarray, x: int, y: int, rad: int = 2) -> np.ndarray:
        h, w = image.shape[:2]
        x0, x1 = max(0, x - rad), min(w, x + rad + 1)
        y0, y1 = max(0, y - rad), min(h, y + rad + 1)
        return image[y0:y1, x0:x1].mean(axis=(0, 1))

    def test_compose_scales_document_roi_and_does_not_use_raw_1920_coords(self):
        frame = np.full((self._SRC_H, self._SRC_W, 3), 18, dtype=np.uint8)
        original = frame.copy()
        snap = self._snapshot()
        scale = self._MAX_W / float(self._SRC_W)
        expected = tuple(int(round(v * scale)) for v in self._ROI)
        # Top-left of the ROI sits under the HUD panel. Use the bottom edge instead.
        src_bl = (self._ROI[0], self._ROI[1] + self._ROI[3])
        scaled_bl = (int(round(src_bl[0] * scale)), int(round(src_bl[1] * scale)))
        # Unscaled 1920-space point that remains inside a 1280x720 canvas.
        raw_on_720 = (self._ROI[0], min(self._ROI[1] + self._ROI[3] - 40, 700))

        composed = compose_debug_frame(frame, snap, fps=5.0, max_width=self._MAX_W, overlay=True)

        assert np.array_equal(frame, original)
        assert composed is not frame
        assert composed.shape[1] == self._MAX_W
        assert composed.shape[0] == int(round(self._SRC_H * scale))
        assert expected == (267, 133, 747, 453)
        assert scaled_bl == (267, 587)

        at_scaled = self._window_mean(composed, scaled_bl[0], scaled_bl[1])
        at_raw = self._window_mean(composed, raw_on_720[0], raw_on_720[1])
        # RGB document stroke is (80, 180, 255); INTER_AREA mixes a 2px line.
        assert at_scaled[2] > 70
        assert at_scaled[2] > at_raw[2] + 30
        assert at_raw[2] < 50

    def test_resize_then_draw_is_the_bug_this_fixes(self):
        """Shrinking first places the 1920 ROI at raw (400,y) on the 1280 image."""
        frame = np.full((self._SRC_H, self._SRC_W, 3), 18, dtype=np.uint8)
        snap = self._snapshot()
        scale = self._MAX_W / float(self._SRC_W)
        scaled_bl = (
            int(round(self._ROI[0] * scale)),
            int(round((self._ROI[1] + self._ROI[3]) * scale)),
        )
        raw_on_720 = (self._ROI[0], 700)
        broken = draw_overlay(resize_debug_frame(frame, self._MAX_W), snap, 5.0)
        at_raw = self._window_mean(broken, raw_on_720[0], raw_on_720[1])
        at_scaled = self._window_mean(broken, scaled_bl[0], scaled_bl[1])
        assert at_raw[2] > 90
        assert at_raw[2] > at_scaled[2] + 40


    def test_compose_keeps_full_1920_frame_when_max_width_is_1920(self):
        frame = np.full((self._SRC_H, self._SRC_W, 3), 18, dtype=np.uint8)
        snap = self._snapshot()
        composed = compose_debug_frame(frame, snap, fps=5.0, max_width=1920, overlay=True)
        assert composed.shape[1] == 1920
        assert composed.shape[0] == 1080
        at_bl = self._window_mean(composed, self._ROI[0], self._ROI[1] + self._ROI[3])
        assert at_bl[2] > 70


class TestDebugHttp:
    def test_status_json_and_no_secrets(self, debug_viewer):
        debug_viewer.publish(
            np.full((80, 120, 3), 30, dtype=np.uint8),
            state="IDLE",
            document_detected=False,
            motion=1.25,
            edge_density=0.01,
            motion_threshold=8.0,
            edge_density_threshold=0.02,
            stable_count=0,
            stable_required=8,
            stable=False,
            camera_healthy=True,
            autofocus_available=True,
        )
        url = f"http://127.0.0.1:{debug_viewer.config.port}/preview/status"
        with urllib.request.urlopen(url, timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        dumped = json.dumps(payload)
        assert "IDCAM_AUTH_TOKEN" not in dumped
        assert "Bearer" not in dumped
        assert payload["enabled"] is True
        assert payload["state"] == "IDLE"
        assert payload["document_detected"] is False
        assert payload["camera_healthy"] is True
        assert payload["autofocus_available"] is True
        assert "motion" in payload
        assert "edge_density" in payload
        assert "stable_count" in payload
        assert "reject_reason" in payload
        assert payload["frame"]["channel_order"] == "RGB"
        assert payload["frame"]["shape"] == [80, 120, 3]
        assert "mjpeg_fps" in payload
        assert "detect_hz" in payload
        assert "tick_ms" in payload

    def test_html_page(self, debug_viewer):
        url = f"http://127.0.0.1:{debug_viewer.config.port}/preview/live"
        with urllib.request.urlopen(url, timeout=3) as resp:
            body = resp.read().decode("utf-8")
        assert "idCam V3 — Live Debug Viewer" in body
        assert "/preview/live.mjpg" in body
        assert "Capture &amp; Send" in body
        assert "IDCAM_AUTH_TOKEN" not in body
        assert "Bearer" not in body

    def test_mjpeg_boundary_and_jpeg(self, debug_viewer):
        frame = np.full((96, 128, 3), 180, dtype=np.uint8)
        debug_viewer.publish(
            frame,
            state="IDLE",
            document_detected=False,
            motion=0.0,
            edge_density=0.0,
            motion_threshold=8.0,
            edge_density_threshold=0.02,
            stable_count=0,
            stable_required=8,
            stable=False,
            camera_healthy=True,
            autofocus_available=True,
        )
        _wait_jpeg(debug_viewer)
        url = f"http://127.0.0.1:{debug_viewer.config.port}/preview/live.mjpg"
        resp = urllib.request.urlopen(url, timeout=5)
        try:
            data = _read_mjpeg_part(resp)
        finally:
            resp.close()
        assert b"--frame" in data
        assert b"Content-Type: image/jpeg" in data
        assert b"\xff\xd8" in data
        assert b"\xff\xd9" in data
        debug_viewer.publish(
            frame,
            state="DOCUMENT_DETECTED",
            document_detected=True,
            motion=20.0,
            edge_density=0.1,
            motion_threshold=8.0,
            edge_density_threshold=0.02,
            stable_count=1,
            stable_required=8,
            stable=False,
            camera_healthy=True,
            autofocus_available=True,
        )
        _wait_jpeg(debug_viewer)
        assert debug_viewer.is_running

    def test_multiple_clients_same_source(self, debug_viewer):
        frame = np.full((64, 80, 3), 100, dtype=np.uint8)
        debug_viewer.publish(
            frame,
            state="IDLE",
            document_detected=False,
            motion=0.0,
            edge_density=0.0,
            motion_threshold=8.0,
            edge_density_threshold=0.02,
            stable_count=0,
            stable_required=8,
            stable=False,
            camera_healthy=True,
            autofocus_available=True,
        )
        _wait_jpeg(debug_viewer)
        url = f"http://127.0.0.1:{debug_viewer.config.port}/preview/live.mjpg"
        results: list[bytes] = []
        errors: list[BaseException] = []

        def consume() -> None:
            try:
                resp = urllib.request.urlopen(url, timeout=5)
                try:
                    results.append(_read_mjpeg_part(resp))
                finally:
                    resp.close()
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=consume), threading.Thread(target=consume)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=6.0)
        assert not errors
        assert len(results) == 2
        for part in results:
            assert b"--frame" in part
            assert b"\xff\xd8" in part

    def test_waiting_placeholder_encodes(self):
        image = make_placeholder("WAITING FOR CAMERA...")
        jpeg = encode_jpeg(image, 70)
        assert jpeg[:2] == b"\xff\xd8"
        assert jpeg[-2:] == b"\xff\xd9"

    def test_encode_jpeg_keeps_rgb_red(self):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:] = (220, 15, 15)  # RGB red
        jpeg = encode_jpeg(frame, 95)
        decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        b_mean, g_mean, r_mean = decoded.mean(axis=(0, 1))
        assert r_mean > b_mean
        assert r_mean > g_mean
        assert r_mean > 150

    def test_capture_post_unavailable_without_handler(self, debug_viewer):
        url = f"http://127.0.0.1:{debug_viewer.config.port}/preview/capture"
        req = urllib.request.Request(url, data=b"", method="POST")
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(req, timeout=3)
        assert excinfo.value.code == 503

    def test_capture_post_busy(self, debug_viewer):
        debug_viewer.set_capture_handler(
            lambda: {"ok": False, "status": "busy", "state": "FOCUSING", "message": "Busy (FOCUSING)"}
        )
        url = f"http://127.0.0.1:{debug_viewer.config.port}/preview/capture"
        req = urllib.request.Request(url, data=b"", method="POST")
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(req, timeout=3)
        assert excinfo.value.code == 409
        payload = json.loads(excinfo.value.read().decode("utf-8"))
        assert payload["status"] == "busy"
        assert "Bearer" not in json.dumps(payload)

    def test_capture_post_success_strips_unknown_fields(self, debug_viewer):
        debug_viewer.set_capture_handler(
            lambda: {
                "ok": True,
                "status": "success",
                "capture_id": "cap-1",
                "message": "Upload accepted",
                "token": "SECRET",
                "Authorization": "Bearer SECRET",
            }
        )
        url = f"http://127.0.0.1:{debug_viewer.config.port}/preview/capture"
        req = urllib.request.Request(url, data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload["ok"] is True
        assert payload["capture_id"] == "cap-1"
        dumped = json.dumps(payload)
        assert "SECRET" not in dumped
        assert "Bearer" not in dumped
        assert "token" not in payload

    def test_html_includes_manual_focus_controls(self, debug_viewer):
        url = f"http://127.0.0.1:{debug_viewer.config.port}/preview/live"
        with urllib.request.urlopen(url, timeout=3) as resp:
            html = resp.read().decode("utf-8")
        assert "SET MANUAL" in html
        assert "SCAN FOCUS" in html
        assert "CAPTURE FOCUS TEST" in html
        assert "/preview/focus" in html

    def test_focus_post_unavailable_without_handler(self, debug_viewer):
        url = f"http://127.0.0.1:{debug_viewer.config.port}/preview/focus"
        req = urllib.request.Request(
            url, data=b'{"action":"manual"}', method="POST", headers={"Content-Type": "application/json"}
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(req, timeout=3)
        assert excinfo.value.code == 503

    def test_focus_post_nudge_and_clamp(self, debug_viewer, tmp_path):
        from idcam.camera.manager import CameraManager
        from idcam.config.schema import CameraConfig
        from idcam.preview.manual_focus import ManualFocusCalibrator

        cam = CameraManager(
            CameraConfig(preview_width=320, preview_height=240, capture_width=640, capture_height=480),
            use_mock=True,
        )
        cam.initialize()
        frame = np.full((240, 320, 3), 190, dtype=np.uint8)
        cal = ManualFocusCalibrator(
            camera=cam,
            get_preview=lambda: frame,
            get_document=lambda: (False, None),
            still_dir=tmp_path / "focus_debug",
        )
        debug_viewer.set_focus_handler(cal.handle)
        url = f"http://127.0.0.1:{debug_viewer.config.port}/preview/focus"
        req = urllib.request.Request(
            url,
            data=b'{"action":"set","lens":7.8}',
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload["ok"] is True
        assert abs(float(payload["requested_lens"]) - 7.8) < 1e-6
        req2 = urllib.request.Request(
            url,
            data=b'{"action":"set","lens":50}',
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req2, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert float(payload["requested_lens"]) == 10.0
        assert float(payload["actual_lens"]) == 10.0


class TestDetectorFrameIsolation:
    def test_overlay_does_not_change_detector_result(self):
        config = DetectionConfig(
            motion_threshold=5.0,
            edge_density_threshold=0.01,
            stable_frames_required=3,
        )
        detector = DocumentDetector(config)
        baseline = np.full((240, 320, 3), 200, dtype=np.uint8)
        detector.update_baseline(baseline)
        frame = baseline.copy()
        frame[40:200, 40:280] = 40
        before = detector.detect(frame.copy())
        snap = DebugSnapshot(state="IDLE", document_detected=before, camera_healthy=True)
        draw_overlay(frame, snap, 5.0)
        after = detector.detect(frame)
        assert before is after
        stab = StabilityDetector(config)
        moving = frame.copy()
        assert stab.update(frame) is False
        draw_overlay(moving, snap, 5.0)
        assert stab.update(moving) in {True, False}
