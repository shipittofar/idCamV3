"""DEBUG-ONLY manual focus calibration helpers and camera mock behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from idcam.camera.manager import CameraManager
from idcam.config.schema import CameraConfig
from idcam.preview.manual_focus import (
    ManualFocusCalibrator,
    clamp_lens,
    interior_document_roi,
    lens_is_settled,
    parse_lens_limits,
    pick_best_scan,
    select_focus_roi,
)


def _tiny_camera(tmp_path: Path) -> CameraManager:
    cfg = CameraConfig(
        preview_width=320,
        preview_height=240,
        capture_width=640,
        capture_height=480,
    )
    mgr = CameraManager(cfg, use_mock=True)
    mgr.initialize()
    return mgr


def test_parse_and_clamp_lens_limits():
    lo, hi = parse_lens_limits({"LensPosition": (0.0, 10.0, 1.0)})
    assert lo == 0.0
    assert hi == 10.0
    assert clamp_lens(-1.0, lo, hi) == 0.0
    assert clamp_lens(99.0, lo, hi) == 10.0
    assert clamp_lens(7.8, lo, hi) == 7.8


def test_select_focus_roi_document_vs_center():
    shape = (1080, 1920, 3)
    roi, source = select_focus_roi(shape, (100, 80, 800, 500), document_present=True)
    assert source == "document_geometry"
    assert roi[0] > 100
    assert roi[1] > 80
    assert roi[2] < 800
    inner = interior_document_roi((100, 80, 800, 500), shape)
    assert inner == roi
    center, csrc = select_focus_roi(shape, None, document_present=False)
    assert csrc == "center_fallback"
    assert center[2] > 1000


def test_scan_picks_maximum_sharpness():
    rows = [
        {"lens_position": 7.6, "focus_fom": 300, "sharpness": 180.0},
        {"lens_position": 8.0, "focus_fom": 330, "sharpness": 203.1},
        {"lens_position": 8.2, "focus_fom": 318, "sharpness": 197.2},
    ]
    best = pick_best_scan(rows)
    assert best is not None
    assert best["lens_position"] == 8.0
    assert best["sharpness"] == 203.1


def test_lens_settled_requires_stable_window():
    assert lens_is_settled([7.5, 7.71, 7.79]) is False
    assert lens_is_settled([7.80, 7.80, 7.80]) is True
    assert lens_is_settled([7.80, 7.82, 7.80]) is True


def test_manual_mode_nudge_clamp_and_readback(tmp_path):
    cam = _tiny_camera(tmp_path)
    frame = np.full((240, 320, 3), 180, dtype=np.uint8)
    cal = ManualFocusCalibrator(
        camera=cam,
        get_preview=lambda: frame,
        get_document=lambda: (False, None),
        still_dir=tmp_path / "focus_debug",
    )
    out = cal.handle("manual")
    assert out["ok"] is True
    assert cal.mode == "manual"
    assert cam.capture_metadata()["AfMode"] == 0
    cal.handle("set", {"lens": 7.8})
    actual = cam.capture_metadata()["LensPosition"]
    assert abs(actual - 7.8) < 1e-6
    cal.handle("nudge", {"delta": 0.1})
    assert abs(cal.requested_lens - 7.9) < 1e-6
    cal.handle("set", {"lens": 99.0})
    assert cal.requested_lens == 10.0
    cal.handle("set", {"lens": -5.0})
    assert cal.requested_lens == 0.0


def test_scan_and_capture_test(tmp_path):
    cam = _tiny_camera(tmp_path)
    frame = np.full((240, 320, 3), 200, dtype=np.uint8)
    frame[60:180, 80:240] = 40
    cal = ManualFocusCalibrator(
        camera=cam,
        get_preview=lambda: frame,
        get_document=lambda: (True, (80, 60, 160, 120)),
        still_dir=tmp_path / "focus_debug",
    )
    scanned = cal.handle("scan", {"step": 5.0})
    assert scanned["ok"] is True
    assert cal.last_scan is not None
    assert cal.last_scan["best_lens"] is not None
    captured = cal.handle("capture_test")
    assert captured["ok"] is True
    assert cal.last_capture is not None
    assert Path(cal.last_capture["path"]).is_file()
    assert cal.last_capture["image_width"] == 640
    assert cal.last_capture["preview_sharpness"] is not None
    assert cal.last_capture["still_sharpness"] is not None


def test_debug_disabled_does_not_attach_calibrator(tmp_path):
    from idcam.app.orchestrator import ApplicationOrchestrator
    from idcam.config.schema import AppConfig, ServerConfig, UIConfig

    cfg = AppConfig(
        device_id="test-device",
        temp_dir=tmp_path / "tmp",
        server=ServerConfig(url="https://api.example.com"),
        ui=UIConfig(
            status_file=tmp_path / "run" / "status.json",
            admin_socket=tmp_path / "run" / "admin.sock",
        ),
    )
    assert cfg.debug.enabled is False
    orch = ApplicationOrchestrator(cfg, use_mock_camera=True)
    assert orch._debug_viewer is None
    assert orch._manual_focus is None
