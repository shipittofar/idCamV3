#!/usr/bin/env python3
"""Prove where an opened-passport preview is rejected.

Uses the same find_document_quad / DocumentDetector.detect_geometry path as production.
Does not start the state machine, AF, still capture, or upload.

On the Raspberry Pi (stop the service first so the camera is free):

    sudo systemctl stop idcam
    sudo -u idcam IDCAM_GEOMETRY_DEBUG=1 \\
        /opt/idcam/venv/bin/python /opt/idcam/scripts/passport_detection_debug.py \\
        --save-dir /tmp --frames 5

systemd PrivateTmp=true hides /tmp for the service itself. This script must run
outside systemd so /tmp/idcam_passport_debug.jpg is visible.

Inspect which code the service would run:

    /opt/idcam/venv/bin/python -c "import idcam.vision.geometry as g; print(g.__file__, g.MAX_ASPECT)"
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from idcam import __version__  # noqa: E402
from idcam.camera.manager import PICAMERA2_AVAILABLE, CameraManager  # noqa: E402
from idcam.config.schema import CameraConfig, DetectionConfig  # noqa: E402
from idcam.vision.document_detector import DocumentDetector  # noqa: E402
from idcam.vision.geometry import find_document_quad  # noqa: E402
from idcam.vision.geometry_debug import (  # noqa: E402
    classify_failure,
    diagnose_frame,
    effective_geometry_config,
    render_overlay,
    save_rgb_jpeg,
)


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=8)
        return out.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"(unavailable: {exc})"


def runtime_code_report() -> dict:
    import idcam.vision.geometry as g

    src = inspect.getsource(g)
    return {
        "idcam_version": __version__,
        "python_executable": sys.executable,
        "geometry_module": getattr(g, "__file__", None),
        "MAX_ASPECT": g.MAX_ASPECT,
        "MIN_AREA_FRACTION": g.MIN_AREA_FRACTION,
        "MAX_AREA_FRACTION": g.MAX_AREA_FRACTION,
        "source_has_merged_two_page": "merged_two_page" in src,
        "source_has_suppress_inner": "_suppress_inner_quads" in src,
        "source_has_large_occupancy_lobe": "0.72" in src and "_area_term" in src,
        "systemd_show": _run(["systemctl", "show", "idcam", "-p", "FragmentPath", "-p", "ExecStart", "-p", "MainPID", "-p", "ActiveState"]),
        "idcam_which": _run(["sh", "-c", "command -v idcam || true"]),
    }


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def grab_preview(frames: int) -> np.ndarray:
    if not PICAMERA2_AVAILABLE:
        raise RuntimeError("Picamera2 is not available; pass --image PATH")
    cam = CameraManager(CameraConfig(), use_mock=False)
    cam.initialize()
    try:
        last = None
        for _ in range(max(1, frames)):
            last = cam.capture_preview_array()
        if last is None:
            raise RuntimeError("No preview frame")
        return np.ascontiguousarray(last.copy())
    finally:
        cam.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description="Passport detection runtime debug")
    parser.add_argument("--image", type=Path, default=None, help="RGB/BGR still instead of live preview")
    parser.add_argument("--frames", type=int, default=3, help="Live preview grabs before saving the last")
    parser.add_argument("--save-dir", type=Path, default=Path("/tmp"))
    args = parser.parse_args()

    code = runtime_code_report()
    cfg = effective_geometry_config()
    det_cfg = DetectionConfig()
    detection_runtime = {
        "geometry_hold_frames": det_cfg.geometry_hold_frames,
        "geometry_smooth_alpha": det_cfg.geometry_smooth_alpha,
        "geometry_stability_threshold_px": det_cfg.geometry_stability_threshold_px,
        "geometry_stable_frames_required": det_cfg.geometry_stable_frames_required,
        "removal_frames_required": det_cfg.removal_frames_required,
        "note": "DetectionConfig defaults unless the process loaded /etc/idcam/config.yaml",
    }
    print("=== RUNTIME CODE VERIFICATION ===")
    print(json.dumps(code, indent=2))
    print("=== EFFECTIVE GEOMETRY CONSTANTS (imported module) ===")
    print(json.dumps(cfg, indent=2))
    print("=== EFFECTIVE DETECTION CONFIG (schema defaults) ===")
    print(json.dumps(detection_runtime, indent=2))

    if args.image:
        frame = load_rgb(args.image)
        source = str(args.image)
    else:
        frame = grab_preview(args.frames)
        source = "live_preview"

    raw_path = args.save_dir / "idcam_passport_debug.jpg"
    overlay_path = args.save_dir / "idcam_passport_debug_overlay.jpg"
    args.save_dir.mkdir(parents=True, exist_ok=True)
    save_rgb_jpeg(raw_path, frame)

    best, n = find_document_quad(frame)
    trace = diagnose_frame(frame)
    detector = DocumentDetector(DetectionConfig())
    geom = detector.detect_geometry(frame)
    case = classify_failure(trace, detector_present=geom.present)
    save_rgb_jpeg(overlay_path, render_overlay(frame, trace))

    print("=== FRAME ===")
    print(json.dumps({"source": source, "shape": list(frame.shape), "saved": str(raw_path), "overlay": str(overlay_path)}, indent=2))
    print("=== FINDER ===")
    print(
        json.dumps(
            {
                "candidate_count_returned": n,
                "trace_candidates": len(trace.candidates),
                "accepted": sum(1 for c in trace.candidates if c.get("accepted")),
                "merge_attempts": len(trace.merges),
                "merge_ok": sum(1 for m in trace.merges if not m.get("merge_rejection_reason")),
                "best_mode": None if best is None else best.geometry_mode,
                "best_bbox": None if best is None else best.bbox,
                "detector_present": geom.present,
                "detector_reason": geom.reason,
                "detector_mode": geom.geometry_mode,
                "case": case,
            },
            indent=2,
        )
    )
    print("=== CANDIDATES ===")
    print(json.dumps(trace.candidates, indent=2, default=str))
    print("=== MERGES ===")
    print(json.dumps(trace.merges, indent=2, default=str))
    print("=== CLASSIFICATION ===")
    print(case)
    print("ROOT CAUSE IDENTIFIED — FIX NOT YET IMPLEMENTED" if case in {"A", "B", "C", "D", "E", "F"} else "CASE G (includes success or unclassified). FIX NOT APPLIED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
