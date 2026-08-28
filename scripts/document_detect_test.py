#!/usr/bin/env python3
"""Isolated document-geometry diagnostic.

Acquires preview frames (or loads an image) and runs DocumentDetector.detect_geometry
only. It does not start the state machine, autofocus, 64MP capture, or upload.

Usage on the Raspberry Pi:

    sudo systemctl stop idcam
    sudo -u idcam /opt/idcam/venv/bin/python scripts/document_detect_test.py \\
        --frames 80 --output /tmp/idcam-document-test --save-overlay --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from idcam.camera.manager import PICAMERA2_AVAILABLE, CameraManager  # noqa: E402
from idcam.config.schema import CameraConfig, DetectionConfig  # noqa: E402
from idcam.vision.document_detector import DocumentDetector, DocumentGeometryResult  # noqa: E402


def configure_logging(verbose: bool) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("         log | %(name)s | %(message)s"))
    root = logging.getLogger("idcam")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.propagate = False


def load_rgb_image(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def annotate(frame: np.ndarray, geom: DocumentGeometryResult) -> np.ndarray:
    """Draw one document quadrilateral. Search-ROI rectangles are not the document."""
    vis = frame.copy()
    if geom.corners is not None:
        pts = np.asarray(geom.corners, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, (40, 220, 80), 3, cv2.LINE_AA)
        for i, (x, y) in enumerate(geom.corners):
            cv2.circle(vis, (int(x), int(y)), 6, (220, 40, 40), -1, cv2.LINE_AA)
            cv2.putText(
                vis, ("TL", "TR", "BR", "BL")[i], (int(x) + 8, int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )
    if geom.bbox is not None:
        x, y, w, h = geom.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (80, 180, 255), 1)
    status = "PRESENT" if geom.present else "ABSENT"
    lines = [
        f"{status}  conf={geom.confidence:.2f}  {geom.reason}",
        f"candidates={geom.candidate_count}  area={geom.selected_area_ratio:.3f}  "
        f"aspect={geom.aspect_ratio:.2f}  rect={geom.rectangularity:.2f}",
    ]
    y = 32
    for line in lines:
        cv2.putText(vis, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(vis, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        y += 32
    return vis


def geom_dict(geom: DocumentGeometryResult) -> dict[str, Any]:
    return {
        "present": geom.present,
        "confidence": round(geom.confidence, 4),
        "corners": [list(p) for p in geom.corners] if geom.corners else None,
        "bbox": list(geom.bbox) if geom.bbox else None,
        "reason": geom.reason,
        "candidate_count": geom.candidate_count,
        "selected_area_ratio": round(geom.selected_area_ratio, 4),
        "aspect_ratio": round(geom.aspect_ratio, 4),
        "rectangularity": round(geom.rectangularity, 4),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="idCam document-geometry diagnostic (no state machine).")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("./idcam-document-test"))
    parser.add_argument("--save-overlay", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--input", type=Path, help="offline RGB/BGR image instead of the camera")
    parser.add_argument("--preview-width", type=int, default=1920)
    parser.add_argument("--preview-height", type=int, default=1080)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    args.output.mkdir(parents=True, exist_ok=True)
    detector = DocumentDetector(DetectionConfig())
    records: list[dict[str, Any]] = []

    if args.input is not None:
        frame = load_rgb_image(args.input)
        geom = detector.detect_geometry(frame)
        records.append({"source": str(args.input), **geom_dict(geom)})
        if args.save_overlay:
            rgb = annotate(frame, geom)
            out = args.output / f"{args.input.stem}_overlay.png"
            cv2.imwrite(str(out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            records[-1]["overlay"] = str(out)
        print(json.dumps(records[0], indent=2) if args.json else records[0])
        _write_json(args, {"ok": True, "frames": records, "detect_hz": 0})
        return 0

    camera = CameraManager(
        CameraConfig(preview_width=args.preview_width, preview_height=args.preview_height),
        use_mock=args.mock,
    )
    if camera.is_mock and not args.mock and not PICAMERA2_AVAILABLE:
        print("FAIL: Picamera2 not available (pass --mock for CI)", file=sys.stderr)
        return 1

    camera.initialize()
    t0 = time.monotonic()
    try:
        for i in range(max(1, args.frames)):
            frame = camera.capture_preview_array()
            geom = detector.detect_geometry(frame)
            rec = {"index": i, **geom_dict(geom), "shape": list(frame.shape)}
            if args.save_overlay and (i == 0 or i == args.frames - 1 or geom.present):
                rgb = annotate(frame, geom)
                out = args.output / f"frame_{i:04d}.png"
                cv2.imwrite(str(out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                rec["overlay"] = str(out)
            records.append(rec)
            print(
                f"[{i:03d}] present={geom.present} conf={geom.confidence:.2f} "
                f"reason={geom.reason} bbox={geom.bbox}"
            )
    finally:
        camera.shutdown()

    elapsed = max(1e-6, time.monotonic() - t0)
    detect_hz = len(records) / elapsed
    present_n = sum(1 for r in records if r["present"])
    summary = {
        "ok": True,
        "frames": len(records),
        "present_frames": present_n,
        "detect_hz": round(detect_hz, 2),
        "note": "detect_hz is detector+preview sample rate, not camera hardware FPS",
        "results": records if args.json else None,
    }
    print(f"RESULT: {present_n}/{len(records)} frames present  detect_hz={detect_hz:.2f}")
    _write_json(args, summary if args.json else {k: v for k, v in summary.items() if k != "results"})
    return 0


def _write_json(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"summary written to {args.json_out}")


if __name__ == "__main__":
    raise SystemExit(main())
