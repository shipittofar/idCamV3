#!/usr/bin/env python3
"""Standalone camera diagnostic for idCam.

Exercises only the camera layer: acquire -> preview -> full-resolution JPEG ->
release. It does not import or run the orchestrator, state machine, document
detector, autofocus controller or uploader, so a failure here is unambiguously
a camera-layer failure.

The capture step deliberately goes through the same TempFileStore partial-path
and finalize sequence that production uses, because the historic capture bug
lived in the *filename*, not in the camera. A diagnostic that writes straight
to "out.jpg" would pass while production still failed.

Usage on the Raspberry Pi:

    sudo -u idcam /opt/idcam/venv/bin/python \
        /opt/idcam/scripts/camera_capture_test.py --output /var/lib/idcam/tmp
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import numpy as np  # noqa: E402

from idcam.camera.manager import (  # noqa: E402
    PICAMERA2_AVAILABLE,
    CameraManager,
    describe_preview_frame,
)
from idcam.config.schema import CameraConfig  # noqa: E402
from idcam.storage.temp_files import TempFileStore  # noqa: E402

CHANNEL_NAMES = ("R", "G", "B")


@dataclass
class StepResult:
    name: str
    passed: bool
    message: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        head = f"[{status}] {self.name}"
        if self.message:
            head += f": {self.message}"
        lines = [head]
        for key, value in self.detail.items():
            lines.append(f"         {key} = {value}")
        return "\n".join(lines)


class Report:
    def __init__(self) -> None:
        self.steps: list[StepResult] = []

    def add(self, result: StepResult) -> StepResult:
        self.steps.append(result)
        print(result.render(), flush=True)
        return result

    @property
    def ok(self) -> bool:
        return all(step.passed for step in self.steps)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "steps": [
                {"name": s.name, "passed": s.passed, "message": s.message, "detail": s.detail}
                for s in self.steps
            ],
        }


def configure_logging(verbose: bool) -> None:
    """Send idcam logs to stdout so camera-layer events are visible inline."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("         log | %(name)s | %(message)s"))
    root = logging.getLogger("idcam")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.propagate = False


def step_initialize(report: Report, config: CameraConfig, use_mock: bool) -> CameraManager | None:
    manager = CameraManager(config, use_mock=use_mock)
    detail = {
        "picamera2_available": PICAMERA2_AVAILABLE,
        "backend": "mock" if manager.is_mock else "picamera2",
    }
    if manager.is_mock and not use_mock:
        report.add(
            StepResult(
                "initialize camera",
                False,
                "Picamera2 is not importable, so this run cannot validate hardware",
                detail,
            )
        )
        return None
    try:
        manager.initialize()
    except Exception as exc:
        detail["error"] = f"{type(exc).__name__}: {exc}"
        report.add(StepResult("initialize camera", False, "camera could not be acquired", detail))
        return None
    detail["is_started"] = manager.is_started
    report.add(StepResult("initialize camera", manager.is_started, "camera acquired", detail))
    return manager


def step_stream_geometry(report: Report, manager: CameraManager, config: CameraConfig) -> None:
    geometry = manager.stream_geometry()
    preview = tuple(geometry.get("preview") or ())
    still = tuple(geometry.get("still") or ())
    requested_preview = (config.preview_width, config.preview_height)
    requested_still = (config.capture_width, config.capture_height)
    source = geometry.get("source")
    detail = {
        "source": source,
        "requested_preview": requested_preview,
        "configured_preview": preview,
        "preview_format": geometry.get("preview_format"),
        "preview_stride": geometry.get("preview_stride"),
        "requested_still": requested_still,
        "configured_still": still,
        "still_format": geometry.get("still_format"),
    }
    # libcamera is allowed to align sizes; that is reported, not failed. Only a
    # missing configuration is a failure.
    passed = bool(preview) and bool(still)
    if source == "requested":
        message = "could not read back configured geometry - values below are the REQUEST, not confirmation"
    elif preview != requested_preview or still != requested_still:
        message = "streams configured (libcamera aligned the requested sizes - see values)"
    else:
        message = "streams configured as requested"
    report.add(StepResult("stream geometry", passed, message, detail))


def step_preview_frame(report: Report, manager: CameraManager) -> np.ndarray | None:
    try:
        frame = manager.capture_preview_array()
    except Exception as exc:
        report.add(
            StepResult(
                "acquire preview frame",
                False,
                "preview frame could not be read",
                {"error": f"{type(exc).__name__}: {exc}"},
            )
        )
        return None

    info = describe_preview_frame(frame)
    detail: dict[str, Any] = {
        "shape (h, w, c)": info.get("shape"),
        "dtype": info.get("dtype"),
    }
    means = info.get("channel_means") or {}
    if len(means) == 3:
        detail["channel_means (R, G, B)"] = {
            name: means[key] for name, key in zip(CHANNEL_NAMES, ("c0", "c1", "c2"))
        }
    detail["min"] = int(frame.min())
    detail["max"] = int(frame.max())
    detail["std"] = round(float(frame.std()), 2)

    problems = []
    if frame.ndim != 3 or frame.shape[-1] != 3:
        problems.append("frame is not 3-channel")
    if frame.dtype != np.uint8:
        problems.append(f"dtype is {frame.dtype}, expected uint8")
    if float(frame.std()) < 1.0:
        problems.append("frame is uniform (lens cap on, or stream not producing data)")

    passed = not problems
    report.add(
        StepResult(
            "acquire preview frame",
            passed,
            "; ".join(problems) if problems else "preview frame is valid",
            detail,
        )
    )
    return frame if passed else None


def step_preview_fps(report: Report, manager: CameraManager, frames: int) -> None:
    timestamps: list[float] = []
    start = time.monotonic()
    try:
        for _ in range(frames):
            manager.capture_preview_array()
            timestamps.append(time.monotonic())
    except Exception as exc:
        report.add(
            StepResult(
                "preview fps",
                False,
                "preview stopped updating during measurement",
                {"frames_read": len(timestamps), "error": f"{type(exc).__name__}: {exc}"},
            )
        )
        return

    elapsed = timestamps[-1] - start if timestamps else 0.0
    fps = len(timestamps) / elapsed if elapsed > 0 else 0.0
    gaps = np.diff(np.array(timestamps)) if len(timestamps) > 1 else np.array([0.0])
    detail = {
        "frames": len(timestamps),
        "elapsed_sec": round(elapsed, 3),
        "fps": round(fps, 2),
        "frame_interval_ms_mean": round(float(gaps.mean()) * 1000, 2),
        "frame_interval_ms_max": round(float(gaps.max()) * 1000, 2),
    }
    passed = fps > 0
    report.add(
        StepResult(
            "preview fps",
            passed,
            f"measured {fps:.2f} fps" if passed else "no frames delivered",
            detail,
        )
    )


def step_rgb_order(
    report: Report, frame: np.ndarray, expect_color: str | None, save_preview: Path | None
) -> None:
    """Report channel ordering as delivered to consumers.

    ``capture_preview_array`` returns RGB. Picamera2's RGB888 buffer is B,G,R in
    memory, so the byte order in the buffer is the reverse of what is reported
    here. Channel statistics alone cannot prove correctness on an arbitrary
    scene, so this step is informational unless --expect-color is given.
    """
    means = frame.reshape(-1, 3).mean(axis=0)
    detail: dict[str, Any] = {
        "as delivered to detector/viewer (R, G, B)": [round(float(v), 2) for v in means],
        "buffer byte order (picamera2 RGB888 = B, G, R)": [
            round(float(v), 2) for v in means[::-1]
        ],
        "dominant_channel": CHANNEL_NAMES[int(np.argmax(means))],
    }

    if save_preview is not None:
        try:
            from PIL import Image

            Image.fromarray(frame, mode="RGB").save(save_preview, format="PNG")
            detail["preview_png"] = str(save_preview)
            detail["visual_check"] = "open this PNG: colours must look natural, not blue/orange swapped"
        except Exception as exc:
            detail["preview_png_error"] = f"{type(exc).__name__}: {exc}"

    if expect_color is None:
        report.add(
            StepResult(
                "rgb channel ordering",
                True,
                "reported for inspection (pass --expect-color for a hard check)",
                detail,
            )
        )
        return

    index = CHANNEL_NAMES.index(expect_color[0].upper())
    dominant = int(np.argmax(means))
    margin = float(means[index] - np.delete(means, index).max())
    detail["expected_dominant"] = CHANNEL_NAMES[index]
    detail["margin_over_next_channel"] = round(margin, 2)
    passed = dominant == index and margin > 5.0
    report.add(
        StepResult(
            "rgb channel ordering",
            passed,
            (
                f"{CHANNEL_NAMES[index]} target reads as {CHANNEL_NAMES[index]}"
                if passed
                else f"{CHANNEL_NAMES[index]} target reads as {CHANNEL_NAMES[dominant]} - channels are swapped"
            ),
            detail,
        )
    )


def step_full_res_capture(
    report: Report, manager: CameraManager, output_dir: Path
) -> Path | None:
    """Capture through the production partial-path + finalize sequence."""
    try:
        store = TempFileStore(output_dir)
    except Exception as exc:
        report.add(
            StepResult(
                "full-resolution capture",
                False,
                "temp store unusable",
                {"output_dir": str(output_dir), "error": f"{type(exc).__name__}: {exc}"},
            )
        )
        return None

    final_path = store.create_temp_path(".jpg")
    partial_path = store.create_partial_path(final_path)
    detail: dict[str, Any] = {
        "partial_path": str(partial_path),
        "final_path": str(final_path),
        "encoder_sees_extension": partial_path.suffix,
    }

    started = time.monotonic()
    try:
        manager.capture_still_to_file(partial_path)
        saved = store.finalize_capture(partial_path, final_path)
    except Exception as exc:
        detail["error"] = f"{type(exc).__name__}: {exc}"
        detail["elapsed_sec"] = round(time.monotonic() - started, 2)
        store.cleanup(partial_path)
        report.add(StepResult("full-resolution capture", False, "capture failed", detail))
        return None

    detail["elapsed_sec"] = round(time.monotonic() - started, 2)
    detail["exists"] = saved.exists()
    detail["file_bytes"] = saved.stat().st_size if saved.exists() else 0
    detail["partial_removed"] = not partial_path.exists()
    passed = saved.exists() and detail["file_bytes"] > 0 and detail["partial_removed"]
    report.add(
        StepResult(
            "full-resolution capture",
            passed,
            "JPEG written and finalized" if passed else "capture did not produce a usable file",
            detail,
        )
    )
    return saved if passed else None


def step_validate_jpeg(report: Report, path: Path, config: CameraConfig) -> None:
    import cv2
    from PIL import Image

    detail: dict[str, Any] = {"path": str(path), "file_bytes": path.stat().st_size}
    problems: list[str] = []

    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
            width, height = img.size
            detail["pil_format"] = img.format
            detail["pil_size (w, h)"] = (width, height)
    except Exception as exc:
        detail["pil_error"] = f"{type(exc).__name__}: {exc}"
        report.add(StepResult("validate JPEG", False, "PIL could not decode the file", detail))
        return

    if detail.get("pil_format") != "JPEG":
        problems.append(f"container is {detail.get('pil_format')}, expected JPEG")

    frame = cv2.imread(str(path))
    if frame is None:
        problems.append("OpenCV could not decode the file")
    else:
        detail["opencv_size (w, h)"] = (int(frame.shape[1]), int(frame.shape[0]))
        if (frame.shape[1], frame.shape[0]) != (width, height):
            problems.append("PIL and OpenCV disagree on dimensions")
        detail["sharpness_center_crop"] = round(_center_crop_sharpness(frame), 1)
        detail["mean_brightness"] = round(float(frame.mean()), 1)

    expected = (config.capture_width, config.capture_height)
    detail["expected_size (w, h)"] = expected
    if (width, height) != expected:
        detail["size_delta"] = (width - expected[0], height - expected[1])
        problems.append(
            f"captured {width}x{height} but config expects {expected[0]}x{expected[1]}"
        )

    passed = not problems
    report.add(
        StepResult(
            "validate JPEG",
            passed,
            "; ".join(problems) if problems else "JPEG decodes and matches configured resolution",
            detail,
        )
    )


def _center_crop_sharpness(frame: np.ndarray, size: int = 1024) -> float:
    """Laplacian variance on a bounded centre crop.

    Running this over a full 64 MP frame would allocate a multi-hundred-MB
    float64 buffer, so the crop keeps the diagnostic memory-safe on a Pi.
    """
    import cv2

    h, w = frame.shape[:2]
    ch, cw = min(size, h), min(size, w)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    patch = frame[y0 : y0 + ch, x0 : x0 + cw]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def step_shutdown(report: Report, manager: CameraManager, verify_release: bool) -> None:
    detail: dict[str, Any] = {}
    try:
        manager.shutdown()
    except Exception as exc:
        report.add(
            StepResult(
                "shutdown camera",
                False,
                "shutdown raised",
                {"error": f"{type(exc).__name__}: {exc}"},
            )
        )
        return
    detail["is_started"] = manager.is_started
    passed = not manager.is_started

    if passed and verify_release and not manager.is_mock:
        # Reacquiring proves the device was actually released rather than left
        # busy by a dangling handle.
        try:
            manager.initialize()
            detail["reacquire_after_release"] = "ok"
            manager.shutdown()
        except Exception as exc:
            detail["reacquire_after_release"] = f"{type(exc).__name__}: {exc}"
            passed = False

    report.add(
        StepResult(
            "shutdown camera",
            passed,
            "device released" if passed else "device was not released cleanly",
            detail,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    defaults = CameraConfig()
    parser = argparse.ArgumentParser(
        description="idCam camera-layer diagnostic (no state machine involved).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preview-width", type=int, default=defaults.preview_width)
    parser.add_argument("--preview-height", type=int, default=defaults.preview_height)
    parser.add_argument("--capture-width", type=int, default=defaults.capture_width)
    parser.add_argument("--capture-height", type=int, default=defaults.capture_height)
    parser.add_argument("--jpeg-quality", type=int, default=defaults.jpeg_quality)
    parser.add_argument(
        "--output", type=Path, default=Path("./idcam-camera-test"), help="directory for captures"
    )
    parser.add_argument("--fps-frames", type=int, default=30, help="frames used to measure fps")
    parser.add_argument("--skip-capture", action="store_true", help="preview checks only")
    parser.add_argument(
        "--expect-color",
        choices=["red", "green", "blue"],
        help="fill the frame with this colour to hard-check channel ordering",
    )
    parser.add_argument(
        "--save-preview", action="store_true", help="write preview.png for visual colour inspection"
    )
    parser.add_argument(
        "--no-verify-release",
        action="store_true",
        help="skip reacquiring the camera after shutdown",
    )
    parser.add_argument("--mock", action="store_true", help="run against the mock camera")
    parser.add_argument("--json", action="store_true", help="print a machine-readable summary")
    parser.add_argument("--json-out", type=Path, help="write the machine-readable summary here")
    parser.add_argument("--verbose", action="store_true", help="debug-level idcam logs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)

    config = CameraConfig(
        preview_width=args.preview_width,
        preview_height=args.preview_height,
        capture_width=args.capture_width,
        capture_height=args.capture_height,
        jpeg_quality=args.jpeg_quality,
    )
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("idCam camera diagnostic")
    print(f"  preview  : {config.preview_width}x{config.preview_height}")
    print(f"  capture  : {config.capture_width}x{config.capture_height}")
    print(f"  output   : {output_dir.resolve()}")
    print("=" * 72)

    report = Report()
    manager = step_initialize(report, config, use_mock=args.mock)
    if manager is None:
        return _finish(report, args.json, args.json_out)

    try:
        step_stream_geometry(report, manager, config)
        frame = step_preview_frame(report, manager)
        step_preview_fps(report, manager, args.fps_frames)
        if frame is not None:
            step_rgb_order(
                report,
                frame,
                args.expect_color,
                output_dir / "preview.png" if args.save_preview else None,
            )
        if not args.skip_capture:
            captured = step_full_res_capture(report, manager, output_dir)
            if captured is not None:
                step_validate_jpeg(report, captured, config)
    finally:
        step_shutdown(report, manager, verify_release=not args.no_verify_release)

    return _finish(report, args.json, args.json_out)


def _finish(report: Report, as_json: bool, json_out: Path | None = None) -> int:
    print("=" * 72)
    passed = sum(1 for s in report.steps if s.passed)
    print(f"RESULT: {'PASS' if report.ok else 'FAIL'}  ({passed}/{len(report.steps)} steps passed)")
    print("=" * 72)
    payload = json.dumps(report.as_dict(), indent=2, default=str)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(payload, encoding="utf-8")
        print(f"summary written to {json_out}")
    if as_json:
        print(payload)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
