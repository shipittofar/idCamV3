"""DEBUG-ONLY manual lens calibration. Not the production autofocus algorithm."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

from idcam.camera.manager import preview_roi_to_still_roi
from idcam.logging.setup import get_logger
from idcam.vision.sharpness import clip_roi, laplacian_variance, resize_patch_to_preview, roi_rect

logger = get_logger("preview.manual_focus")

AF_STATE_NAMES = {0: "Idle", 1: "Scanning", 2: "Focused", 3: "Failed"}
DEFAULT_LENS_MIN = 0.0
DEFAULT_LENS_MAX = 10.0
LENS_SETTLE_EPS = 0.05
LENS_SETTLE_SAMPLES = 3
INTERIOR_INSET = 0.12


def parse_lens_limits(controls: dict[str, Any] | None) -> tuple[float, float]:
    """Read LensPosition min/max from Picamera2 camera_controls tuples."""
    if not controls:
        return DEFAULT_LENS_MIN, DEFAULT_LENS_MAX
    info = controls.get("LensPosition")
    if info is None:
        return DEFAULT_LENS_MIN, DEFAULT_LENS_MAX
    try:
        lo, hi = float(info[0]), float(info[1])
    except (TypeError, ValueError, IndexError):
        return DEFAULT_LENS_MIN, DEFAULT_LENS_MAX
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def clamp_lens(value: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, float(value))))


def interior_document_roi(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    *,
    inset_frac: float = INTERIOR_INSET,
) -> tuple[int, int, int, int]:
    """Shrink a document AABB to exclude table, border, and extreme page edges."""
    x, y, w, h = (int(v) for v in bbox)
    ix = max(1, int(round(w * inset_frac)))
    iy = max(1, int(round(h * inset_frac)))
    inner = (x + ix, y + iy, max(1, w - 2 * ix), max(1, h - 2 * iy))
    clipped = clip_roi(inner, frame_shape)
    if clipped[2] < 16 or clipped[3] < 16:
        return clip_roi(bbox, frame_shape)
    return clipped


def select_focus_roi(
    frame_shape: tuple[int, ...],
    document_bbox: tuple[int, int, int, int] | None,
    *,
    document_present: bool,
) -> tuple[tuple[int, int, int, int], str]:
    if document_present and document_bbox is not None:
        w, h = int(document_bbox[2]), int(document_bbox[3])
        if w >= 24 and h >= 24:
            return interior_document_roi(document_bbox, frame_shape), "document_geometry"
    return roi_rect(frame_shape, 0.7), "center_fallback"


def lens_is_settled(samples: list[float], *, eps: float = LENS_SETTLE_EPS, need: int = LENS_SETTLE_SAMPLES) -> bool:
    if len(samples) < need:
        return False
    window = samples[-need:]
    return max(window) - min(window) < eps


def pick_best_scan(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda r: (float(r.get("sharpness") or 0.0), float(r.get("focus_fom") or 0.0)))


def af_state_name(value: Any) -> str:
    try:
        return AF_STATE_NAMES.get(int(value), str(value))
    except (TypeError, ValueError):
        return "—"


def _meta_float(meta: dict[str, Any], key: str) -> float | None:
    raw = meta.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def still_roi_sharpness(
    jpeg_path: Path,
    preview_roi: tuple[int, int, int, int],
    preview_size: tuple[int, int],
    scaler_crop: tuple[int, int, int, int],
) -> tuple[float, tuple[int, int], tuple[int, int, int, int]]:
    """Same preview-equivalent Laplacian used by still validation (no pass/fail)."""
    with Image.open(jpeg_path) as img:
        width, height = img.size
        still_size = (int(width), int(height))
        scaled = preview_roi_to_still_roi(preview_roi, preview_size, scaler_crop, still_size)
        img.load()
        x, y, rw, rh = scaled
        patch = img.crop((x, y, x + rw, y + rh))
        if patch.mode != "RGB":
            patch = patch.convert("RGB")
        patch_rgb = np.asarray(patch)
    preview_patch = resize_patch_to_preview(patch_rgb, (int(preview_roi[2]), int(preview_roi[3])))
    if preview_patch.ndim == 3:
        gray = cv2.cvtColor(preview_patch, cv2.COLOR_RGB2GRAY)
    else:
        gray = preview_patch
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()), still_size, scaled


@dataclass
class ManualFocusCalibrator:
    """Operator-driven lens control for the live debug viewer."""

    camera: Any
    get_preview: Callable[[], np.ndarray | None]
    get_document: Callable[[], tuple[bool, tuple[int, int, int, int] | None]]
    still_dir: Path
    mode: str = "auto"
    requested_lens: float | None = None
    last_actual_lens: float | None = None
    last_fom: float | None = None
    last_af_state: Any = None
    last_af_mode: Any = None
    last_sharpness: float | None = None
    last_roi: tuple[int, int, int, int] | None = None
    last_roi_source: str = "center_fallback"
    last_scan: dict[str, Any] | None = None
    last_capture: dict[str, Any] | None = None
    busy: bool = False
    message: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def blocks_production_af(self) -> bool:
        return self.mode in {"manual", "lock"}

    def lens_limits(self) -> tuple[float, float]:
        getter = getattr(self.camera, "lens_position_limits", None)
        if callable(getter):
            return getter()
        controls = self.camera.get_controls() if hasattr(self.camera, "get_controls") else {}
        return parse_lens_limits(controls)

    def snapshot(self) -> dict[str, Any]:
        lo, hi = self.lens_limits()
        return {
            "focus_mode": self.mode,
            "requested_lens": self.requested_lens,
            "actual_lens": self.last_actual_lens,
            "af_state": self.last_af_state,
            "af_state_name": af_state_name(self.last_af_state),
            "af_mode": self.last_af_mode,
            "focus_fom": self.last_fom,
            "document_sharpness": self.last_sharpness,
            "focus_roi": None if self.last_roi is None else "{},{},{},{}".format(*self.last_roi),
            "focus_roi_source": self.last_roi_source,
            "lens_min": lo,
            "lens_max": hi,
            "busy": self.busy,
            "message": self.message,
            "scan": self.last_scan,
            "capture_test": self.last_capture,
        }

    def observe(self, frame: np.ndarray | None) -> dict[str, Any]:
        meta = {}
        try:
            meta = dict(self.camera.capture_metadata())
        except Exception:
            meta = {}
        self.last_actual_lens = _meta_float(meta, "LensPosition")
        self.last_fom = _meta_float(meta, "FocusFoM")
        self.last_af_state = meta.get("AfState")
        self.last_af_mode = meta.get("AfMode")
        present, bbox = self.get_document()
        if frame is not None:
            roi, source = select_focus_roi(frame.shape, bbox, document_present=present)
            self.last_roi = roi
            self.last_roi_source = source
            self.last_sharpness = laplacian_variance(frame, roi=roi)
        return self.snapshot()

    def handle(self, action: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        with self._lock:
            if self.busy and action not in {"status"}:
                return {**self.snapshot(), "ok": False, "status": "busy", "message": "Focus calibration busy"}
        try:
            if action == "status":
                return {**self.snapshot(), "ok": True}
            if action in {"manual", "set_manual"}:
                return self._enter_manual()
            if action in {"autofocus", "auto"}:
                return self._run_autofocus()
            if action == "lock":
                return self._lock_lens()
            if action == "nudge":
                return self._nudge(float(payload.get("delta", 0.0)))
            if action == "set":
                return self._set_lens(float(payload["lens"]))
            if action == "scan":
                step = float(payload.get("step") or 0.2)
                return self._scan(step)
            if action in {"capture_test", "capture"}:
                return self._capture_test()
            return {**self.snapshot(), "ok": False, "status": "error", "message": f"Unknown action {action}"}
        except Exception as exc:
            logger.exception("event=manual_focus_failed action=%s", action)
            return {**self.snapshot(), "ok": False, "status": "error", "message": str(exc)}

    def _enter_manual(self) -> dict[str, Any]:
        lo, hi = self.lens_limits()
        current = self._read_lens()
        if current is None:
            current = clamp_lens(1.0, lo, hi)
        actual = self.camera.set_manual_lens(current)
        self.mode = "manual"
        self.requested_lens = current
        self.last_actual_lens = actual if actual is not None else current
        self.message = "MANUAL"
        logger.info(
            "event=manual_focus_set requested_lens=%.4f actual_lens=%s AfMode=Manual AfState=%s FocusFoM=%s",
            current,
            self.last_actual_lens,
            self.last_af_state,
            self.last_fom,
        )
        self._refresh_meta()
        return {**self.snapshot(), "ok": True, "status": "manual"}

    def _run_autofocus(self) -> dict[str, Any]:
        self.mode = "auto"
        trigger = getattr(self.camera, "trigger_autofocus", None)
        if callable(trigger):
            trigger()
        self.message = "AUTOFOCUS triggered (production algorithm, not redesigned)"
        logger.info("event=manual_focus_set requested_lens=%s actual_lens=%s AfMode=Auto", self.requested_lens, self._read_lens())
        self._refresh_meta()
        return {**self.snapshot(), "ok": True, "status": "autofocus"}

    def _lock_lens(self) -> dict[str, Any]:
        pause = getattr(self.camera, "pause_autofocus", None)
        if callable(pause):
            pause()
        lens = self._read_lens()
        if lens is not None:
            self.camera.set_manual_lens(lens)
            self.requested_lens = lens
            self.last_actual_lens = lens
        self.mode = "lock"
        self.message = "LOCK"
        logger.info(
            "event=manual_focus_set requested_lens=%s actual_lens=%s AfMode=Manual AfState=%s FocusFoM=%s",
            self.requested_lens,
            self.last_actual_lens,
            self.last_af_state,
            self.last_fom,
        )
        return {**self.snapshot(), "ok": True, "status": "lock"}

    def _nudge(self, delta: float) -> dict[str, Any]:
        lo, hi = self.lens_limits()
        base = self.requested_lens if self.requested_lens is not None else self._read_lens()
        if base is None:
            base = lo
        return self._set_lens(clamp_lens(base + delta, lo, hi))

    def _set_lens(self, value: float) -> dict[str, Any]:
        lo, hi = self.lens_limits()
        requested = clamp_lens(value, lo, hi)
        self.mode = "manual"
        self.requested_lens = requested
        t0 = time.monotonic()
        actual = self.camera.set_manual_lens(requested)
        settled = self.camera.wait_for_lens_settle(1.6)
        settle_ms = (time.monotonic() - t0) * 1000.0
        if settled is not None:
            actual = settled
        self.last_actual_lens = actual if actual is not None else requested
        logger.info(
            "event=manual_focus_set requested_lens=%.4f actual_lens=%s AfMode=Manual AfState=%s FocusFoM=%s",
            requested,
            self.last_actual_lens,
            self.last_af_state,
            self.last_fom,
        )
        logger.info(
            "event=manual_focus_settle requested_lens=%.4f actual_lens=%s settle_duration_ms=%.0f",
            requested,
            self.last_actual_lens,
            settle_ms,
        )
        frame = self.get_preview()
        self.observe(frame)
        logger.info(
            "event=manual_focus_measurement lens_position=%s focus_fom=%s sharpness=%s roi=%s roi_source=%s",
            self.last_actual_lens,
            self.last_fom,
            self.last_sharpness,
            self.last_roi,
            self.last_roi_source,
        )
        self.message = f"Lens {self.last_actual_lens:.3f}" if self.last_actual_lens is not None else "Lens set"
        return {**self.snapshot(), "ok": True, "status": "set", "settle_duration_ms": round(settle_ms, 1)}

    def _scan(self, step: float) -> dict[str, Any]:
        self.busy = True
        self.message = "SCAN FOCUS..."
        try:
            lo, hi = self.lens_limits()
            step = min(max(abs(step), 0.05), max(0.05, hi - lo))
            start = lo
            rows: list[dict[str, Any]] = []
            pos = start
            while pos <= hi + 1e-9:
                self._set_lens(pos)
                rows.append(
                    {
                        "lens_position": self.last_actual_lens,
                        "focus_fom": self.last_fom,
                        "sharpness": self.last_sharpness,
                    }
                )
                pos = round(pos + step, 4)
            best = pick_best_scan(rows)
            self.last_scan = {
                "scan_start": start,
                "scan_end": hi,
                "step": step,
                "samples": rows,
                "best_lens": None if best is None else best.get("lens_position"),
                "best_sharpness": None if best is None else best.get("sharpness"),
            }
            logger.info(
                "event=manual_focus_scan scan_start=%.3f scan_end=%.3f step=%.3f best_lens=%s best_sharpness=%s",
                start,
                hi,
                step,
                None if best is None else best.get("lens_position"),
                None if best is None else best.get("sharpness"),
            )
            if best and best.get("lens_position") is not None:
                self._set_lens(float(best["lens_position"]))
            self.message = (
                f"Best lens={self.last_scan['best_lens']} sharpness={self.last_scan['best_sharpness']}"
            )
            return {**self.snapshot(), "ok": True, "status": "scan"}
        finally:
            self.busy = False

    def _capture_test(self) -> dict[str, Any]:
        self.busy = True
        self.message = "CAPTURE FOCUS TEST..."
        try:
            if self.mode != "manual":
                self._enter_manual()
            frame = self.get_preview()
            self.observe(frame)
            preview_sharp = self.last_sharpness
            roi = self.last_roi or (0, 0, 1, 1)
            preview_size = (int(frame.shape[1]), int(frame.shape[0])) if frame is not None else (1920, 1080)
            scaler = self.camera.current_scaler_crop() if hasattr(self.camera, "current_scaler_crop") else (0, 0, *preview_size)
            self.still_dir.mkdir(parents=True, exist_ok=True)
            path = self.still_dir / f"focus_test_lens_{self.last_actual_lens or 0:.3f}.jpg".replace(" ", "")
            t0 = time.monotonic()
            capture = self.camera.capture_still_to_file
            capture(path, restore_preview_af=False)
            duration_ms = (time.monotonic() - t0) * 1000.0
            if self.requested_lens is not None:
                self.camera.set_manual_lens(self.requested_lens)
            still_sharp, still_hw, mapped = still_roi_sharpness(path, roi, preview_size, scaler)
            self.last_capture = {
                "path": str(path),
                "lens_position": self.last_actual_lens,
                "preview_sharpness": preview_sharp,
                "still_sharpness": still_sharp,
                "image_width": still_hw[0],
                "image_height": still_hw[1],
                "roi": "{},{},{},{}".format(*roi),
                "roi_source": self.last_roi_source,
                "still_roi": "{},{},{},{}".format(*mapped),
                "duration_ms": round(duration_ms, 1),
            }
            logger.info(
                "event=focus_test_capture lens_position=%s preview_sharpness=%s still_sharpness=%s "
                "image_width=%d image_height=%d duration_ms=%.0f path=%s",
                self.last_actual_lens,
                preview_sharp,
                still_sharp,
                still_hw[0],
                still_hw[1],
                duration_ms,
                path,
            )
            self.message = (
                f"Still {still_hw[0]}x{still_hw[1]} preview={preview_sharp:.1f} still={still_sharp:.1f}"
            )
            return {**self.snapshot(), "ok": True, "status": "capture_test"}
        finally:
            self.busy = False

    def _read_lens(self) -> float | None:
        meta = {}
        try:
            meta = dict(self.camera.capture_metadata())
        except Exception:
            return self.last_actual_lens
        return _meta_float(meta, "LensPosition")

    def _refresh_meta(self) -> None:
        try:
            meta = dict(self.camera.capture_metadata())
        except Exception:
            return
        self.last_actual_lens = _meta_float(meta, "LensPosition") or self.last_actual_lens
        self.last_fom = _meta_float(meta, "FocusFoM")
        self.last_af_state = meta.get("AfState")
        self.last_af_mode = meta.get("AfMode")
