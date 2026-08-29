"""Live Debug Viewer — observer of the existing preview pipeline.

ONE producer (the orchestrator preview path) publishes the latest frame.
Multiple HTTP clients consume the latest JPEG. There is no per-client camera
capture and no unbounded queue.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

from idcam.camera.manager import describe_preview_frame
from idcam.config.schema import DebugConfig
from idcam.logging.setup import get_logger

logger = get_logger("preview.live_viewer")

MJPEG_BOUNDARY = "frame"
PLACEHOLDER_WIDTH = 1920
PLACEHOLDER_HEIGHT = 1080
_MAX_CLIENT_LOGS_PER_MINUTE = 20


@dataclass
class DebugSnapshot:
    """Latest debug view of the capture pipeline. Frame may be None."""

    frame: np.ndarray | None = None
    timestamp: float = 0.0
    state: str = "BOOT"
    document_detected: bool = False
    geometry_candidate: bool = False
    document_confirmed: bool = False
    confirmation_reason: str = ""
    evidence_edge_support: float = 0.0
    evidence_edge_sides: int = 0
    evidence_paper_contrast: float = 0.0
    evidence_print: float = 0.0
    evidence_line_structure: float = 0.0
    evidence_fold: float = 0.0
    motion: float = 0.0
    edge_density: float = 0.0
    motion_threshold: float = 8.0
    edge_density_threshold: float = 0.02
    stable_count: int = 0
    stable_required: int = 8
    stable: bool = False
    camera_healthy: bool = False
    autofocus_available: bool = False
    camera_error: str | None = None
    roi_center_fraction: float = 0.7
    waiting_for_camera: bool = True
    reject_reason: str = ""
    roi_width: int = 0
    roi_height: int = 0
    gray_std: float = 0.0
    gray_mean: float = 0.0
    frame_width: int = 0
    frame_height: int = 0
    frame_motion: float = 0.0
    manual_capture_available: bool = False
    manual_phase: str = "idle"
    manual_message: str = ""
    removal_detected: bool = False
    removal_count: int = 0
    autofocus_running: bool = False
    autofocus_method: str = ""
    autofocus_attempt: int = 0
    focus_before: float = 0.0
    focus_after: float = 0.0
    focus_roi: str = ""
    lens_before: float | None = None
    lens_after: float | None = None
    improvement_ratio: float = 0.0
    preview_focus_roi: str = ""
    sensor_focus_roi: str = ""
    preview_roi_wh: str = ""
    document_roi: str = ""
    stability_motion: float = 0.0
    stability_threshold: float = 0.0
    state_reason: str = ""
    manual_capture_requested: bool = False
    capture_started: bool = False
    upload_started: bool = False
    upload_result: str = ""
    detect_hz: float = 0.0
    tick_ms: float = 0.0
    focus_mode: str = ""
    requested_lens: float | None = None
    actual_lens: float | None = None
    af_state_name: str = ""
    focus_fom: float | None = None
    document_sharpness: float | None = None
    focus_roi_source: str = ""
    focus_roi_live: str = ""
    lens_min: float | None = None
    lens_max: float | None = None
    focus_message: str = ""


class LiveFrameBuffer:
    """Thread-safe latest-slot buffer. Slow consumers skip to the newest item."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._seq = 0
        self._payload: Any = None

    @property
    def sequence(self) -> int:
        with self._lock:
            return self._seq

    def publish(self, payload: Any) -> None:
        with self._cond:
            self._payload = payload
            self._seq += 1
            self._cond.notify_all()

    def latest(self) -> tuple[int, Any]:
        with self._lock:
            return self._seq, self._payload

    def wait_latest(self, last_seq: int, timeout: float) -> tuple[int, Any]:
        """Return the newest payload. Never queues older frames."""
        with self._cond:
            if self._seq <= last_seq:
                self._cond.wait(timeout)
            return self._seq, self._payload

    def notify_all(self) -> None:
        with self._cond:
            self._cond.notify_all()


def resize_debug_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Resize only a debug copy. Never upscale."""
    if frame.ndim < 2:
        return frame
    height, width = frame.shape[:2]
    if width <= max_width or max_width <= 0:
        return frame
    scale = max_width / float(width)
    new_size = (max_width, max(1, int(round(height * scale))))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _to_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert an RGB preview frame to BGR for OpenCV JPEG encoding.

    CameraManager.normalize_picamera_preview_frame() delivers RGB (or gray).
    OpenCV imencode expects BGR.
    """
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def encode_jpeg(frame: np.ndarray, quality: int) -> bytes:
    bgr = _to_bgr(frame)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def _draw_panel(image: np.ndarray, lines: list[str], *, banner: str | None = None) -> None:
    overlay = image.copy()
    line_h = 22
    pad = 12
    panel_w = 620
    panel_h = pad * 2 + line_h * len(lines)
    if banner:
        panel_h += line_h + 8
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, dst=image)

    y = 8 + pad + 18
    if banner:
        cv2.putText(
            image,
            banner,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 80, 80),
            2,
            cv2.LINE_AA,
        )
        y += line_h + 4
    for line in lines:
        cv2.putText(
            image,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += line_h


def draw_overlay(frame: np.ndarray, snapshot: DebugSnapshot, fps: float) -> np.ndarray:
    """Draw diagnostics on a copy of ``frame``. The input array is not modified."""
    debug_frame = frame.copy()
    if debug_frame.ndim == 2:
        debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_GRAY2RGB)

    frac = snapshot.roi_center_fraction
    h, w = debug_frame.shape[:2]
    rh, rw = int(h * frac), int(w * frac)
    y0, x0 = (h - rh) // 2, (w - rw) // 2
    cv2.rectangle(debug_frame, (x0, y0), (x0 + rw, y0 + rh), (70, 70, 70), 1)
    doc_roi = _parse_roi(snapshot.document_roi or snapshot.preview_focus_roi or snapshot.focus_roi)
    if doc_roi is not None:
        dx, dy, dw, dh = doc_roi
        color = (40, 220, 90) if snapshot.document_confirmed else (40, 200, 255)
        cv2.rectangle(debug_frame, (dx, dy), (dx + dw, dy + dh), color, 2)
    focus_live = _parse_roi(snapshot.focus_roi_live)
    if focus_live is not None:
        fx, fy, fw, fh = focus_live
        cv2.rectangle(debug_frame, (fx, fy), (fx + fw, fy + fh), (255, 200, 40), 2)

    camera = "OK" if snapshot.camera_healthy else "ERROR"
    if not snapshot.camera_healthy and snapshot.waiting_for_camera and not snapshot.camera_error:
        camera = "WAITING"
    document = "CONFIRMED" if snapshot.document_confirmed else ("CANDIDATE" if snapshot.geometry_candidate or snapshot.document_detected else "NO")
    banner = None
    if snapshot.waiting_for_camera and not snapshot.camera_error:
        banner = "WAITING FOR CAMERA..."
    if snapshot.camera_error or (not snapshot.camera_healthy and not snapshot.waiting_for_camera):
        banner = "CAMERA ERROR"

    lines = [
        f"STATE: {snapshot.state}",
        f"GEOMETRY: {'candidate' if (snapshot.geometry_candidate or snapshot.document_detected) else 'none'}",
        f"DOCUMENT: {document}",
        f"CONFIRM: {snapshot.confirmation_reason or '-'}",
        f"EDGE SIDES: {snapshot.evidence_edge_sides} support={snapshot.evidence_edge_support:.2f}",
        f"PAPER CONTRAST: {snapshot.evidence_paper_contrast:.2f}",
        f"PRINT/LINE/FOLD: {snapshot.evidence_print:.2f}/{snapshot.evidence_line_structure:.2f}/{snapshot.evidence_fold:.2f}",
        f"MOTION: {snapshot.motion:.2f}",
        f"MOTION THRESHOLD: {snapshot.motion_threshold:.2f}",
        f"EDGE: {snapshot.edge_density:.4f}",
        f"EDGE THRESHOLD: {snapshot.edge_density_threshold:.4f}",
        f"REJECT: {snapshot.reject_reason or '-'}",
        f"ROI: {snapshot.roi_width}x{snapshot.roi_height}",
        f"FRAME: {snapshot.frame_width}x{snapshot.frame_height}",
        f"FRAME MOTION: {snapshot.frame_motion:.2f}",
        f"STABLE: {snapshot.stable_count}/{snapshot.stable_required} motion={snapshot.stability_motion:.2f}/{snapshot.stability_threshold:.2f}",
        f"REMOVAL: {snapshot.removal_count} empty={snapshot.removal_detected}",
        f"AF: {snapshot.autofocus_method or '-'} att={snapshot.autofocus_attempt}",
        f"FOCUS: {snapshot.focus_before:.1f}->{snapshot.focus_after:.1f} x{snapshot.improvement_ratio:.2f}",
        f"LENS: {snapshot.lens_before}->{snapshot.lens_after}",
        f"PREVIEW ROI: {snapshot.preview_focus_roi or snapshot.focus_roi or '-'}",
        f"SENSOR ROI: {snapshot.sensor_focus_roi or '-'}",
        f"REASON: {snapshot.state_reason or '-'}",
        f"MANUAL: req={snapshot.manual_capture_requested} cap={snapshot.capture_started} up={snapshot.upload_started} {snapshot.upload_result or '-'}",
        f"MJPEG FPS: {fps:.1f} (encoder, not camera)",
        f"DETECT Hz: {snapshot.detect_hz:.1f}  TICK: {snapshot.tick_ms:.0f}ms",
        f"CAMERA: {camera}",
    ]
    if snapshot.focus_mode:
        lens = snapshot.actual_lens
        req = snapshot.requested_lens
        lines = [
            "FOCUS DEBUG",
            f"Mode: {snapshot.focus_mode.upper()}",
            f"Requested: {'—' if req is None else f'{req:.3f}'}",
            f"Lens: {'—' if lens is None else f'{lens:.3f}'}",
            f"AfState: {snapshot.af_state_name or '—'}",
            f"FocusFoM: {'—' if snapshot.focus_fom is None else f'{snapshot.focus_fom:.0f}'}",
            f"Document Sharpness: {'—' if snapshot.document_sharpness is None else f'{snapshot.document_sharpness:.1f}'}",
            f"Focus ROI: {snapshot.focus_roi_live or '—'}",
            f"Focus ROI source: {snapshot.focus_roi_source or '—'}",
            f"Lens range: {snapshot.lens_min}..{snapshot.lens_max}",
        ] + lines
    if snapshot.camera_error and not snapshot.camera_healthy:
        err = snapshot.camera_error.replace("\n", " ")[:60]
        lines.append(f"DETAIL: {err}")
    _draw_panel(debug_frame, lines, banner=banner)
    return debug_frame


def compose_debug_frame(
    frame: np.ndarray,
    snapshot: DebugSnapshot,
    fps: float,
    max_width: int,
    *,
    overlay: bool = True,
) -> np.ndarray:
    """Draw overlay in ``frame`` preview coordinates, then shrink for MJPEG.

    Document/focus ROIs are preview pixels of ``frame``. Draw first, then resize,
    so a 1920-space box is not painted onto a narrower JPEG.
    """
    if overlay:
        image = draw_overlay(frame, snapshot, fps)
    else:
        image = frame.copy()
    return resize_debug_frame(image, max_width)


def _parse_roi(value: str) -> tuple[int, int, int, int] | None:
    if not value or "," not in value:
        return None
    parts = value.replace(" ", "").split(",")
    if len(parts) != 4:
        return None
    try:
        x, y, w, h = (int(float(p)) for p in parts)
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def make_placeholder(message: str) -> np.ndarray:
    image = np.full((PLACEHOLDER_HEIGHT, PLACEHOLDER_WIDTH, 3), 28, dtype=np.uint8)
    cv2.putText(
        image,
        message,
        (48, PLACEHOLDER_HEIGHT // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return image


_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>idCam V3 — Live Debug Viewer</title>
  <style>
    :root { color-scheme: dark; }
    body { margin: 0; font-family: ui-sans-serif, system-ui, sans-serif;
           background: #111; color: #eee; }
    header { padding: 16px 20px; background: #1b1b1b; border-bottom: 1px solid #333; }
    h1 { margin: 0; font-size: 1.25rem; font-weight: 600; }
    .note { color: #aaa; font-size: 0.85rem; margin-top: 4px; }
    main { display: grid; grid-template-columns: 1fr 320px; gap: 16px; padding: 16px; }
    img { width: auto; max-width: min(100%, 1920px); height: auto; object-fit: contain;
           background: #000; border: 1px solid #333; }
    dl { margin: 0; }
    dt { color: #888; font-size: 0.75rem; text-transform: uppercase; margin-top: 12px; }
    dd { margin: 2px 0 0; font-variant-numeric: tabular-nums; font-size: 1.05rem; }
    button {
      margin-top: 16px; width: 100%; padding: 12px 16px; font-size: 1rem;
      font-weight: 600; border: 0; border-radius: 8px; cursor: pointer;
      background: #3b82f6; color: #fff;
    }
    button:disabled { background: #334155; color: #94a3b8; cursor: not-allowed; }
    .focus-box { margin-top: 18px; padding-top: 12px; border-top: 1px solid #333; }
    .focus-box h2 { margin: 0 0 8px; font-size: 0.85rem; letter-spacing: 0.04em; color: #fbbf24; }
    .focus-row { display: flex; gap: 6px; margin: 6px 0; }
    .focus-row button { margin-top: 0; padding: 8px 6px; font-size: 0.85rem; background: #334155; }
    .keys { color: #888; font-size: 0.75rem; margin-top: 8px; line-height: 1.4; }
    .manual { margin-top: 8px; font-size: 0.9rem; color: #ddd; min-height: 2.4em; }
    .manual.error { color: #f87171; }
    .manual.ok { color: #4ade80; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>idCam V3 — Live Debug Viewer</h1>
    <div class="note">Debug only. Live camera preview — do not enable on untrusted networks.</div>
  </header>
  <main>
    <section>
      <img src="/preview/live.mjpg" alt="Live preview">
    </section>
    <aside>
      <dl>
        <dt>State</dt><dd id="state">—</dd>
        <dt>Camera</dt><dd id="camera">—</dd>
        <dt>Detection</dt><dd id="detection">—</dd>
        <dt>Geometry</dt><dd id="geometry">—</dd>
        <dt>Document confirmed</dt><dd id="confirmed">—</dd>
        <dt>Confirmation</dt><dd id="confirmreason">—</dd>
        <dt>Edge support</dt><dd id="edgesupport">—</dd>
        <dt>Paper contrast</dt><dd id="papercontrast">—</dd>
        <dt>Print / fold</dt><dd id="printfold">—</dd>
        <dt>Reject reason</dt><dd id="reject">—</dd>
        <dt>Motion</dt><dd id="motion">—</dd>
        <dt>Frame motion</dt><dd id="framemotion">—</dd>
        <dt>Edge density</dt><dd id="edge">—</dd>
        <dt>ROI</dt><dd id="roi">—</dd>
        <dt>Frame</dt><dd id="frame">—</dd>
        <dt>Gray std</dt><dd id="graystd">—</dd>
        <dt>Stable frames</dt><dd id="stable">—</dd>
        <dt>Stability motion</dt><dd id="stabmotion">—</dd>
        <dt>Removal</dt><dd id="removal">—</dd>
        <dt>Autofocus</dt><dd id="af">—</dd>
        <dt>Focus ROI (preview)</dt><dd id="focusroi">—</dd>
        <dt>Sensor AF ROI</dt><dd id="sensorroi">—</dd>
        <dt>Document ROI</dt><dd id="docroi">—</dd>
        <dt>Lens</dt><dd id="lens">—</dd>
        <dt>Sharpness ratio</dt><dd id="ratio">—</dd>
        <dt>State reason</dt><dd id="reason">—</dd>
        <dt>Capture / upload</dt><dd id="pipeline">—</dd>
        <dt>MJPEG FPS (encoder)</dt><dd id="fps">—</dd>
        <dt>Detect Hz</dt><dd id="detecthz">—</dd>
        <dt>Tick ms</dt><dd id="tickms">—</dd>
        <dt>Timestamp</dt><dd id="timestamp">—</dd>
      </dl>
      <button id="capture" type="button">Capture &amp; Send</button>
      <div id="manual" class="manual">Idle</div>
      <div class="focus-box">
        <h2>Manual focus (debug)</h2>
        <dt style="display:none"></dt>
        <div>Mode: <span id="fmode">—</span></div>
        <div>Requested: <span id="freq">—</span></div>
        <div>Lens: <span id="flens">—</span></div>
        <div>AfState: <span id="fstate">—</span></div>
        <div>FocusFoM: <span id="ffom">—</span></div>
        <div>Doc sharpness: <span id="fsharp">—</span></div>
        <div>Focus ROI: <span id="froi">—</span></div>
        <div>ROI source: <span id="froisrc">—</span></div>
        <div>Range: <span id="frange">—</span></div>
        <div class="focus-row">
          <button type="button" data-focus='{"action":"nudge","delta":-1}'>-1.0</button>
          <button type="button" data-focus='{"action":"nudge","delta":-0.1}'>-0.1</button>
          <button type="button" data-focus='{"action":"nudge","delta":-0.01}'>-0.01</button>
        </div>
        <div class="focus-row">
          <button type="button" data-focus='{"action":"nudge","delta":0.01}'>+0.01</button>
          <button type="button" data-focus='{"action":"nudge","delta":0.1}'>+0.1</button>
          <button type="button" data-focus='{"action":"nudge","delta":1}'>+1.0</button>
        </div>
        <div class="focus-row">
          <button type="button" data-focus='{"action":"manual"}'>SET MANUAL</button>
        </div>
        <div class="focus-row">
          <button type="button" data-focus='{"action":"autofocus"}'>AUTOFOCUS</button>
          <button type="button" data-focus='{"action":"lock"}'>LOCK</button>
        </div>
        <div class="focus-row">
          <button type="button" data-focus='{"action":"scan","step":0.2}'>SCAN FOCUS</button>
        </div>
        <div class="focus-row">
          <button type="button" data-focus='{"action":"capture_test"}'>CAPTURE FOCUS TEST</button>
        </div>
        <div id="focusmsg" class="manual"></div>
        <div class="keys">←/→ ±0.01 · Shift+arrows ±0.1 · M manual · A AF · L lock · S scan · T still test</div>
      </div>
    </aside>
  </main>
  <script>
    const el = (id) => document.getElementById(id);
    let capturing = false;
    function phaseLabel(d) {
      const phase = d.manual_phase || "";
      const map = {
        queued: "Queued...",
        starting: "Starting...",
        detected: "Document detected...",
        focusing: "Focusing...",
        capturing: "Capturing...",
        validating: "Validating...",
        uploading: "Uploading...",
        success: "Success",
        failed: "Failure",
        timeout: "Timed out",
        busy: "Busy",
        idle: "Idle"
      };
      return map[phase] || phase || "Idle";
    }
    async function refresh() {
      try {
        const r = await fetch("/preview/status", { cache: "no-store" });
        if (!r.ok) throw new Error("status " + r.status);
        const d = await r.json();
        el("state").textContent = d.state || "—";
        el("camera").textContent = d.camera_healthy ? "healthy" : "unhealthy";
        el("detection").textContent = d.document_confirmed
          ? "confirmed"
          : (d.geometry_candidate || d.document_detected ? "candidate" : "none");
        if (el("geometry")) el("geometry").textContent = (d.geometry_candidate || d.document_detected) ? "candidate" : "none";
        if (el("confirmed")) el("confirmed").textContent = d.document_confirmed ? "yes" : "no";
        if (el("confirmreason")) el("confirmreason").textContent = d.confirmation_reason || "—";
        if (el("edgesupport")) el("edgesupport").textContent = (d.evidence_edge_sides ?? 0) + " sides / " + Number(d.evidence_edge_support || 0).toFixed(2);
        if (el("papercontrast")) el("papercontrast").textContent = Number(d.evidence_paper_contrast || 0).toFixed(2);
        if (el("printfold")) el("printfold").textContent = Number(d.evidence_print || 0).toFixed(2) + " / " + Number(d.evidence_fold || 0).toFixed(2);
        el("reject").textContent = d.reject_reason || "—";
        el("motion").textContent = Number(d.motion).toFixed(2);
        el("framemotion").textContent = Number(d.frame_motion || 0).toFixed(2);
        el("edge").textContent = Number(d.edge_density).toFixed(4);
        el("roi").textContent = (d.roi_width || 0) + " x " + (d.roi_height || 0);
        const fr = d.frame || {};
        const shape = fr.shape ? fr.shape.join("x") : ((d.frame_width || "—") + "x" + (d.frame_height || "—"));
        el("frame").textContent = shape + " " + (fr.dtype || "") + " " + (fr.channel_order || "");
        el("graystd").textContent = Number(d.gray_std || 0).toFixed(2);
        el("stable").textContent = (d.stable_count ?? 0) + " / " + (d.stable_required ?? 0);
        el("stabmotion").textContent = Number(d.stability_motion || 0).toFixed(2) + " / " + Number(d.stability_threshold || 0).toFixed(2);
        el("removal").textContent = (d.removal_detected ? "empty" : "document still present") + " (" + (d.removal_count ?? 0) + ")";
        el("af").textContent = (d.autofocus_method || "—") + " " + Number(d.focus_before || 0).toFixed(1) + "→" + Number(d.focus_after || 0).toFixed(1);
        el("focusroi").textContent = d.preview_focus_roi || d.focus_roi || "—";
        el("sensorroi").textContent = d.sensor_focus_roi || "—";
        el("docroi").textContent = d.document_roi || "—";
        el("lens").textContent = (d.lens_before == null ? "—" : Number(d.lens_before).toFixed(2)) + " → " + (d.lens_after == null ? "—" : Number(d.lens_after).toFixed(2));
        el("ratio").textContent = Number(d.improvement_ratio || 0).toFixed(2);
        el("reason").textContent = d.state_reason || "—";
        el("pipeline").textContent = "manual=" + (d.manual_capture_requested ? "yes" : "no") + " capture=" + (d.capture_started ? "yes" : "no") + " upload=" + (d.upload_started ? "yes" : "no") + " " + (d.upload_result || "—");
        el("fps").textContent = Number(d.mjpeg_fps != null ? d.mjpeg_fps : d.fps).toFixed(1);
        el("detecthz").textContent = Number(d.detect_hz || 0).toFixed(1);
        el("tickms").textContent = Number(d.tick_ms || 0).toFixed(0);
        const ts = d.timestamp ? new Date(d.timestamp * 1000).toISOString() : "—";
        el("timestamp").textContent = ts;
        const fd = d.focus || {};
        el("fmode").textContent = fd.focus_mode || "—";
        el("freq").textContent = fd.requested_lens == null ? "—" : Number(fd.requested_lens).toFixed(3);
        el("flens").textContent = fd.actual_lens == null ? "—" : Number(fd.actual_lens).toFixed(3);
        el("fstate").textContent = fd.af_state_name || "—";
        el("ffom").textContent = fd.focus_fom == null ? "—" : Number(fd.focus_fom).toFixed(0);
        el("fsharp").textContent = fd.document_sharpness == null ? "—" : Number(fd.document_sharpness).toFixed(1);
        el("froi").textContent = fd.focus_roi || "—";
        el("froisrc").textContent = fd.focus_roi_source || "—";
        el("frange").textContent = (fd.lens_min == null ? "—" : Number(fd.lens_min).toFixed(2)) + " .. " + (fd.lens_max == null ? "—" : Number(fd.lens_max).toFixed(2));
        if (!window._focusBusy) {
          const fm = el("focusmsg");
          fm.textContent = fd.message || "";
        }
        const btn = el("capture");
        btn.disabled = capturing || d.manual_capture_available === false;
        const box = el("manual");
        const live = d.manual_message || phaseLabel(d);
        if (capturing) {
          if (live && live !== "Idle") box.textContent = live;
        } else {
          box.textContent = live;
          box.className = "manual" + (d.manual_phase === "failed" ? " error" : d.manual_phase === "success" ? " ok" : "");
        }
      } catch (err) {
        el("camera").textContent = "unhealthy";
      }
    }
    el("capture").addEventListener("click", async () => {
      if (capturing) return;
      capturing = true;
      const btn = el("capture");
      const box = el("manual");
      btn.disabled = true;
      box.className = "manual";
      box.textContent = "Queued...";
      const poll = setInterval(refresh, 300);
      try {
        const r = await fetch("/preview/capture", { method: "POST", cache: "no-store" });
        const d = await r.json();
        if (d.ok) {
          box.className = "manual ok";
          box.textContent = "Success" + (d.capture_id ? " (" + d.capture_id + ")" : "");
        } else {
          box.className = "manual error";
          box.textContent = (d.status === "busy" ? "Busy" : "Failure") + ": " + (d.message || "unknown error");
        }
      } catch (err) {
        box.className = "manual error";
        box.textContent = "Failure: request failed";
      } finally {
        clearInterval(poll);
        capturing = false;
        refresh();
      }
    });
    async function postFocus(body) {
      window._focusBusy = true;
      const fm = el("focusmsg");
      fm.className = "manual";
      fm.textContent = "Working...";
      try {
        const r = await fetch("/preview/focus", {
          method: "POST",
          cache: "no-store",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body)
        });
        const d = await r.json();
        fm.className = d.ok ? "manual ok" : "manual error";
        fm.textContent = d.message || d.status || (d.ok ? "OK" : "Failed");
        refresh();
      } catch (err) {
        fm.className = "manual error";
        fm.textContent = "Focus request failed";
      } finally {
        window._focusBusy = false;
      }
    }
    document.querySelectorAll("[data-focus]").forEach((btn) => {
      btn.addEventListener("click", () => postFocus(JSON.parse(btn.getAttribute("data-focus"))));
    });
    window.addEventListener("keydown", (ev) => {
      if (ev.target && (ev.target.tagName === "INPUT" || ev.target.tagName === "TEXTAREA")) return;
      const k = ev.key;
      if (k === "ArrowLeft") { ev.preventDefault(); postFocus({ action: "nudge", delta: ev.shiftKey ? -0.1 : -0.01 }); }
      else if (k === "ArrowRight") { ev.preventDefault(); postFocus({ action: "nudge", delta: ev.shiftKey ? 0.1 : 0.01 }); }
      else if (k === "a" || k === "A") postFocus({ action: "autofocus" });
      else if (k === "l" || k === "L") postFocus({ action: "lock" });
      else if (k === "m" || k === "M") postFocus({ action: "manual" });
      else if (k === "s" || k === "S") postFocus({ action: "scan", step: 0.2 });
      else if (k === "t" || k === "T") postFocus({ action: "capture_test" });
    });
    refresh();
    setInterval(refresh, 500);
  </script>
</body>
</html>
"""


def _make_handler(viewer: LiveDebugViewer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/preview/live":
                self._serve_html()
            elif path == "/preview/live.mjpg":
                self._serve_mjpeg()
            elif path == "/preview/status":
                self._serve_status()
            else:
                self.send_error(404, "Not Found")

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/preview/capture":
                self._serve_capture()
            elif path == "/preview/focus":
                self._serve_focus()
            else:
                self.send_error(404, "Not Found")

        def _json_bytes(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_capture(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length > 0:
                remaining = min(length, 65536)
                self.rfile.read(remaining)
            handler = viewer.capture_handler
            if handler is None:
                self._json_bytes(
                    {"ok": False, "status": "unavailable", "message": "Manual capture is not attached"},
                    503,
                )
                return
            try:
                result = handler()
            except Exception:
                logger.exception("event=debug_manual_capture_failed")
                self._json_bytes(
                    {"ok": False, "status": "error", "message": "Capture request failed"},
                    500,
                )
                return
            if not isinstance(result, dict):
                self._json_bytes({"ok": False, "status": "error", "message": "Invalid capture result"}, 500)
                return
            safe = {
                "ok": bool(result.get("ok")),
                "status": str(result.get("status") or ""),
                "state": str(result.get("state") or ""),
                "message": str(result.get("message") or ""),
            }
            if result.get("capture_id"):
                safe["capture_id"] = str(result["capture_id"])
            code = 200
            if safe["status"] == "busy":
                code = 409
            elif safe["status"] == "timeout":
                code = 504
            elif not safe["ok"] and safe["status"] not in {"failed"}:
                code = 500
            self._json_bytes(safe, code)

        def _serve_focus(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            raw = b"{}"
            if length > 0:
                raw = self.rfile.read(min(length, 65536))
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            action = str(payload.get("action") or "status")
            handler = viewer.focus_handler
            if handler is None:
                self._json_bytes(
                    {"ok": False, "status": "unavailable", "message": "Manual focus is not attached"},
                    503,
                )
                return
            try:
                result = handler(action, payload)
            except Exception:
                logger.exception("event=debug_manual_focus_failed")
                self._json_bytes(
                    {"ok": False, "status": "error", "message": "Focus request failed"},
                    500,
                )
                return
            if not isinstance(result, dict):
                self._json_bytes({"ok": False, "status": "error", "message": "Invalid focus result"}, 500)
                return
            allowed = {
                "ok",
                "status",
                "message",
                "focus_mode",
                "requested_lens",
                "actual_lens",
                "af_state",
                "af_state_name",
                "af_mode",
                "focus_fom",
                "document_sharpness",
                "focus_roi",
                "focus_roi_source",
                "lens_min",
                "lens_max",
                "busy",
                "scan",
                "capture_test",
                "settle_duration_ms",
            }
            safe = {k: result[k] for k in allowed if k in result}
            if "ok" not in safe:
                safe["ok"] = bool(result.get("ok"))
            code = 200
            if result.get("status") == "busy":
                code = 409
            elif result.get("status") == "unavailable":
                code = 503
            elif not safe.get("ok") and result.get("status") == "error":
                code = 500
            self._json_bytes(safe, code)

        def _serve_html(self) -> None:
            body = _HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_status(self) -> None:
            payload = json.dumps(viewer.status_dict(), separators=(",", ":")).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _serve_mjpeg(self) -> None:
            viewer.client_connected(self.client_address)
            try:
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
                )
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.end_headers()
                last_seq = -1
                while viewer.is_running:
                    seq, jpeg = viewer.wait_jpeg(last_seq, timeout=1.0)
                    if jpeg is None or seq == last_seq:
                        continue
                    last_seq = seq
                    try:
                        self.wfile.write(
                            (
                                f"--{MJPEG_BOUNDARY}\r\n"
                                "Content-Type: image/jpeg\r\n"
                                f"Content-Length: {len(jpeg)}\r\n"
                                "\r\n"
                            ).encode("ascii")
                        )
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
                        break
                    except OSError:
                        break
            finally:
                viewer.client_disconnected(self.client_address)

    return Handler


class LiveDebugViewer:
    """Background JPEG encoder + HTTP observer. Never captures from the camera."""

    def __init__(self, config: DebugConfig) -> None:
        self.config = config
        self._meta_lock = threading.Lock()
        self._frame_seq = 0
        self._latest_frame: np.ndarray | None = None
        self._snapshot = DebugSnapshot()
        self._jpeg_buffer = LiveFrameBuffer()
        self._stop = threading.Event()
        self._encoder_thread: threading.Thread | None = None
        self._http_thread: threading.Thread | None = None
        self._server: ThreadingHTTPServer | None = None
        self._clients = 0
        self._clients_lock = threading.Lock()
        self._encode_times: deque[float] = deque(maxlen=12)
        self._actual_fps = 0.0
        self._client_log_times: deque[float] = deque(maxlen=_MAX_CLIENT_LOGS_PER_MINUTE)
        self._started = False
        self._frame_info: dict[str, Any] | None = None
        self._logged_encode_format = False
        self._capture_handler: Any = None
        self._focus_handler: Any = None

    @property
    def is_running(self) -> bool:
        return self._started and not self._stop.is_set()

    @property
    def capture_handler(self) -> Any:
        return self._capture_handler

    def set_capture_handler(self, handler: Any) -> None:
        self._capture_handler = handler

    @property
    def focus_handler(self) -> Any:
        return self._focus_handler

    def set_focus_handler(self, handler: Any) -> None:
        self._focus_handler = handler

    def apply_focus_debug(self, data: dict[str, Any]) -> None:
        with self._meta_lock:
            snap = self._snapshot
            snap.focus_mode = str(data.get("focus_mode") or "")
            snap.requested_lens = data.get("requested_lens")
            snap.actual_lens = data.get("actual_lens")
            snap.af_state_name = str(data.get("af_state_name") or "")
            snap.focus_fom = data.get("focus_fom")
            snap.document_sharpness = data.get("document_sharpness")
            snap.focus_roi_source = str(data.get("focus_roi_source") or "")
            snap.focus_roi_live = str(data.get("focus_roi") or "")
            snap.lens_min = data.get("lens_min")
            snap.lens_max = data.get("lens_max")
            snap.focus_message = str(data.get("message") or "")

    def start(self) -> None:
        if self._started:
            return
        handler = _make_handler(self)
        try:
            self._server = ThreadingHTTPServer((self.config.host, self.config.port), handler)
            self._server.daemon_threads = True
            self._server.timeout = 0.5
        except OSError as exc:
            logger.error(
                "event=debug_viewer_bind_failed host=%s port=%s error=%s",
                self.config.host,
                self.config.port,
                exc,
            )
            self._server = None
            return

        self._stop.clear()
        self._encoder_thread = threading.Thread(
            target=self._encoder_loop, name="idcam-debug-encoder", daemon=True
        )
        self._http_thread = threading.Thread(
            target=self._http_loop, name="idcam-debug-http", daemon=True
        )
        self._encoder_thread.start()
        self._http_thread.start()
        self._started = True
        logger.info(
            "Live Debug Viewer enabled: http://%s:%s/preview/live",
            self.config.host,
            self.config.port,
        )
        logger.info(
            "event=debug_viewer_started host=%s port=%s fps=%.1f overlay=%s",
            self.config.host,
            self.config.port,
            self.config.fps,
            self.config.overlay,
        )

    def stop(self) -> None:
        if not self._started and self._server is None:
            return
        self._stop.set()
        self._jpeg_buffer.notify_all()
        server = self._server
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                logger.debug("event=debug_viewer_shutdown_error", exc_info=True)
            try:
                server.server_close()
            except Exception:
                logger.debug("event=debug_viewer_close_error", exc_info=True)
        if self._http_thread is not None:
            self._http_thread.join(timeout=2.0)
        if self._encoder_thread is not None:
            self._encoder_thread.join(timeout=2.0)
        self._server = None
        self._started = False
        logger.info("event=debug_viewer_stopped")

    def publish(
        self,
        frame: np.ndarray | None = None,
        *,
        state: str,
        document_detected: bool,
        geometry_candidate: bool = False,
        document_confirmed: bool = False,
        confirmation_reason: str = "",
        evidence_edge_support: float = 0.0,
        evidence_edge_sides: int = 0,
        evidence_paper_contrast: float = 0.0,
        evidence_print: float = 0.0,
        evidence_line_structure: float = 0.0,
        evidence_fold: float = 0.0,
        motion: float,
        edge_density: float,
        motion_threshold: float,
        edge_density_threshold: float,
        stable_count: int,
        stable_required: int,
        stable: bool,
        camera_healthy: bool,
        autofocus_available: bool,
        camera_error: str | None = None,
        roi_center_fraction: float = 0.7,
        reject_reason: str = "",
        roi_width: int = 0,
        roi_height: int = 0,
        gray_std: float = 0.0,
        gray_mean: float = 0.0,
        frame_width: int = 0,
        frame_height: int = 0,
        frame_motion: float = 0.0,
        manual_capture_available: bool = False,
        manual_phase: str = "idle",
        manual_message: str = "",
        removal_detected: bool = False,
        removal_count: int = 0,
        autofocus_running: bool = False,
        autofocus_method: str = "",
        autofocus_attempt: int = 0,
        focus_before: float = 0.0,
        focus_after: float = 0.0,
        focus_roi: str = "",
        lens_before: float | None = None,
        lens_after: float | None = None,
        improvement_ratio: float = 0.0,
        preview_focus_roi: str = "",
        sensor_focus_roi: str = "",
        preview_roi_wh: str = "",
        document_roi: str = "",
        stability_motion: float = 0.0,
        stability_threshold: float = 0.0,
        state_reason: str = "",
        manual_capture_requested: bool = False,
        capture_started: bool = False,
        upload_started: bool = False,
        upload_result: str = "",
        detect_hz: float = 0.0,
        tick_ms: float = 0.0,
    ) -> None:
        """Store the latest preview copy and metadata. Never blocks on HTTP clients."""
        with self._meta_lock:
            prev = self._snapshot
            if frame is not None:
                self._latest_frame = frame.copy()
                self._frame_seq += 1
                info = describe_preview_frame(self._latest_frame)
                if self._latest_frame.ndim == 3:
                    info["channel_order"] = "RGB"
                elif self._latest_frame.ndim == 2:
                    info["channel_order"] = "GRAY"
                self._frame_info = info
            self._snapshot = DebugSnapshot(
                frame=None,
                timestamp=time.time(),
                state=state,
                document_detected=document_detected,
                geometry_candidate=geometry_candidate,
                document_confirmed=document_confirmed,
                confirmation_reason=confirmation_reason,
                evidence_edge_support=evidence_edge_support,
                evidence_edge_sides=evidence_edge_sides,
                evidence_paper_contrast=evidence_paper_contrast,
                evidence_print=evidence_print,
                evidence_line_structure=evidence_line_structure,
                evidence_fold=evidence_fold,
                motion=motion,
                edge_density=edge_density,
                motion_threshold=motion_threshold,
                edge_density_threshold=edge_density_threshold,
                stable_count=stable_count,
                stable_required=stable_required,
                stable=stable,
                camera_healthy=camera_healthy,
                autofocus_available=autofocus_available,
                camera_error=camera_error,
                roi_center_fraction=roi_center_fraction,
                waiting_for_camera=self._latest_frame is None,
                reject_reason=reject_reason,
                roi_width=roi_width,
                roi_height=roi_height,
                gray_std=gray_std,
                gray_mean=gray_mean,
                frame_width=frame_width,
                frame_height=frame_height,
                frame_motion=frame_motion,
                manual_capture_available=manual_capture_available,
                manual_phase=manual_phase,
                manual_message=manual_message,
                removal_detected=removal_detected,
                removal_count=removal_count,
                autofocus_running=autofocus_running,
                autofocus_method=autofocus_method,
                autofocus_attempt=autofocus_attempt,
                focus_before=focus_before,
                focus_after=focus_after,
                focus_roi=preview_focus_roi or focus_roi,
                lens_before=lens_before,
                lens_after=lens_after,
                improvement_ratio=improvement_ratio,
                preview_focus_roi=preview_focus_roi or focus_roi,
                sensor_focus_roi=sensor_focus_roi,
                preview_roi_wh=preview_roi_wh,
                document_roi=document_roi,
                stability_motion=stability_motion,
                stability_threshold=stability_threshold,
                state_reason=state_reason,
                manual_capture_requested=manual_capture_requested,
                capture_started=capture_started,
                upload_started=upload_started,
                upload_result=upload_result,
                detect_hz=detect_hz,
                tick_ms=tick_ms,
                focus_mode=prev.focus_mode,
                requested_lens=prev.requested_lens,
                actual_lens=prev.actual_lens,
                af_state_name=prev.af_state_name,
                focus_fom=prev.focus_fom,
                document_sharpness=prev.document_sharpness,
                focus_roi_source=prev.focus_roi_source,
                focus_roi_live=prev.focus_roi_live,
                lens_min=prev.lens_min,
                lens_max=prev.lens_max,
                focus_message=prev.focus_message,
            )

    def snapshot_meta(self) -> DebugSnapshot:
        with self._meta_lock:
            return self._snapshot

    def latest_frame_copy(self) -> np.ndarray | None:
        with self._meta_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def status_dict(self) -> dict[str, Any]:
        snap = self.snapshot_meta()
        with self._meta_lock:
            frame_info = dict(self._frame_info) if self._frame_info else None
        return {
            "enabled": True,
            "state": snap.state,
            "document_detected": snap.document_detected,
            "geometry_candidate": snap.geometry_candidate,
            "document_confirmed": snap.document_confirmed,
            "confirmation_reason": snap.confirmation_reason,
            "evidence_edge_support": round(snap.evidence_edge_support, 3),
            "evidence_edge_sides": snap.evidence_edge_sides,
            "evidence_paper_contrast": round(snap.evidence_paper_contrast, 3),
            "evidence_print": round(snap.evidence_print, 3),
            "evidence_line_structure": round(snap.evidence_line_structure, 3),
            "evidence_fold": round(snap.evidence_fold, 3),
            "motion": snap.motion,
            "edge_density": snap.edge_density,
            "motion_threshold": snap.motion_threshold,
            "edge_density_threshold": snap.edge_density_threshold,
            "stable_count": snap.stable_count,
            "stable_required": snap.stable_required,
            "stable": snap.stable,
            "camera_healthy": snap.camera_healthy,
            "autofocus_available": snap.autofocus_available,
            "fps": round(self._actual_fps or self.config.fps, 2),
            "mjpeg_fps": round(self._actual_fps or self.config.fps, 2),
            "detect_hz": round(snap.detect_hz, 2),
            "tick_ms": round(snap.tick_ms, 1),
            "timestamp": snap.timestamp,
            "frame": frame_info,
            "reject_reason": snap.reject_reason,
            "roi_width": snap.roi_width,
            "roi_height": snap.roi_height,
            "gray_std": snap.gray_std,
            "gray_mean": snap.gray_mean,
            "frame_width": snap.frame_width,
            "frame_height": snap.frame_height,
            "frame_motion": snap.frame_motion,
            "manual_capture_available": snap.manual_capture_available,
            "manual_phase": snap.manual_phase,
            "manual_message": snap.manual_message,
            "removal_detected": snap.removal_detected,
            "removal_count": snap.removal_count,
            "autofocus_running": snap.autofocus_running,
            "autofocus_method": snap.autofocus_method,
            "autofocus_attempt": snap.autofocus_attempt,
            "focus_before": snap.focus_before,
            "focus_after": snap.focus_after,
            "focus_roi": snap.preview_focus_roi or snap.focus_roi,
            "lens_before": snap.lens_before,
            "lens_after": snap.lens_after,
            "improvement_ratio": snap.improvement_ratio,
            "preview_focus_roi": snap.preview_focus_roi or snap.focus_roi,
            "sensor_focus_roi": snap.sensor_focus_roi,
            "preview_roi_wh": snap.preview_roi_wh,
            "document_roi": snap.document_roi,
            "stability_motion": snap.stability_motion,
            "stability_threshold": snap.stability_threshold,
            "state_reason": snap.state_reason,
            "manual_capture_requested": snap.manual_capture_requested,
            "capture_started": snap.capture_started,
            "upload_started": snap.upload_started,
            "upload_result": snap.upload_result,
            "focus": {
                "focus_mode": snap.focus_mode,
                "requested_lens": snap.requested_lens,
                "actual_lens": snap.actual_lens,
                "af_state_name": snap.af_state_name,
                "focus_fom": snap.focus_fom,
                "document_sharpness": snap.document_sharpness,
                "focus_roi": snap.focus_roi_live,
                "focus_roi_source": snap.focus_roi_source,
                "lens_min": snap.lens_min,
                "lens_max": snap.lens_max,
                "message": snap.focus_message,
            },
        }

    def wait_jpeg(self, last_seq: int, timeout: float) -> tuple[int, bytes | None]:
        seq, payload = self._jpeg_buffer.wait_latest(last_seq, timeout)
        jpeg = payload if isinstance(payload, (bytes, bytearray)) else None
        return seq, jpeg

    def client_connected(self, address: tuple[str, int]) -> None:
        with self._clients_lock:
            self._clients += 1
            count = self._clients
        self._log_client("event=debug_client_connected", address, count)

    def client_disconnected(self, address: tuple[str, int]) -> None:
        with self._clients_lock:
            self._clients = max(0, self._clients - 1)
            count = self._clients
        self._log_client("event=debug_client_disconnected", address, count)

    def _log_client(self, event: str, address: tuple[str, int], count: int) -> None:
        now = time.monotonic()
        recent = sum(1 for t in self._client_log_times if now - t < 60.0)
        if recent >= _MAX_CLIENT_LOGS_PER_MINUTE:
            logger.debug("%s clients=%d peer=%s", event, count, address[0])
        else:
            logger.info("%s clients=%d peer=%s", event, count, address[0])
        self._client_log_times.append(now)

    def _http_loop(self) -> None:
        server = self._server
        if server is None:
            return
        try:
            server.serve_forever(poll_interval=0.3)
        except Exception:
            if not self._stop.is_set():
                logger.exception("event=debug_http_server_failed")

    def _encoder_loop(self) -> None:
        interval = 1.0 / max(self.config.fps, 0.1)
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                jpeg = self._encode_latest()
                if jpeg is not None:
                    self._jpeg_buffer.publish(jpeg)
                    now = time.monotonic()
                    self._encode_times.append(now)
                    if len(self._encode_times) >= 2:
                        span = self._encode_times[-1] - self._encode_times[0]
                        if span > 0:
                            self._actual_fps = (len(self._encode_times) - 1) / span
            except Exception:
                logger.exception("event=debug_encode_failed")
            remaining = interval - (time.monotonic() - t0)
            self._stop.wait(timeout=max(0.0, remaining))

    def _encode_latest(self) -> bytes | None:
        with self._meta_lock:
            frame = self._latest_frame
            snap = self._snapshot
        if frame is None:
            message = "WAITING FOR CAMERA..."
            if snap.camera_error:
                message = "CAMERA ERROR"
            image = make_placeholder(message)
            if self.config.overlay:
                image = draw_overlay(image, snap, self._actual_fps or self.config.fps)
            else:
                image = resize_debug_frame(image, self.config.max_width)
            return encode_jpeg(image, self.config.jpeg_quality)

        if not self._logged_encode_format:
            info = describe_preview_frame(frame)
            logger.info(
                "event=preview_encode_format shape=%s dtype=%s means=%s assumed_order=RGB "
                "note=preview_contract_before_overlay_resize",
                info.get("shape"),
                info.get("dtype"),
                info.get("channel_means"),
            )
            self._logged_encode_format = True
        image = compose_debug_frame(
            frame,
            snap,
            self._actual_fps or self.config.fps,
            self.config.max_width,
            overlay=self.config.overlay,
        )
        return encode_jpeg(image, self.config.jpeg_quality)
