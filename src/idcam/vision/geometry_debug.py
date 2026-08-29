"""Rate-limited passport/geometry diagnostics. Does not change accept/reject rules."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from idcam.logging.setup import get_logger
from idcam.vision import geometry as geom

logger = get_logger("vision.geometry_debug")

_last_persist_at = 0.0
_last_log_at = 0.0


def geometry_debug_enabled() -> bool:
    flag = os.environ.get("IDCAM_GEOMETRY_DEBUG", "").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def geometry_debug_dir() -> Path:
    raw = os.environ.get("IDCAM_GEOMETRY_DEBUG_DIR", "").strip()
    if raw:
        return Path(raw)
    return Path(os.environ.get("TMP", os.environ.get("TEMP", "/tmp")))


def effective_geometry_config() -> dict[str, Any]:
    return {
        "WORKING_WIDTH": geom.WORKING_WIDTH,
        "MIN_AREA_FRACTION": geom.MIN_AREA_FRACTION,
        "MAX_AREA_FRACTION": geom.MAX_AREA_FRACTION,
        "MIN_ASPECT": geom.MIN_ASPECT,
        "MAX_ASPECT": geom.MAX_ASPECT,
        "MIN_RECTANGULARITY": geom.MIN_RECTANGULARITY,
        "MIN_CONFIDENCE": geom.MIN_CONFIDENCE,
        "MIN_RIGHT_ANGLE": geom.MIN_RIGHT_ANGLE,
        "MERGE_MIN_HEIGHT_RATIO": geom.MERGE_MIN_HEIGHT_RATIO,
        "MERGE_MIN_VERTICAL_OVERLAP": geom.MERGE_MIN_VERTICAL_OVERLAP,
        "MERGE_MAX_GAP_FRAC": geom.MERGE_MAX_GAP_FRAC,
        "MERGE_MIN_GAP_FRAC": geom.MERGE_MIN_GAP_FRAC,
        "MERGE_MIN_ASPECT": geom.MERGE_MIN_ASPECT,
        "MERGE_MAX_ASPECT": geom.MERGE_MAX_ASPECT,
        "MERGE_MIN_AREA": geom.MERGE_MIN_AREA,
        "LARGE_MIN_AREA": geom.LARGE_MIN_AREA,
        "LARGE_MAX_AREA": geom.LARGE_MAX_AREA,
        "LARGE_MIN_QUAD_ASPECT": geom.LARGE_MIN_QUAD_ASPECT,
        "LARGE_MAX_QUAD_ASPECT": geom.LARGE_MAX_QUAD_ASPECT,
        "SMALL_EVIDENCE_MIN_AREA": geom.SMALL_EVIDENCE_MIN_AREA,
        "geometry_module": getattr(geom, "__file__", None),
        "has_merged_two_page": hasattr(geom, "_try_merge_two_page"),
        "has_suppress_inner": hasattr(geom, "_suppress_inner_quads"),
        "has_large_document": hasattr(geom, "_build_large_document_candidate"),
    }


@dataclass
class GeometryTrace:
    src_wh: tuple[int, int] = (0, 0)
    work_wh: tuple[int, int] = (0, 0)
    scale: float = 1.0
    candidates: list[dict[str, Any]] = field(default_factory=list)
    merges: list[dict[str, Any]] = field(default_factory=list)
    suppressed: list[str] = field(default_factory=list)
    final: dict[str, Any] | None = None
    accepted_before_merge: int = 0
    _next: int = 1
    _bbox_ids: dict[tuple[int, int, int, int], str] = field(default_factory=dict)

    def next_id(self) -> str:
        cid = f"C{self._next}"
        self._next += 1
        return cid

    def remember(self, bbox: tuple[int, int, int, int], cid: str) -> None:
        self._bbox_ids[bbox] = cid

    def id_for(self, bbox: tuple[int, int, int, int]) -> str:
        return self._bbox_ids.get(bbox, "?")

    def record_candidate(self, record: dict[str, Any]) -> None:
        if len(self.candidates) >= 80:
            return
        self.candidates.append(record)
        bbox = record.get("bbox")
        if record.get("accepted") and isinstance(bbox, tuple):
            self.remember(bbox, record["candidate_id"])


def classify_failure(trace: GeometryTrace, *, detector_present: bool | None = None) -> str:
    """Map a single-frame trace to CASE A–G. Success is reported as G with note."""
    accepted = [c for c in trace.candidates if c.get("accepted")]
    rejected = [c for c in trace.candidates if not c.get("accepted")]
    merge_ok = [m for m in trace.merges if not m.get("merge_rejection_reason")]
    merge_fail = [m for m in trace.merges if m.get("merge_rejection_reason")]
    page_like = [
        c
        for c in accepted
        if 1.02 <= float(c.get("quad_aspect") or c.get("aspect_ratio") or 0) <= 1.85
        and float(c.get("area_ratio") or 0) >= 0.08
    ]
    wide_rejected = [
        c
        for c in rejected
        if float(c.get("aspect_ratio") or 0) >= 1.8 and float(c.get("area_ratio") or 0) >= 0.18
    ]
    hull_rejected = [
        c
        for c in rejected
        if c.get("rejection_reason") in {"not_quadrilateral", "low_rectangularity", "low_right_angle"}
        and float(c.get("area_ratio") or 0) >= 0.18
    ]
    merge_built_fail = [
        m
        for m in trace.merges
        if m.get("merge_rejection_reason")
        in {
            "aspect_out_of_range",
            "below_geometry_confidence",
            "low_rectangularity",
            "low_right_angle",
            "frame_aligned",
            "merge_geometry_rejected",
        }
    ]

    raw_ok = trace.final is not None
    if detector_present is False and raw_ok:
        return "D"
    if not raw_ok and not accepted:
        if hull_rejected or wide_rejected:
            return "A"
        if not rejected:
            return "F"
        if all(c.get("rejection_reason") in {"area_too_small", "perimeter_too_small"} for c in rejected):
            return "F"
        return "G"
    if len(page_like) >= 2 and not merge_ok and (merge_fail or not trace.merges):
        return "B"
    if merge_built_fail:
        return "C"
    if raw_ok and trace.final and trace.final.get("geometry_mode") != "merged_two_page" and len(page_like) >= 2:
        return "B"
    if raw_ok:
        return "G"
    return "G"


def render_overlay(frame: np.ndarray, trace: GeometryTrace) -> np.ndarray:
    vis = frame.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
    for rec in trace.candidates:
        bbox = rec.get("bbox")
        if not bbox:
            continue
        x, y, w, h = [int(v) for v in bbox]
        accepted = bool(rec.get("accepted"))
        color = (40, 200, 80) if accepted else (200, 60, 60)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)
        cid = rec.get("candidate_id", "?")
        if accepted:
            label = f"{cid} ACCEPT conf={float(rec.get('confidence') or 0):.2f}"
        else:
            label = f"{cid} REJECT {rec.get('rejection_reason')}"
        _put(vis, label, x, max(14, y - 4), color)
    for rec in trace.merges:
        a, b = rec.get("candidate_a"), rec.get("candidate_b")
        reason = rec.get("merge_rejection_reason") or "ok"
        cb = rec.get("combined_bbox")
        if cb:
            x, y, w, h = [int(v) for v in cb]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (40, 180, 220), 2)
            _put(
                vis,
                f"MERGE {a}+{b} gap={rec.get('horizontal_gap')} {reason}",
                x,
                min(vis.shape[0] - 8, y + h + 16),
                (40, 180, 220),
            )
    if trace.final and trace.final.get("bbox"):
        x, y, w, h = [int(v) for v in trace.final["bbox"]]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 3)
        mode = trace.final.get("geometry_mode")
        _put(vis, f"FINAL geometry_mode={mode} aspect={trace.final.get('oriented_aspect') or trace.final.get('quad_aspect')} rot={trace.final.get('rotation_angle')}", 12, 28, (255, 255, 255))
        corners = trace.final.get("quad")
        if corners is not None:
            pts = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)
            for i, (cx, cy) in enumerate(np.asarray(corners)):
                cv2.putText(
                    vis,
                    ("TL", "TR", "BR", "BL")[i],
                    (int(cx) + 6, int(cy) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
    case = classify_failure(trace)
    _put(vis, f"CASE {case}  accepted={sum(1 for c in trace.candidates if c.get('accepted'))}", 12, 56, (255, 220, 80))
    return vis


def _put(img: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def save_rgb_jpeg(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = frame
    if rgb.ndim == 2:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def log_trace(trace: GeometryTrace, *, case: str, extra: dict[str, Any] | None = None) -> None:
    global _last_log_at
    now = time.monotonic()
    if now - _last_log_at < 2.0:
        return
    _last_log_at = now
    payload = {
        "event": "document_detection_debug",
        "case": case,
        "accepted_count": sum(1 for c in trace.candidates if c.get("accepted")),
        "rejected_count": sum(1 for c in trace.candidates if not c.get("accepted")),
        "merge_count": len(trace.merges),
        "merge_ok": sum(1 for m in trace.merges if not m.get("merge_rejection_reason")),
        "final": trace.final,
        "candidates": trace.candidates[:40],
        "merges": trace.merges[:20],
        "suppressed": trace.suppressed,
        "config": effective_geometry_config(),
    }
    if extra:
        payload.update(extra)
    logger.info("event=document_detection_debug payload=%s", json.dumps(payload, default=str))


def diagnose_frame(frame: np.ndarray) -> GeometryTrace:
    """Run the production finder and collect per-candidate evidence. Same accept/reject rules."""
    trace = GeometryTrace()
    best, _count = geom.find_document_quad(frame, trace=trace)
    if best is not None:
        trace.final = {
            "geometry_mode": best.geometry_mode,
            "quad": best.corners.tolist(),
            "bbox": best.bbox,
            "area_ratio": best.area_ratio,
            "aspect_ratio": best.aspect_ratio,
            "confidence": best.confidence,
            "rejection_reason": "",
            "rectangularity": best.rectangularity,
            "border_distance": best.border_distance,
            "quad_aspect": best.quad_aspect,
            "edge_support": best.edge_support,
            "border_sides": best.border_sides,
            "perspective_score": best.perspective_score,
            "rotation_angle": best.rotation_angle,
            "recovered_corner_source": best.recovered_corner_source,
            "large_document_candidate": best.geometry_mode == "large_document_outer",
            "oriented_aspect": best.quad_aspect,
            "occupancy": best.area_ratio,
            "border_contacts": best.border_sides,
        }
    return trace


def maybe_persist_from_detector(frame: np.ndarray, geom_result: Any, detector: Any) -> None:
    """Called from detect_geometry when IDCAM_GEOMETRY_DEBUG is set. Rate-limited."""
    if not geometry_debug_enabled():
        return
    global _last_persist_at
    now = time.monotonic()
    if now - _last_persist_at < 2.0:
        return
    _last_persist_at = now
    trace = diagnose_frame(frame)
    case = classify_failure(trace, detector_present=bool(getattr(geom_result, "present", False)))
    out_dir = geometry_debug_dir()
    raw_path = out_dir / "idcam_passport_debug.jpg"
    overlay_path = out_dir / "idcam_passport_debug_overlay.jpg"
    save_rgb_jpeg(raw_path, frame)
    save_rgb_jpeg(overlay_path, render_overlay(frame, trace))
    extra = {
        "raw_detection": bool(trace.final),
        "smoothed_detection": bool(getattr(geom_result, "present", False)),
        "stable_frames": int(getattr(detector, "_geom_misses", 0)),
        "stability_reason": getattr(geom_result, "reason", ""),
        "hold_frames": int(getattr(getattr(detector, "config", None), "geometry_hold_frames", 0) or 0),
        "smooth_alpha": float(getattr(getattr(detector, "config", None), "geometry_smooth_alpha", 0) or 0),
        "debug_frame": str(raw_path),
        "debug_overlay": str(overlay_path),
    }
    log_trace(trace, case=case, extra=extra)
