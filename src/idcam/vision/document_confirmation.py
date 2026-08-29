"""Candidate-relative document confirmation (not center-ROI, not OCR).

Geometry may propose a quadrilateral. Auto-capture must not start unless this
module agrees the quadrilateral behaves like a document against the current
preview pixels.

All measurements are taken in the candidate's own polygon, anywhere in frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from idcam.logging.setup import get_logger
from idcam.vision.document_detector import DocumentGeometryResult
from idcam.vision.geometry import _fold_evidence, _is_full_frame_quad, order_corners

logger = get_logger("vision.document_confirmation")

CONFIRM_WORKING_WIDTH = 640
BOOKLET_MODES = frozenset({"merged_two_page", "large_document_outer"})

_MIN_SIDE_ON = 14.0
_MIN_SIDE_DELTA_OUT = 6.0
_MIN_SIDE_DELTA_IN = 3.0
_MIN_SIDES_NORMAL = 3
_MIN_SIDES_BOOKLET = 2
_MIN_PAPER_DELTA = 16.0
_MIN_PAPER_DELTA_BOOKLET = 12.0
_MIN_PAPER_NORM = 0.18
_FULL_FRAME_AREA = 0.94
_INTERIOR_INSET = 0.18


@dataclass(frozen=True)
class DocumentConfirmation:
    accepted: bool
    reason: str
    candidate_present: bool
    evidence_edge_support: float
    evidence_edge_sides: int
    evidence_edge_score: float
    evidence_paper_contrast: float
    evidence_interior_mean: float
    evidence_surround_mean: float
    evidence_interior_std: float
    evidence_surround_std: float
    evidence_print: float
    evidence_line_structure: float
    evidence_fold: float
    roi: tuple[int, int, int, int]
    geometry_mode: str
    geometry_aspect: float
    geometry_area_fraction: float


def _empty_confirmation(
    *,
    accepted: bool,
    reason: str,
    candidate_present: bool = False,
    geom: DocumentGeometryResult | None = None,
) -> DocumentConfirmation:
    bbox = (0, 0, 0, 0)
    mode = "none"
    aspect = 0.0
    area = 0.0
    if geom is not None:
        if geom.bbox is not None:
            bbox = geom.bbox
        mode = geom.geometry_mode
        aspect = float(geom.aspect_ratio)
        area = float(geom.selected_area_ratio)
    return DocumentConfirmation(
        accepted=accepted,
        reason=reason,
        candidate_present=candidate_present,
        evidence_edge_support=0.0,
        evidence_edge_sides=0,
        evidence_edge_score=0.0,
        evidence_paper_contrast=0.0,
        evidence_interior_mean=0.0,
        evidence_surround_mean=0.0,
        evidence_interior_std=0.0,
        evidence_surround_std=0.0,
        evidence_print=0.0,
        evidence_line_structure=0.0,
        evidence_fold=0.0,
        roi=bbox,
        geometry_mode=mode,
        geometry_aspect=aspect,
        geometry_area_fraction=area,
    )


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY if frame.shape[2] == 3 else cv2.COLOR_RGBA2GRAY)
    else:
        gray = np.ascontiguousarray(frame)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def _prepare(gray: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    h, w = gray.shape[:2]
    pts = order_corners(corners)
    if w <= CONFIRM_WORKING_WIDTH:
        return gray, pts, 1.0
    scale = CONFIRM_WORKING_WIDTH / float(w)
    work = cv2.resize(
        gray,
        (CONFIRM_WORKING_WIDTH, max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return work, (pts * scale).astype(np.float32), scale


def _sobel_mag(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return np.clip(mag, 0.0, 255.0)


def _sample_at(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    xs = np.clip(np.rint(points[:, 0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.rint(points[:, 1]).astype(np.int32), 0, h - 1)
    return image[ys, xs].astype(np.float32)


def _side_edge_scores(mag: np.ndarray, corners: np.ndarray) -> tuple[list[float], int, float]:
    """Per-side localized edge near the proposed polygon (not axis-aligned).

    Geometry quads (especially minAreaRect / paper_region) can sit a few pixels
    inside the true ink/paper boundary. Search a short normal band for the
    strongest edge, then compare that peak to farther inside/outside samples.
    """
    h, w = mag.shape[:2]
    pts = order_corners(corners)
    side_scores: list[float] = []
    supported = 0
    samples = 16
    for i in range(4):
        a, b = pts[i], pts[(i + 1) % 4]
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length < 8.0:
            side_scores.append(0.0)
            continue
        inward = np.array([-edge[1], edge[0]], dtype=np.float32) / length
        outward = -inward
        near = min(10.0, max(4.0, 0.025 * max(h, w)))
        far = min(22.0, max(10.0, 0.05 * max(h, w)))
        ts = np.linspace(0.12, 0.88, samples, dtype=np.float32)
        on_pts = a[None, :] + ts[:, None] * edge[None, :]
        peak = 0.0
        span = int(round(near))
        for delta in range(-span, span + 1, 1):
            band = on_pts + outward[None, :] * float(delta)
            valid = (band[:, 0] >= 0) & (band[:, 0] < w) & (band[:, 1] >= 0) & (band[:, 1] < h)
            if int(np.count_nonzero(valid)) >= 6:
                peak = max(peak, float(np.median(_sample_at(mag, band[valid]))))
        out_pts = on_pts + outward[None, :] * far
        in_pts = on_pts + inward[None, :] * far
        out_valid = (out_pts[:, 0] >= 0) & (out_pts[:, 0] < w) & (out_pts[:, 1] >= 0) & (out_pts[:, 1] < h)
        in_valid = (in_pts[:, 0] >= 0) & (in_pts[:, 0] < w) & (in_pts[:, 1] >= 0) & (in_pts[:, 1] < h)
        out_v = float(np.median(_sample_at(mag, out_pts[out_valid]))) if np.any(out_valid) else 0.0
        in_v = float(np.median(_sample_at(mag, in_pts[in_valid]))) if np.any(in_valid) else 0.0
        ok = peak >= _MIN_SIDE_ON and (peak - out_v) >= _MIN_SIDE_DELTA_OUT and (peak - in_v) >= _MIN_SIDE_DELTA_IN
        score = 0.0
        if ok:
            score = float(np.clip(0.5 * ((peak - out_v) / 40.0) + 0.5 * ((peak - in_v) / 40.0), 0.0, 1.0))
            supported += 1
        elif peak >= _MIN_SIDE_ON and (peak - max(out_v, in_v)) >= _MIN_SIDE_DELTA_OUT:
            # Clipped / one-sided comparison when the far band is off-frame or similar.
            score = float(np.clip((peak - max(out_v, in_v)) / 50.0, 0.0, 1.0))
            if score >= 0.12:
                supported += 1
        side_scores.append(score)
    mean_score = float(np.mean(side_scores)) if side_scores else 0.0
    return side_scores, supported, mean_score


def _quad_region_stats(gray: np.ndarray, corners: np.ndarray) -> tuple[float, float, np.ndarray]:
    pts = order_corners(corners)
    us = np.linspace(_INTERIOR_INSET, 1.0 - _INTERIOR_INSET, 12)
    vs = np.linspace(_INTERIOR_INSET, 1.0 - _INTERIOR_INSET, 12)
    grid = []
    for v in vs:
        top = (1.0 - v) * pts[0] + v * pts[3]
        bot = (1.0 - v) * pts[1] + v * pts[2]
        for u in us:
            p = (1.0 - u) * top + u * bot
            grid.append(p)
    samples = np.asarray(grid, dtype=np.float32)
    h, w = gray.shape[:2]
    inside = (samples[:, 0] >= 0) & (samples[:, 0] < w) & (samples[:, 1] >= 0) & (samples[:, 1] < h)
    vals = _sample_at(gray, samples[inside]) if np.any(inside) else np.array([0.0], dtype=np.float32)
    return float(np.mean(vals)), float(np.std(vals)), samples[inside]


def _surround_stats(gray: np.ndarray, corners: np.ndarray) -> tuple[float, float]:
    h, w = gray.shape[:2]
    pts = order_corners(corners)
    offset = min(22.0, max(8.0, 0.05 * max(h, w)))
    collected: list[np.ndarray] = []
    for i in range(4):
        a, b = pts[i], pts[(i + 1) % 4]
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length < 8.0:
            continue
        inward = np.array([-edge[1], edge[0]], dtype=np.float32) / length
        outward = -inward
        ts = np.linspace(0.12, 0.88, 14, dtype=np.float32)
        on_pts = a[None, :] + ts[:, None] * edge[None, :]
        for mul in (1.0, 1.6):
            out_pts = on_pts + outward[None, :] * (offset * mul)
            valid = (out_pts[:, 0] >= 1) & (out_pts[:, 0] < w - 1) & (out_pts[:, 1] >= 1) & (out_pts[:, 1] < h - 1)
            if np.any(valid):
                collected.append(out_pts[valid])
    if not collected:
        return 0.0, 0.0
    pts_out = np.concatenate(collected, axis=0)
    vals = _sample_at(gray, pts_out)
    return float(np.mean(vals)), float(np.std(vals))


def _print_and_lines(gray: np.ndarray, mag: np.ndarray, interior_pts: np.ndarray) -> tuple[float, float]:
    if interior_pts.size == 0:
        return 0.0, 0.0
    mag_v = _sample_at(mag, interior_pts)
    print_e = float(np.clip(np.mean(mag_v > 28.0), 0.0, 1.0))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gx_v = np.abs(_sample_at(gx, interior_pts))
    gy_v = np.abs(_sample_at(gy, interior_pts))
    line_e = float(np.clip(max(float(np.mean(gx_v > 22.0)), float(np.mean(gy_v > 22.0))), 0.0, 1.0))
    return print_e, line_e


def confirm_document(frame: np.ndarray, geom: DocumentGeometryResult) -> DocumentConfirmation:
    """Return whether ``geom`` is a document in ``frame`` (preview pixels)."""
    if not geom.present or geom.corners is None:
        return _empty_confirmation(accepted=False, reason="invalid_geometry", candidate_present=False, geom=geom)

    gray = _to_gray(frame)
    src_h, src_w = gray.shape[:2]
    corners = np.asarray(geom.corners, dtype=np.float32)
    if _is_full_frame_quad(corners, src_w, src_h) or float(geom.selected_area_ratio) >= _FULL_FRAME_AREA:
        return _empty_confirmation(accepted=False, reason="full_frame", candidate_present=True, geom=geom)

    work, work_pts, _scale = _prepare(gray, corners)
    mag = _sobel_mag(work)
    _side_scores, sides, edge_score = _side_edge_scores(mag, work_pts)
    edge_support = float(sides) / 4.0
    interior_mean, interior_std, interior_pts = _quad_region_stats(work, work_pts)
    surround_mean, surround_std = _surround_stats(work, work_pts)
    paper_delta = abs(interior_mean - surround_mean)
    paper_norm = float(np.clip(paper_delta / 40.0, 0.0, 1.0))
    print_e, line_e = _print_and_lines(work, mag, interior_pts)
    fold = _fold_evidence(work, work_pts)
    booklet = geom.geometry_mode in BOOKLET_MODES
    roi = geom.bbox if geom.bbox is not None else (0, 0, 0, 0)

    reason = "insufficient_combined_evidence"
    accepted = False
    min_sides = _MIN_SIDES_BOOKLET if booklet else _MIN_SIDES_NORMAL
    min_paper = _MIN_PAPER_DELTA_BOOKLET if booklet else _MIN_PAPER_DELTA

    if sides < min_sides:
        # Strong candidate-relative paper/table split can compensate for a
        # minAreaRect that only finds 2 clean sides (open booklet / paper_region).
        if paper_delta >= 28.0 and sides >= 2:
            reason = "accepted"
            accepted = True
        else:
            reason = "insufficient_edge_support"
    elif paper_delta < min_paper and paper_norm < _MIN_PAPER_NORM:
        reason = "insufficient_paper_contrast"
    elif booklet:
        fold_ok = fold >= 0.18 and sides >= 2 and paper_delta >= _MIN_PAPER_DELTA_BOOKLET
        outer_ok = sides >= 3 and paper_delta >= _MIN_PAPER_DELTA
        if fold_ok or outer_ok:
            accepted = True
            reason = "accepted"
        else:
            reason = "insufficient_combined_evidence"
    else:
        blank_ok = sides >= 3 and paper_delta >= _MIN_PAPER_DELTA
        printed_ok = sides >= 3 and paper_delta >= 12.0 and (print_e >= 0.10 or line_e >= 0.12)
        if blank_ok or printed_ok:
            accepted = True
            reason = "accepted"
        elif sides >= 3:
            reason = "insufficient_paper_contrast"
        else:
            reason = "weak_document_structure"

    return DocumentConfirmation(
        accepted=accepted,
        reason=reason,
        candidate_present=True,
        evidence_edge_support=edge_support,
        evidence_edge_sides=sides,
        evidence_edge_score=edge_score,
        evidence_paper_contrast=paper_norm,
        evidence_interior_mean=interior_mean,
        evidence_surround_mean=surround_mean,
        evidence_interior_std=interior_std,
        evidence_surround_std=surround_std,
        evidence_print=print_e,
        evidence_line_structure=line_e,
        evidence_fold=fold,
        roi=roi,
        geometry_mode=str(geom.geometry_mode),
        geometry_aspect=float(geom.aspect_ratio),
        geometry_area_fraction=float(geom.selected_area_ratio),
    )


class DocumentConfirmationTracker:
    """Require confirmation to hold across the geometry-stability window."""

    def __init__(self, frames_required: int) -> None:
        self.frames_required = max(1, int(frames_required))
        self.count = 0
        self.last = _empty_confirmation(accepted=False, reason="no_candidate")
        self._last_log_key: tuple | None = None

    def reset(self) -> None:
        self.count = 0
        self.last = _empty_confirmation(accepted=False, reason="no_candidate")
        self._last_log_key = None

    @property
    def temporally_confirmed(self) -> bool:
        return bool(self.last.accepted and self.count >= self.frames_required)

    def update(self, frame: np.ndarray, geom: DocumentGeometryResult) -> DocumentConfirmation:
        if not geom.present or geom.corners is None:
            self.count = 0
            self.last = _empty_confirmation(
                accepted=False, reason="invalid_geometry", candidate_present=False, geom=geom
            )
            self._log_if_changed(geom, self.last, candidate_event=False)
            return self.last
        result = confirm_document(frame, geom)
        if result.accepted:
            self.count += 1
        else:
            self.count = 0
        self.last = result
        self._log_if_changed(geom, result, candidate_event=True)
        return result

    def _log_if_changed(
        self, geom: DocumentGeometryResult, result: DocumentConfirmation, *, candidate_event: bool
    ) -> None:
        key = (
            int(result.candidate_present),
            int(result.accepted),
            result.reason,
            result.geometry_mode,
            result.roi,
        )
        if key == self._last_log_key:
            return
        self._last_log_key = key
        bbox = result.roi
        if candidate_event:
            logger.info(
                "event=document_candidate candidate_present=1 geometry_mode=%s "
                "geometry_aspect=%.3f geometry_area_fraction=%.3f roi_x=%d roi_y=%d "
                "roi_width=%d roi_height=%d",
                result.geometry_mode,
                result.geometry_aspect,
                result.geometry_area_fraction,
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
            )
        status = "accepted" if result.accepted else "rejected"
        logger.info(
            "event=document_confirmation result=%s reason=%s document_confirmed=%s "
            "candidate_present=%s confirmation_reason=%s evidence_edge_support=%.3f "
            "evidence_edge_sides=%d evidence_edge_score=%.3f evidence_paper_contrast=%.3f "
            "evidence_interior_mean=%.1f evidence_surround_mean=%.1f "
            "evidence_interior_std=%.1f evidence_surround_std=%.1f evidence_print=%.3f "
            "evidence_line_structure=%.3f evidence_fold=%.3f roi_x=%d roi_y=%d "
            "roi_width=%d roi_height=%d geometry_mode=%s geometry_aspect=%.3f "
            "geometry_area_fraction=%.3f",
            status,
            result.reason,
            int(result.accepted),
            int(result.candidate_present),
            result.reason,
            result.evidence_edge_support,
            result.evidence_edge_sides,
            result.evidence_edge_score,
            result.evidence_paper_contrast,
            result.evidence_interior_mean,
            result.evidence_surround_mean,
            result.evidence_interior_std,
            result.evidence_surround_std,
            result.evidence_print,
            result.evidence_line_structure,
            result.evidence_fold,
            bbox[0],
            bbox[1],
            bbox[2],
            bbox[3],
            result.geometry_mode,
            result.geometry_aspect,
            result.geometry_area_fraction,
        )
