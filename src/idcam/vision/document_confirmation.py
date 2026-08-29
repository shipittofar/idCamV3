"""Optional candidate-relative document evidence (debug / offline analysis).

Auto-capture is gated by geometry + GeometryStability only. This module is not
a hard auto-capture blocker. ``confirm_document`` remains available for debug
scripts and unit tests of evidence metrics.

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

# Match geometry working width so confirmation does not re-blur 1920→640.
CONFIRM_WORKING_WIDTH = 1280
BOOKLET_MODES = frozenset({"merged_two_page", "large_document_outer"})

_MIN_SIDE_ON = 14.0
_MIN_SIDE_DELTA_OUT = 6.0
_MIN_SIDE_DELTA_IN = 3.0
_MIN_SIDES_NORMAL = 3
_MIN_SIDES_BOOKLET = 2
_TWO_SIDE_MEAN_SCORE = 0.28
_MIN_BAND_SAMPLES = 5
_MIN_PAPER_DELTA = 16.0
_MIN_PAPER_DELTA_BOOKLET = 12.0
_MIN_PAPER_NORM = 0.18
_FULL_FRAME_AREA = 0.94
_INTERIOR_INSET = 0.18
_BORDER_PX = 8.0
_BORDER_CONTACT_FRAC = 0.40
_OUTWARD_INFRAME_MIN = 0.30
_ON_SIDE_INFRAME_MIN = 0.35
_LARGE_BORDER_MIN_OCC = 0.28
_LARGE_BORDER_MAX_OCC = 0.92
_LARGE_BORDER_MIN_CONF = 0.70
_LARGE_BORDER_MIN_GEOM_EDGE = 0.32
_LARGE_BORDER_MIN_PAPER_NORM = 0.30
_SIDE_NAMES = ("top", "right", "bottom", "left")


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
    edge_support_top: float = 0.0
    edge_support_right: float = 0.0
    edge_support_bottom: float = 0.0
    edge_support_left: float = 0.0
    edge_support_mean: float = 0.0
    edge_support_min: float = 0.0
    working_width: int = 0
    working_height: int = 0
    edge_band_width_px: float = 0.0
    edge_band_inner_offset_px: float = 0.0
    edge_band_outer_offset_px: float = 0.0
    observed_side_count: int = 0
    occluded_side_count: int = 0
    edge_support_observable_mean: float = 0.0
    geometry_confidence: float = 0.0
    geometry_edge_support: float = 0.0
    side_observable_frac: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    side_valid_samples: tuple[int, int, int, int] = (0, 0, 0, 0)
    side_border_occluded: tuple[int, int, int, int] = (0, 0, 0, 0)


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
    # LINEAR keeps Sobel peaks; AREA at 1920→640 was anti-aliasing the boundary away.
    interp = cv2.INTER_AREA if scale <= 0.5 else cv2.INTER_LINEAR
    work = cv2.resize(
        gray,
        (CONFIRM_WORKING_WIDTH, max(1, int(round(h * scale)))),
        interpolation=interp,
    )
    return work, (pts * scale).astype(np.float32), scale


def _median_valid(image: np.ndarray, points: np.ndarray) -> tuple[float | None, int]:
    """Median of in-frame samples only. Does not clip OOB points onto the border."""
    h, w = image.shape[:2]
    valid = (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
    n = int(np.count_nonzero(valid))
    if n < _MIN_BAND_SAMPLES:
        return None, n
    xs = np.clip(np.rint(points[valid, 0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.rint(points[valid, 1]).astype(np.int32), 0, h - 1)
    return float(np.median(image[ys, xs].astype(np.float32))), n


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


@dataclass
class SideEval:
    name: str
    score: float
    observable_fraction: float
    valid_sample_count: int
    border_occluded: bool
    status: str


@dataclass
class EdgeBandResult:
    scores: list[float]
    supported: int
    mean: float
    band_width: float
    inner_offset: float
    outer_offset: float
    side_lengths: list[float]
    sides: list[SideEval]
    observable_count: int
    occluded_count: int
    observable_mean: float


def _in_frame_mask(points: np.ndarray, width: int, height: int) -> np.ndarray:
    return (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)


def _border_contact_fraction(a: np.ndarray, b: np.ndarray, width: int, height: int, *, px: float, samples: int = 20) -> float:
    ts = np.linspace(0.05, 0.95, samples, dtype=np.float32)
    pts = a[None, :] + ts[:, None] * (b - a)[None, :]
    dist = np.minimum.reduce(
        [pts[:, 0], pts[:, 1], (width - 1) - pts[:, 0], (height - 1) - pts[:, 1]]
    )
    return float(np.mean(dist <= px))


def _side_edge_scores(mag: np.ndarray, corners: np.ndarray) -> EdgeBandResult:
    """Per-side localized *boundary* support (not generic texture energy).

    Border-occluded sides are unevaluable (not score 0). Scores and means use
    observable sides only.
    """
    h_m, w_m = mag.shape[:2]
    pts = order_corners(corners)
    evals: list[SideEval] = []
    lengths: list[float] = []
    near_acc: list[float] = []
    far_acc: list[float] = []
    samples = 20
    for i in range(4):
        name = _SIDE_NAMES[i]
        a, b = pts[i], pts[(i + 1) % 4]
        edge = b - a
        length = float(np.linalg.norm(edge))
        lengths.append(length)
        if length < 8.0:
            evals.append(SideEval(name, 0.0, 0.0, 0, True, "border_occluded"))
            continue
        inward = np.array([-edge[1], edge[0]], dtype=np.float32) / length
        outward = -inward
        near = float(np.clip(0.03 * length, 4.0, 20.0))
        far_cap = min(22.0, 0.08 * float(min(h_m, w_m)), 0.18 * length)
        far = float(np.clip(max(0.06 * length, 1.6 * near), 8.0, max(8.0, far_cap)))
        near_acc.append(near)
        far_acc.append(far)
        ts = np.linspace(0.15, 0.85, samples, dtype=np.float32)
        on_pts = a[None, :] + ts[:, None] * edge[None, :]
        on_mask = _in_frame_mask(on_pts, w_m, h_m)
        on_frac = float(np.mean(on_mask)) if on_mask.size else 0.0
        out_pts = on_pts + outward[None, :] * far
        out_frac = float(np.mean(_in_frame_mask(out_pts, w_m, h_m)))
        contact = _border_contact_fraction(a, b, w_m, h_m, px=_BORDER_PX)
        occluded = contact >= _BORDER_CONTACT_FRAC or on_frac < _ON_SIDE_INFRAME_MIN
        if occluded:
            evals.append(
                SideEval(name, 0.0, on_frac, int(np.count_nonzero(on_mask)), True, "border_occluded")
            )
            continue
        status = "fully_observable" if on_frac >= 0.85 and out_frac >= 0.6 else "partially_observable"
        peak = 0.0
        peak_n = 0
        span = max(1, int(round(near)))
        for delta in range(-span, span + 1, 1):
            band = on_pts + outward[None, :] * float(delta)
            med, n = _median_valid(mag, band)
            if med is not None:
                peak = max(peak, med)
                peak_n = max(peak_n, n)
        if peak_n < _MIN_BAND_SAMPLES:
            evals.append(SideEval(name, 0.0, on_frac, peak_n, True, "border_occluded"))
            continue
        out_v, n_out = _median_valid(mag, out_pts)
        in_v, n_in = _median_valid(mag, on_pts + inward[None, :] * far)
        out_ok = out_v is not None
        in_ok = in_v is not None
        score = 0.0
        ok = False
        if peak >= _MIN_SIDE_ON:
            if out_ok and in_ok:
                d_out = peak - float(out_v)
                d_in = peak - float(in_v)
                ok = (d_out >= _MIN_SIDE_DELTA_OUT and d_in >= _MIN_SIDE_DELTA_IN) or (
                    d_in >= _MIN_SIDE_DELTA_OUT and d_out >= 0.0
                )
                if ok:
                    score = float(np.clip(0.5 * (max(d_out, 0.0) / 40.0) + 0.5 * (max(d_in, 0.0) / 40.0), 0.0, 1.0))
            elif out_ok and not in_ok:
                d_out = peak - float(out_v)
                ok = d_out >= _MIN_SIDE_DELTA_OUT
                if ok:
                    score = float(np.clip(d_out / 50.0, 0.0, 1.0))
            elif in_ok and not out_ok:
                d_in = peak - float(in_v)
                ok = d_in >= _MIN_SIDE_DELTA_OUT
                if ok:
                    score = float(np.clip(d_in / 50.0, 0.0, 1.0))
        if ok and score >= 0.08:
            evals.append(SideEval(name, score, on_frac, peak_n, False, status))
        else:
            evals.append(SideEval(name, 0.0, on_frac, peak_n, False, status))
    scores = [s.score for s in evals]
    observable = [s for s in evals if not s.border_occluded]
    supported = sum(1 for s in observable if s.score >= 0.08)
    obs_mean = float(np.mean([s.score for s in observable])) if observable else 0.0
    return EdgeBandResult(
        scores=scores,
        supported=supported,
        mean=obs_mean,
        band_width=float(np.mean(near_acc)) if near_acc else 0.0,
        inner_offset=float(np.mean(far_acc)) if far_acc else 0.0,
        outer_offset=float(np.mean(far_acc)) if far_acc else 0.0,
        side_lengths=lengths,
        sides=evals,
        observable_count=len(observable),
        occluded_count=sum(1 for s in evals if s.border_occluded),
        observable_mean=obs_mean,
    )


def _quad_region_stats(gray: np.ndarray, corners: np.ndarray) -> tuple[float, float, np.ndarray]:
    pts = order_corners(corners)
    mean_side = 0.25 * sum(float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4))
    inset = float(np.clip(6.0 / max(mean_side, 1.0), 0.10, 0.22))
    us = np.linspace(inset, 1.0 - inset, 12)
    vs = np.linspace(inset, 1.0 - inset, 12)
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
    collected: list[np.ndarray] = []
    for i in range(4):
        a, b = pts[i], pts[(i + 1) % 4]
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length < 8.0:
            continue
        outward = np.array([edge[1], -edge[0]], dtype=np.float32) / length
        offset = float(np.clip(0.08 * length, 6.0, 28.0))
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


def _max_corner_inset(corners: np.ndarray, width: int, height: int) -> float:
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    dist = np.minimum.reduce(
        [pts[:, 0], pts[:, 1], (width - 1) - pts[:, 0], (height - 1) - pts[:, 1]]
    )
    return float(np.max(dist))


def _side_tuple(sides: list[SideEval], attr: str) -> tuple[float, float, float, float]:
    vals = [float(getattr(s, attr)) for s in sides[:4]]
    vals.extend([0.0] * (4 - len(vals)))
    return (vals[0], vals[1], vals[2], vals[3])


def _side_int_tuple(sides: list[SideEval], attr: str) -> tuple[int, int, int, int]:
    vals = [int(getattr(s, attr)) for s in sides[:4]]
    vals.extend([0] * (4 - len(vals)))
    return (vals[0], vals[1], vals[2], vals[3])


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
    edge = _side_edge_scores(mag, work_pts)
    scores = (edge.scores + [0.0, 0.0, 0.0, 0.0])[:4]
    sides = edge.supported
    edge_score = edge.observable_mean
    n_obs = edge.observable_count
    n_occ = edge.occluded_count
    edge_support = float(sides) / float(max(n_obs, 1)) if n_obs else 0.0
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
        two_side_ok = (
            n_obs >= 2
            and sides >= 2
            and paper_delta >= min_paper
            and edge_score >= _TWO_SIDE_MEAN_SCORE
        )
        strong_paper = paper_delta >= 28.0 and sides >= 2 and n_obs >= 2
        inset = _max_corner_inset(corners, src_w, src_h)
        large_border_ok = (
            n_occ >= 2
            and sides >= 1
            and n_obs <= 2
            and inset >= 16.0
            and _LARGE_BORDER_MIN_OCC <= float(geom.selected_area_ratio) <= _LARGE_BORDER_MAX_OCC
            and int(geom.border_sides) >= 2
            and float(geom.confidence) >= _LARGE_BORDER_MIN_CONF
            and (
                float(geom.edge_support) >= _LARGE_BORDER_MIN_GEOM_EDGE
                or geom.geometry_mode in {"paper_region", "large_document_outer", "merged_two_page"}
            )
            and paper_norm >= _LARGE_BORDER_MIN_PAPER_NORM
            and paper_delta >= min_paper
        )
        if two_side_ok or strong_paper or large_border_ok:
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
        edge_support_top=float(scores[0]),
        edge_support_right=float(scores[1]),
        edge_support_bottom=float(scores[2]),
        edge_support_left=float(scores[3]),
        edge_support_mean=float(edge_score),
        edge_support_min=float(min(scores)) if scores else 0.0,
        working_width=int(work.shape[1]),
        working_height=int(work.shape[0]),
        edge_band_width_px=float(edge.band_width),
        edge_band_inner_offset_px=float(edge.inner_offset),
        edge_band_outer_offset_px=float(edge.outer_offset),
        observed_side_count=int(n_obs),
        occluded_side_count=int(n_occ),
        edge_support_observable_mean=float(edge.observable_mean),
        geometry_confidence=float(geom.confidence),
        geometry_edge_support=float(geom.edge_support),
        side_observable_frac=_side_tuple(edge.sides, "observable_fraction"),
        side_valid_samples=_side_int_tuple(edge.sides, "valid_sample_count"),
        side_border_occluded=_side_int_tuple(edge.sides, "border_occluded"),
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
            "geometry_area_fraction=%.3f geometry_rotation_deg=%.1f geometry_border_sides=%d "
            "working_resolution=%dx%d edge_support_top=%.3f edge_support_right=%.3f "
            "edge_support_bottom=%.3f edge_support_left=%.3f edge_support_mean=%.3f "
            "edge_support_min=%.3f supported_sides=%d interior_surround=%.3f "
            "band_width_px=%.1f inner_offset_px=%.1f outer_offset_px=%.1f "
            "observed_sides=%d occluded_sides=%d observable_mean=%.3f "
            "geometry_confidence=%.3f geometry_edge_support=%.3f "
            "side_occluded=%s",
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
            float(getattr(geom, "rotation_deg", 0.0) or 0.0),
            int(getattr(geom, "border_sides", 0) or 0),
            result.working_width,
            result.working_height,
            result.edge_support_top,
            result.edge_support_right,
            result.edge_support_bottom,
            result.edge_support_left,
            result.edge_support_mean,
            result.edge_support_min,
            result.evidence_edge_sides,
            result.evidence_paper_contrast,
            result.edge_band_width_px,
            result.edge_band_inner_offset_px,
            result.edge_band_outer_offset_px,
            result.observed_side_count,
            result.occluded_side_count,
            result.edge_support_observable_mean,
            result.geometry_confidence,
            result.geometry_edge_support,
            ",".join(str(v) for v in result.side_border_occluded),
        )
        logger.debug(
            "event=document_confirmation_edges reason=%s top=%.3f right=%.3f bottom=%.3f left=%.3f "
            "mean=%.3f min=%.3f sides=%d band=%.1f inner=%.1f outer=%.1f work=%dx%d",
            result.reason,
            result.edge_support_top,
            result.edge_support_right,
            result.edge_support_bottom,
            result.edge_support_left,
            result.edge_support_mean,
            result.edge_support_min,
            result.evidence_edge_sides,
            result.edge_band_width_px,
            result.edge_band_inner_offset_px,
            result.edge_band_outer_offset_px,
            result.working_width,
            result.working_height,
        )
