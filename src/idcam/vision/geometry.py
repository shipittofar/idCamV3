"""Document quadrilateral geometry from a preview frame.

Coordinates are always in the input image's pixel space. Internal work may
use a smaller image; corners are scaled back before return.

Confidence is geometric, not semantic:

    0.35 * rectangularity   contour area / min-area-rect area (1 = rectangle)
    0.25 * area_term        occupancy-neutral (small through very large)
    0.25 * right_angle_term mean |cos| of adjacent edges near 0
    0.15 * aspect_term      oriented opposite-side aspect, no preferred ratio

Motion vs an empty baseline is not used.
The AABB is a convenience box only; ranking and aspect use the perspective quad.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from idcam.logging.setup import get_logger

logger = get_logger("vision.geometry")

# Working width for contour search. 1920×1080 is scaled down then mapped back.
# 1280 keeps Canny/Sobel edges usable on Pi 5 without a full-preview contour pass.
WORKING_WIDTH = 1280

MIN_AREA_FRACTION = 0.08
MAX_AREA_FRACTION = 0.97
MIN_ASPECT = 0.40
MAX_ASPECT = 3.35
MIN_RECTANGULARITY = 0.72
MIN_CONFIDENCE = 0.48
MIN_RIGHT_ANGLE = 0.55  # 1 - mean|cos|; 90° edges → 1.0

# Extra evidence path only. Does not change MIN_AREA_FRACTION for the normal gate.
SMALL_EVIDENCE_MIN_AREA = 0.030
SMALL_MIN_REL_SIDE = 0.09
SMALL_MIN_DIAG_FRAC = 0.14
WORKING_WIDTH_REFINE = 1600

# Two-page merge: opened passport / booklet. Not a generic blob union.
MERGE_MIN_HEIGHT_RATIO = 0.82
MERGE_MIN_VERTICAL_OVERLAP = 0.78
MERGE_MAX_GAP_FRAC = 0.12
MERGE_MIN_GAP_FRAC = -0.08
MERGE_MIN_ASPECT = 1.45
MERGE_MAX_ASPECT = 3.45
MERGE_MIN_AREA = 0.16

# Dedicated open-passport / large-document path. Does not change the constants above.
LARGE_MIN_AREA = 0.12
LARGE_MAX_AREA = 0.92
LARGE_MIN_QUAD_ASPECT = 1.12
LARGE_MAX_QUAD_ASPECT = 4.20
LARGE_MIN_EDGE_SUPPORT = 0.36
LARGE_MIN_RIGHT = 0.42
LARGE_MIN_CONFIDENCE = 0.42
LARGE_MIN_SIDE_CONSISTENCY = 0.52
LARGE_MIN_FOLD = 0.20
LARGE_STRONG_EDGE = 0.52

_last_reject_log_at = 0.0


@dataclass(frozen=True)
class QuadCandidate:
    corners: np.ndarray  # (4, 2) float32, TL, TR, BR, BL in source pixels
    bbox: tuple[int, int, int, int]
    confidence: float
    area_ratio: float
    aspect_ratio: float
    rectangularity: float
    right_angle: float
    reason: str
    geometry_mode: str = "quad"
    border_distance: float = 0.0
    edge_support: float = 0.0
    quad_aspect: float = 0.0
    border_sides: int = 0
    rotation_angle: float = 0.0
    perspective_score: float = 0.0
    recovered_corner_source: str = ""


def order_corners(points: np.ndarray) -> np.ndarray:
    """Order four points as a cyclic perspective quad (TL, TR, BR, BL when upright).

    Primary order is angular around the centroid (winding), not independent
    ``x+y`` / ``x-y`` argmin/argmax which can pair the wrong vertices at ~45°.
    The start vertex is the highest point (then leftmost) so an axis-aligned
    rectangle still maps to TL, TR, BR, BL.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    rel = pts - center
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    ordered = pts[np.argsort(angles)]
    if float(cv2.contourArea(ordered.reshape(-1, 1, 2))) < 0:
        ordered = ordered[::-1]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    return np.roll(ordered, -start, axis=0)


def bbox_from_corners(corners: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    x0 = int(np.floor(float(pts[:, 0].min())))
    y0 = int(np.floor(float(pts[:, 1].min())))
    x1 = int(np.ceil(float(pts[:, 0].max())))
    y1 = int(np.ceil(float(pts[:, 1].max())))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def clip_corners(corners: np.ndarray, width: int, height: int) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2).copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, max(0, width - 1))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(0, height - 1))
    return pts


def _border_distance(corners: np.ndarray, width: int, height: int) -> float:
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    d = np.minimum.reduce(
        [
            pts[:, 0],
            pts[:, 1],
            (width - 1) - pts[:, 0],
            (height - 1) - pts[:, 1],
        ]
    )
    return float(np.min(d))


def _quad_edge_lengths(corners: np.ndarray) -> tuple[float, float, float, float]:
    pts = order_corners(corners)
    top = float(np.linalg.norm(pts[1] - pts[0]))
    right = float(np.linalg.norm(pts[2] - pts[1]))
    bottom = float(np.linalg.norm(pts[2] - pts[3]))
    left = float(np.linalg.norm(pts[3] - pts[0]))
    return top, bottom, left, right


def _quad_aspect_from_sides(corners: np.ndarray) -> float:
    """Perspective-aware aspect: max(avg width, avg height) / min(...). Always >= 1."""
    top, bottom, left, right = _quad_edge_lengths(corners)
    width_est = 0.5 * (top + bottom)
    height_est = 0.5 * (left + right)
    short = min(width_est, height_est)
    if short < 1.0:
        return 0.0
    return float(max(width_est, height_est) / short)


def _side_consistency(corners: np.ndarray) -> float:
    top, bottom, left, right = _quad_edge_lengths(corners)
    tb = min(top, bottom) / max(top, bottom, 1.0)
    lr = min(left, right) / max(left, right, 1.0)
    return float(min(tb, lr))


def _border_sides(corners: np.ndarray, width: int, height: int, *, px: float = 4.0) -> tuple[int, list[str]]:
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    names: list[str] = []
    if x0 <= px:
        names.append("left")
    if x1 >= width - 1 - px:
        names.append("right")
    if y0 <= px:
        names.append("top")
    if y1 >= height - 1 - px:
        names.append("bottom")
    return len(names), names


def _rotation_angle_deg(corners: np.ndarray) -> float:
    pts = order_corners(corners)
    v = pts[1] - pts[0]
    return float(np.degrees(np.arctan2(float(v[1]), float(v[0]))))


def _perspective_score(corners: np.ndarray) -> float:
    cons = _side_consistency(corners)
    right = _right_angle_score(corners)
    return float(np.clip(0.55 * cons + 0.45 * right, 0.0, 1.0))


def _edge_support_along_quad(edge: np.ndarray, corners: np.ndarray, *, samples: int = 28, band: int = 2) -> float:
    if edge.size == 0:
        return 0.0
    h, w = edge.shape[:2]
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    hits = 0
    total = 0
    for i in range(4):
        a, b = pts[i], pts[(i + 1) % 4]
        side_len = float(np.linalg.norm(b - a))
        side_band = max(band, int(round(0.025 * side_len)))
        for t in np.linspace(0.06, 0.94, samples):
            p = a * (1.0 - t) + b * t
            x, y = int(round(float(p[0]))), int(round(float(p[1])))
            x0, x1 = max(0, x - side_band), min(w, x + side_band + 1)
            y0, y1 = max(0, y - side_band), min(h, y + side_band + 1)
            total += 1
            if x1 > x0 and y1 > y0 and int(edge[y0:y1, x0:x1].max()) > 0:
                hits += 1
    return float(hits / total) if total else 0.0


def _fold_evidence(gray: np.ndarray, corners: np.ndarray) -> float:
    """Dark band near the page midline (fold), in the recovered quad's coordinates."""
    if gray.size == 0:
        return 0.0
    pts = order_corners(corners)
    h, w = gray.shape[:2]
    left_mid = 0.5 * (pts[0] + pts[3])
    right_mid = 0.5 * (pts[1] + pts[2])
    top_mid = 0.5 * (pts[0] + pts[1])
    bot_mid = 0.5 * (pts[3] + pts[2])
    interior: list[float] = []
    fold_line: list[float] = []
    for t in np.linspace(0.12, 0.88, 24):
        for s in (0.22, 0.78):
            p = (1 - s) * ((1 - t) * left_mid + t * right_mid) + s * ((1 - t) * top_mid + t * bot_mid)
            x, y = int(round(float(p[0]))), int(round(float(p[1])))
            if 0 <= x < w and 0 <= y < h:
                interior.append(float(gray[y, x]))
        q = (1 - t) * left_mid + t * right_mid
        # vertical booklet fold: sample along height at horizontal center
        c = (1 - t) * top_mid + t * bot_mid
        for p in (q, c):
            x, y = int(round(float(p[0]))), int(round(float(p[1])))
            if 0 <= x < w and 0 <= y < h:
                fold_line.append(float(gray[y, x]))
    if len(interior) < 6 or len(fold_line) < 6:
        return 0.0
    delta = float(np.median(interior) - np.median(fold_line))
    return float(np.clip(delta / 40.0, 0.0, 1.0))


def _is_full_frame_quad(corners: np.ndarray, width: int, height: int, *, px: float = 4.0) -> bool:
    """True when the quad is the image rectangle (all four sides), not a clipped page.

    A passport that hits the top and bottom of a 16:9 preview has four corners on
    the border; rejecting that as ``frame_aligned`` was the Pi CASE A false reject.
    """
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    spans_w = x0 <= px and x1 >= width - 1 - px
    spans_h = y0 <= px and y1 >= height - 1 - px
    return bool(spans_w and spans_h)


def _right_angle_score(corners: np.ndarray) -> float:
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    cos_abs = []
    for i in range(4):
        prev_p = pts[(i - 1) % 4]
        cur = pts[i]
        nxt = pts[(i + 1) % 4]
        v1 = prev_p - cur
        v2 = nxt - cur
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1.0 or n2 < 1.0:
            return 0.0
        cos_abs.append(abs(float(np.dot(v1, v2) / (n1 * n2))))
    return float(max(0.0, 1.0 - float(np.mean(cos_abs))))


def _aspect_of_quad(corners: np.ndarray) -> float:
    """Oriented aspect: max(opposite-side mean) / min(...). Independent of AABB."""
    return _quad_aspect_from_sides(corners)


def _mean_diagonal(corners: np.ndarray) -> float:
    pts = order_corners(corners)
    d1 = float(np.linalg.norm(pts[2] - pts[0]))
    d2 = float(np.linalg.norm(pts[3] - pts[1]))
    return 0.5 * (d1 + d2)


def _area_term(area_ratio: float) -> float:
    """Occupancy is not a size preference: small through very large pages score similarly.

    Near-zero blobs and near-full-frame patches stay low so ranking cannot
    prefer the camera rectangle.
    """
    if area_ratio <= 0.0:
        return 0.0
    if area_ratio < 0.025:
        return float(np.clip(area_ratio / 0.025, 0.0, 1.0)) * 0.35
    if area_ratio > 0.94:
        return float(np.clip((0.97 - area_ratio) / 0.03, 0.0, 1.0)) * 0.45
    if area_ratio > 0.90:
        return 0.70 + 0.22 * float((0.94 - area_ratio) / 0.04)
    return 0.92


def _aspect_term(aspect: float) -> float:
    """No preferred ID vs booklet ratio. Any recoverable oriented aspect in range."""
    if aspect <= 0:
        return 0.0
    oriented = aspect if aspect >= 1.0 else (1.0 / aspect)
    if oriented > MAX_ASPECT:
        return 0.12
    return 0.90


def score_quad(
    corners: np.ndarray,
    contour_area: float,
    frame_area: float,
    min_rect_area: float,
) -> tuple[float, float, float, float]:
    rectangularity = float(contour_area / min_rect_area) if min_rect_area > 1 else 0.0
    rectangularity = float(np.clip(rectangularity, 0.0, 1.0))
    area_ratio = float(contour_area / frame_area) if frame_area > 0 else 0.0
    aspect = _aspect_of_quad(corners)
    right = _right_angle_score(corners)
    confidence = (
        0.35 * rectangularity
        + 0.25 * _area_term(area_ratio)
        + 0.25 * right
        + 0.15 * _aspect_term(aspect)
    )
    return float(np.clip(confidence, 0.0, 1.0)), area_ratio, aspect, rectangularity


def _illumination_normalize(gray: np.ndarray) -> np.ndarray:
    sigma = max(5.0, 0.04 * max(gray.shape[0], gray.shape[1]))
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    blur = np.maximum(blur, 1)
    norm = cv2.divide(gray, blur, scale=128)
    return norm


def _edge_maps(gray: np.ndarray) -> list[np.ndarray]:
    """Candidate binary maps. Edges are a feature, not presence."""
    maps: list[np.ndarray] = []
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    maps.append(cv2.Canny(blurred, 40, 120))
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    maps.append((mag > 28).astype(np.uint8) * 255)
    return maps


def _paper_mask(gray: np.ndarray) -> np.ndarray:
    """Filled paper vs table using unnormalized gray.

    Illumination normalization flattens page/table brightness, so Canny-only
    search sees hollow outlines. A vertically clipped passport never closes.
    The darkest of the four border strips is the table reference so a document
    that hits top/bottom still leaves left/right table samples (and vice versa).
    """
    h, w = gray.shape[:2]
    mx = max(6, w // 80)
    my = max(6, h // 80)
    meds = [
        float(np.median(gray[:, :mx])),
        float(np.median(gray[:, w - mx :])),
        float(np.median(gray[:my, :])),
        float(np.median(gray[h - my :, :])),
    ]
    ref = min(meds)
    thr = ref + 28.0
    mask = (gray.astype(np.float32) > thr).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    k_fold = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_fold, iterations=1)
    return mask


def _approx_quad(contour: np.ndarray, *, allow_min_rect: bool) -> tuple[np.ndarray | None, str, int]:
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    if peri < 8:
        return None, "perimeter_too_small", 0
    n_pts = 0
    for eps in (0.02, 0.03, 0.04, 0.055, 0.07):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        n_pts = int(len(approx))
        if n_pts == 4:
            return approx.reshape(4, 2).astype(np.float32), "quad", 4
    if allow_min_rect:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.float32)
        return box, "min_rect_fallback", 4
    return None, "not_quadrilateral", n_pts


def _maybe_log_reject(
    *,
    reason: str,
    area_ratio: float,
    bbox: tuple[int, int, int, int] | None,
    aspect_ratio: float,
    rectangularity: float,
    corner_count: int,
    right_angle: float,
    border_distance: float,
) -> None:
    global _last_reject_log_at
    now = time.monotonic()
    if now - _last_reject_log_at < 1.0:
        return
    _last_reject_log_at = now
    logger.debug(
        "event=geometry_candidate_rejected rejection_reason=%s candidate_area_ratio=%.3f "
        "candidate_bbox=%s candidate_aspect_ratio=%.3f candidate_rectangularity=%.3f "
        "candidate_corner_count=%d candidate_edge_score=%.3f candidate_border_distance=%.1f",
        reason,
        area_ratio,
        bbox,
        aspect_ratio,
        rectangularity,
        corner_count,
        right_angle,
        border_distance,
    )


def _build_candidate(
    corners_work: np.ndarray,
    contour_area: float,
    work_area: float,
    min_rect_area: float,
    scale: float,
    src_width: int,
    src_height: int,
    geometry_mode: str,
    *,
    log_reject: bool,
    trace: object | None = None,
    candidate_id: str = "",
    small_evidence: bool = False,
    edge_map: np.ndarray | None = None,
) -> QuadCandidate | None:
    corners_work = order_corners(corners_work)
    conf, area_ratio, aspect, rectangularity = score_quad(
        corners_work, contour_area, work_area, min_rect_area
    )
    right = _right_angle_score(corners_work)
    corners = corners_work / float(scale)
    corners = clip_corners(corners, src_width, src_height)
    bbox = bbox_from_corners(corners)
    border = _border_distance(corners, src_width, src_height)
    n_sides, side_names = _border_sides(corners, src_width, src_height, px=4.0)
    quad_aspect = _quad_aspect_from_sides(corners)
    rot = _rotation_angle_deg(corners)
    persp = _perspective_score(corners_work)
    edge_s = _edge_support_along_quad(edge_map, corners_work) if edge_map is not None else 0.0
    cid = candidate_id or (trace.next_id() if trace is not None else "")  # type: ignore[union-attr]

    def _emit(accepted: bool, reason: str, **extra: object) -> None:
        if log_reject and not accepted:
            _maybe_log_reject(
                reason=reason,
                area_ratio=area_ratio,
                bbox=bbox,
                aspect_ratio=aspect,
                rectangularity=rectangularity,
                corner_count=4,
                right_angle=right,
                border_distance=border,
            )
        if trace is None:
            return
        rec = {
            "candidate_id": cid,
            "bbox": bbox,
            "x": bbox[0],
            "y": bbox[1],
            "width": bbox[2],
            "height": bbox[3],
            "area": float(contour_area),
            "area_ratio": area_ratio,
            "aspect_ratio": aspect,
            "rectangularity": rectangularity,
            "corner_count": 4,
            "confidence": conf,
            "border_distance": border,
            "rejection_reason": "" if accepted else reason,
            "accepted": accepted,
            "geometry_mode": geometry_mode,
            "candidate_edge_score": right,
            "quad_aspect": quad_aspect,
            "quad_area_ratio": area_ratio,
            "border_contact": n_sides,
            "border_sides": side_names,
            "rotation_angle": rot,
            "perspective_score": persp,
            "edge_support": edge_s,
            "large_document_candidate": False,
            "recovered_quad": corners.tolist(),
        }
        rec.update(extra)
        trace.record_candidate(rec)  # type: ignore[union-attr]

    def _reject(reason: str) -> None:
        _emit(False, reason)

    if quad_aspect > MAX_ASPECT or quad_aspect < 1.0:
        _reject("aspect_out_of_range")
        return None
    if rectangularity < MIN_RECTANGULARITY:
        _reject("low_rectangularity")
        return None
    if right < MIN_RIGHT_ANGLE:
        _reject("low_right_angle")
        return None
    if conf < MIN_CONFIDENCE:
        _reject("below_geometry_confidence")
        return None
    if _is_full_frame_quad(corners, src_width, src_height, px=4.0):
        _reject("frame_aligned")
        return None
    if small_evidence:
        top, bottom, left, right_len = _quad_edge_lengths(corners)
        min_side = min(top, bottom, left, right_len)
        min_dim = float(min(src_width, src_height))
        frame_diag = float(np.hypot(src_width, src_height))
        diag = _mean_diagonal(corners)
        if min_side < SMALL_MIN_REL_SIDE * min_dim and diag < SMALL_MIN_DIAG_FRAC * frame_diag:
            _reject("small_noise")
            return None
        if edge_map is not None and edge_s < 0.40:
            _reject("small_weak_edge")
            return None
    _emit(True, "")
    return QuadCandidate(
        corners=corners,
        bbox=bbox,
        confidence=conf,
        area_ratio=area_ratio,
        aspect_ratio=quad_aspect,
        rectangularity=rectangularity,
        right_angle=right,
        reason=geometry_mode,
        geometry_mode=geometry_mode,
        border_distance=border,
        edge_support=edge_s,
        quad_aspect=quad_aspect,
        border_sides=n_sides,
        rotation_angle=rot,
        perspective_score=persp,
    )


def _build_large_document_candidate(
    corners_work: np.ndarray,
    contour_area: float,
    work_area: float,
    min_rect_area: float,
    scale: float,
    src_width: int,
    src_height: int,
    work_gray: np.ndarray,
    edge_map: np.ndarray,
    recovered_source: str,
    *,
    log_reject: bool,
    trace: object | None = None,
    candidate_id: str = "",
) -> QuadCandidate | None:
    """Accept a recovered outer quad when raw contour rectangularity is low.

    Normal ``MIN_RECTANGULARITY`` is not used as a hard gate. Full-frame and
    weak-evidence blobs are still rejected.
    """
    corners_work = order_corners(corners_work)
    corners = clip_corners(corners_work / float(scale), src_width, src_height)
    bbox = bbox_from_corners(corners)
    border = _border_distance(corners, src_width, src_height)
    n_sides, side_names = _border_sides(corners, src_width, src_height, px=4.0)
    quad_aspect = _quad_aspect_from_sides(corners)
    right = _right_angle_score(corners_work)
    cons = _side_consistency(corners_work)
    edge_s = _edge_support_along_quad(edge_map, corners_work)
    fold = _fold_evidence(work_gray, corners_work)
    persp = _perspective_score(corners_work)
    rot = _rotation_angle_deg(corners)
    raw_rect = float(np.clip(contour_area / min_rect_area if min_rect_area > 1 else 0.0, 0.0, 1.0))
    quad_poly = float(cv2.contourArea(corners_work.astype(np.float32)))
    mr = cv2.minAreaRect(corners_work.astype(np.float32))
    mra = max(1.0, float(mr[1][0]) * float(mr[1][1]))
    quad_rect = float(np.clip(quad_poly / mra, 0.0, 1.0))
    frame_area = float(max(src_width * src_height, 1))
    quad_area_ratio = float(cv2.contourArea(corners.astype(np.float32)) / frame_area)
    wh_aspect = _aspect_of_quad(corners_work)
    conf = float(
        np.clip(
            0.28 * right
            + 0.22 * edge_s
            + 0.18 * _area_term(quad_area_ratio)
            + 0.14 * _aspect_term(wh_aspect)
            + 0.10 * fold
            + 0.08 * quad_rect,
            0.0,
            1.0,
        )
    )
    cid = candidate_id or (trace.next_id() if trace is not None else "")  # type: ignore[union-attr]
    extras = {
        "large_document_candidate": True,
        "recovered_corner_source": recovered_source,
        "recovered_quad": corners.tolist(),
        "quad_aspect": quad_aspect,
        "quad_area_ratio": quad_area_ratio,
        "edge_support": edge_s,
        "border_contact": n_sides,
        "border_sides": side_names,
        "perspective_score": persp,
        "rotation_angle": rot,
        "outer_quad_reconstruction": True,
        "fold_evidence": fold,
        "raw_rectangularity": raw_rect,
        "side_consistency": cons,
    }

    def _emit(accepted: bool, reason: str) -> None:
        if log_reject and not accepted:
            _maybe_log_reject(
                reason=reason,
                area_ratio=quad_area_ratio,
                bbox=bbox,
                aspect_ratio=quad_aspect,
                rectangularity=raw_rect,
                corner_count=4,
                right_angle=right,
                border_distance=border,
            )
        if trace is None:
            return
        rec = {
            "candidate_id": cid,
            "bbox": bbox,
            "x": bbox[0],
            "y": bbox[1],
            "width": bbox[2],
            "height": bbox[3],
            "area": float(contour_area),
            "area_ratio": quad_area_ratio,
            "aspect_ratio": wh_aspect,
            "rectangularity": raw_rect,
            "corner_count": 4,
            "confidence": conf,
            "border_distance": border,
            "rejection_reason": "" if accepted else reason,
            "accepted": accepted,
            "geometry_mode": "large_document_outer",
            "candidate_edge_score": right,
            "large_document_rejection_reason": "" if accepted else reason,
        }
        rec.update(extras)
        trace.record_candidate(rec)  # type: ignore[union-attr]

    if _is_full_frame_quad(corners, src_width, src_height, px=6.0) or n_sides >= 4:
        _emit(False, "frame_aligned")
        return None
    if n_sides >= 3 and fold < LARGE_MIN_FOLD:
        _emit(False, "frame_aligned")
        return None
    if quad_area_ratio > LARGE_MAX_AREA and n_sides >= 3:
        _emit(False, "frame_aligned")
        return None
    if quad_area_ratio < LARGE_MIN_AREA:
        _emit(False, "large_area_too_small")
        return None
    if quad_area_ratio > LARGE_MAX_AREA:
        _emit(False, "large_area_too_large")
        return None
    if quad_aspect < LARGE_MIN_QUAD_ASPECT or quad_aspect > LARGE_MAX_QUAD_ASPECT:
        _emit(False, "large_aspect_out_of_range")
        return None
    top, bottom, left, right_len = _quad_edge_lengths(corners)
    width_est = 0.5 * (top + bottom)
    height_est = 0.5 * (left + right_len)
    min_span = min(width_est, height_est)
    max_span = max(width_est, height_est)
    min_dim = float(min(src_width, src_height))
    max_dim = float(max(src_width, src_height))
    # Occupancy already gated; do not require a preferred fraction of the frame axes
    # (that rejected rotated / off-center large pages). Thin noise is still rejected.
    if min_span < 0.12 * min_dim:
        _emit(False, "large_too_thin")
        return None
    if max_span < 0.18 * max_dim and quad_area_ratio < 0.28:
        _emit(False, "large_span_too_small")
        return None
    if fold < LARGE_MIN_FOLD and quad_aspect < 1.55:
        _emit(False, "large_aspect_out_of_range")
        return None
    if cons < LARGE_MIN_SIDE_CONSISTENCY:
        _emit(False, "large_side_inconsistent")
        return None
    if right < LARGE_MIN_RIGHT:
        _emit(False, "large_low_right_angle")
        return None
    if edge_s < LARGE_MIN_EDGE_SUPPORT:
        _emit(False, "large_weak_edge_support")
        return None
    if raw_rect < MIN_RECTANGULARITY and fold < LARGE_MIN_FOLD and edge_s < LARGE_STRONG_EDGE:
        _emit(False, "large_low_evidence")
        return None
    if conf < LARGE_MIN_CONFIDENCE:
        _emit(False, "large_below_confidence")
        return None
    _emit(True, "")
    return QuadCandidate(
        corners=corners,
        bbox=bbox,
        confidence=conf,
        area_ratio=quad_area_ratio,
        aspect_ratio=quad_aspect,
        rectangularity=max(raw_rect, quad_rect),
        right_angle=right,
        reason="large_document_outer",
        geometry_mode="large_document_outer",
        border_distance=border,
        edge_support=edge_s,
        quad_aspect=quad_aspect,
        border_sides=n_sides,
        rotation_angle=rot,
        perspective_score=persp,
        recovered_corner_source=recovered_source,
    )


def _candidate_from_contour(
    contour: np.ndarray,
    work_shape: tuple[int, int],
    scale: float,
    src_width: int,
    src_height: int,
    *,
    log_reject: bool = False,
    trace: object | None = None,
    force_min_rect: bool = False,
    area_from_hull: bool = False,
    geometry_mode: str | None = None,
    work_gray: np.ndarray | None = None,
    edge_map: np.ndarray | None = None,
) -> QuadCandidate | None:
    fh, fw = int(work_shape[0]), int(work_shape[1])
    work_area = float(fh * fw)
    area = float(cv2.contourArea(contour))
    if area_from_hull:
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        if hull_area > area:
            area = hull_area
    area_frac = area / work_area if work_area else 0.0

    def _early(reason: str, *, corners_n: int = 0, bbox=None, aspect=0.0, rectangularity=0.0, right=0.0) -> None:
        if log_reject:
            _maybe_log_reject(
                reason=reason,
                area_ratio=area_frac,
                bbox=bbox,
                aspect_ratio=aspect,
                rectangularity=rectangularity,
                corner_count=corners_n,
                right_angle=right,
                border_distance=0.0,
            )
        if trace is None or area_frac < 0.04:
            return
        cid = trace.next_id()  # type: ignore[union-attr]
        rec = {
            "candidate_id": cid,
            "bbox": bbox,
            "x": None if bbox is None else bbox[0],
            "y": None if bbox is None else bbox[1],
            "width": None if bbox is None else bbox[2],
            "height": None if bbox is None else bbox[3],
            "area": area,
            "area_ratio": area_frac,
            "aspect_ratio": aspect,
            "rectangularity": rectangularity,
            "corner_count": corners_n,
            "confidence": 0.0,
            "border_distance": 0.0,
            "rejection_reason": reason,
            "accepted": False,
            "geometry_mode": "",
            "candidate_edge_score": right,
        }
        trace.record_candidate(rec)  # type: ignore[union-attr]

    small_evidence = False
    if area_frac < MIN_AREA_FRACTION:
        if area_frac < SMALL_EVIDENCE_MIN_AREA:
            _early("area_too_small")
            return None
        small_evidence = True
    if area_frac > MAX_AREA_FRACTION:
        _early("area_too_large")
        return None
    allow_min_rect = force_min_rect or (SMALL_EVIDENCE_MIN_AREA <= area_frac <= 0.90)
    quad, mode, n_pts = _approx_quad(contour, allow_min_rect=allow_min_rect)
    recovered = "hull_approx" if mode == "quad" else "min_rect"
    if quad is None and area_frac >= LARGE_MIN_AREA:
        quad, mode, n_pts = _approx_quad(contour, allow_min_rect=True)
        recovered = "min_rect"
    if quad is None:
        _early(mode, corners_n=n_pts)
        return None
    hull_quad = cv2.convexHull(quad.astype(np.float32))
    if len(hull_quad) != 4:
        if area_frac >= LARGE_MIN_AREA:
            quad, mode, n_pts = _approx_quad(contour, allow_min_rect=True)
            recovered = "min_rect"
            hull_quad = cv2.convexHull(quad.astype(np.float32)) if quad is not None else hull_quad
        if quad is None or len(hull_quad) != 4:
            _early("hull_not_quad", corners_n=int(len(hull_quad)))
            return None
    rect = cv2.minAreaRect(contour)
    rw, rh = float(rect[1][0]), float(rect[1][1])
    min_rect_area = max(1.0, rw * rh)
    if geometry_mode:
        geom_mode = geometry_mode
    else:
        geom_mode = "min_rect_fallback" if mode == "min_rect_fallback" else "quad"
    cid = trace.next_id() if trace is not None else ""  # type: ignore[union-attr]
    normal = _build_candidate(
        order_corners(quad),
        area,
        work_area,
        min_rect_area,
        scale,
        src_width,
        src_height,
        geom_mode,
        log_reject=log_reject,
        trace=trace,
        candidate_id=cid,
        small_evidence=small_evidence,
        edge_map=edge_map,
    )
    if normal is not None:
        return normal
    if work_gray is None or edge_map is None:
        return None
    large_id = trace.next_id() if trace is not None else ""  # type: ignore[union-attr]
    return _build_large_document_candidate(
        order_corners(quad),
        area,
        work_area,
        min_rect_area,
        scale,
        src_width,
        src_height,
        work_gray,
        edge_map,
        recovered,
        log_reject=log_reject,
        trace=trace,
        candidate_id=large_id,
    )


def _bbox_containment_frac(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> float:
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    x0, y0 = max(ix, ox), max(iy, oy)
    x1, y1 = min(ix + iw, ox + ow), min(iy + ih, oy + oh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    inner_area = max(1, iw * ih)
    return float(inter) / float(inner_area)


def _suppress_inner_quads(candidates: list[QuadCandidate], trace: object | None = None) -> list[QuadCandidate]:
    """Prefer the outer document boundary over text blocks and single pages inside it."""
    if len(candidates) < 2:
        return candidates
    keep = [True] * len(candidates)
    for i, inner in enumerate(candidates):
        for j, outer in enumerate(candidates):
            if i == j:
                continue
            if outer.geometry_mode == "large_document_outer" and inner.geometry_mode in {
                "quad",
                "min_rect_fallback",
                "paper_region",
            }:
                if inner.rectangularity >= MIN_RECTANGULARITY and inner.area_ratio >= SMALL_EVIDENCE_MIN_AREA:
                    continue
            if (
                outer.geometry_mode not in {"merged_two_page", "large_document_outer", "paper_region"}
                and outer.area_ratio < inner.area_ratio * 1.75
            ):
                continue
            if _bbox_containment_frac(inner.bbox, outer.bbox) >= 0.88:
                keep[i] = False
                break
    if trace is not None:
        for c, ok in zip(candidates, keep):
            if not ok:
                trace.suppressed.append(trace.id_for(c.bbox))  # type: ignore[union-attr]
    return [c for c, ok in zip(candidates, keep) if ok]


def _centroid(corners: np.ndarray) -> np.ndarray:
    return order_corners(corners).mean(axis=0)


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    lo = max(min(a0, a1), min(b0, b1))
    hi = min(max(a0, a1), max(b0, b1))
    return max(0.0, hi - lo)


def _try_merge_oriented(
    a: QuadCandidate,
    b: QuadCandidate,
    work_area: float,
    scale: float,
    src_width: int,
    src_height: int,
    trace: object | None,
    id_a: str,
    id_b: str,
) -> QuadCandidate | None:
    """Merge two adjacent page quads using centroids and projections, not AABB order."""
    ca = _centroid(a.corners)
    cb = _centroid(b.corners)
    delta = cb - ca
    dist = float(np.linalg.norm(delta))
    if dist < 8.0:
        return None
    direction = delta / dist
    perp = np.array([-direction[1], direction[0]], dtype=np.float32)

    def _proj_span(cand: QuadCandidate, axis: np.ndarray) -> tuple[float, float, float]:
        pts = np.asarray(cand.corners, dtype=np.float32)
        proj = pts @ axis
        return float(proj.min()), float(proj.max()), float(proj.max() - proj.min())

    a0, a1, aw = _proj_span(a, direction)
    b0, b1, bw = _proj_span(b, direction)
    gap = min(abs(a1 - b0), abs(b1 - a0), abs(a0 - b1), abs(b0 - a1))
    # If intervals overlap on the connecting axis, gap is negative-like; use edge distance.
    overlap_dir = _interval_overlap(a0, a1, b0, b1)
    if overlap_dir <= 1.0:
        gap = min(abs(a1 - b0), abs(b1 - a0))
    else:
        gap = -overlap_dir
    min_width = min(aw, bw)
    max_gap = MERGE_MAX_GAP_FRAC * min_width
    min_gap = MERGE_MIN_GAP_FRAC * min_width
    pa0, pa1, ah = _proj_span(a, perp)
    pb0, pb1, bh = _proj_span(b, perp)
    overlap_p = _interval_overlap(pa0, pa1, pb0, pb1)
    height_ratio = min(ah, bh) / max(ah, bh, 1.0)
    vert_overlap = overlap_p / min(ah, bh) if min(ah, bh) else 0.0
    area_ratio = min(a.area_ratio, b.area_ratio) / max(a.area_ratio, b.area_ratio, 1e-6)

    def _merge_fail(reason: str, *, score: float = 0.0) -> None:
        if trace is None:
            return
        trace.merges.append(  # type: ignore[union-attr]
            {
                "candidate_a": id_a,
                "candidate_b": id_b,
                "horizontal_gap": gap,
                "vertical_overlap": vert_overlap,
                "height_ratio": height_ratio,
                "combined_bbox": None,
                "merge_score": score,
                "merge_rejection_reason": reason,
            }
        )

    if area_ratio < 0.45:
        _merge_fail("page_size")
        return None
    if height_ratio < MERGE_MIN_HEIGHT_RATIO:
        _merge_fail("height_ratio")
        return None
    if vert_overlap < MERGE_MIN_VERTICAL_OVERLAP:
        _merge_fail("vertical_overlap")
        return None
    if gap > max_gap or gap < min_gap:
        _merge_fail("centroid_gap")
        return None
    hull = cv2.convexHull(np.vstack([a.corners, b.corners]).astype(np.float32))
    peri = cv2.arcLength(hull, True)
    merged = None
    for eps in (0.02, 0.03, 0.045, 0.06):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            merged = order_corners(approx.reshape(4, 2).astype(np.float32))
            break
    if merged is None:
        box = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
        merged = order_corners(box)
    return _finish_merged_quad(
        merged, work_area, scale, src_width, src_height, trace, id_a, id_b, gap, vert_overlap, height_ratio
    )


def _finish_merged_quad(
    merged: np.ndarray,
    work_area: float,
    scale: float,
    src_width: int,
    src_height: int,
    trace: object | None,
    id_left: str,
    id_right: str,
    gap: float,
    vert_overlap: float,
    height_ratio: float,
) -> QuadCandidate | None:
    merged = order_corners(merged)
    frame_area = float(max(src_width * src_height, 1))
    poly_area = float(cv2.contourArea(merged.astype(np.float32)))

    def _merge_fail(reason: str, *, score: float = 0.0) -> None:
        if trace is None:
            return
        trace.merges.append(  # type: ignore[union-attr]
            {
                "candidate_a": id_left,
                "candidate_b": id_right,
                "horizontal_gap": gap,
                "vertical_overlap": vert_overlap,
                "height_ratio": height_ratio,
                "combined_bbox": bbox_from_corners(merged),
                "merge_score": score,
                "merge_rejection_reason": reason,
            }
        )

    if poly_area / frame_area < MERGE_MIN_AREA:
        _merge_fail("combined_area")
        return None
    aspect = _aspect_of_quad(merged)
    if aspect < MERGE_MIN_ASPECT or aspect > MERGE_MAX_ASPECT:
        _merge_fail("combined_aspect")
        return None
    rect = cv2.minAreaRect(merged.astype(np.float32))
    rw_r, rh_r = float(rect[1][0]), float(rect[1][1])
    min_rect_area_src = max(1.0, rw_r * rh_r)
    work_corners = merged * float(scale)
    work_area_px = poly_area * (scale * scale)
    cand = _build_candidate(
        work_corners,
        work_area_px,
        work_area,
        min_rect_area_src * (scale * scale),
        scale,
        src_width,
        src_height,
        "merged_two_page",
        log_reject=False,
        trace=trace,
    )
    if cand is None:
        _merge_fail("merge_geometry_rejected")
        return None
    if trace is not None:
        trace.merges.append(  # type: ignore[union-attr]
            {
                "candidate_a": id_left,
                "candidate_b": id_right,
                "horizontal_gap": gap,
                "vertical_overlap": vert_overlap,
                "height_ratio": height_ratio,
                "combined_bbox": cand.bbox,
                "merge_score": cand.confidence,
                "merge_rejection_reason": "",
            }
        )
    return cand


def _try_merge_two_page(
    a: QuadCandidate,
    b: QuadCandidate,
    work_area: float,
    scale: float,
    src_width: int,
    src_height: int,
    trace: object | None = None,
    id_a: str = "",
    id_b: str = "",
) -> QuadCandidate | None:
    left, right = (a, b) if a.bbox[0] <= b.bbox[0] else (b, a)
    if a.bbox[0] <= b.bbox[0]:
        id_left, id_right = id_a, id_b
    else:
        id_left, id_right = id_b, id_a
    lx, ly, lw, lh = left.bbox
    rx, ry, rw, rh = right.bbox
    height_ratio = min(lh, rh) / max(lh, rh)
    y0, y1 = max(ly, ry), min(ly + lh, ry + rh)
    overlap_h = max(0, y1 - y0)
    vert_overlap = overlap_h / min(lh, rh) if min(lh, rh) else 0.0
    gap = rx - (lx + lw)
    ux, uy = min(lx, rx), min(ly, ry)
    combined = (ux, uy, max(lx + lw, rx + rw) - ux, max(ly + lh, ry + rh) - uy)
    max_gap = MERGE_MAX_GAP_FRAC * min(lw, rw)
    min_gap = MERGE_MIN_GAP_FRAC * min(lw, rw)

    def _merge_fail(reason: str, *, score: float = 0.0) -> None:
        if trace is None:
            return
        trace.merges.append(  # type: ignore[union-attr]
            {
                "candidate_a": id_left,
                "candidate_b": id_right,
                "horizontal_gap": gap,
                "vertical_overlap": vert_overlap,
                "height_ratio": height_ratio,
                "combined_bbox": combined,
                "merge_score": score,
                "merge_rejection_reason": reason,
            }
        )

    aabb_ok = (
        height_ratio >= MERGE_MIN_HEIGHT_RATIO
        and vert_overlap >= MERGE_MIN_VERTICAL_OVERLAP
        and min_gap <= gap <= max_gap
    )
    if aabb_ok:
        lc = order_corners(left.corners)
        rc = order_corners(right.corners)
        merged = order_corners(np.stack([lc[0], rc[1], rc[2], lc[3]], axis=0))
        cand = _finish_merged_quad(
            merged, work_area, scale, src_width, src_height, trace, id_left, id_right, gap, vert_overlap, height_ratio
        )
        if cand is not None:
            return cand
    else:
        if height_ratio < MERGE_MIN_HEIGHT_RATIO:
            _merge_fail("height_ratio")
        elif vert_overlap < MERGE_MIN_VERTICAL_OVERLAP:
            _merge_fail("vertical_overlap")
        else:
            _merge_fail("horizontal_gap")
    return _try_merge_oriented(
        a, b, work_area, scale, src_width, src_height, trace, id_a, id_b
    )


def _merge_two_page_candidates(
    candidates: list[QuadCandidate],
    work_area: float,
    scale: float,
    src_width: int,
    src_height: int,
    trace: object | None = None,
) -> list[QuadCandidate]:
    extra: list[QuadCandidate] = []
    n = len(candidates)
    for i in range(n):
        for j in range(i + 1, n):
            id_a = trace.id_for(candidates[i].bbox) if trace is not None else ""  # type: ignore[union-attr]
            id_b = trace.id_for(candidates[j].bbox) if trace is not None else ""  # type: ignore[union-attr]
            merged = _try_merge_two_page(
                candidates[i],
                candidates[j],
                work_area,
                scale,
                src_width,
                src_height,
                trace=trace,
                id_a=id_a,
                id_b=id_b,
            )
            if merged is not None:
                extra.append(merged)
    if not extra:
        return candidates
    return candidates + extra


def _rank_key(cand: QuadCandidate) -> tuple[float, float]:
    booklet = 0.0
    if cand.geometry_mode == "merged_two_page":
        booklet = 0.05
    elif cand.geometry_mode == "paper_region":
        booklet = 0.02
    elif cand.geometry_mode == "large_document_outer":
        # Booklet-like outer only. Do not let a frame-sized hypothesis beat an ID quad.
        booklet = 0.04 if cand.quad_aspect >= 1.55 else -0.10
    swallowed = 0.0
    if (
        cand.geometry_mode == "large_document_outer"
        and cand.area_ratio >= 0.50
        and cand.border_sides >= 3
        and cand.edge_support < 0.55
    ):
        swallowed = 0.12
    evidence = 0.04 * float(cand.rectangularity) + 0.03 * float(cand.edge_support)
    return (cand.confidence + booklet + evidence - swallowed, cand.rectangularity)


def find_document_quad(frame: np.ndarray, *, trace: object | None = None) -> tuple[QuadCandidate | None, int]:
    """Return the best document-like quadrilateral in ``frame`` pixel space.

    ``frame`` is RGB or gray. The input array is not written.
    ``trace`` is optional diagnostics only; it does not change accept/reject rules.
    """
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY if frame.shape[2] == 3 else cv2.COLOR_RGBA2GRAY)
    else:
        gray = np.ascontiguousarray(frame)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

def _downsample_gray(gray: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    src_h, src_w = gray.shape[:2]
    if src_w > width:
        scale = width / float(src_w)
        work = cv2.resize(
            gray,
            (width, max(1, int(round(src_h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        return work, scale
    return gray, 1.0


def _small_blob_hint(work: np.ndarray) -> bool:
    """True when a paper-like blob is in the small-document occupancy band."""
    paper = _paper_mask(work)
    contours, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    work_area = float(max(work.shape[0] * work.shape[1], 1))
    for contour in contours:
        frac = float(cv2.contourArea(contour)) / work_area
        if SMALL_EVIDENCE_MIN_AREA <= frac <= MIN_AREA_FRACTION + 0.05:
            return True
    return False


def _collect_candidates(
    work: np.ndarray,
    scale: float,
    src_w: int,
    src_h: int,
    trace: object | None,
) -> list[QuadCandidate]:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    candidates: list[QuadCandidate] = []
    seen: set[tuple[int, int, int, int]] = set()
    work_area = float(work.shape[0] * work.shape[1])
    norm = _illumination_normalize(work)
    edge_maps = _edge_maps(norm)
    edge_combo = np.zeros(work.shape[:2], dtype=np.uint8)
    for edge in edge_maps:
        edge_combo = cv2.bitwise_or(edge_combo, edge)
    edge_combo = cv2.bitwise_or(
        edge_combo, cv2.Canny(cv2.GaussianBlur(work, (5, 5), 0), 40, 120)
    )

    def _add_from_edges(edge: np.ndarray, *, close_kernel, iterations: int) -> None:
        closed = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, close_kernel, iterations=iterations)
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cand = _candidate_from_contour(
                contour,
                work.shape,
                scale,
                src_w,
                src_h,
                log_reject=not candidates,
                trace=trace,
                work_gray=work,
                edge_map=edge_combo,
            )
            if cand is None:
                continue
            key = cand.bbox
            if key in seen:
                continue
            seen.add(key)
            candidates.append(cand)

    for edge in edge_maps:
        _add_from_edges(edge, close_kernel=kernel, iterations=2)

    paper = _paper_mask(work)
    paper_contours, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in paper_contours:
        cand = _candidate_from_contour(
            contour,
            work.shape,
            scale,
            src_w,
            src_h,
            log_reject=not candidates,
            trace=trace,
            force_min_rect=True,
            area_from_hull=True,
            geometry_mode="paper_region",
            work_gray=work,
            edge_map=edge_combo,
        )
        if cand is None:
            continue
        key = cand.bbox
        if key in seen:
            continue
        seen.add(key)
        candidates.append(cand)

    if candidates:
        if trace is not None:
            trace.accepted_before_merge = len(candidates)  # type: ignore[union-attr]
        candidates = _merge_two_page_candidates(
            candidates, work_area, scale, src_w, src_h, trace=trace
        )
        candidates = _suppress_inner_quads(candidates, trace=trace)
    return candidates


def find_document_quad(frame: np.ndarray, *, trace: object | None = None) -> tuple[QuadCandidate | None, int]:
    """Return the best document-like quadrilateral in ``frame`` pixel space.

    ``frame`` is RGB or gray. The input array is not written.
    ``trace`` is optional diagnostics only; it does not change accept/reject rules.
    """
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY if frame.shape[2] == 3 else cv2.COLOR_RGBA2GRAY)
    else:
        gray = np.ascontiguousarray(frame)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    src_h, src_w = gray.shape[:2]
    work, scale = _downsample_gray(gray, WORKING_WIDTH)
    if trace is not None:
        trace.src_wh = (src_w, src_h)  # type: ignore[union-attr]
        trace.work_wh = (int(work.shape[1]), int(work.shape[0]))  # type: ignore[union-attr]
        trace.scale = float(scale)  # type: ignore[union-attr]

    candidates = _collect_candidates(work, scale, src_w, src_h, trace)
    if not candidates and scale < 0.999 and src_w > WORKING_WIDTH:
        target = min(src_w, WORKING_WIDTH_REFINE)
        if target > work.shape[1] + 16 and _small_blob_hint(work):
            work2, scale2 = _downsample_gray(gray, target)
            extra = _collect_candidates(work2, scale2, src_w, src_h, trace)
            if extra:
                candidates = extra

    if not candidates:
        return None, 0
    best = max(candidates, key=_rank_key)
    return best, len(candidates)
