"""Document quadrilateral geometry from a preview frame.

Coordinates are always in the input image's pixel space. Internal work may
use a smaller image; corners are scaled back before return.

Confidence is geometric, not semantic:

    0.35 * rectangularity   contour area / min-area-rect area (1 = rectangle)
    0.25 * area_term        accepts typical pages through large occupancy
    0.25 * right_angle_term mean |cos| of adjacent edges near 0
    0.15 * aspect_term      ID/A4 and opened two-page booklets

Motion vs an empty baseline is not used.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from idcam.logging.setup import get_logger

logger = get_logger("vision.geometry")

# Working width for contour search. 1920×1080 is scaled down then mapped back.
WORKING_WIDTH = 960

MIN_AREA_FRACTION = 0.08
MAX_AREA_FRACTION = 0.97
MIN_ASPECT = 0.40
MAX_ASPECT = 3.35
MIN_RECTANGULARITY = 0.72
MIN_CONFIDENCE = 0.48
MIN_RIGHT_ANGLE = 0.55  # 1 - mean|cos|; 90° edges → 1.0

# Two-page merge: opened passport / booklet. Not a generic blob union.
MERGE_MIN_HEIGHT_RATIO = 0.82
MERGE_MIN_VERTICAL_OVERLAP = 0.78
MERGE_MAX_GAP_FRAC = 0.12
MERGE_MIN_GAP_FRAC = -0.08
MERGE_MIN_ASPECT = 1.45
MERGE_MAX_ASPECT = 3.45
MERGE_MIN_AREA = 0.16

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


def order_corners(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]
    ordered = np.stack(
        [
            pts[int(np.argmin(s))],  # TL: smallest x+y
            pts[int(np.argmax(d))],  # TR: largest x-y
            pts[int(np.argmax(s))],  # BR: largest x+y
            pts[int(np.argmin(d))],  # BL: smallest x-y
        ],
        axis=0,
    )
    return ordered


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


def _corner_on_border_count(corners: np.ndarray, width: int, height: int, *, px: float = 3.0) -> int:
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    n = 0
    for x, y in pts:
        if x <= px or y <= px or x >= width - 1 - px or y >= height - 1 - px:
            n += 1
    return n


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
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    w = (np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[3])) / 2.0
    h = (np.linalg.norm(pts[3] - pts[0]) + np.linalg.norm(pts[2] - pts[1])) / 2.0
    if h < 1.0:
        return 0.0
    return float(w / h)


def _area_term(area_ratio: float) -> float:
    """Typical placed page (~0.2–0.45) and large occupancy (~0.55–0.85)."""
    if area_ratio <= 0:
        return 0.0
    typical = float(np.exp(-((area_ratio - 0.35) ** 2) / (2 * 0.18**2)))
    large = float(np.exp(-((area_ratio - 0.72) ** 2) / (2 * 0.16**2)))
    return max(typical, 0.94 * large)


def _aspect_term(aspect: float) -> float:
    if aspect <= 0:
        return 0.0
    # Single-page ID/A4 (~1.25) and opened booklet / passport (~2.5–2.9).
    log_err_page = abs(np.log(aspect) - np.log(1.25))
    log_err_spread = abs(np.log(aspect) - np.log(2.55))
    page = float(np.exp(-(log_err_page**2) / (2 * 0.45**2)))
    spread = float(np.exp(-(log_err_spread**2) / (2 * 0.38**2)))
    return max(page, spread)


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


def _approx_quad(contour: np.ndarray, *, allow_min_rect: bool) -> tuple[np.ndarray | None, str]:
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    if peri < 8:
        return None, "perimeter_too_small"
    for eps in (0.02, 0.03, 0.04, 0.055, 0.07):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32), "quad"
    if allow_min_rect:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.float32)
        return box, "min_rect_fallback"
    return None, "not_quadrilateral"


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

    def _reject(reason: str) -> None:
        if log_reject:
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

    if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
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
    if _corner_on_border_count(corners, src_width, src_height, px=4.0) >= 3:
        _reject("frame_aligned")
        return None
    return QuadCandidate(
        corners=corners,
        bbox=bbox,
        confidence=conf,
        area_ratio=area_ratio,
        aspect_ratio=aspect,
        rectangularity=rectangularity,
        right_angle=right,
        reason=geometry_mode,
        geometry_mode=geometry_mode,
        border_distance=border,
    )


def _candidate_from_contour(
    contour: np.ndarray,
    work_shape: tuple[int, int],
    scale: float,
    src_width: int,
    src_height: int,
    *,
    log_reject: bool = False,
) -> QuadCandidate | None:
    fh, fw = int(work_shape[0]), int(work_shape[1])
    work_area = float(fh * fw)
    area = float(cv2.contourArea(contour))
    area_frac = area / work_area if work_area else 0.0
    if area_frac < MIN_AREA_FRACTION:
        if log_reject:
            _maybe_log_reject(
                reason="area_too_small",
                area_ratio=area_frac,
                bbox=None,
                aspect_ratio=0.0,
                rectangularity=0.0,
                corner_count=0,
                right_angle=0.0,
                border_distance=0.0,
            )
        return None
    if area_frac > MAX_AREA_FRACTION:
        if log_reject:
            _maybe_log_reject(
                reason="area_too_large",
                area_ratio=area_frac,
                bbox=None,
                aspect_ratio=0.0,
                rectangularity=0.0,
                corner_count=0,
                right_angle=0.0,
                border_distance=0.0,
            )
        return None
    allow_min_rect = 0.20 <= area_frac <= 0.85
    quad, mode = _approx_quad(contour, allow_min_rect=allow_min_rect)
    if quad is None:
        if log_reject:
            _maybe_log_reject(
                reason=mode,
                area_ratio=area_frac,
                bbox=None,
                aspect_ratio=0.0,
                rectangularity=0.0,
                corner_count=0,
                right_angle=0.0,
                border_distance=0.0,
            )
        return None
    hull_quad = cv2.convexHull(quad.astype(np.float32))
    if len(hull_quad) != 4:
        return None
    rect = cv2.minAreaRect(contour)
    rw, rh = float(rect[1][0]), float(rect[1][1])
    min_rect_area = max(1.0, rw * rh)
    geom_mode = "min_rect_fallback" if mode == "min_rect_fallback" else "quad"
    return _build_candidate(
        order_corners(quad),
        area,
        work_area,
        min_rect_area,
        scale,
        src_width,
        src_height,
        geom_mode,
        log_reject=log_reject,
    )


def _bbox_containment_frac(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> float:
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    x0, y0 = max(ix, ox), max(iy, oy)
    x1, y1 = min(ix + iw, ox + ow), min(iy + ih, oy + oh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    inner_area = max(1, iw * ih)
    return float(inter) / float(inner_area)


def _suppress_inner_quads(candidates: list[QuadCandidate]) -> list[QuadCandidate]:
    """Prefer the outer document boundary over text blocks and single pages inside it."""
    if len(candidates) < 2:
        return candidates
    keep = [True] * len(candidates)
    for i, inner in enumerate(candidates):
        for j, outer in enumerate(candidates):
            if i == j:
                continue
            if outer.geometry_mode != "merged_two_page" and outer.area_ratio < inner.area_ratio * 1.75:
                continue
            if _bbox_containment_frac(inner.bbox, outer.bbox) >= 0.88:
                keep[i] = False
                break
    return [c for c, ok in zip(candidates, keep) if ok]


def _try_merge_two_page(
    a: QuadCandidate,
    b: QuadCandidate,
    work_area: float,
    scale: float,
    src_width: int,
    src_height: int,
) -> QuadCandidate | None:
    left, right = (a, b) if a.bbox[0] <= b.bbox[0] else (b, a)
    lx, ly, lw, lh = left.bbox
    rx, ry, rw, rh = right.bbox
    height_ratio = min(lh, rh) / max(lh, rh)
    if height_ratio < MERGE_MIN_HEIGHT_RATIO:
        return None
    y0, y1 = max(ly, ry), min(ly + lh, ry + rh)
    overlap_h = max(0, y1 - y0)
    if overlap_h / min(lh, rh) < MERGE_MIN_VERTICAL_OVERLAP:
        return None
    gap = rx - (lx + lw)
    max_gap = MERGE_MAX_GAP_FRAC * min(lw, rw)
    min_gap = MERGE_MIN_GAP_FRAC * min(lw, rw)
    if gap > max_gap or gap < min_gap:
        return None
    lc = order_corners(left.corners)
    rc = order_corners(right.corners)
    merged = np.stack([lc[0], rc[1], rc[2], lc[3]], axis=0)
    merged = order_corners(merged)
    frame_area = float(max(src_width * src_height, 1))
    poly_area = float(cv2.contourArea(merged.astype(np.float32)))
    if poly_area / frame_area < MERGE_MIN_AREA:
        return None
    aspect = _aspect_of_quad(merged)
    if aspect < MERGE_MIN_ASPECT or aspect > MERGE_MAX_ASPECT:
        return None
    rect = cv2.minAreaRect(merged.astype(np.float32))
    rw_r, rh_r = float(rect[1][0]), float(rect[1][1])
    min_rect_area_src = max(1.0, rw_r * rh_r)
    work_corners = merged * float(scale)
    work_area_px = poly_area * (scale * scale)
    return _build_candidate(
        work_corners,
        work_area_px,
        work_area,
        min_rect_area_src * (scale * scale),
        scale,
        src_width,
        src_height,
        "merged_two_page",
        log_reject=False,
    )


def _merge_two_page_candidates(
    candidates: list[QuadCandidate],
    work_area: float,
    scale: float,
    src_width: int,
    src_height: int,
) -> list[QuadCandidate]:
    extra: list[QuadCandidate] = []
    n = len(candidates)
    for i in range(n):
        for j in range(i + 1, n):
            merged = _try_merge_two_page(
                candidates[i], candidates[j], work_area, scale, src_width, src_height
            )
            if merged is not None:
                extra.append(merged)
    if not extra:
        return candidates
    return candidates + extra


def _rank_key(cand: QuadCandidate) -> tuple[float, float]:
    booklet = 0.04 if cand.geometry_mode == "merged_two_page" else 0.0
    # Prefer a true page inset from the frame over a quad that swallowed the margin.
    flush = 0.06 if cand.border_distance < 2.5 else 0.0
    return (cand.confidence + booklet - flush, cand.border_distance)


def find_document_quad(frame: np.ndarray) -> tuple[QuadCandidate | None, int]:
    """Return the best document-like quadrilateral in ``frame`` pixel space.

    ``frame`` is RGB or gray. The input array is not written.
    """
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY if frame.shape[2] == 3 else cv2.COLOR_RGBA2GRAY)
    else:
        gray = np.ascontiguousarray(frame)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    src_h, src_w = gray.shape[:2]
    if src_w > WORKING_WIDTH:
        scale = WORKING_WIDTH / float(src_w)
        work = cv2.resize(
            gray,
            (WORKING_WIDTH, max(1, int(round(src_h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        scale = 1.0
        work = gray

    norm = _illumination_normalize(work)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    candidates: list[QuadCandidate] = []
    seen: set[tuple[int, int, int, int]] = set()
    work_area = float(work.shape[0] * work.shape[1])

    def _add_from_edges(edge: np.ndarray, *, close_kernel, iterations: int) -> None:
        closed = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, close_kernel, iterations=iterations)
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cand = _candidate_from_contour(
                contour, work.shape, scale, src_w, src_h, log_reject=not candidates
            )
            if cand is None:
                continue
            key = cand.bbox
            if key in seen:
                continue
            seen.add(key)
            candidates.append(cand)

    for edge in _edge_maps(norm):
        _add_from_edges(edge, close_kernel=kernel, iterations=2)

    if candidates:
        candidates = _merge_two_page_candidates(candidates, work_area, scale, src_w, src_h)
        candidates = _suppress_inner_quads(candidates)

    if not candidates:
        return None, 0
    best = max(candidates, key=_rank_key)
    return best, len(candidates)
