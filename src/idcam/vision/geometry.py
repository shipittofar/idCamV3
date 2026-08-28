"""Document quadrilateral geometry from a preview frame.

Coordinates are always in the input image's pixel space. Internal work may
use a smaller image; corners are scaled back before return.

Confidence is geometric, not semantic:

    0.35 * rectangularity   contour area / min-area-rect area (1 = rectangle)
    0.25 * area_term        prefers ~15–55% of the frame
    0.25 * right_angle_term mean |cos| of adjacent edges near 0
    0.15 * aspect_term      ID/A4-like width/height

Motion vs an empty baseline is not used.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Working width for contour search. 1920×1080 is scaled down then mapped back.
WORKING_WIDTH = 960

MIN_AREA_FRACTION = 0.08
MAX_AREA_FRACTION = 0.90
MIN_ASPECT = 0.40
MAX_ASPECT = 2.60
MIN_RECTANGULARITY = 0.72
MIN_CONFIDENCE = 0.48
MIN_RIGHT_ANGLE = 0.55  # 1 - mean|cos|; 90° edges → 1.0


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
    # 90° → |cos|=0 → score 1
    return float(max(0.0, 1.0 - float(np.mean(cos_abs))))


def _aspect_of_quad(corners: np.ndarray) -> float:
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    w = (np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[3])) / 2.0
    h = (np.linalg.norm(pts[3] - pts[0]) + np.linalg.norm(pts[2] - pts[1])) / 2.0
    if h < 1.0:
        return 0.0
    return float(w / h)


def _area_term(area_ratio: float) -> float:
    """Peak near a typical placed page (~0.25–0.50 of the frame)."""
    if area_ratio <= 0:
        return 0.0
    target = 0.35
    # Gaussian-like falloff; 8% area still scores something.
    return float(np.exp(-((area_ratio - target) ** 2) / (2 * 0.18**2)))


def _aspect_term(aspect: float) -> float:
    if aspect <= 0:
        return 0.0
    # 1.0 (square ID-ish) through ~1.4 (A4 landscape) are preferred.
    log_err = abs(np.log(aspect) - np.log(1.25))
    return float(np.exp(-(log_err**2) / (2 * 0.45**2)))


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


def _approx_quad(contour: np.ndarray) -> np.ndarray | None:
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    if peri < 8:
        return None
    for eps in (0.02, 0.03, 0.04, 0.055, 0.07):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return None


def _candidate_from_contour(
    contour: np.ndarray,
    work_shape: tuple[int, int],
    scale: float,
    src_width: int,
    src_height: int,
) -> QuadCandidate | None:
    fh, fw = int(work_shape[0]), int(work_shape[1])
    work_area = float(fh * fw)
    area = float(cv2.contourArea(contour))
    area_frac = area / work_area if work_area else 0.0
    if area_frac < MIN_AREA_FRACTION or area_frac > MAX_AREA_FRACTION:
        return None
    quad = _approx_quad(contour)
    if quad is None:
        return None
    hull_quad = cv2.convexHull(quad.astype(np.float32))
    if len(hull_quad) != 4:
        return None
    rect = cv2.minAreaRect(contour)
    rw, rh = float(rect[1][0]), float(rect[1][1])
    min_rect_area = max(1.0, rw * rh)
    corners_work = order_corners(quad)
    conf, area_ratio, aspect, rectangularity = score_quad(
        corners_work, area, work_area, min_rect_area
    )
    right = _right_angle_score(corners_work)
    if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
        return None
    if rectangularity < MIN_RECTANGULARITY:
        return None
    if right < MIN_RIGHT_ANGLE:
        return None
    if conf < MIN_CONFIDENCE:
        return None
    corners = corners_work / float(scale)
    corners = clip_corners(corners, src_width, src_height)
    bbox = bbox_from_corners(corners)
    return QuadCandidate(
        corners=corners,
        bbox=bbox,
        confidence=conf,
        area_ratio=area_ratio,
        aspect_ratio=aspect,
        rectangularity=rectangularity,
        right_angle=right,
        reason="quad",
    )


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

    for edge in _edge_maps(norm):
        closed = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cand = _candidate_from_contour(contour, work.shape, scale, src_w, src_h)
            if cand is None:
                continue
            key = cand.bbox
            if key in seen:
                continue
            seen.add(key)
            candidates.append(cand)

    if not candidates:
        return None, 0
    best = max(candidates, key=lambda c: c.confidence)
    return best, len(candidates)
