"""Production geometry-stability gate (Phase 2C).

Readiness is consecutive preview frames whose document quad stays on the
*same* page: matched-corner travel below a pixel floor or a fraction of
the quad's own size. This is independent of center-ROI motion metrics.

Corner labels can rotate by one vertex (especially near 45°). Displacement
uses the best cyclic correspondence so labeling flips are not treated as
the document moving.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from idcam.vision.document_detector import DocumentGeometryResult
from idcam.vision.geometry import bbox_from_corners, bbox_iou, order_corners


def mean_corner_displacement_px(
    previous: np.ndarray | tuple,
    current: np.ndarray | tuple,
) -> float:
    """Mean L2 of corresponding TL/TR/BR/BL corners (fixed labeling)."""
    prev = order_corners(np.asarray(previous, dtype=np.float32))
    cur = order_corners(np.asarray(current, dtype=np.float32))
    return float(np.mean(np.linalg.norm(cur - prev, axis=1)))


def matched_corner_displacement_px(
    previous: np.ndarray | tuple,
    current: np.ndarray | tuple,
) -> float:
    """Smallest mean L2 after cyclic rolls / reverse winding.

    A start-vertex flip at ~45° must not look like a large document move.
    """
    prev = order_corners(np.asarray(previous, dtype=np.float32))
    cur = order_corners(np.asarray(current, dtype=np.float32))
    best = float("inf")
    for seq in (cur, cur[::-1]):
        for shift in range(4):
            rolled = np.roll(seq, shift, axis=0)
            best = min(best, float(np.mean(np.linalg.norm(rolled - prev, axis=1))))
    return best


def _mean_side_px(corners: np.ndarray) -> float:
    pts = order_corners(np.asarray(corners, dtype=np.float32))
    lengths = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]
    return float(np.mean(lengths)) if lengths else 0.0


@dataclass(frozen=True)
class GeometryStabilityDebug:
    displacement_px: float
    threshold_px: float
    stable_count: int
    stable_frames_required: int
    stable: bool
    effective_threshold_px: float = 0.0


class GeometryStability:
    """Count consecutive low-movement geometry frames.

    The first present observation only seeds the previous corners and never
    reports ready. Later frames use matched-corner displacement versus
    ``max(threshold_px, relative_fraction * mean_side)``.
    """

    def __init__(
        self,
        *,
        threshold_px: float,
        frames_required: int,
        miss_tolerance: int = 2,
        relative_fraction: float = 0.05,
    ) -> None:
        self.threshold_px = float(threshold_px)
        self.frames_required = int(frames_required)
        self.miss_tolerance = max(0, int(miss_tolerance))
        self.relative_fraction = max(0.0, float(relative_fraction))
        self._prev_corners: np.ndarray | None = None
        self._stable_count = 0
        self._last_displacement = 0.0
        self._last_effective_threshold = float(threshold_px)
        self._absent_streak = 0
        self._jitter_streak = 0
        self._last_same_document = True

    def reset(self) -> None:
        self._prev_corners = None
        self._stable_count = 0
        self._last_displacement = 0.0
        self._last_effective_threshold = self.threshold_px
        self._absent_streak = 0
        self._jitter_streak = 0
        self._last_same_document = True

    @property
    def stable_count(self) -> int:
        return self._stable_count

    @property
    def last_displacement_px(self) -> float:
        return self._last_displacement

    @property
    def last_effective_threshold_px(self) -> float:
        return self._last_effective_threshold

    @property
    def corners(self) -> np.ndarray | None:
        return None if self._prev_corners is None else self._prev_corners.copy()

    @property
    def last_same_document(self) -> bool:
        return self._last_same_document

    def effective_threshold_px(self, corners: np.ndarray | None = None) -> float:
        pts = corners if corners is not None else self._prev_corners
        if pts is None:
            return self.threshold_px
        return max(self.threshold_px, self.relative_fraction * _mean_side_px(pts))

    def same_document(self, geometry: DocumentGeometryResult) -> bool:
        """True when the new quad is the same page (IoU or matched displacement)."""
        if not geometry.present or geometry.corners is None or self._prev_corners is None:
            return bool(geometry.present and geometry.corners is not None)
        cur = np.asarray(geometry.corners, dtype=np.float32)
        iou = bbox_iou(bbox_from_corners(self._prev_corners), bbox_from_corners(cur))
        disp = matched_corner_displacement_px(self._prev_corners, cur)
        thr = self.effective_threshold_px(self._prev_corners)
        return iou >= 0.40 or disp <= thr

    def update(self, geometry: DocumentGeometryResult) -> bool:
        """Return True when consecutive stable geometry frames meet the requirement."""
        if not geometry.present or geometry.corners is None:
            self._absent_streak += 1
            if self._absent_streak > self.miss_tolerance:
                self.reset()
                return False
            return self._stable_count >= self.frames_required
        self._absent_streak = 0
        corners = order_corners(np.asarray(geometry.corners, dtype=np.float32))
        if self._prev_corners is None:
            self._prev_corners = corners
            self._stable_count = 0
            self._last_displacement = 0.0
            self._last_effective_threshold = self.effective_threshold_px(corners)
            self._jitter_streak = 0
            self._last_same_document = False
            return False
        displacement = matched_corner_displacement_px(self._prev_corners, corners)
        self._last_displacement = displacement
        thr = self.effective_threshold_px(self._prev_corners)
        self._last_effective_threshold = thr
        self._last_same_document = displacement <= thr
        self._prev_corners = corners
        if displacement <= thr:
            self._jitter_streak = 0
            self._stable_count += 1
        else:
            self._jitter_streak += 1
            if self._jitter_streak > self.miss_tolerance:
                self._stable_count = 0
        return self._stable_count >= self.frames_required

    def debug_info(self) -> GeometryStabilityDebug:
        return GeometryStabilityDebug(
            displacement_px=self._last_displacement,
            threshold_px=self.threshold_px,
            stable_count=self._stable_count,
            stable_frames_required=self.frames_required,
            stable=self._stable_count >= self.frames_required,
            effective_threshold_px=self._last_effective_threshold,
        )
