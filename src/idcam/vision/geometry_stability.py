"""Production geometry-stability gate (Phase 2C).

One definition of document readiness: consecutive preview frames whose
detected quadrilateral moves less than ``geometry_stability_threshold_px``.

This is independent of:

- DetectionConfig.motion_threshold (empty-scene / ROI gray motion)
- StabilityDetector (center-ROI absdiff)
- geometry_hold_frames / geometry_smooth_alpha (detector hysteresis only)

Displacement is the mean Euclidean distance of the four ordered corners
in preview pixels (same space as detect_geometry). It is not normalized
by image size so the threshold stays an explicit pixel budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from idcam.vision.document_detector import DocumentGeometryResult
from idcam.vision.geometry import order_corners


def mean_corner_displacement_px(
    previous: np.ndarray | tuple,
    current: np.ndarray | tuple,
) -> float:
    """Mean L2 distance of corresponding TL/TR/BR/BL corners, in pixels."""
    prev = order_corners(np.asarray(previous, dtype=np.float32))
    cur = order_corners(np.asarray(current, dtype=np.float32))
    return float(np.mean(np.linalg.norm(cur - prev, axis=1)))


@dataclass(frozen=True)
class GeometryStabilityDebug:
    displacement_px: float
    threshold_px: float
    stable_count: int
    stable_frames_required: int
    stable: bool


class GeometryStability:
    """Count consecutive low-movement geometry frames.

    The first present observation only seeds the previous corners and never
    reports ready. A later frame is stable when mean corner motion is at or
    below ``threshold_px``; otherwise the consecutive counter resets.
    """

    def __init__(self, *, threshold_px: float, frames_required: int) -> None:
        self.threshold_px = float(threshold_px)
        self.frames_required = int(frames_required)
        self._prev_corners: np.ndarray | None = None
        self._stable_count = 0
        self._last_displacement = 0.0

    def reset(self) -> None:
        self._prev_corners = None
        self._stable_count = 0
        self._last_displacement = 0.0

    @property
    def stable_count(self) -> int:
        return self._stable_count

    @property
    def last_displacement_px(self) -> float:
        return self._last_displacement

    @property
    def corners(self) -> np.ndarray | None:
        return None if self._prev_corners is None else self._prev_corners.copy()

    def update(self, geometry: DocumentGeometryResult) -> bool:
        """Return True when consecutive stable geometry frames meet the requirement."""
        if not geometry.present or geometry.corners is None:
            self.reset()
            return False
        corners = order_corners(np.asarray(geometry.corners, dtype=np.float32))
        if self._prev_corners is None:
            self._prev_corners = corners
            self._stable_count = 0
            self._last_displacement = 0.0
            return False
        displacement = mean_corner_displacement_px(self._prev_corners, corners)
        self._last_displacement = displacement
        self._prev_corners = corners
        if displacement <= self.threshold_px:
            self._stable_count += 1
        else:
            self._stable_count = 0
        return self._stable_count >= self.frames_required

    def debug_info(self) -> GeometryStabilityDebug:
        return GeometryStabilityDebug(
            displacement_px=self._last_displacement,
            threshold_px=self.threshold_px,
            stable_count=self._stable_count,
            stable_frames_required=self.frames_required,
            stable=self._stable_count >= self.frames_required,
        )
