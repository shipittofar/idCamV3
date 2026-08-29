"""Document presence detection from preview frames."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from idcam.config.schema import DetectionConfig
from idcam.logging.setup import get_logger
from idcam.vision.geometry import (
    bbox_from_corners,
    find_document_quad,
    order_corners,
)
from idcam.vision.sharpness import clip_roi

logger = get_logger("vision.document_detector")


class DocumentPresence(str, Enum):
    NO_DOCUMENT = "no_document"
    DOCUMENT_ENTERING = "document_entering"
    DOCUMENT_MOVING = "document_moving"
    DOCUMENT_STABLE = "document_stable"
    DOCUMENT_READY = "document_ready"


@dataclass(frozen=True)
class DocumentDetectionDebug:
    """Latest detector metrics for debug overlay. Does not affect detection."""

    motion: float
    edge_density: float
    motion_threshold: float
    edge_density_threshold: float
    presence_frames: int
    stable_frames_required: int
    detected: bool
    reject_reason: str = ""
    roi_width: int = 0
    roi_height: int = 0
    frame_width: int = 0
    frame_height: int = 0
    gray_std: float = 0.0
    gray_mean: float = 0.0
    frame_motion: float = 0.0
    document_roi: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass(frozen=True)
class DocumentGeometryResult:
    """Canonical document geometry in preview pixels. Not motion-based."""

    present: bool
    confidence: float
    corners: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]] | None
    bbox: tuple[int, int, int, int] | None
    reason: str
    candidate_count: int = 0
    selected_area_ratio: float = 0.0
    aspect_ratio: float = 0.0
    rectangularity: float = 0.0
    geometry_mode: str = "none"

    def corners_array(self) -> np.ndarray | None:
        if self.corners is None:
            return None
        return np.asarray(self.corners, dtype=np.float32)


def _corners_tuple(
    corners: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    pts = order_corners(corners)
    return (
        (float(pts[0, 0]), float(pts[0, 1])),
        (float(pts[1, 0]), float(pts[1, 1])),
        (float(pts[2, 0]), float(pts[2, 1])),
        (float(pts[3, 0]), float(pts[3, 1])),
    )


def gradient_edge_density(gray: np.ndarray, magnitude_threshold: float = 30.0) -> float:
    """Fraction of ROI pixels with Sobel magnitude above ``magnitude_threshold``.

    Canny(50, 150) misses printed texture on a mildly defocused 1080p preview.
    Sobel magnitude still measures document structure without lowering the
    configured edge_density_threshold.
    """
    if gray.size == 0:
        return 0.0
    work = np.ascontiguousarray(gray)
    if work.dtype != np.uint8:
        work = np.clip(work, 0, 255).astype(np.uint8)
    gx = cv2.Sobel(work, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(work, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return float(np.count_nonzero(mag > magnitude_threshold)) / mag.size


class DocumentDetector:
    """Document presence from preview frames.

    ``detect_geometry`` is the production presence API (Phase 2B/2C).
    ``detect`` / ``classify`` remain the legacy motion+edge path and must not
    be used for orchestrator presence, stability, or removal.
    """

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self._baseline: np.ndarray | None = None
        self._roi_slice: tuple[slice, slice] | None = None
        self._presence_frames = 0
        self._last_motion = 0.0
        self._last_edge_density = 0.0
        self._last_reject_reason = "no_frame"
        self._last_roi_size = (0, 0)
        self._last_frame_size = (0, 0)
        self._last_gray_std = 0.0
        self._last_gray_mean = 0.0
        self._prev_gray: np.ndarray | None = None
        self._last_frame_motion = 0.0
        self._last_document_roi: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._geom_corners: np.ndarray | None = None
        self._geom_misses = 0
        self._last_geometry = DocumentGeometryResult(
            present=False, confidence=0.0, corners=None, bbox=None, reason="no_frame"
        )

    @property
    def last_motion(self) -> float:
        return self._last_motion

    @property
    def last_edge_density(self) -> float:
        return self._last_edge_density

    def roi_rect(self, frame_shape: tuple[int, ...] | None = None) -> tuple[int, int, int, int]:
        """Centered document ROI as (x, y, width, height) in frame pixels."""
        if frame_shape is not None and len(frame_shape) >= 2:
            h, w = int(frame_shape[0]), int(frame_shape[1])
        else:
            w, h = self._last_frame_size
        frac = self.config.roi_center_fraction
        rh, rw = int(h * frac), int(w * frac)
        y0, x0 = (h - rh) // 2, (w - rw) // 2
        return int(x0), int(y0), int(rw), int(rh)

    def document_preview_roi(self, frame: np.ndarray | None = None) -> tuple[int, int, int, int]:
        """Axis-aligned document box in preview pixels, or the search ROI.

        When ``detect_geometry`` has a live quad, that bbox is canonical.
        The Otsu largest-blob box is no longer a competing document ROI.
        """
        shape = None if frame is None else frame.shape
        if self._last_geometry.present and self._last_geometry.bbox is not None:
            if shape is not None:
                return clip_roi(self._last_geometry.bbox, shape)
            return self._last_geometry.bbox
        if self._last_document_roi != (0, 0, 0, 0):
            if shape is not None:
                return clip_roi(self._last_document_roi, shape)
            return self._last_document_roi
        return self.roi_rect(shape)

    def detect_geometry(self, frame: np.ndarray) -> DocumentGeometryResult:
        """Detect a document quadrilateral in ``frame`` (preview pixels).

        Does not require an empty-scene baseline. Does not mutate ``frame``.
        Temporal hysteresis is local to this detector (not StabilityDetector).
        """
        h, w = int(frame.shape[0]), int(frame.shape[1])
        self._last_frame_size = (w, h)
        was_present = self._last_geometry.present
        raw, candidate_count = find_document_quad(frame)
        alpha = float(self.config.geometry_smooth_alpha)
        hold = int(self.config.geometry_hold_frames)

        if raw is not None:
            corners = self._smooth_corners(raw.corners, alpha)
            bbox = bbox_from_corners(corners)
            bbox = clip_roi(bbox, frame.shape)
            self._geom_corners = corners
            self._geom_misses = 0
            self._last_document_roi = bbox
            result = DocumentGeometryResult(
                present=True,
                confidence=float(raw.confidence),
                corners=_corners_tuple(corners),
                bbox=bbox,
                reason="quad",
                candidate_count=candidate_count,
                selected_area_ratio=float(raw.area_ratio),
                aspect_ratio=float(raw.aspect_ratio),
                rectangularity=float(raw.rectangularity),
                geometry_mode=str(raw.geometry_mode),
            )
            self._last_geometry = result
            if not was_present:
                logger.info(
                    "event=document_geometry present=1 confidence=%.3f reason=quad "
                    "geometry_mode=%s candidates=%d document_area_ratio=%.3f "
                    "document_aspect_ratio=%.3f document_rectangularity=%.3f "
                    "document_bbox=%d,%d,%d,%d document_quad=%s",
                    result.confidence,
                    result.geometry_mode,
                    candidate_count,
                    result.selected_area_ratio,
                    result.aspect_ratio,
                    result.rectangularity,
                    *bbox,
                    result.corners,
                )
            return result

        self._geom_misses += 1
        if self._geom_corners is not None and self._geom_misses < hold:
            bbox = bbox_from_corners(self._geom_corners)
            bbox = clip_roi(bbox, frame.shape)
            self._last_document_roi = bbox
            result = DocumentGeometryResult(
                present=True,
                confidence=self._last_geometry.confidence,
                corners=_corners_tuple(self._geom_corners),
                bbox=bbox,
                reason="geometry_held",
                candidate_count=0,
                selected_area_ratio=self._last_geometry.selected_area_ratio,
                aspect_ratio=self._last_geometry.aspect_ratio,
                rectangularity=self._last_geometry.rectangularity,
                geometry_mode=self._last_geometry.geometry_mode,
            )
            self._last_geometry = result
            return result

        self._geom_corners = None
        self._last_document_roi = (0, 0, 0, 0)
        reason = "no_quadrilateral" if candidate_count == 0 else "below_geometry_confidence"
        result = DocumentGeometryResult(
            present=False,
            confidence=0.0,
            corners=None,
            bbox=None,
            reason=reason,
            candidate_count=candidate_count,
        )
        if self._last_geometry.present:
            logger.info(
                "event=document_geometry present=0 reason=%s hold_expired misses=%d",
                reason,
                self._geom_misses,
            )
        self._last_geometry = result
        return result

    def last_geometry(self) -> DocumentGeometryResult:
        return self._last_geometry

    def _smooth_corners(self, corners: np.ndarray, alpha: float) -> np.ndarray:
        ordered = order_corners(corners)
        if self._geom_corners is None or self._geom_corners.shape != ordered.shape:
            return ordered
        prev = order_corners(self._geom_corners)
        return (alpha * ordered + (1.0 - alpha) * prev).astype(np.float32)

    def debug_info(self) -> DocumentDetectionDebug:
        """Return last computed metrics. Does not run detection or mutate state."""
        detected = (
            self._last_motion > self.config.motion_threshold
            and self._last_edge_density > self.config.edge_density_threshold
        )
        return DocumentDetectionDebug(
            motion=self._last_motion,
            edge_density=self._last_edge_density,
            motion_threshold=self.config.motion_threshold,
            edge_density_threshold=self.config.edge_density_threshold,
            presence_frames=self._presence_frames,
            stable_frames_required=self.config.stable_frames_required,
            detected=detected,
            reject_reason="" if detected else self._last_reject_reason,
            roi_width=self._last_roi_size[0],
            roi_height=self._last_roi_size[1],
            frame_width=self._last_frame_size[0],
            frame_height=self._last_frame_size[1],
            gray_std=self._last_gray_std,
            gray_mean=self._last_gray_mean,
            frame_motion=self._last_frame_motion,
            document_roi=self._last_document_roi,
        )

    def reset_baseline(self) -> None:
        self._baseline = None
        self._presence_frames = 0
        self._prev_gray = None
        self._last_frame_motion = 0.0
        self._last_document_roi = (0, 0, 0, 0)
        self._last_reject_reason = "no_frame"
        self._geom_corners = None
        self._geom_misses = 0
        self._last_geometry = DocumentGeometryResult(
            present=False, confidence=0.0, corners=None, bbox=None, reason="reset"
        )

    def update_baseline(self, frame: np.ndarray) -> None:
        """Update baseline from a stable empty scene."""
        gray = self._to_gray(self._get_roi(frame))
        if self._baseline is None:
            self._baseline = gray.astype(np.float32)
        else:
            alpha = self.config.baseline_alpha
            cv2.accumulateWeighted(gray, self._baseline, alpha)

    def _get_roi(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self._last_frame_size = (int(w), int(h))
        frac = self.config.roi_center_fraction
        rh, rw = int(h * frac), int(w * frac)
        y0, x0 = (h - rh) // 2, (w - rw) // 2
        self._roi_slice = (slice(y0, y0 + rh), slice(x0, x0 + rw))
        self._last_roi_size = (int(rw), int(rh))
        return frame[y0 : y0 + rh, x0 : x0 + rw]

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            if frame.shape[2] == 4:
                return cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
            return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame

    def _metrics(self, frame: np.ndarray) -> tuple[float, float]:
        gray = self._to_gray(self._get_roi(frame))
        gray = np.ascontiguousarray(gray)
        edge_density = gradient_edge_density(gray)
        self._last_gray_std = float(np.std(gray))
        self._last_gray_mean = float(np.mean(gray))
        self._last_edge_density = edge_density
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            self._last_frame_motion = float(np.mean(cv2.absdiff(gray, self._prev_gray)))
        else:
            self._last_frame_motion = 0.0
        self._prev_gray = gray.copy()
        if self._baseline is None:
            self._baseline = gray.astype(np.float32)
            self._last_motion = 0.0
            self._last_reject_reason = "baseline_initialized"
            return 0.0, edge_density
        diff = cv2.absdiff(gray, self._baseline.astype(np.uint8))
        motion = float(np.mean(diff))
        self._last_motion = motion
        if motion <= self.config.motion_threshold and edge_density <= self.config.edge_density_threshold:
            self._last_reject_reason = "below_motion_and_edge"
        elif motion <= self.config.motion_threshold:
            self._last_reject_reason = "below_motion_threshold"
        elif edge_density <= self.config.edge_density_threshold:
            self._last_reject_reason = "below_edge_density_threshold"
        else:
            self._last_reject_reason = ""
        return motion, edge_density

    def classify(self, frame: np.ndarray) -> DocumentPresence:
        motion, edge_density = self._metrics(frame)
        present = motion > self.config.motion_threshold and edge_density > self.config.edge_density_threshold

        if not present:
            self._presence_frames = 0
            if motion < self.config.motion_threshold * 0.5:
                gray = self._to_gray(self._get_roi(frame))
                alpha = self.config.baseline_alpha
                cv2.accumulateWeighted(gray, self._baseline, alpha)
            return DocumentPresence.NO_DOCUMENT

        self._presence_frames += 1
        if self._presence_frames < 2:
            return DocumentPresence.DOCUMENT_ENTERING
        if self._last_frame_motion > self.config.motion_threshold * 1.5:
            return DocumentPresence.DOCUMENT_MOVING
        if self._presence_frames < self.config.stable_frames_required:
            return DocumentPresence.DOCUMENT_STABLE
        return DocumentPresence.DOCUMENT_READY

    def detect(self, frame: np.ndarray) -> bool:
        """Returns True when document is present and entering detection."""
        presence = self.classify(frame)
        return presence in {
            DocumentPresence.DOCUMENT_ENTERING,
            DocumentPresence.DOCUMENT_MOVING,
            DocumentPresence.DOCUMENT_STABLE,
            DocumentPresence.DOCUMENT_READY,
        }

    def is_document_present(self, frame: np.ndarray) -> bool:
        presence = self.classify(frame)
        return presence != DocumentPresence.NO_DOCUMENT

    def is_scene_empty(self, frame: np.ndarray) -> bool:
        """True when a document is no longer detected versus the empty baseline.

        Table texture (Sobel edges) must not block removal. Presence uses the
        same AND of motion-vs-baseline and edge density as auto-detection.
        A textured empty table has low motion vs baseline, so it is empty even
        when absolute edge_density is above the detection threshold.
        """
        motion, edge_density = self._metrics(frame)
        present = motion > self.config.motion_threshold and edge_density > self.config.edge_density_threshold
        empty = not present
        if empty:
            logger.debug(
                "Scene empty: motion=%.2f edge=%.4f (present requires motion>%.2f and edge>%.4f)",
                motion,
                edge_density,
                self.config.motion_threshold,
                self.config.edge_density_threshold,
            )
        return empty

    def is_document_removed(self, frame: np.ndarray, consecutive_frames: int) -> bool:
        """True when document has been absent for required consecutive frames."""
        return consecutive_frames >= self.config.removal_frames_required
