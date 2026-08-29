"""Large-document and opened two-page (passport) geometry regression tests."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from idcam.config.schema import DetectionConfig
from idcam.vision.document_detector import DocumentDetector
from idcam.vision.geometry import (
    MAX_AREA_FRACTION,
    MAX_ASPECT,
    MIN_AREA_FRACTION,
    MIN_ASPECT,
    MIN_CONFIDENCE,
    MIN_RECTANGULARITY,
    MIN_RIGHT_ANGLE,
    _build_large_document_candidate,
    bbox_iou,
    find_document_quad,
    order_corners,
)


@pytest.fixture
def detection_config():
    return DetectionConfig(
        motion_threshold=5.0,
        edge_density_threshold=0.01,
        stable_frames_required=3,
        removal_frames_required=3,
    )


def _wood_table(shape: tuple[int, int], *, brightness: int = 0) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices((h, w))
    base = (85 + (xx * 3 + yy * 5) % 45).astype(np.int16) + brightness
    base = np.clip(base, 0, 255).astype(np.uint8)
    g = (base.astype(np.int16) * 70 // 100).clip(0, 255).astype(np.uint8)
    b = (base.astype(np.int16) * 48 // 100).clip(0, 255).astype(np.uint8)
    return np.stack([base, g, b], axis=-1)


def _draw_page(
    frame: np.ndarray,
    x: int,
    y: int,
    pw: int,
    ph: int,
    *,
    fill=(242, 238, 230),
    printed: bool = False,
) -> np.ndarray:
    img = frame.copy()
    img[y : y + ph, x : x + pw] = fill
    img[y : y + 3, x : x + pw] = 35
    img[y + ph - 3 : y + ph, x : x + pw] = 35
    img[y : y + ph, x : x + 3] = 35
    img[y : y + ph, x + pw - 3 : x + pw] = 35
    if printed:
        for row in range(y + 16, y + ph - 16, 12):
            img[row : row + 2, x + 18 : x + pw - 18] = (30, 30, 36)
    return img


def _page(shape, box, **kwargs) -> np.ndarray:
    return _draw_page(_wood_table(shape), *box, **kwargs)


def _opened_passport(
    shape: tuple[int, int],
    box: tuple[int, int, int, int],
    *,
    seam_px: int = 6,
    left_fill=(242, 238, 230),
    right_fill=(236, 228, 214),
    fold_shadow: bool = True,
) -> np.ndarray:
    x, y, pw, ph = box
    img = _wood_table(shape)
    mid = x + pw // 2
    img = _draw_page(img, x, y, mid - x, ph, fill=left_fill, printed=True)
    img = _draw_page(img, mid, y, x + pw - mid, ph, fill=right_fill, printed=True)
    seam0 = max(x, mid - seam_px // 2)
    seam1 = min(x + pw, seam0 + max(2, seam_px))
    img[y : y + ph, seam0:seam1] = (28, 24, 22)
    if fold_shadow:
        shade = np.linspace(18, 0, min(40, pw // 8), dtype=np.int16)
        for i, s in enumerate(shade):
            col = mid - i
            if x <= col < x + pw:
                img[y : y + ph, col] = np.clip(img[y : y + ph, col].astype(np.int16) - s, 0, 255).astype(np.uint8)
            col = mid + i
            if x <= col < x + pw:
                img[y : y + ph, col] = np.clip(img[y : y + ph, col].astype(np.int16) - s, 0, 255).astype(np.uint8)
    return img


class TestSinglePageRegression:
    def test_normal_single_page(self, detection_config):
        geom = DocumentDetector(detection_config).detect_geometry(_page((480, 640), (80, 60, 480, 360)))
        assert geom.present is True
        assert geom.corners is not None
        assert len(geom.corners) == 4

    def test_small_document(self, detection_config):
        # ~11% of 640×480
        geom = DocumentDetector(detection_config).detect_geometry(_page((480, 640), (200, 150, 220, 160)))
        assert geom.present is True

    def test_large_single_page(self, detection_config):
        box = (120, 80, 1400, 880)
        geom = DocumentDetector(detection_config).detect_geometry(_page((1080, 1920), box))
        assert geom.present is True
        assert bbox_iou(geom.bbox, box) > 0.75
        assert geom.selected_area_ratio > 0.45

    def test_very_large_document(self, detection_config):
        box = (40, 30, 1840, 1020)
        geom = DocumentDetector(detection_config).detect_geometry(_page((1080, 1920), box))
        assert geom.present is True
        assert bbox_iou(geom.bbox, box) > 0.70
        assert geom.selected_area_ratio > 0.70


class TestAreaOccupancy:
    @pytest.mark.parametrize(
        "box,min_ratio",
        [
            ((480, 270, 960, 540), 0.22),  # ~25%
            ((288, 162, 1344, 756), 0.40),  # ~49%
            ((192, 108, 1536, 864), 0.55),  # ~64%
            ((64, 40, 1792, 1000), 0.70),  # ~82%
        ],
    )
    def test_occupancy_bands(self, detection_config, box, min_ratio):
        geom = DocumentDetector(detection_config).detect_geometry(_page((1080, 1920), box))
        assert geom.present is True
        assert geom.selected_area_ratio >= min_ratio * 0.75
        assert bbox_iou(geom.bbox, box) > 0.65


class TestOpenedPassport:
    def test_two_page_is_one_document(self, detection_config):
        box = (80, 90, 1480, 720)
        frame = _opened_passport((1080, 1920), box, seam_px=8)
        geom = DocumentDetector(detection_config).detect_geometry(frame)
        assert geom.present is True
        assert geom.corners is not None
        assert bbox_iou(geom.bbox, box) > 0.70
        x, y, w, h = geom.bbox
        assert w > box[2] * 0.70
        cx = box[0] + box[2] // 2
        assert x < cx < x + w

    def test_strong_center_seam(self, detection_config):
        box = (100, 120, 1400, 680)
        frame = _opened_passport((1080, 1920), box, seam_px=18)
        geom = DocumentDetector(detection_config).detect_geometry(frame)
        assert geom.present is True
        assert bbox_iou(geom.bbox, box) > 0.65
        raw, _n = find_document_quad(frame)
        assert raw is not None
        assert raw.geometry_mode in {
            "quad",
            "merged_two_page",
            "min_rect_fallback",
            "paper_region",
            "large_document_outer",
        }

    def test_uneven_page_brightness(self, detection_config):
        box = (90, 100, 1500, 740)
        frame = _opened_passport(
            (1080, 1920),
            box,
            left_fill=(250, 248, 240),
            right_fill=(190, 175, 150),
            fold_shadow=True,
        )
        geom = DocumentDetector(detection_config).detect_geometry(frame)
        assert geom.present is True
        assert bbox_iou(geom.bbox, box) > 0.65

    def test_vertically_clipped_opened_passport(self, detection_config):
        """Pi CASE A: passport hits the top and bottom of 1920×1080 preview."""
        box = (80, 0, 1400, 1080)
        frame = _opened_passport((1080, 1920), box, seam_px=10)
        geom = DocumentDetector(detection_config).detect_geometry(frame)
        assert geom.present is True
        assert geom.bbox is not None
        assert geom.bbox[2] > 900
        assert bbox_iou(geom.bbox, box) > 0.55

    def test_clipped_page_without_drawn_border(self, detection_config):
        """Hawkeye preview has no synthetic dark outline; paper vs table must carry it."""
        img = _wood_table((1080, 1920))
        box = (80, 0, 1400, 1080)
        x, y, pw, ph = box
        img[y : y + ph, x : x + pw] = (242, 238, 230)
        geom = DocumentDetector(detection_config).detect_geometry(img)
        assert geom.present is True
        assert geom.bbox is not None
        assert bbox_iou(geom.bbox, box) > 0.55
        assert geom.geometry_mode in (
            "paper_region",
            "min_rect_fallback",
            "quad",
            "merged_two_page",
            "large_document_outer",
        )


class TestNegatives:
    def test_background_large_rectangle_on_table_is_the_document_or_none(self, detection_config):
        """A single large page on wood is a document; wood alone is not."""
        table = _wood_table((1080, 1920))
        assert DocumentDetector(detection_config).detect_geometry(table).present is False
        page = _page((1080, 1920), (400, 250, 900, 560))
        assert DocumentDetector(detection_config).detect_geometry(page).present is True

    def test_two_unrelated_rectangles_are_not_merged(self, detection_config):
        img = _wood_table((1080, 1920))
        left = (80, 200, 520, 400)
        right = (1280, 220, 520, 400)
        img = _draw_page(img, *left, printed=True)
        img = _draw_page(img, *right, printed=True)
        geom = DocumentDetector(detection_config).detect_geometry(img)
        union = (80, 200, 1720, 420)
        if geom.present and geom.bbox is not None:
            assert bbox_iou(geom.bbox, union) < 0.55
            assert geom.bbox[2] < 1100


class TestAfRoiCompatibility:
    def test_bbox_stays_inside_passport(self, detection_config):
        box = (120, 140, 1400, 700)
        frame = _opened_passport((1080, 1920), box)
        detector = DocumentDetector(detection_config)
        geom = detector.detect_geometry(frame)
        assert geom.present is True
        roi = detector.document_preview_roi(frame)
        rx, ry, rw, rh = roi
        assert rx >= box[0] - 30
        assert ry >= box[1] - 30
        assert rx + rw <= box[0] + box[2] + 30
        assert ry + rh <= box[1] + box[3] + 30
        assert rw * rh > 0.4 * box[2] * box[3]


_PASSPORT_MODES = {
    "quad",
    "merged_two_page",
    "min_rect_fallback",
    "paper_region",
    "large_document_outer",
}


def _rotate_frame(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _perspective_warp_box(img: np.ndarray, box: tuple[int, int, int, int], *, skew: int = 90) -> np.ndarray:
    h, w = img.shape[:2]
    x, y, pw, ph = box
    src = np.float32([[x, y], [x + pw, y], [x + pw, y + ph], [x, y + ph]])
    dst = np.float32(
        [
            [x + skew, y + 12],
            [x + pw - skew // 3, y - 8],
            [x + pw + 10, y + ph - 18],
            [x + 16, y + ph + 8],
        ]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


class TestNormalThresholdsUnchanged:
    def test_hard_filters_are_the_published_values(self):
        assert MIN_AREA_FRACTION == 0.08
        assert MAX_AREA_FRACTION == 0.97
        assert MIN_ASPECT == 0.40
        assert MAX_ASPECT == 3.35
        assert MIN_RECTANGULARITY == 0.72
        assert MIN_CONFIDENCE == 0.48
        assert MIN_RIGHT_ANGLE == 0.55

    def test_landscape_id_still_detected(self, detection_config):
        geom = DocumentDetector(detection_config).detect_geometry(_page((480, 640), (80, 60, 480, 360)))
        assert geom.present is True

    def test_portrait_document_still_detected(self, detection_config):
        geom = DocumentDetector(detection_config).detect_geometry(_page((480, 640), (210, 30, 220, 420)))
        assert geom.present is True


class TestRotatedAndSkewedPassport:
    def test_open_passport_rotated_10_deg(self, detection_config):
        box = (220, 200, 1480, 680)
        frame = _rotate_frame(_opened_passport((1080, 1920), box, seam_px=8), 10)
        geom = DocumentDetector(detection_config).detect_geometry(frame)
        assert geom.present is True
        assert geom.bbox is not None
        assert geom.bbox[2] > 900
        raw, _n = find_document_quad(frame)
        assert raw is not None
        assert raw.geometry_mode in _PASSPORT_MODES

    def test_open_passport_rotated_25_deg(self, detection_config):
        box = (280, 230, 1360, 620)
        frame = _rotate_frame(_opened_passport((1080, 1920), box, seam_px=8), 25)
        geom = DocumentDetector(detection_config).detect_geometry(frame)
        assert geom.present is True
        assert geom.bbox is not None
        assert max(geom.bbox[2], geom.bbox[3]) > 700
        raw, _n = find_document_quad(frame)
        assert raw is not None
        assert raw.geometry_mode in _PASSPORT_MODES

    def test_perspective_skewed_open_passport(self, detection_config):
        box = (160, 160, 1500, 720)
        frame = _perspective_warp_box(_opened_passport((1080, 1920), box, seam_px=8), box, skew=110)
        geom = DocumentDetector(detection_config).detect_geometry(frame)
        assert geom.present is True
        assert geom.bbox is not None
        assert geom.bbox[2] > 800


class TestOversizedAndBorderContact:
    def test_slightly_oversized_passport(self, detection_config):
        box = (20, 12, 1880, 1054)
        geom = DocumentDetector(detection_config).detect_geometry(_opened_passport((1080, 1920), box, seam_px=8))
        assert geom.present is True
        assert geom.bbox is not None
        assert geom.bbox[2] > 1200

    def test_passport_touching_one_border(self, detection_config):
        box = (180, 0, 1480, 820)
        geom = DocumentDetector(detection_config).detect_geometry(_opened_passport((1080, 1920), box, seam_px=8))
        assert geom.present is True
        assert geom.bbox is not None
        assert geom.bbox[1] <= 8

    def test_passport_touching_two_borders(self, detection_config):
        box = (0, 0, 1500, 900)
        geom = DocumentDetector(detection_config).detect_geometry(_opened_passport((1080, 1920), box, seam_px=8))
        assert geom.present is True
        assert geom.bbox is not None
        assert geom.bbox[2] > 900


class TestLargeDocumentNegatives:
    def test_full_frame_rectangle_rejected(self, detection_config):
        frame = np.full((1080, 1920, 3), 242, dtype=np.uint8)
        geom = DocumentDetector(detection_config).detect_geometry(frame)
        assert geom.present is False

    def test_random_wide_rectangle_rejected(self, detection_config):
        img = _wood_table((1080, 1920))
        img[440:660, 50:1870] = (242, 238, 230)
        geom = DocumentDetector(detection_config).detect_geometry(img)
        assert geom.present is False

    def test_random_rotated_wide_rectangle_rejected(self, detection_config):
        img = _wood_table((1080, 1920))
        img[450:670, 70:1850] = (242, 238, 230)
        geom = DocumentDetector(detection_config).detect_geometry(_rotate_frame(img, 18))
        assert geom.present is False

    def test_fold_does_not_split_into_two_documents(self, detection_config):
        box = (100, 140, 1500, 700)
        frame = _opened_passport((1080, 1920), box, seam_px=16)
        raw, count = find_document_quad(frame)
        assert raw is not None
        assert raw.geometry_mode in _PASSPORT_MODES
        geom = DocumentDetector(detection_config).detect_geometry(frame)
        assert geom.present is True
        assert geom.bbox[2] > box[2] * 0.65


class TestLargeDocumentEvidenceGates:
    def test_low_raw_rectangularity_alone_is_rejected(self):
        corners = order_corners(np.array([[80.0, 80.0], [880.0, 80.0], [880.0, 500.0], [80.0, 500.0]]))
        gray = np.full((540, 960), 140, dtype=np.uint8)
        edges = np.zeros((540, 960), dtype=np.uint8)
        cand = _build_large_document_candidate(
            corners,
            contour_area=8000.0,
            work_area=540.0 * 960.0,
            min_rect_area=800.0 * 420.0,
            scale=0.5,
            src_width=1920,
            src_height=1080,
            work_gray=gray,
            edge_map=edges,
            recovered_source="min_rect",
            log_reject=False,
        )
        assert cand is None

    def test_recovered_quad_with_fold_and_edges_can_accept(self):
        corners = order_corners(np.array([[80.0, 80.0], [880.0, 90.0], [870.0, 500.0], [90.0, 490.0]]))
        gray = np.full((540, 960), 210, dtype=np.uint8)
        gray[:, 470:490] = 40
        edges = np.zeros((540, 960), dtype=np.uint8)
        pts = corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(edges, [pts], True, 255, 3)
        cand = _build_large_document_candidate(
            corners,
            contour_area=8000.0,
            work_area=540.0 * 960.0,
            min_rect_area=800.0 * 420.0,
            scale=0.5,
            src_width=1920,
            src_height=1080,
            work_gray=gray,
            edge_map=edges,
            recovered_source="min_rect",
            log_reject=False,
        )
        assert cand is not None
        assert cand.geometry_mode == "large_document_outer"
        assert cand.quad_aspect >= 1.12
