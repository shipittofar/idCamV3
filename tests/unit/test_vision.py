"""Vision pipeline tests."""

import cv2
import numpy as np
import pytest

from idcam.config.schema import DetectionConfig
from idcam.vision.document_detector import (
    DocumentDetector,
    DocumentDetectionDebug,
    DocumentGeometryResult,
    DocumentPresence,
    gradient_edge_density,
)
from idcam.vision.geometry import bbox_from_corners, bbox_iou, order_corners
from idcam.vision.sharpness import laplacian_variance, mean_brightness
from idcam.vision.document_detector import DocumentDetector
from idcam.vision.geometry_stability import GeometryStability, mean_corner_displacement_px
from idcam.vision.stability_detector import StabilityDetector


@pytest.fixture
def detection_config():
    return DetectionConfig(
        motion_threshold=5.0,
        edge_density_threshold=0.01,
        stable_frames_required=3,
        removal_frames_required=3,
    )


class TestDocumentDetector:
    def test_no_document_on_uniform_frame(self, detection_config):
        detector = DocumentDetector(detection_config)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        detector.update_baseline(frame)
        assert not detector.detect(frame)

    def test_document_detected_on_change(self, detection_config):
        detector = DocumentDetector(detection_config)
        baseline = np.full((480, 640, 3), 200, dtype=np.uint8)
        detector.update_baseline(baseline)
        frame = baseline.copy()
        # Add high-contrast document region
        frame[100:380, 100:540] = 50
        frame[100:120, 100:540] = 0
        frame[360:380, 100:540] = 0
        frame[100:380, 100:120] = 0
        frame[100:380, 520:540] = 0
        assert detector.detect(frame)

    def test_scene_empty_after_removal(self, detection_config):
        detector = DocumentDetector(detection_config)
        frame = np.full((480, 640, 3), 150, dtype=np.uint8)
        for _ in range(20):
            detector.update_baseline(frame)
        assert detector.is_scene_empty(frame)


class TestDocumentDetectionDebug:
    def test_debug_values_match_detection_without_changing_result(self, detection_config):
        detector = DocumentDetector(detection_config)
        baseline = np.full((480, 640, 3), 200, dtype=np.uint8)
        detector.update_baseline(baseline)
        frame = baseline.copy()
        frame[100:380, 100:540] = 50
        frame[100:120, 100:540] = 0
        frame[360:380, 100:540] = 0
        frame[100:380, 100:120] = 0
        frame[100:380, 520:540] = 0
        result = detector.detect(frame)
        info = detector.debug_info()
        assert isinstance(info, DocumentDetectionDebug)
        assert result is True
        assert info.detected is True
        assert info.motion > info.motion_threshold
        assert info.edge_density > info.edge_density_threshold
        assert detector.detect(frame) is True

    def test_debug_info_does_not_run_detection(self, detection_config):
        detector = DocumentDetector(detection_config)
        before = detector.debug_info()
        again = detector.debug_info()
        assert before == again
        assert before.detected is False
        assert before.motion == 0.0

    def test_reject_reason_on_uniform_scene(self, detection_config):
        detector = DocumentDetector(detection_config)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        detector.detect(frame)
        detector.detect(frame)
        info = detector.debug_info()
        assert info.detected is False
        assert info.reject_reason
        assert info.roi_width > 0
        assert info.frame_width == 640

    def test_rgb_preview_format_detects_document(self, detection_config):
        detector = DocumentDetector(detection_config)
        baseline = np.zeros((240, 320, 3), dtype=np.uint8)
        baseline[:] = (180, 40, 40)  # RGB red-ish empty scene
        detector.update_baseline(baseline)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:] = (40, 180, 40)
        frame[20:220:4, :] = (10, 10, 10)
        frame[:, 20:300:6] = (240, 240, 240)
        assert detector.detect(frame)
        info = detector.debug_info()
        assert info.edge_density > info.edge_density_threshold
        assert info.reject_reason == ""

    def test_static_document_is_not_moving(self, detection_config):
        detector = DocumentDetector(detection_config)
        empty = np.full((240, 320, 3), 200, dtype=np.uint8)
        doc = empty.copy()
        doc[40:200, 40:280] = 50
        doc[40:50, 40:280] = 0
        doc[190:200, 40:280] = 0
        detector.update_baseline(empty)
        first = detector.classify(doc)
        assert first != DocumentPresence.NO_DOCUMENT
        still = detector.classify(doc)
        assert still != DocumentPresence.DOCUMENT_MOVING

    def test_textured_empty_table_is_scene_empty(self, detection_config):
        detector = DocumentDetector(detection_config)
        yy, xx = np.indices((240, 320))
        table = np.stack(
            [
                (80 + (xx % 16)).astype(np.uint8),
                (70 + (yy % 16)).astype(np.uint8),
                np.full((240, 320), 60, dtype=np.uint8),
            ],
            axis=-1,
        )
        detector.update_baseline(table)
        assert detector.is_scene_empty(table)
        assert detector.last_edge_density > 0.0

    def test_mild_defocus_has_gradient_edges(self, detection_config):
        sharp = np.full((240, 320, 3), 200, dtype=np.uint8)
        sharp[40:200, 40:280] = 40
        for y in range(50, 190, 8):
            sharp[y : y + 2, 50:270] = 0
        mild = cv2.GaussianBlur(sharp, (9, 9), 0)
        gray = cv2.cvtColor(mild, cv2.COLOR_RGB2GRAY)
        canny = cv2.Canny(gray, 50, 150)
        canny_density = float(np.count_nonzero(canny)) / canny.size
        sobel_density = gradient_edge_density(gray)
        assert sobel_density > canny_density
        detector = DocumentDetector(detection_config)
        detector.update_baseline(np.full((240, 320, 3), 180, dtype=np.uint8))
        assert detector.detect(mild)


class TestStabilityDetector:
    def test_not_stable_initially(self, detection_config):
        sd = StabilityDetector(detection_config)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        assert not sd.update(frame)

    def test_stable_after_consecutive_frames(self, detection_config):
        sd = StabilityDetector(detection_config)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = False
        for _ in range(10):
            result = sd.update(frame)
        assert result

    def test_reset_on_motion(self, detection_config):
        sd = StabilityDetector(detection_config)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        for _ in range(2):
            sd.update(frame)
        moving = frame.copy()
        moving[200:300, 200:300] = 255
        sd.update(moving)
        assert sd.stable_count == 0

    def test_stability_debug_info(self, detection_config):
        sd = StabilityDetector(detection_config)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        results = [sd.update(frame) for _ in range(10)]
        info = sd.debug_info()
        assert results[-1] is True
        assert info.stable is True
        assert info.stable_count == sd.stable_count
        assert info.stable_frames_required == detection_config.stable_frames_required
        assert info.threshold == detection_config.motion_threshold
        assert sd.update(frame) is True

    def test_small_preview_noise_keeps_stable_document(self, detection_config):
        sd = StabilityDetector(detection_config)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        for _ in range(10):
            sd.update(frame)
        assert sd.stable_count >= detection_config.stable_frames_required
        noisy = frame.copy()
        noisy[200:210, 200:210] = 129
        assert sd.update(noisy) is True
        assert sd.stable_count >= detection_config.stable_frames_required


class TestDocumentPreviewRoi:
    def test_roi_is_preview_pixels_inside_frame(self, detection_config):
        detector = DocumentDetector(detection_config)
        empty = np.full((1080, 1920, 3), 200, dtype=np.uint8)
        doc = _page_on_table((1080, 1920), (400, 200, 1120, 680))
        geom = detector.detect_geometry(doc)
        assert geom.present is True
        roi = detector.document_preview_roi(doc)
        x, y, w, h = roi
        assert x >= 0 and y >= 0
        assert x + w <= 1920
        assert y + h <= 1080
        assert w > 100 and h > 100
        assert roi != (1936, 1960, 5376, 3024)


def _wood_table(shape: tuple[int, int], *, brightness: int = 0) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices((h, w))
    base = (85 + (xx * 3 + yy * 5) % 45).astype(np.int16) + brightness
    base = np.clip(base, 0, 255).astype(np.uint8)
    g = (base.astype(np.int16) * 70 // 100).clip(0, 255).astype(np.uint8)
    b = (base.astype(np.int16) * 48 // 100).clip(0, 255).astype(np.uint8)
    return np.stack([base, g, b], axis=-1)


def _draw_page(frame: np.ndarray, x: int, y: int, pw: int, ph: int, *, printed: bool = False) -> np.ndarray:
    img = frame.copy()
    img[y : y + ph, x : x + pw] = (242, 238, 230)
    img[y : y + 3, x : x + pw] = 35
    img[y + ph - 3 : y + ph, x : x + pw] = 35
    img[y : y + ph, x : x + 3] = 35
    img[y : y + ph, x + pw - 3 : x + pw] = 35
    if printed:
        for row in range(y + 20, y + ph - 20, 14):
            img[row : row + 3, x + 24 : x + pw - 24] = (25, 25, 30)
    return img


def _page_on_table(shape: tuple[int, int], box: tuple[int, int, int, int], *, printed: bool = False) -> np.ndarray:
    return _draw_page(_wood_table(shape), *box, printed=printed)


def _perspective_page(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """White page with mild perspective. Corners are the drawn polygon (TL TR BR BL)."""
    h, w = shape
    img = _wood_table(shape)
    corners = np.array(
        [
            [int(w * 0.18), int(h * 0.16)],
            [int(w * 0.84), int(h * 0.12)],
            [int(w * 0.88), int(h * 0.86)],
            [int(w * 0.14), int(h * 0.82)],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(img, corners, (244, 240, 232))
    cv2.polylines(img, [corners], True, (28, 28, 28), 4, cv2.LINE_AA)
    return img, corners.astype(np.float32)


class TestDocumentGeometry:
    def test_page_present_at_t0_without_baseline(self, detection_config):
        detector = DocumentDetector(detection_config)
        frame = _page_on_table((480, 640), (80, 60, 480, 360))
        geom = detector.detect_geometry(frame)
        assert geom.present is True
        assert geom.corners is not None
        assert geom.bbox is not None
        assert geom.confidence > 0.48

    def test_page_after_empty_frame(self, detection_config):
        detector = DocumentDetector(detection_config)
        empty = _wood_table((480, 640))
        assert detector.detect_geometry(empty).present is False
        page = _page_on_table((480, 640), (80, 60, 480, 360))
        assert detector.detect_geometry(page).present is True

    def test_slowly_introduced_page_does_not_use_motion_threshold(self, detection_config):
        detector = DocumentDetector(detection_config)
        table = _wood_table((480, 640))
        found = False
        for width in (40, 80, 160, 280, 420):
            frame = _draw_page(table, 80, 70, width, int(width * 0.72))
            geom = detector.detect_geometry(frame)
            if geom.present:
                found = True
                break
        assert found is True

    def test_empty_uniform_scene(self, detection_config):
        detector = DocumentDetector(detection_config)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        geom = detector.detect_geometry(frame)
        assert geom.present is False
        assert geom.corners is None

    def test_textured_empty_table(self, detection_config):
        detector = DocumentDetector(detection_config)
        geom = detector.detect_geometry(_wood_table((480, 640)))
        assert geom.present is False

    def test_textured_table_brightness_shift_is_not_a_document(self, detection_config):
        detector = DocumentDetector(detection_config)
        table = _wood_table((480, 640), brightness=0)
        assert detector.detect_geometry(table).present is False
        shifted = _wood_table((480, 640), brightness=28)
        geom = detector.detect_geometry(shifted)
        assert geom.present is False

    def test_hand_like_blob_rejected(self, detection_config):
        detector = DocumentDetector(detection_config)
        frame = _wood_table((480, 640))
        cv2.circle(frame, (320, 240), 70, (210, 160, 140), -1)
        cv2.circle(frame, (370, 200), 35, (210, 160, 140), -1)
        cv2.circle(frame, (390, 250), 28, (210, 160, 140), -1)
        geom = detector.detect_geometry(frame)
        assert geom.present is False

    def test_shadow_gradient_is_not_a_document(self, detection_config):
        detector = DocumentDetector(detection_config)
        frame = _wood_table((480, 640)).astype(np.int16)
        shade = np.linspace(0, 50, 640, dtype=np.int16)
        frame -= shade[None, :, None]
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        geom = detector.detect_geometry(frame)
        assert geom.present is False

    def test_perspective_document_returns_four_ordered_corners(self, detection_config):
        detector = DocumentDetector(detection_config)
        frame, truth = _perspective_page((720, 960))
        geom = detector.detect_geometry(frame)
        assert geom.present is True
        assert geom.corners is not None
        assert len(geom.corners) == 4
        ordered = order_corners(np.asarray(geom.corners))
        assert ordered[0, 0] + ordered[0, 1] <= ordered[2, 0] + ordered[2, 1]
        assert ordered[1, 0] >= ordered[0, 0]
        assert ordered[3, 1] >= ordered[0, 1]

    def test_corner_order_is_tl_tr_br_bl(self):
        pts = np.array([[200.0, 10.0], [10.0, 10.0], [10.0, 90.0], [200.0, 90.0]])
        ordered = order_corners(pts)
        np.testing.assert_allclose(ordered[0], [10, 10])
        np.testing.assert_allclose(ordered[1], [200, 10])
        np.testing.assert_allclose(ordered[2], [200, 90])
        np.testing.assert_allclose(ordered[3], [10, 90])

    def test_bbox_encloses_corners(self, detection_config):
        detector = DocumentDetector(detection_config)
        frame = _page_on_table((480, 640), (80, 60, 480, 360))
        geom = detector.detect_geometry(frame)
        assert geom.present and geom.bbox and geom.corners
        x, y, w, h = geom.bbox
        for cx, cy in geom.corners:
            assert x - 2 <= cx <= x + w + 2
            assert y - 2 <= cy <= y + h + 2
        enclosed = bbox_from_corners(np.asarray(geom.corners))
        assert bbox_iou(geom.bbox, enclosed) > 0.98

    def test_forty_percent_page_iou(self, detection_config):
        detector = DocumentDetector(detection_config)
        # 1200×720 on 1920×1080 ≈ 41.7% of the frame.
        box = (360, 180, 1200, 720)
        frame = _page_on_table((1080, 1920), box)
        geom = detector.detect_geometry(frame)
        assert geom.present is True
        assert geom.bbox is not None
        assert bbox_iou(geom.bbox, box) > 0.80

    def test_printed_document(self, detection_config):
        detector = DocumentDetector(detection_config)
        frame = _page_on_table((480, 640), (70, 50, 500, 380), printed=True)
        assert detector.detect_geometry(frame).present is True

    def test_mildly_defocused_document(self, detection_config):
        detector = DocumentDetector(detection_config)
        sharp = _page_on_table((480, 640), (80, 60, 480, 360), printed=True)
        mild = cv2.GaussianBlur(sharp, (9, 9), 0)
        geom = detector.detect_geometry(mild)
        assert geom.present is True

    def test_corner_jitter_is_smoothed(self, detection_config):
        detector = DocumentDetector(detection_config)
        base = _page_on_table((480, 640), (80, 60, 480, 360))
        detector.detect_geometry(base)
        outputs = []
        for dx in (0, 3, -2, 4, -3, 1):
            shifted = np.roll(base, dx, axis=1)
            geom = detector.detect_geometry(shifted)
            assert geom.present is True
            outputs.append(np.asarray(geom.corners)[:, 0].mean())
        raw_span = 4 - (-3)
        out_span = max(outputs) - min(outputs)
        assert out_span < raw_span

    def test_one_bad_frame_does_not_drop_document(self, detection_config):
        detector = DocumentDetector(detection_config)
        page = _page_on_table((480, 640), (80, 60, 480, 360))
        empty = _wood_table((480, 640))
        for _ in range(3):
            assert detector.detect_geometry(page).present is True
        held = detector.detect_geometry(empty)
        assert held.present is True
        assert held.reason == "geometry_held"

    def test_removal_requires_consecutive_misses(self, detection_config):
        detector = DocumentDetector(detection_config)
        page = _page_on_table((480, 640), (80, 60, 480, 360))
        empty = _wood_table((480, 640))
        detector.detect_geometry(page)
        hold = detection_config.geometry_hold_frames
        still = []
        gone = None
        for i in range(hold + 2):
            geom = detector.detect_geometry(empty)
            still.append(geom.present)
            if not geom.present:
                gone = i
                break
        assert gone is not None
        assert gone >= hold - 1
        assert still[0] is True

    def test_does_not_mutate_input(self, detection_config):
        detector = DocumentDetector(detection_config)
        frame = _page_on_table((480, 640), (80, 60, 480, 360))
        original = frame.copy()
        detector.detect_geometry(frame)
        assert np.array_equal(frame, original)

    def test_no_quad_is_not_present(self, detection_config):
        detector = DocumentDetector(detection_config)
        geom = detector.detect_geometry(np.full((240, 320, 3), 40, dtype=np.uint8))
        assert geom.present is False
        assert isinstance(geom, DocumentGeometryResult)

    def test_coordinates_are_preview_pixels_on_1080p(self, detection_config):
        detector = DocumentDetector(detection_config)
        box = (400, 200, 1120, 680)
        frame = _page_on_table((1080, 1920), box)
        geom = detector.detect_geometry(frame)
        assert geom.present is True
        for cx, cy in geom.corners or ():
            assert 0 <= cx < 1920
            assert 0 <= cy < 1080


class TestGeometryStability:
    def test_first_present_frame_is_not_ready(self, detection_config):
        gate = GeometryStability(threshold_px=12.0, frames_required=2)
        detector = DocumentDetector(detection_config)
        geom = detector.detect_geometry(_page_on_table((480, 640), (80, 60, 480, 360)))
        assert geom.present is True
        assert gate.update(geom) is False
        assert gate.stable_count == 0

    def test_stable_quad_reaches_required_count(self, detection_config):
        gate = GeometryStability(threshold_px=12.0, frames_required=2)
        detector = DocumentDetector(detection_config)
        page = _page_on_table((480, 640), (80, 60, 480, 360))
        assert gate.update(detector.detect_geometry(page)) is False
        assert gate.update(detector.detect_geometry(page)) is False
        assert gate.update(detector.detect_geometry(page)) is True
        assert gate.stable_count >= 2

    def test_moving_quad_resets_counter(self, detection_config):
        gate = GeometryStability(threshold_px=12.0, frames_required=3)
        detector = DocumentDetector(detection_config)
        a = detector.detect_geometry(_page_on_table((480, 640), (40, 40, 480, 360)))
        b = detector.detect_geometry(_page_on_table((480, 640), (120, 80, 480, 360)))
        assert a.present and b.present
        assert mean_corner_displacement_px(a.corners, b.corners) > 12.0
        gate.update(a)
        gate.update(a)
        assert gate.stable_count >= 1
        gate.update(b)
        assert gate.stable_count == 0
        assert gate.update(b) is False

    def test_absent_resets(self, detection_config):
        gate = GeometryStability(threshold_px=12.0, frames_required=2)
        detector = DocumentDetector(detection_config)
        page = _page_on_table((480, 640), (80, 60, 480, 360))
        gate.update(detector.detect_geometry(page))
        gate.update(detector.detect_geometry(page))
        absent = DocumentGeometryResult(
            present=False, confidence=0.0, corners=None, bbox=None, reason="no_quadrilateral"
        )
        gate.update(absent)
        assert gate.stable_count == 0


class TestSharpness:
    def test_blurry_lower_than_sharp(self):
        blurry = np.full((480, 640, 3), 128, dtype=np.uint8)
        sharp = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        assert laplacian_variance(blurry) < laplacian_variance(sharp)

    def test_brightness(self):
        dark = np.full((100, 100), 30, dtype=np.uint8)
        bright = np.full((100, 100), 200, dtype=np.uint8)
        assert mean_brightness(bright) > mean_brightness(dark)


class TestBlurKillsEdgeDensity:
    def test_heavy_blur_reduces_textured_edge_density(self):
        """Sobel density of a printed card drops under heavy defocus; a solid
        rectangle's outline can *spread* after Gaussian blur, so this uses texture."""
        h, w = 240, 320
        yy, xx = np.indices((h, w))
        checker = np.where(((xx // 8) + (yy // 8)) % 2 == 0, 220, 40).astype(np.uint8)
        textured = np.stack([checker, checker, checker], axis=-1)
        frame = np.full((h, w, 3), 30, dtype=np.uint8)
        frame[40:200, 40:280] = textured[40:200, 40:280]
        blurred = cv2.GaussianBlur(frame, (51, 51), 18)
        sharp = gradient_edge_density(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
        blur = gradient_edge_density(cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY))
        assert sharp > 0.15
        assert blur < sharp * 0.25

