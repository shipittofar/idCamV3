"""Position / scale / rotation invariance of geometry + confirmation."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from idcam.config.schema import DetectionConfig
from idcam.vision.document_confirmation import confirm_document
from idcam.vision.document_detector import DocumentDetector
from idcam.vision.geometry import (
    MAX_AREA_FRACTION,
    MAX_ASPECT,
    MIN_AREA_FRACTION,
    MIN_ASPECT,
    MIN_CONFIDENCE,
    MIN_RECTANGULARITY,
    MIN_RIGHT_ANGLE,
    _quad_aspect_from_sides,
    find_document_quad,
    order_corners,
)

from tests.unit.test_document_confirmation import (
    _full_frame_rect,
    _grain_rectangle,
    _id_card,
    _mat_frame,
    _perspective_id,
    _two_distant_rects,
    _window_frame,
)
from tests.unit.test_large_two_page_geometry import _opened_passport, _page, _wood_table


@pytest.fixture
def detection_config():
    return DetectionConfig(
        motion_threshold=5.0,
        edge_density_threshold=0.01,
        stable_frames_required=2,
        removal_frames_required=2,
        geometry_stable_frames_required=2,
        geometry_stability_threshold_px=12.0,
    )


def _detect_confirm(frame, detection_config):
    geom = DocumentDetector(detection_config).detect_geometry(frame)
    conf = confirm_document(frame, geom)
    return geom, conf


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    border = (int(img[0, 0, 0]), int(img[0, 0, 1]), int(img[0, 0, 2]))
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=border)


class TestGatesUnchanged:
    def test_normal_thresholds_frozen(self):
        assert MIN_AREA_FRACTION == 0.08
        assert MAX_AREA_FRACTION == 0.97
        assert MIN_ASPECT == 0.40
        assert MAX_ASPECT == 3.35
        assert MIN_RECTANGULARITY == 0.72
        assert MIN_CONFIDENCE == 0.48
        assert MIN_RIGHT_ANGLE == 0.55


class TestCornerOrdering:
    def test_axis_aligned_tl_tr_br_bl(self):
        pts = np.array([[200.0, 10.0], [10.0, 10.0], [10.0, 90.0], [200.0, 90.0]])
        ordered = order_corners(pts)
        np.testing.assert_allclose(ordered[0], [10, 10])
        np.testing.assert_allclose(ordered[1], [200, 10])
        np.testing.assert_allclose(ordered[2], [200, 90])
        np.testing.assert_allclose(ordered[3], [10, 90])

    def test_diamond_is_cyclic_not_xy_paired(self):
        pts = np.array([[100.0, 40.0], [160.0, 100.0], [100.0, 160.0], [40.0, 100.0]])
        ordered = order_corners(pts)
        unique = {tuple(np.round(p, 3)) for p in ordered}
        assert len(unique) == 4
        aspect = _quad_aspect_from_sides(ordered)
        assert aspect == pytest.approx(1.0, abs=0.08)


class TestPositionMatrix:
    @pytest.mark.parametrize(
        "name,box",
        [
            ("center", (180, 140, 280, 180)),
            ("left", (12, 140, 260, 170)),
            ("right", (360, 140, 260, 170)),
            ("top", (180, 8, 280, 150)),
            ("bottom", (180, 310, 280, 150)),
            ("top_left", (8, 8, 240, 150)),
            ("top_right", (390, 8, 240, 150)),
            ("bottom_left", (8, 320, 240, 150)),
            ("bottom_right", (390, 320, 240, 150)),
        ],
    )
    def test_id_anywhere(self, detection_config, name, box):
        geom, conf = _detect_confirm(_id_card(box=box, printed=True), detection_config)
        assert geom.present is True, name
        assert conf.accepted is True, (name, conf.reason)


class TestSizeMatrix:
    @pytest.mark.parametrize(
        "name,box",
        [
            ("small", (230, 170, 160, 110)),
            ("medium", (160, 120, 320, 220)),
            ("large", (40, 30, 560, 420)),
            ("very_large", (16, 12, 608, 456)),
        ],
    )
    def test_size_bands(self, detection_config, name, box):
        geom, conf = _detect_confirm(_id_card(box=box, printed=True), detection_config)
        assert geom.present is True, name
        assert conf.accepted is True, (name, conf.reason)


class TestOrientationMatrix:
    @pytest.mark.parametrize("angle", [0, 10, 20, 30, 45])
    def test_rotated_id(self, detection_config, angle):
        frame = _id_card(box=(160, 120, 300, 200), printed=True, angle=float(angle))
        geom, conf = _detect_confirm(frame, detection_config)
        assert geom.present is True, angle
        assert conf.accepted is True, (angle, conf.reason)

    def test_portrait_and_landscape(self, detection_config):
        land = _detect_confirm(_page((480, 640), (80, 140, 480, 220), printed=True), detection_config)
        port = _detect_confirm(_page((480, 640), (220, 40, 200, 400), printed=True), detection_config)
        assert land[0].present and land[1].accepted
        assert port[0].present and port[1].accepted


class TestPerspectiveMatrix:
    def test_mild_perspective(self, detection_config):
        geom, conf = _detect_confirm(_perspective_id(), detection_config)
        assert geom.present is True
        assert conf.accepted is True

    def test_keystone_page(self, detection_config):
        h, w = 480, 640
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dst = np.float32([[70, 40], [w - 50, 5], [w - 20, h - 25], [30, h - 50]])
        page = _id_card((h, w), (100, 80, 400, 260), printed=True)
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(page, matrix, (w, h), borderValue=(90, 63, 43))
        geom, conf = _detect_confirm(warped, detection_config)
        assert geom.present is True
        assert conf.accepted is True


class TestBorderContact:
    def test_one_border(self, detection_config):
        geom, conf = _detect_confirm(_id_card(box=(0, 140, 280, 180), printed=True), detection_config)
        assert conf.accepted is True
        assert geom.present is True

    def test_two_borders(self, detection_config):
        geom, conf = _detect_confirm(_id_card(box=(0, 0, 300, 200), printed=True), detection_config)
        assert geom.present is True
        assert conf.accepted is True

    def test_clipped_one_side(self, detection_config):
        geom, conf = _detect_confirm(_id_card(box=(0, 140, 260, 180), printed=True), detection_config)
        assert geom.present is True
        assert conf.accepted is True


class TestDocumentTypes:
    def test_blank_sheet(self, detection_config):
        geom, conf = _detect_confirm(_id_card(printed=False), detection_config)
        assert conf.accepted is True

    def test_open_passport(self, detection_config):
        frame = _opened_passport((1080, 1920), (80, 180, 1760, 720))
        geom, conf = _detect_confirm(frame, detection_config)
        assert conf.accepted is True
        assert geom.present is True

    def test_open_passport_rotated(self, detection_config):
        frame = _rotate(_opened_passport((1080, 1920), (200, 220, 1500, 640)), 15)
        geom, conf = _detect_confirm(frame, detection_config)
        assert geom.present is True
        assert conf.accepted is True


class TestNegatives:
    @pytest.mark.parametrize(
        "name,factory",
        [
            ("empty", lambda: _wood_table((480, 640))),
            ("grain", _grain_rectangle),
            ("mat", _mat_frame),
            ("wall", _window_frame),
            ("full_frame", _full_frame_rect),
        ],
    )
    def test_background_rejected(self, detection_config, name, factory):
        _geom, conf = _detect_confirm(factory(), detection_config)
        assert conf.accepted is False, name

    def test_two_distant_not_merged_span(self, detection_config):
        geom, conf = _detect_confirm(_two_distant_rects(), detection_config)
        if geom.present and geom.bbox is not None:
            x, y, w, h = geom.bbox
            assert not (x < 40 and x + w > 560 and y < 60 and y + h > 380)
        if conf.accepted and geom.bbox is not None:
            x, y, w, h = geom.bbox
            assert w < 500

    def test_table_edge_bar(self, detection_config):
        img = _wood_table((480, 640))
        img[200:230, :] = (40, 32, 28)
        _geom, conf = _detect_confirm(img, detection_config)
        assert conf.accepted is False

    def test_finder_does_not_prefer_center(self, detection_config):
        left = find_document_quad(_id_card(box=(12, 140, 260, 170), printed=True))[0]
        center = find_document_quad(_id_card(box=(190, 140, 260, 170), printed=True))[0]
        assert left is not None and center is not None
        assert left.bbox[0] < center.bbox[0]
