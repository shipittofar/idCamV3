"""Confirmation edge-support: boundary contrast, not generic Sobel energy."""

from __future__ import annotations

import inspect
import time

import cv2
import numpy as np
import pytest

from idcam.config.schema import DetectionConfig
from idcam.vision.document_confirmation import (
    CONFIRM_WORKING_WIDTH,
    _MIN_SIDE_ON,
    _MIN_SIDES_NORMAL,
    confirm_document,
)
from idcam.vision.document_detector import DocumentDetector, DocumentGeometryResult
from idcam.vision.geometry import WORKING_WIDTH
from idcam.vision.geometry_debug import effective_resolution_chain

from tests.unit.test_document_confirmation import (
    _full_frame_rect,
    _grain_rectangle,
    _id_card,
    _mat_frame,
    _perspective_id,
    _two_distant_rects,
    _window_frame,
)
from tests.unit.test_geometry_invariance import _rotate
from tests.unit.test_large_two_page_geometry import _opened_passport, _wood_table


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


def _dc(frame, detection_config):
    geom = DocumentDetector(detection_config).detect_geometry(frame)
    return geom, confirm_document(frame, geom)


def _weak_right_side(shape=(480, 640)) -> np.ndarray:
    img = _id_card(shape, (90, 80, 420, 280), printed=True)
    x, y, pw, ph = 90, 80, 420, 280
    img[y : y + ph, x + pw - 8 : x + pw] = (236, 230, 220)
    return img


class TestResolutionChain:
    def test_working_widths(self):
        assert WORKING_WIDTH == 1280
        assert CONFIRM_WORKING_WIDTH == 1280
        chain = effective_resolution_chain(1920, 1080)
        assert chain["geometry_working"] == "1280x720"
        assert chain["confirmation_working"] == "1280x720"

    def test_small_preview_is_not_upscaled(self):
        chain = effective_resolution_chain(640, 480)
        assert chain["geometry_working"] == "640x480"
        assert chain["confirmation_working"] == "640x480"


class TestPositiveEdgeSupport:
    @pytest.mark.parametrize(
        "name,factory",
        [
            ("center", lambda: _id_card()),
            ("left", lambda: _id_card(box=(12, 90, 280, 180), printed=True)),
            ("right", lambda: _id_card(box=(340, 90, 260, 180), printed=True)),
            ("top", lambda: _id_card(box=(160, 8, 280, 160), printed=True)),
            ("bottom", lambda: _id_card(box=(160, 300, 280, 160), printed=True)),
            ("corner", lambda: _id_card(box=(8, 8, 240, 150), printed=True)),
            ("small", lambda: _id_card(box=(230, 170, 160, 110), printed=True)),
            ("large", lambda: _id_card(box=(20, 16, 600, 450), printed=True)),
            ("rot10", lambda: _id_card(angle=10, printed=True)),
            ("rot20", lambda: _id_card(angle=20, printed=True)),
            ("rot30", lambda: _id_card(angle=30, printed=True)),
            ("rot45", lambda: _id_card(box=(140, 100, 320, 220), angle=45, printed=True)),
            ("perspective", _perspective_id),
            ("border", lambda: _id_card(box=(0, 140, 280, 180), printed=True)),
        ],
    )
    def test_real_document_confirms(self, detection_config, name, factory):
        geom, conf = _dc(factory(), detection_config)
        assert geom.present is True, name
        assert conf.accepted is True, (name, conf.reason, conf.evidence_edge_sides)
        assert conf.reason != "insufficient_edge_support"
        assert conf.working_width > 0
        assert len((conf.edge_support_top, conf.edge_support_right, conf.edge_support_bottom, conf.edge_support_left)) == 4

    def test_keystone(self, detection_config):
        h, w = 480, 640
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dst = np.float32([[70, 40], [w - 50, 5], [w - 20, h - 25], [30, h - 50]])
        page = _id_card((h, w), (100, 80, 400, 260), printed=True)
        warped = cv2.warpPerspective(page, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(90, 63, 43))
        geom, conf = _dc(warped, detection_config)
        assert geom.present and conf.accepted

    def test_one_weak_side_still_confirms(self, detection_config):
        geom, conf = _dc(_weak_right_side(), detection_config)
        assert geom.present is True
        assert conf.accepted is True
        assert conf.evidence_edge_sides >= 2

    def test_open_passport(self, detection_config):
        frame = _opened_passport((1080, 1920), (80, 180, 1760, 720))
        geom, conf = _dc(frame, detection_config)
        assert conf.accepted is True

    def test_rotated_open_passport(self, detection_config):
        frame = _rotate(_opened_passport((1080, 1920), (200, 220, 1500, 640)), 15)
        geom, conf = _dc(frame, detection_config)
        assert geom.present and conf.accepted


class TestNegativeEdgeSupport:
    @pytest.mark.parametrize(
        "name,factory",
        [
            ("empty", lambda: _wood_table((480, 640))),
            ("grain", _grain_rectangle),
            ("mat", _mat_frame),
            ("window", _window_frame),
            ("full_frame", _full_frame_rect),
        ],
    )
    def test_background_rejected(self, detection_config, name, factory):
        _geom, conf = _dc(factory(), detection_config)
        assert conf.accepted is False, name

    def test_distant_rects_not_one_sheet(self, detection_config):
        geom, conf = _dc(_two_distant_rects(), detection_config)
        if geom.present and geom.bbox is not None:
            x, _y, w, _h = geom.bbox
            assert w < 500 or not conf.accepted

    def test_table_edge(self, detection_config):
        img = _wood_table((480, 640))
        img[200:230, :] = (40, 32, 28)
        _geom, conf = _dc(img, detection_config)
        assert conf.accepted is False


class TestLatencySanity:
    def test_geometry_and_confirm_on_1080p_are_bounded(self, detection_config):
        frame = cv2.resize(_id_card(printed=True), (1920, 1080), interpolation=cv2.INTER_LINEAR)
        detector = DocumentDetector(detection_config)
        t0 = time.perf_counter()
        geom = detector.detect_geometry(frame)
        t1 = time.perf_counter()
        conf = confirm_document(frame, geom)
        t2 = time.perf_counter()
        assert geom.present and conf.accepted
        assert (t1 - t0) < 2.5
        assert (t2 - t1) < 0.75


class TestBorderAwareConfirmation:
    def test_normal_side_thresholds_unchanged(self):
        assert _MIN_SIDES_NORMAL == 3
        assert _MIN_SIDE_ON == 14.0

    def test_confirmation_has_no_center_roi_gate(self):
        from idcam.vision import document_confirmation as mod

        src = inspect.getsource(mod)
        assert "roi_center_fraction" not in src
        assert "center_70" not in src
        assert "center70" not in src

    def _three_border_sheet(self) -> np.ndarray:
        from tests.unit.test_large_two_page_geometry import _wood_table

        h, w = 1080, 1920
        img = _wood_table((h, w))
        x0 = 340
        img[:, x0:] = (242, 238, 230)
        img[:, x0 : x0 + 5] = 32
        for row in range(50, h - 50, 16):
            img[row : row + 2, x0 + 24 : w - 24] = (28, 28, 34)
        return img

    def test_pi_like_three_border_quad_confirms(self):
        frame = self._three_border_sheet()
        geom = DocumentGeometryResult(
            present=True,
            confidence=0.9239,
            corners=((342.0, 159.0), (1918.5, 0.0), (1918.5, 1078.5), (379.5, 1078.5)),
            bbox=(342, 0, 1577, 1080),
            reason="quad",
            selected_area_ratio=0.761,
            aspect_ratio=1.563,
            rectangularity=0.55,
            geometry_mode="paper_region",
            rotation_deg=-5.759,
            border_sides=3,
            quad_aspect=1.563,
            edge_support=0.607,
        )
        conf = confirm_document(frame, geom)
        assert conf.accepted is True
        assert conf.occluded_side_count >= 2
        assert conf.observed_side_count >= 1
        assert conf.evidence_edge_sides >= 1
        assert sum(conf.side_border_occluded) >= 2

    def test_one_and_two_border_pages_confirm(self, detection_config):
        one = _dc(_id_card(box=(0, 80, 400, 280), printed=True), detection_config)
        two = _dc(_id_card(box=(0, 0, 420, 300), printed=True), detection_config)
        assert one[1].accepted and two[1].accepted

    def test_clipped_open_passport(self, detection_config):
        frame = _opened_passport((1080, 1920), (80, 0, 1400, 1080))
        geom, conf = _dc(frame, detection_config)
        assert geom.present is True
        assert conf.accepted is True

    def test_weak_three_border_without_document_edge_rejected(self):
        h, w = 480, 640
        img = np.full((h, w, 3), 90, dtype=np.uint8)
        img[:, 80:] = 110
        geom = DocumentGeometryResult(
            present=True,
            confidence=0.40,
            corners=((80.0, 0.0), (639.0, 0.0), (639.0, 479.0), (80.0, 479.0)),
            bbox=(80, 0, 560, 480),
            reason="quad",
            selected_area_ratio=0.75,
            aspect_ratio=1.4,
            rectangularity=0.4,
            geometry_mode="paper_region",
            border_sides=3,
            quad_aspect=1.4,
            edge_support=0.10,
        )
        conf = confirm_document(img, geom)
        assert conf.accepted is False

    def test_occupancy_alone_is_not_enough(self):
        img = np.full((480, 640, 3), 200, dtype=np.uint8)
        geom = DocumentGeometryResult(
            present=True,
            confidence=0.95,
            corners=((0.0, 0.0), (639.0, 0.0), (639.0, 479.0), (0.0, 479.0)),
            bbox=(0, 0, 640, 480),
            reason="quad",
            selected_area_ratio=0.99,
            geometry_mode="paper_region",
            border_sides=4,
            edge_support=0.9,
        )
        conf = confirm_document(img, geom)
        assert conf.accepted is False
        assert conf.reason == "full_frame"

