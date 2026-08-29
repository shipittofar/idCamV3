"""Confirmation edge-support: boundary contrast, not generic Sobel energy."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from idcam.config.schema import DetectionConfig
from idcam.vision.document_confirmation import CONFIRM_WORKING_WIDTH, confirm_document
from idcam.vision.document_detector import DocumentDetector
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
