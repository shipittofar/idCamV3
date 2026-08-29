"""Document confirmation gate: reject empty scenes, keep off-center/rotated pages."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import cv2
import numpy as np
import pytest

from idcam.app.orchestrator import ApplicationOrchestrator
from idcam.app.state_machine import AppState
from idcam.app.transaction import ActiveCaptureTransaction
from idcam.config.schema import AppConfig, CameraConfig, DetectionConfig, QualityConfig, ServerConfig, UIConfig
from idcam.system.component_health import ComponentStatus, HealthLevel
from idcam.vision.document_confirmation import DocumentConfirmation, confirm_document
from idcam.vision.document_detector import DocumentDetector
from idcam.vision.geometry import (
    MAX_AREA_FRACTION,
    MAX_ASPECT,
    MIN_AREA_FRACTION,
    MIN_ASPECT,
    MIN_CONFIDENCE,
    MIN_RECTANGULARITY,
    MIN_RIGHT_ANGLE,
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


@pytest.fixture
def app_config(tmp_path):
    return AppConfig(
        device_id="confirm",
        temp_dir=tmp_path / "tmp",
        server=ServerConfig(url="https://api.example.com"),
        quality=QualityConfig(
            min_sharpness=10.0,
            min_file_bytes=100,
            min_width=64,
            min_height=64,
            require_expected_dimensions=False,
        ),
        detection=DetectionConfig(
            motion_threshold=5.0,
            edge_density_threshold=0.01,
            stable_frames_required=2,
            removal_frames_required=2,
            geometry_stable_frames_required=2,
            geometry_stability_threshold_px=12.0,
        ),
        camera=CameraConfig(preview_fps=30.0, af_timeout_sec=1.0, af_max_retries=1),
        ui=UIConfig(
            status_file=tmp_path / "run" / "status.json",
            admin_socket=tmp_path / "run" / "admin.sock",
        ),
    )


def _orch(app_config):
    orch = ApplicationOrchestrator(app_config, use_mock_camera=True)
    orch.health.camera = ComponentStatus(level=HealthLevel.HEALTHY, details={"healthy": True})
    orch.disk.can_capture = MagicMock(return_value=True)
    return orch


def _id_card(
    shape=(480, 640),
    box=(80, 60, 480, 360),
    *,
    printed: bool = False,
    angle: float = 0.0,
) -> np.ndarray:
    img = _page(shape, box, printed=printed)
    if abs(angle) < 0.5:
        return img
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    border = (int(img[0, 0, 0]), int(img[0, 0, 1]), int(img[0, 0, 2]))
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=border)


def _perspective_id(shape=(480, 640)) -> np.ndarray:
    h, w = shape
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[40, 30], [w - 80, 10], [w - 30, h - 40], [20, h - 20]])
    page = _id_card(shape, (90, 70, 380, 240), printed=True)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(page, matrix, (w, h), borderValue=(90, 63, 43))


def _mat_frame(shape=(480, 640)) -> np.ndarray:
    img = _wood_table(shape)
    h, w = shape
    x, y, pw, ph = 40, 30, w - 80, h - 60
    patch = img[y : y + ph, x : x + pw].astype(np.int16) + 18
    img[y : y + ph, x : x + pw] = np.clip(patch, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (21, 21), 0)


def _window_frame(shape=(480, 640)) -> np.ndarray:
    """Wall with a brighter architectural patch, not a paper sheet with a drawn border."""
    img = np.full((shape[0], shape[1], 3), 52, dtype=np.uint8)
    img[50:270, 90:540] = (68, 70, 72)
    return cv2.GaussianBlur(img, (41, 41), 0)


def _grain_rectangle(shape=(480, 640)) -> np.ndarray:
    img = _wood_table(shape)
    img[80:320, 60:500] = _wood_table((240, 440), brightness=8)
    return img


def _two_distant_rects(shape=(480, 640)) -> np.ndarray:
    img = _page(shape, (20, 40, 160, 120), printed=True)
    x, y, pw, ph = 440, 280, 160, 120
    img[y : y + ph, x : x + pw] = (242, 238, 230)
    img[y : y + 3, x : x + pw] = 35
    img[y + ph - 3 : y + ph, x : x + pw] = 35
    img[y : y + ph, x : x + 3] = 35
    img[y : y + ph, x + pw - 3 : x + pw] = 35
    return img


def _full_frame_rect(shape=(480, 640)) -> np.ndarray:
    img = np.full((shape[0], shape[1], 3), 240, dtype=np.uint8)
    img[:3, :] = 20
    img[-3:, :] = 20
    img[:, :3] = 20
    img[:, -3:] = 20
    return img


def _rejected() -> DocumentConfirmation:
    return DocumentConfirmation(
        accepted=False,
        reason="insufficient_combined_evidence",
        candidate_present=True,
        evidence_edge_support=0.0,
        evidence_edge_sides=1,
        evidence_edge_score=0.0,
        evidence_paper_contrast=0.0,
        evidence_interior_mean=0.0,
        evidence_surround_mean=0.0,
        evidence_interior_std=0.0,
        evidence_surround_std=0.0,
        evidence_print=0.0,
        evidence_line_structure=0.0,
        evidence_fold=0.0,
        roi=(0, 0, 0, 0),
        geometry_mode="quad",
        geometry_aspect=0.0,
        geometry_area_fraction=0.0,
    )


def _detect_and_confirm(frame, detection_config):
    geom = DocumentDetector(detection_config).detect_geometry(frame)
    conf = confirm_document(frame, geom)
    return geom, conf


async def _idle_until_auto(orch, ticks=16):
    for _ in range(ticks):
        await orch._idle_tick()
        if orch.sm.state == AppState.DOCUMENT_DETECTED:
            return
    raise AssertionError(f"auto transaction did not start; state={orch.sm.state}")


class TestGeometryGatesUnchanged:
    def test_normal_thresholds_frozen(self):
        assert MIN_AREA_FRACTION == 0.08
        assert MAX_AREA_FRACTION == 0.97
        assert MIN_ASPECT == 0.40
        assert MAX_ASPECT == 3.35
        assert MIN_RECTANGULARITY == 0.72
        assert MIN_CONFIDENCE == 0.48
        assert MIN_RIGHT_ANGLE == 0.55


class TestFalsePositives:
    def test_empty_desk(self, detection_config):
        _geom, conf = _detect_and_confirm(_wood_table((480, 640)), detection_config)
        assert conf.accepted is False

    def test_table_grain_rectangle(self, detection_config):
        _geom, conf = _detect_and_confirm(_grain_rectangle(), detection_config)
        assert conf.accepted is False

    def test_placemat(self, detection_config):
        _geom, conf = _detect_and_confirm(_mat_frame(), detection_config)
        assert conf.accepted is False

    def test_window_wall(self, detection_config):
        _geom, conf = _detect_and_confirm(_window_frame(), detection_config)
        assert conf.accepted is False

    def test_full_frame(self, detection_config):
        geom, conf = _detect_and_confirm(_full_frame_rect(), detection_config)
        assert conf.accepted is False
        if geom.present:
            assert conf.reason in {"full_frame", "insufficient_edge_support", "insufficient_paper_contrast", "insufficient_combined_evidence"}

    def test_two_distant_rectangles_are_not_one_document(self, detection_config):
        geom, _conf = _detect_and_confirm(_two_distant_rects(), detection_config)
        if geom.present and geom.bbox is not None:
            x, y, w, h = geom.bbox
            assert not (x < 40 and x + w > 560 and y < 60 and y + h > 380)


class TestTrueDocuments:
    def test_centered_id(self, detection_config):
        geom, conf = _detect_and_confirm(_id_card(), detection_config)
        assert geom.present is True
        assert conf.accepted is True

    def test_left_id(self, detection_config):
        geom, conf = _detect_and_confirm(_id_card(box=(12, 90, 280, 180)), detection_config)
        assert conf.accepted is True
        assert geom.bbox is not None
        assert geom.bbox[0] + geom.bbox[2] / 2 < 320

    def test_right_id(self, detection_config):
        geom, conf = _detect_and_confirm(_id_card(box=(340, 90, 260, 180)), detection_config)
        assert conf.accepted is True
        assert geom.bbox is not None
        assert geom.bbox[0] + geom.bbox[2] / 2 > 320

    def test_top_id(self, detection_config):
        _geom, conf = _detect_and_confirm(_id_card(box=(160, 8, 280, 160)), detection_config)
        assert conf.accepted is True

    def test_bottom_id(self, detection_config):
        _geom, conf = _detect_and_confirm(_id_card(box=(160, 300, 280, 160)), detection_config)
        assert conf.accepted is True

    def test_corner_id(self, detection_config):
        _geom, conf = _detect_and_confirm(_id_card(box=(8, 8, 240, 150)), detection_config)
        assert conf.accepted is True

    def test_rotated_10(self, detection_config):
        _geom, conf = _detect_and_confirm(_id_card(angle=10), detection_config)
        assert conf.accepted is True

    def test_rotated_20(self, detection_config):
        _geom, conf = _detect_and_confirm(_id_card(angle=20), detection_config)
        assert conf.accepted is True

    def test_rotated_30(self, detection_config):
        geom, conf = _detect_and_confirm(_id_card(angle=30), detection_config)
        if geom.present:
            assert conf.accepted is True

    def test_perspective(self, detection_config):
        _geom, conf = _detect_and_confirm(_perspective_id(), detection_config)
        assert conf.accepted is True

    def test_oversized(self, detection_config):
        _geom, conf = _detect_and_confirm(_id_card(box=(20, 16, 600, 450), printed=True), detection_config)
        assert conf.accepted is True

    def test_smaller(self, detection_config):
        geom, conf = _detect_and_confirm(_id_card(box=(200, 140, 220, 140), printed=True), detection_config)
        if geom.present:
            assert conf.accepted is True

    def test_open_passport_one_outer(self, detection_config):
        frame = _opened_passport((1080, 1920), (80, 180, 1760, 720))
        geom, conf = _detect_and_confirm(frame, detection_config)
        assert conf.accepted is True
        assert geom.geometry_mode in {"merged_two_page", "large_document_outer", "paper_region", "quad"}

    def test_open_passport_rotated(self, detection_config):
        frame = _opened_passport((1080, 1920), (120, 200, 1600, 640))
        h, w = frame.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 12.0, 1.0)
        rot = cv2.warpAffine(frame, matrix, (w, h), borderValue=(90, 63, 43))
        _geom, conf = _detect_and_confirm(rot, detection_config)
        assert conf.accepted is True

    def test_blank_paper_with_boundary(self, detection_config):
        _geom, conf = _detect_and_confirm(_id_card(printed=False), detection_config)
        assert conf.accepted is True

    def test_not_center_biased(self, detection_config):
        left = _detect_and_confirm(_id_card(box=(10, 100, 260, 170)), detection_config)[1]
        center = _detect_and_confirm(_id_card(box=(190, 100, 260, 170)), detection_config)[1]
        assert left.accepted and center.accepted


class TestAutoCaptureGuards:
    @pytest.mark.asyncio
    async def test_empty_desk_no_auto_capture(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        orch._safe_preview = AsyncMock(return_value=_wood_table((480, 640)))
        orch.autofocus.run_autofocus = AsyncMock()
        for _ in range(10):
            await orch._idle_tick()
        assert orch.sm.state == AppState.IDLE
        assert orch._transaction is None
        orch.autofocus.run_autofocus.assert_not_called()

    @pytest.mark.asyncio
    async def test_stable_false_rectangle_does_not_start_transaction(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        orch._safe_preview = AsyncMock(return_value=_mat_frame())
        for _ in range(12):
            await orch._idle_tick()
        assert orch.sm.state == AppState.IDLE
        assert orch._transaction is None

    @pytest.mark.asyncio
    async def test_geometry_present_without_confirmation_skips_af(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        page = _id_card()
        assert DocumentDetector(app_config.detection).detect_geometry(page).present
        rejected = _rejected()
        orch._safe_preview = AsyncMock(return_value=page)

        def reject_update(_frame, geom):
            orch.document_confirmation.last = rejected
            orch.document_confirmation.count = 0
            return rejected

        orch.document_confirmation.update = reject_update
        for _ in range(8):
            await orch._idle_tick()
        assert orch.sm.state == AppState.IDLE
        assert orch._transaction is None

    @pytest.mark.asyncio
    async def test_centered_id_starts_auto_path(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        orch._safe_preview = AsyncMock(return_value=_id_card())
        await _idle_until_auto(orch)
        assert orch._transaction is not None
        assert orch._transaction.trigger == "auto"
        assert orch._transaction.document_confirmed is True

    @pytest.mark.asyncio
    async def test_off_center_id_starts_auto_path(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        orch._safe_preview = AsyncMock(return_value=_id_card(box=(12, 90, 280, 180)))
        await _idle_until_auto(orch)
        assert orch._transaction.document_confirmed is True

    @pytest.mark.asyncio
    async def test_candidate_gone_before_af(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.FOCUSING
        orch._resume_state = AppState.IDLE
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1", trigger="auto")
        page = _id_card()
        orch._snapshot_accepted_geometry(DocumentDetector(app_config.detection).detect_geometry(page))
        orch._safe_preview = AsyncMock(return_value=_wood_table((480, 640)))
        orch.autofocus.run_autofocus = AsyncMock()
        await orch._focusing_tick()
        orch.autofocus.run_autofocus.assert_not_called()
        assert orch.sm.state != AppState.FOCUS_VERIFICATION

    @pytest.mark.asyncio
    async def test_candidate_gone_before_capture(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.CAPTURING
        orch._resume_state = AppState.IDLE
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1", trigger="auto")
        orch._transaction.mark_focus_completed(90.0, 10.0)
        page = _id_card()
        orch._snapshot_accepted_geometry(DocumentDetector(app_config.detection).detect_geometry(page))
        orch._safe_preview = AsyncMock(return_value=_wood_table((480, 640)))
        orch.capture_service.capture = AsyncMock()
        await orch._capturing_tick()
        orch.capture_service.capture.assert_not_called()

    @pytest.mark.asyncio
    async def test_manual_capture_still_starts(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        orch._safe_preview = AsyncMock(return_value=_wood_table((480, 640)))
        orch._take_manual_request = MagicMock(return_value=True)
        await orch._idle_tick()
        assert orch.sm.state == AppState.DOCUMENT_DETECTED
        assert orch._transaction is not None
        assert orch._transaction.trigger == "manual"
