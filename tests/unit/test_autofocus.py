"""Autofocus controller uses live metadata, not advertised control ranges."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from idcam.camera.autofocus import AutofocusController
from idcam.camera.manager import (
    AutofocusWaitResult,
    expand_preview_roi,
    map_preview_roi_to_sensor,
    parse_xywh,
    preview_roi_is_usable,
    preview_roi_to_af_window,
    clip_xywh_to_bounds,
)
from idcam.config.schema import CameraConfig
from idcam.exceptions import AutofocusError
from idcam.vision.sharpness import accept_preview_focus


def _sharp_frame() -> np.ndarray:
    frame = np.full((120, 160, 3), 200, dtype=np.uint8)
    frame[20:100, 20:30] = 0
    frame[20:30, 20:140] = 0
    frame[90:100, 20:140] = 0
    return frame


class FakeCamera:
    def __init__(
        self,
        *,
        advertised: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        cycle_result: Any = True,
        has_cycle: bool = True,
    ) -> None:
        self.advertised = advertised or {
            "AfMode": (0, 2, 0),
            "AfTrigger": (0, 1, 0),
            "LensPosition": (0.0, 10.0, 1.0),
        }
        self.metadata = metadata or {"AfState": 2, "LensPosition": 1.25}
        self.cycle_result = cycle_result
        self.has_cycle = has_cycle
        self.cycle_calls = 0
        self.set_calls: list[dict[str, Any]] = []
        self.metadata_calls = 0
        self.controls_calls = 0
        self.pause_calls = 0

    def get_controls(self) -> dict[str, Any]:
        self.controls_calls += 1
        return dict(self.advertised)

    def capture_metadata(self) -> dict[str, Any]:
        self.metadata_calls += 1
        return dict(self.metadata)

    def set_controls(self, controls: dict[str, Any]) -> None:
        self.set_calls.append(dict(controls))

    def apply_document_focus_window(
        self,
        preview_roi: tuple[int, int, int, int],
        preview_size: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        self.applied_roi = tuple(int(v) for v in preview_roi)
        size = preview_size or (160, 120)
        return {
            "preview_roi": self.applied_roi,
            "preview_size": size,
            "scaler_crop": (0, 0, int(size[0]), int(size[1])),
            "sensor_roi": self.applied_roi,
        }

    def pause_autofocus(self) -> None:
        self.pause_calls += 1

    def autofocus_cycle(self) -> Any:
        if not self.has_cycle:
            raise AssertionError("autofocus_cycle should not be called")
        self.cycle_calls += 1
        return self.cycle_result

    def capture_preview_array(self) -> np.ndarray:
        return _sharp_frame()


class SequentialAfCamera(FakeCamera):
    """Production-like trigger + AfState wait, Idle → Scanning → Focused."""

    def __init__(self, *, states: list[int] | None = None) -> None:
        super().__init__(has_cycle=False, metadata={"AfState": 0, "LensPosition": 1.0, "FocusFoM": 26})
        self.states = states or [0, 1, 1, 2]
        self.trigger_calls = 0
        self.wait_calls = 0

    def trigger_autofocus(self) -> None:
        self.trigger_calls += 1

    def wait_for_autofocus_completion(self, timeout_sec: float) -> AutofocusWaitResult:
        self.wait_calls += 1
        last = 0
        lens = 1.0
        fom = 26.0
        for i, state in enumerate(self.states):
            last = state
            lens = 8.7 + i * 0.1
            fom = 80.0 + i * 20.0
            self.metadata = {"AfState": state, "LensPosition": lens, "FocusFoM": fom}
            if state == 2:
                return AutofocusWaitResult(2, lens, fom, "af_state", False)
            if state == 3:
                return AutofocusWaitResult(3, lens, fom, "af_state", False)
        return AutofocusWaitResult(last, lens, fom, "timeout", True)


class LensSettleCamera(FakeCamera):
    def __init__(self) -> None:
        super().__init__(has_cycle=False, metadata={"LensPosition": 8.9})

    def trigger_autofocus(self) -> None:
        self.trigger_calls = getattr(self, "trigger_calls", 0) + 1

    def wait_for_autofocus_completion(self, timeout_sec: float) -> AutofocusWaitResult:
        return AutofocusWaitResult(None, 8.9, None, "lens_settle", False)


class TestAutofocusController:
    @pytest.mark.asyncio
    async def test_sharp_preview_still_runs_hardware_cycle(self):
        """Do not skip AF because the full/ROI preview already looks sharp."""
        camera = FakeCamera()
        ctl = AutofocusController(CameraConfig(af_timeout_sec=1.0, af_max_retries=1))
        ctl.probe_capabilities(camera)
        result = await ctl.run_autofocus(camera, sharpness_threshold=1.0)
        assert result.success is True
        assert camera.cycle_calls == 1
        assert result.method == "hardware_cycle"
        assert result.roi[2] > 0

    @pytest.mark.asyncio
    async def test_prefers_autofocus_cycle_when_not_yet_sharp(self):
        class NeedsCycle(FakeCamera):
            def capture_preview_array(self) -> np.ndarray:
                if self.cycle_calls:
                    return _sharp_frame()
                return np.full((120, 160, 3), 128, dtype=np.uint8)

        camera = NeedsCycle()
        ctl = AutofocusController(CameraConfig(af_timeout_sec=1.0, af_max_retries=1))
        ctl.probe_capabilities(camera)
        result = await ctl.run_autofocus(camera, sharpness_threshold=1.0)
        assert result.success is True
        assert camera.cycle_calls == 1
        assert result.method == "hardware_cycle"
        assert result.improved is True
        assert result.sharpness >= 1.0

    @pytest.mark.asyncio
    async def test_wait_reads_af_state_from_metadata_not_control_ranges(self):
        class NeedsTrigger(FakeCamera):
            def capture_preview_array(self) -> np.ndarray:
                if self.set_calls:
                    return _sharp_frame()
                return np.full((120, 160, 3), 128, dtype=np.uint8)

        camera = NeedsTrigger(has_cycle=False)
        camera.advertised["AfState"] = (0, 3, 0)
        ctl = AutofocusController(CameraConfig(af_timeout_sec=1.0, af_max_retries=1))
        caps = ctl.probe_capabilities(camera)
        caps.autofocus_cycle = False
        ctl.capabilities = caps
        result = await ctl.run_autofocus(camera, sharpness_threshold=1.0)
        assert result.success is True
        assert camera.metadata_calls >= 1
        assert camera.set_calls
        assert camera.set_calls[0]["AfMode"] == 1
        assert camera.set_calls[0]["AfTrigger"] == 0

    @pytest.mark.asyncio
    async def test_missing_af_state_does_not_block_until_timeout(self):
        class NeedsTrigger(FakeCamera):
            def capture_preview_array(self) -> np.ndarray:
                if self.set_calls:
                    return _sharp_frame()
                return np.full((120, 160, 3), 128, dtype=np.uint8)

        camera = NeedsTrigger(has_cycle=False, metadata={"LensPosition": 0.8})
        ctl = AutofocusController(CameraConfig(af_timeout_sec=2.0, af_max_retries=1))
        caps = ctl.probe_capabilities(camera)
        caps.autofocus_cycle = False
        ctl.capabilities = caps
        result = await ctl.run_autofocus(camera, sharpness_threshold=1.0)
        assert result.success is True
        assert camera.metadata_calls >= 1

    @pytest.mark.asyncio
    async def test_verify_focus_uses_metadata(self):
        camera = FakeCamera()
        ctl = AutofocusController(CameraConfig())
        ctl.probe_capabilities(camera)
        result = await ctl.verify_focus(camera, sharpness_threshold=1.0)
        assert result.success is True
        assert result.af_state == 2
        assert camera.metadata_calls >= 1

    @pytest.mark.asyncio
    async def test_low_sharpness_retries_then_fails(self):
        class Blurry(FakeCamera):
            def capture_preview_array(self) -> np.ndarray:
                return np.full((80, 80, 3), 128, dtype=np.uint8)

        camera = Blurry(metadata={"LensPosition": 1.0})
        ctl = AutofocusController(CameraConfig(af_max_retries=2, af_timeout_sec=1.0))
        ctl.probe_capabilities(camera)
        with pytest.raises(AutofocusError, match="did not complete") as raised:
            await ctl.run_autofocus(camera, sharpness_threshold=500.0)
        assert camera.cycle_calls == 2
        assert raised.value.reason == "focus_verification_failed"
        err = None
        try:
            await ctl.run_autofocus(camera, sharpness_threshold=500.0)
        except AutofocusError as exc:
            err = str(exc)
        assert err is not None
        assert "before=" in err
        assert "after=" in err
        assert "preview_roi=" in err


class TestPreviewRoiMapping:
    def test_preview_roi_maps_through_live_scaler_crop(self):
        preview_size = (1920, 1080)
        preview_roi = (288, 162, 1344, 756)
        # 16:9 crop on a 4:3 9152x6944 sensor — not ScalerCropMaximum 70%.
        scaler = (0, 898, 9152, 5148)
        mapped = map_preview_roi_to_sensor(preview_roi, preview_size, scaler)
        independent_70_of_sensor = (1373, 1042, 6406, 4860)
        assert mapped != independent_70_of_sensor
        assert mapped[0] >= scaler[0]
        assert mapped[1] >= scaler[1]
        assert mapped[0] + mapped[2] <= scaler[0] + scaler[2] + 1
        assert mapped[1] + mapped[3] <= scaler[1] + scaler[3] + 1
        # 70% of 1920x1080 stays inside the 1920x1080 preview, unlike 1936,1960,5376,3024.
        assert preview_roi[0] + preview_roi[2] <= preview_size[0]
        assert preview_roi[1] + preview_roi[3] <= preview_size[1]
        reported_sensor_window = (1936, 1960, 5376, 3024)
        assert reported_sensor_window[2] > preview_size[0]
        assert mapped != reported_sensor_window

    def test_parse_xywh_tuple(self):
        assert parse_xywh((10, 20, 30, 40)) == (10, 20, 30, 40)
        assert parse_xywh(None) is None


class TestPreviewFocusAcceptance:
    def test_improvement_10_to_30_succeeds_below_floor(self):
        assert accept_preview_focus(before=10.0, after=30.0, absolute_floor=48.0) is True

    def test_degradation_10_to_8_fails(self):
        assert accept_preview_focus(before=10.0, after=8.0, absolute_floor=48.0) is False

    def test_absolute_floor_does_not_veto_improvement(self):
        assert accept_preview_focus(before=11.8, after=11.1, absolute_floor=48.0) is False
        assert accept_preview_focus(before=10.0, after=35.0, absolute_floor=48.0) is True

    @pytest.mark.asyncio
    async def test_controller_accepts_10_to_30_despite_threshold_48(self):
        camera = FakeCamera()
        ctl = AutofocusController(CameraConfig(af_timeout_sec=1.0, af_max_retries=1))
        ctl.probe_capabilities(camera)
        scores = iter([10.0, 30.0])
        with patch("idcam.camera.autofocus.laplacian_variance", side_effect=lambda *_a, **_k: next(scores)):
            result = await ctl.run_autofocus(camera, sharpness_threshold=48.0)
        assert result.success is True
        assert result.before_sharpness == 10.0
        assert result.sharpness == 30.0
        assert result.roi[2] > 0
        assert result.roi[0] + result.roi[2] <= 160

    @pytest.mark.asyncio
    async def test_controller_rejects_10_to_8(self):
        camera = FakeCamera(metadata={"LensPosition": 1.0})
        ctl = AutofocusController(CameraConfig(af_timeout_sec=1.0, af_max_retries=2))
        ctl.probe_capabilities(camera)
        scores = iter([10.0, 8.0, 10.0, 8.0])
        with patch("idcam.camera.autofocus.laplacian_variance", side_effect=lambda *_a, **_k: next(scores)):
            with pytest.raises(AutofocusError, match="before=10.0 after=8.0"):
                await ctl.run_autofocus(camera, sharpness_threshold=48.0)
        assert camera.cycle_calls == 2

    @pytest.mark.asyncio
    async def test_verify_does_not_reject_improved_preview_below_48(self):
        camera = FakeCamera()
        ctl = AutofocusController(CameraConfig())
        ctl.probe_capabilities(camera)
        ctl.debug.focus_before = 10.0
        ctl.debug.focus_after = 30.0
        ctl._last_preview_roi = (20, 20, 80, 60)
        with patch("idcam.camera.autofocus.laplacian_variance", return_value=28.0):
            result = await ctl.verify_focus(camera, sharpness_threshold=48.0)
        assert result.success is True
        assert result.sharpness == 28.0


class TestAfWindowsFromPreview:
    def test_off_center_bbox_maps_to_sensor_not_frame_center(self):
        preview_size = (1920, 1080)
        scaler = (0, 898, 9152, 5148)
        left = (40, 200, 480, 640)
        mapped = preview_roi_to_af_window(left, preview_size, scaler)
        center_search = preview_roi_to_af_window((288, 162, 1344, 756), preview_size, scaler)
        assert mapped["preview_roi"][0] == 40
        assert mapped["sensor_roi"] != center_search["sensor_roi"]
        sx, sy, sw, sh = mapped["sensor_roi"]
        bx, by, bw, bh = scaler
        assert sx >= bx and sy >= by
        assert sx + sw <= bx + bw + 1
        assert sy + sh <= by + bh + 1

    def test_af_windows_clamped_to_scaler_crop(self):
        scaler = (0, 898, 9152, 5148)
        overflowing = map_preview_roi_to_sensor((1900, 1060, 80, 80), (1920, 1080), scaler)
        clamped = clip_xywh_to_bounds(overflowing, scaler)
        assert clamped[0] >= scaler[0]
        assert clamped[1] >= scaler[1]
        assert clamped[0] + clamped[2] <= scaler[0] + scaler[2]
        assert clamped[1] + clamped[3] <= scaler[1] + scaler[3]

    def test_invalid_empty_geometry_rejected(self):
        scaler = (0, 0, 1920, 1080)
        with pytest.raises(ValueError, match="invalid_roi"):
            preview_roi_to_af_window((0, 0, 0, 0), (1920, 1080), scaler)
        with pytest.raises(ValueError, match="invalid_roi"):
            preview_roi_to_af_window((10, 10, -4, 40), (1920, 1080), scaler)
        assert preview_roi_is_usable((0, 0, 0, 0), (1920, 1080)) is False

    def test_preview_to_sensor_scale_is_linear(self):
        preview_size = (1920, 1080)
        scaler = (100, 200, 1920, 1080)
        roi = (96, 108, 192, 216)
        mapped = preview_roi_to_af_window(roi, preview_size, scaler)
        assert mapped["sensor_roi"] == (100 + 96, 200 + 108, 192, 216)


class TestDocumentRoiAutofocusPolicy:
    @pytest.mark.asyncio
    async def test_uses_provided_document_roi_not_center(self):
        camera = FakeCamera()
        ctl = AutofocusController(CameraConfig(af_timeout_sec=1.0, af_max_retries=1))
        ctl.probe_capabilities(camera)
        doc_roi = (20, 10, 90, 70)
        rois: list = []

        def spy(_frame, roi=None, roi_fraction=0.7):
            rois.append(roi)
            return 12.0 if len(rois) == 1 else 30.0

        with patch("idcam.camera.autofocus.laplacian_variance", side_effect=spy):
            result = await ctl.run_autofocus(
                camera,
                sharpness_threshold=48.0,
                preview_roi=doc_roi,
                require_document_roi=True,
            )
        assert result.success is True
        assert result.roi == doc_roi
        assert rois == [doc_roi, doc_roi]
        assert camera.applied_roi == doc_roi
        assert camera.pause_calls == 1
        assert result.lens_locked is True
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_require_document_roi_rejects_missing_and_tiny(self):
        camera = FakeCamera()
        ctl = AutofocusController(CameraConfig(af_timeout_sec=1.0, af_max_retries=1))
        ctl.probe_capabilities(camera)
        with pytest.raises(AutofocusError, match="invalid_roi"):
            await ctl.run_autofocus(camera, sharpness_threshold=1.0, require_document_roi=True)
        with pytest.raises(AutofocusError, match="invalid_roi"):
            await ctl.run_autofocus(
                camera,
                sharpness_threshold=1.0,
                preview_roi=(0, 0, 2, 2),
                require_document_roi=True,
            )
        assert camera.cycle_calls == 0

    @pytest.mark.asyncio
    async def test_already_sharp_document_is_success(self):
        camera = FakeCamera()
        ctl = AutofocusController(CameraConfig(af_timeout_sec=1.0, af_max_retries=1))
        ctl.probe_capabilities(camera)
        scores = iter([32.0, 31.5])
        with patch("idcam.camera.autofocus.laplacian_variance", side_effect=lambda *_a, **_k: next(scores)):
            result = await ctl.run_autofocus(
                camera, sharpness_threshold=48.0, preview_roi=(20, 20, 80, 60), require_document_roi=True
            )
        assert result.success is True
        assert result.improved is False

    @pytest.mark.asyncio
    async def test_failure_is_not_success(self):
        class Blurry(FakeCamera):
            def capture_preview_array(self) -> np.ndarray:
                return np.full((80, 80, 3), 128, dtype=np.uint8)

        camera = Blurry(metadata={"LensPosition": 1.0})
        ctl = AutofocusController(CameraConfig(af_max_retries=1, af_timeout_sec=1.0))
        ctl.probe_capabilities(camera)
        with pytest.raises(AutofocusError):
            await ctl.run_autofocus(
                camera, sharpness_threshold=500.0, preview_roi=(8, 8, 40, 40), require_document_roi=True
            )
        assert camera.pause_calls == 0

    def test_accept_already_sharp_near_floor(self):
        assert accept_preview_focus(before=32.0, after=31.5, absolute_floor=28.0) is True


class TestHardwareAfStateCompletion:
    @pytest.mark.asyncio
    async def test_idle_scanning_focused_succeeds_and_locks_lens(self):
        camera = SequentialAfCamera()
        ctl = AutofocusController(CameraConfig(af_timeout_sec=1.0, af_max_retries=3))
        ctl.probe_capabilities(camera)
        scores = iter([12.4, 12.0])
        with patch("idcam.camera.autofocus.laplacian_variance", side_effect=lambda *_a, **_k: next(scores)):
            result = await ctl.run_autofocus(camera, sharpness_threshold=48.0)
        assert result.success is True
        assert result.af_state == 2
        assert result.hardware_success is True
        assert camera.trigger_calls == 1
        assert camera.wait_calls == 1
        assert camera.pause_calls == 1
        assert result.lens_locked is True
        assert result.focus_fom_after is not None

    @pytest.mark.asyncio
    async def test_af_state_failed_does_not_retry_on_laplacian(self):
        camera = SequentialAfCamera(states=[1, 3])
        ctl = AutofocusController(CameraConfig(af_timeout_sec=1.0, af_max_retries=3))
        ctl.probe_capabilities(camera)
        with pytest.raises(AutofocusError) as raised:
            await ctl.run_autofocus(camera, sharpness_threshold=1.0)
        assert raised.value.reason == "autofocus_hardware_failed"
        assert camera.trigger_calls == 3
        assert camera.pause_calls == 0

    @pytest.mark.asyncio
    async def test_af_state_timeout_is_classified(self):
        camera = SequentialAfCamera(states=[0, 1, 1])
        ctl = AutofocusController(CameraConfig(af_timeout_sec=1.0, af_max_retries=1))
        ctl.probe_capabilities(camera)
        with pytest.raises(AutofocusError) as raised:
            await ctl.run_autofocus(camera, sharpness_threshold=1.0)
        assert raised.value.reason == "autofocus_timeout"

    @pytest.mark.asyncio
    async def test_lens_settle_fallback_uses_preview_sharpness(self):
        camera = LensSettleCamera()
        ctl = AutofocusController(CameraConfig(af_timeout_sec=1.0, af_max_retries=1))
        ctl.probe_capabilities(camera)
        scores = iter([10.0, 30.0])
        with patch("idcam.camera.autofocus.laplacian_variance", side_effect=lambda *_a, **_k: next(scores)):
            result = await ctl.run_autofocus(camera, sharpness_threshold=48.0)
        assert result.success is True
        assert result.completion_method == "lens_settle"

    @pytest.mark.asyncio
    async def test_lens_settle_fallback_rejects_degraded_preview(self):
        camera = LensSettleCamera()
        ctl = AutofocusController(CameraConfig(af_timeout_sec=1.0, af_max_retries=1))
        ctl.probe_capabilities(camera)
        scores = iter([10.0, 8.0])
        with patch("idcam.camera.autofocus.laplacian_variance", side_effect=lambda *_a, **_k: next(scores)):
            with pytest.raises(AutofocusError) as raised:
                await ctl.run_autofocus(camera, sharpness_threshold=48.0)
        assert raised.value.reason == "focus_verification_failed"


class TestPaddedAfWindow:
    def test_expand_keeps_document_center(self):
        preview = (1920, 1080)
        roi = (756, 409, 801, 482)
        expanded = expand_preview_roi(roi, preview)
        assert expanded[2] > roi[2]
        assert expanded[3] > roi[3]
        orig_cx = roi[0] + roi[2] / 2
        new_cx = expanded[0] + expanded[2] / 2
        assert abs(orig_cx - new_cx) < 8
        assert expanded[0] >= 0
        assert expanded[1] >= 0
        assert expanded[0] + expanded[2] <= 1920
        assert expanded[1] + expanded[3] <= 1080

    def test_tiny_roi_grows_to_minimum_fraction(self):
        expanded = expand_preview_roi((100, 100, 20, 20), (1920, 1080))
        assert expanded[2] >= int(1920 * 0.25)
        assert expanded[3] >= int(1080 * 0.25)

    @pytest.mark.asyncio
    async def test_verify_accepts_focused_with_small_laplacian_noise(self):
        camera = FakeCamera()
        ctl = AutofocusController(CameraConfig())
        ctl.probe_capabilities(camera)
        ctl.debug.focus_before = 12.4
        ctl.debug.focus_after = 12.0
        ctl._last_preview_roi = (20, 20, 80, 60)
        with patch("idcam.camera.autofocus.laplacian_variance", return_value=12.1):
            result = await ctl.verify_focus(camera, sharpness_threshold=48.0)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_verify_rejects_collapsed_sharpness_even_if_focused(self):
        camera = FakeCamera()
        ctl = AutofocusController(CameraConfig())
        ctl.probe_capabilities(camera)
        ctl.debug.focus_before = 40.0
        ctl.debug.focus_after = 40.0
        ctl._last_preview_roi = (20, 20, 80, 60)
        with patch("idcam.camera.autofocus.laplacian_variance", return_value=5.0):
            with pytest.raises(AutofocusError) as raised:
                await ctl.verify_focus(camera, sharpness_threshold=48.0)
        assert raised.value.reason == "focus_verification_failed"


class TestBoundedAfLensSearch:
    def test_search_positions_include_boundaries(self):
        from idcam.camera.manager import af_lens_search_positions

        pts = af_lens_search_positions(8.7, 9.2, 0.05)
        assert pts[0] == pytest.approx(8.7, abs=1e-6)
        assert pts[-1] == pytest.approx(9.2, abs=1e-6)
        assert all(8.7 - 1e-6 <= p <= 9.2 + 1e-6 for p in pts)

    def test_manager_search_never_commands_outside_range(self):
        from idcam.camera.manager import CameraManager

        mgr = CameraManager(CameraConfig(preview_width=320, preview_height=240), use_mock=True)
        mgr.initialize()
        mgr.apply_document_focus_window((40, 20, 200, 160), (320, 240))
        mgr.trigger_autofocus()
        result = mgr.wait_for_autofocus_completion(1.0)
        assert result.timed_out is False
        assert result.af_state == 2
        assert result.lens_position is not None
        assert 8.7 <= result.lens_position <= 9.2
        assert any(abs(c - 8.7) < 1e-6 for c in mgr._af_lens_commands)
        assert any(abs(c - 9.2) < 1e-6 for c in mgr._af_lens_commands)
        assert any(abs(c - 8.905) < 1e-6 for c in mgr._af_lens_commands)
        assert all(8.7 - 1e-6 <= c <= 9.2 + 1e-6 for c in mgr._af_lens_commands)
        assert result.lens_position == pytest.approx(8.905, abs=0.002)
        mgr.shutdown()

    def test_out_of_range_command_is_blocked(self, caplog):
        from idcam.camera.manager import CameraManager

        mgr = CameraManager(CameraConfig(preview_width=320, preview_height=240), use_mock=True)
        mgr.initialize()
        before = list(mgr._af_lens_commands)
        lens_before = mgr.capture_metadata()["LensPosition"]
        assert mgr._command_production_af_lens(5.0) is None
        assert mgr._command_production_af_lens(12.0) is None
        assert mgr._af_lens_commands == before
        assert mgr.capture_metadata()["LensPosition"] == lens_before
        assert "autofocus_lens_range_violation" in caplog.text
        sent = mgr._command_production_af_lens(8.905)
        assert sent == pytest.approx(8.905)
        assert mgr.capture_metadata()["LensPosition"] == pytest.approx(8.905)
        mgr.shutdown()

    def test_af_windows_mapping_unchanged(self):
        preview_size = (1920, 1080)
        scaler = (0, 898, 9152, 5148)
        left = (40, 200, 480, 640)
        mapped = preview_roi_to_af_window(left, preview_size, scaler)
        center_search = preview_roi_to_af_window((288, 162, 1344, 756), preview_size, scaler)
        assert mapped["preview_roi"][0] == 40
        assert mapped["sensor_roi"] != center_search["sensor_roi"]

    @pytest.mark.asyncio
    async def test_timeout_and_failed_af_state_unchanged(self):
        camera = SequentialAfCamera(states=[0, 1, 1])
        ctl = AutofocusController(CameraConfig(af_timeout_sec=1.0, af_max_retries=1))
        ctl.probe_capabilities(camera)
        with pytest.raises(AutofocusError) as raised:
            await ctl.run_autofocus(camera, sharpness_threshold=1.0)
        assert raised.value.reason == "autofocus_timeout"

        camera = SequentialAfCamera(states=[1, 3])
        ctl = AutofocusController(CameraConfig(af_timeout_sec=1.0, af_max_retries=3))
        ctl.probe_capabilities(camera)
        with pytest.raises(AutofocusError) as raised:
            await ctl.run_autofocus(camera, sharpness_threshold=1.0)
        assert raised.value.reason == "autofocus_hardware_failed"
        assert camera.trigger_calls == 3


