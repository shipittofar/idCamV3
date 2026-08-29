# Autofocus lens range (8.7–9.2)

## 1. Current AF implementation

Production AF is owned by `AutofocusController.run_autofocus` (`src/idcam/camera/autofocus.py`).

It:

1. Resolves the **document preview ROI** (not center-70% when a document ROI is provided).
2. Calls `CameraManager.apply_document_focus_window` to set **AfWindows** via live ScalerCrop.
3. Triggers hardware AF and waits for completion.
4. Classifies success from AfState (Focused / Failed / timeout) and preview Laplacian.

Previously, `CameraManager.trigger_autofocus` started **libcamera Auto** with **`AfRange=Full`**. That lets the driver scan the **entire advertised LensPosition range** (typically 0–10). The document-sharp region on this Hawkeye setup is ~8.9, so a full scan spent most of its time far from focus.

## 2. Where the AF lens range is controlled

| Path | Role |
| --- | --- |
| `CameraConfig.af_min_lens_position` / `af_max_lens_position` / `af_lens_search_step` | Single config source of truth (defaults 8.7, 9.2, 0.05). |
| `CameraManager.AF_MIN_LENS_POSITION` / `AF_MAX_LENS_POSITION` | Same values, for diagnostics (`getattr(cam, "AF_MIN_LENS_POSITION")`). |
| `CameraManager.trigger_autofocus` | Sets **AfMode=Manual**; does **not** start `AfRange=Full` Auto. |
| `CameraManager.wait_for_autofocus_completion` | Runs the **bounded Manual sweep** and locks the best FocusFoM. |
| `CameraManager._command_production_af_lens` | Only production-AF lens commands; rejects values outside [8.7, 9.2]. |

Debug `set_manual_lens` still uses **advertised** camera limits so the live-viewer calibrator is unchanged.

## 3. Before range

Hardware Auto, **AfRange=Full**, advertised LensPosition (mock/tests: **0.0–10.0**).

## 4. New range

**8.7 → 9.2** inclusive (dioptres), plus a sample at **8.905** (measured sharp point).

## 5. Code/config changes

- `src/idcam/config/schema.py` — AF min/max/step fields.
- `src/idcam/camera/manager.py` — bounded search; range guard; mock FoM peaks at 8.905.
- `src/idcam/camera/autofocus.py` — log `af_min_lens` / `af_max_lens` on `event=autofocus_result`.
- `tests/unit/test_autofocus.py` — range, guard, AfWindows, timeout/Failed unchanged.
- `docs/AUTOFOCUS_LENS_RANGE_FIX.md` — this file.

`apply_document_focus_window` still expands the document bbox and maps AfWindows. It still may set `AfRange=Full` on the **metering** control block; that no longer drives the search because Auto is not triggered.

## 6. Search vs final clamp

The search **visits** positions in [8.7, 9.2] only. There is no full-range Auto cycle followed by a clamp. Out-of-range `_command_production_af_lens` calls are **not sent**; they log `event=autofocus_lens_range_violation`.

## 7. Document geometry / AfWindows

Unchanged: `expand_preview_roi`, `preview_roi_to_af_window`, `apply_document_focus_window` mapping. ROI source remains **document_geometry** when the orchestrator passes the accepted document bbox.

## 8. Tests

`python -m pytest tests -q --tb=line` → **450 passed, 6 skipped**.

Bounded-search coverage is in `tests/unit/test_autofocus.py` (`TestBoundedAfLensSearch`): inclusive 8.7/9.2 grid, no commands outside the band, 8.905 accepted, out-of-range commands blocked with `event=autofocus_lens_range_violation`, AfWindows mapping unchanged, timeout/`autofocus_hardware_failed` unchanged.

Camera initialization unit tests remain in `tests/unit/test_camera.py` and still pass.

## 9. Hardware validation

**NOT RUN** on Raspberry Pi in this session.

On the Pi (service stopped, production venv):

```bash
sudo -u idcam /opt/idcam/venv/bin/python - <<'PY'
from idcam.camera.manager import CameraManager
from idcam.config.schema import CameraConfig

cam = CameraManager(CameraConfig())
cam.initialize()
print("AF MIN:", cam.AF_MIN_LENS_POSITION)
print("AF MAX:", cam.AF_MAX_LENS_POSITION)
cam.shutdown()
PY
```

Then a real document AF cycle. Confirm logs: `event=autofocus_search_range af_min_lens=8.700 af_max_lens=9.200`, no `autofocus_lens_range_violation`, `roi_source=document_geometry`, final lens in [8.7, 9.2].

## 10. Limitations

- Preview **Continuous AF** during IDLE is unchanged and may still hunt the full advertised range until a capture transaction runs production AF.
- Still capture still **holds whatever lens was last read** when switching modes; after a successful AF that should already be in 8.7–9.2.
- Best focus is **max FocusFoM** on the discrete grid (0.05 plus 8.905), not a sub-step interpolator.
- Unit tests do **not** prove Pi optics.
