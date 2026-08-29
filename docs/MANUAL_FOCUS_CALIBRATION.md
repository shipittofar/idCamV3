# Manual focus calibration (debug)

**Autofocus is not claimed fixed.** This document describes a **debug-only** lens control in the Live Debug Viewer so the Raspberry Pi can be used to find the physically sharp `LensPosition`. Automatic AF is unchanged.

## 1. Why existing autofocus is not considered solved

Hardware still shows `AfState` 0→1→2 and `LensPosition` moving (~1.0 → ~7.8) with rising `FocusFoM`, while the **saved document JPEG is not reliably sharp**. That means we do not yet know whether the failure is:

- wrong lens position vs true peak sharpness
- `FocusFoM` not tracking document text
- AF ROI / Laplacian ROI mismatch
- preview-sharp vs 9152×6944 still-sharp mismatch
- distance-dependent focus

Those questions are **experimental**. This tool exists to measure them.

## 2. Manual focus controls

Enable the existing Live Debug Viewer (`debug.enabled: true`). Open `/preview/live`.

Buttons (and keys):

| Control | Key |
|---|---|
| −1.0 / −0.1 / −0.01 | Shift+← / ← |
| +0.01 / +0.1 / +1.0 | → / Shift+→ |
| SET MANUAL FOCUS | `M` |
| AUTOFOCUS | `A` |
| LOCK | `L` |
| SCAN FOCUS | `S` |
| CAPTURE FOCUS TEST | `T` |

Manual mode sets `AfMode=Manual` and keeps the current lens until you change it. Overlay and sidebar show **requested** vs **actual** (metadata) `LensPosition`, `AfState`, `FocusFoM`, and document-region Laplacian (not a pass/fail gate).

`Capture & Send` is still the production upload path. Do **not** use it for this calibration; use **CAPTURE FOCUS TEST** (local JPEG only, no upload, no state machine).

## 3. LensPosition range

Limits come from Picamera2 `camera_controls["LensPosition"]` `(min, max, default)`. Requests are clamped. Mock/dev fallback is **0.0–10.0**. Record the advertised range from `/preview/status` → `focus.lens_min` / `focus.lens_max` on the Pi.

## 4. FocusFoM behavior

`FocusFoM` is **displayed and logged only**. It is not used as a success criterion. After a scan, compare FoM vs document Laplacian vs visual stills.

## 5. Document ROI used

If Phase 2 reports a document bbox, sharpness uses an **interior** crop (12% inset) of that box (`roi_source=document_geometry`), not the full frame center. If no document: labeled `center_fallback` (70% center). Overlay: cyan = focus ROI, blue = document bbox.

## 6. Scan results

**SCAN FOCUS** switches to Manual, steps through the advertised range (default step 0.2), waits for lens settle, measures preview Laplacian, logs `event=manual_focus_scan`, then moves to the best sharpness. It does **not** change the production AF algorithm.

*Fill in on the Pi:*

| Lens | FoM | Sharpness |
|---|---|---|
| | | |

Best lens / sharpness: **not measured in this environment**

## 7. Best LensPosition

**Unknown until Pi calibration.** Do not copy the AF endpoint (~7.8) as “correct.”

## 8. Preview vs full-resolution sharpness

**CAPTURE FOCUS TEST** holds Manual lens, captures one still JPEG under `temp_dir/focus_debug/`, maps the same interior ROI through `ScalerCrop` (preview-equivalent Laplacian, same idea as validation, **no** blur threshold). Logs `event=focus_test_capture`.

*Fill in on the Pi:*

| Lens | Preview sharpness | Still sharpness | JPEG size |
|---|---|---|---|
| | | | |

## 9. Results at different document distances

Repeat manual peak + scan + stills at normal / closer / farther working distance.

| Distance | Best lens | Preview | Still |
|---|---|---|---|
| normal | | | |
| closer | | | |
| farther | | | |

## 10. Does FocusFoM correlate with document sharpness?

**Not answered until tables above are filled.** Possible outcomes: FoM peaks with Laplacian; FoM peaks elsewhere; FoM saturates while text is still soft.

## 11. Hardware observations

Windows checkout cannot run libcamera. After deploy:

```bash
# debug.enabled true, then:
# browser → http://<pi>:8088/preview/live
# M, start ~1.0, step up, note visual peak, then S, then T at nearby lenses
```

Stop competing capture (`Capture & Send`) while calibrating so Continuous AF / still restore does not steal the lens. SCAN/TEST restore **Manual** (`restore_preview_af=False`).

## 12. Recommendations for the next automatic AF (after Pi data)

Wait for §6–10. Then choose among: lock a measured diopter band, maximize **document** Laplacian instead of FoM, fix ROI mapping, or add a still-confirm step. Do not retune AF thresholds until those measurements exist.

## Software status

Unit tests cover clamp, ROI, settle helper, scan argmax, mock lens readback, debug-disabled orchestrator (no calibrator). **Hardware scan: not performed here.**
