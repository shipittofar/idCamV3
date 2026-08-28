# Autofocus & Capture Fix

## Root Cause

Hardware AF on the Arducam 64MP Hawkeye was already working. Production rejected a completed scan because **preview Laplacian was treated as the AF success signal**.

Typical production failure:

```
before=12.4 after=12.0 ratio=0.97
lens=8.17→8.12
method=hardware_trigger
```

What actually happened:

1. Preview Continuous AF had already parked the lens near 8.
2. `trigger_autofocus()` started a real scan, but completion waited only on **LensPosition settle** (capped at 3s) and **ignored AfState**.
3. A lens already at ~8 looks “settled” immediately, so the wait returned before (or without) treating `AfState=Focused` as success.
4. `accept_preview_focus()` then required a Laplacian ratio ≥ 1.2 or `after ≥ 28` with ≥ 0.95× before. 12.4→12.0 failed both.
5. The controller retried hardware AF three times for a −3% preview fluctuation.
6. `run_autofocus` raised `AutofocusError` → orchestrator `focus_failed` → **no still capture**.

Capture was skipped in `_focusing_tick` after that exception. It never reached `FOCUS_VERIFICATION` or `CAPTURING`.

Laplacian on a 1920×1080 RGB document ROI is **not** FocusFoM. Hardware diagnostics showed FocusFoM 26→321 with `AfState` Idle→Scanning→Focused. Those metrics must not be required to match.

## Hardware Validation

Direct Pi tests (provided, not re-run in this change) prove:

| Claim | Evidence |
|---|---|
| Lens moves | Manual `LensPosition` 1.0 and 8.0 |
| Auto scan works | 1.0 → 7.83 after `AfMode=Auto`, `AfTrigger=Start` |
| AfState available | 0 → 1 (Scanning) → 2 (Focused) |
| FocusFoM rises | ~26 → ~321 |

**This change’s Raspberry Pi smoke test: NOT RUN.**

Do not treat the table below as hardware PASS.

## Changes

| File | Why |
|---|---|
| `src/idcam/camera/manager.py` | `wait_for_autofocus_completion()`: wait on AfState Focused/Failed; lens-settle fallback; timeout. Pad document bbox for `AfWindows` only (`expand_preview_roi`). `autofocus_cycle()` uses the new wait. |
| `src/idcam/camera/autofocus.py` | Hardware success = AfState Focused (or lens-settle + Laplacian fallback). Do not retry Focused scans because preview Laplacian dipped. Separate failure reasons. Log FoM. |
| `src/idcam/exceptions.py` | `AutofocusError.reason` for classification. |
| `src/idcam/app/orchestrator.py` | Map AF reasons; log still-capture / API upload aliases. Invalid auto ROI → `invalid_document_roi`. |
| `src/idcam/ui/messages.py` | User strings for the new failure keys. |
| `tests/unit/test_autofocus.py` | AfState, fallback, pad, verification noise. |
| `tests/unit/test_camera.py` | Padded `af_preview_roi` still keeps document `preview_roi`. |

Not changed: upload protocol, camera drivers, FPS, geometry detector, still 9152×6944 JPEG path.

## AF Completion Logic

```
trigger AfMode=Auto, AfSpeed=Fast, AfRange=Full, AfTrigger=Start
        ↓
if live metadata has AfState:
    wait until Focused or Failed or timeout
    leftover Focused from preview Continuous AF is not accepted
    until Scanning is seen, or a 0.6s grace if the lens never leaves Focused
else:
    wait for LensPosition to stabilize
```

The camera lock is **not** held across the wait (only brief `capture_metadata` / `set_controls`). Timeout remains `af_timeout_sec` (default 10s). The process cannot wait forever.

## ROI Handling

- Document bbox (preview pixels) is unchanged for Laplacian.
- Hardware `AfWindows` uses `expand_preview_roi` (+20% each side, at least 25% of preview), still centered on the document, clipped to the frame, then mapped through live `ScalerCrop`.
- Example production window `756,409,801,482` is large enough; padding reduces texture-starved interiors. Mapping math is unchanged (`preview_roi_to_af_window`).
- Empty/tiny document ROI is still rejected (`invalid_document_roi`). No 70% search fallback on the auto path.

## Sharpness Verification

Preview Laplacian is optional **after** hardware completion:

- `AfState=Focused` → hardware success even if Laplacian is 12.4→12.0.
- Verification later accepts Focused unless sharpness **collapses** below 40% of the post-AF value.
- If AfState is missing → keep `accept_preview_focus` (relative improvement or already-sharp near the preview floor).
- `FocusFoM` is logged only. No FoM threshold.

## Capture Pipeline

Existing topology, unchanged:

```
IDLE → geometry present → WAITING_FOR_STABILITY → READY
    → FOCUSING (document ROI + hardware AF + lens lock)
    → FOCUS_VERIFICATION
    → CAPTURING (full-res still JPEG, atomic .partial.jpg)
    → IMAGE_VALIDATION
    → UPLOADING (existing ApiClient; not redesigned)
```

Still capture still switches to the 9152×6944 still config, holds `LensPosition` in Manual, encodes JPEG, restores preview.

## Error Classification

| Reason | Meaning |
|---|---|
| `autofocus_hardware_failed` | `AfState=Failed` after retries |
| `autofocus_timeout` | No Focused/Failed before timeout |
| `invalid_document_roi` | Missing/unusable accepted geometry |
| `focus_verification_failed` | Preview check failed (no AfState, or sharpness collapsed) |
| `still_capture_failed` | Logged on `CaptureError` (recovery policy unchanged) |
| `image_invalid` | JPEG validation (existing) |
| `upload_failed` / `server_rejected` | Existing API path |

`focus_failed` remains the fallback key for UI/timeouts.

## Tests

Added/updated in `tests/unit/test_autofocus.py` and `tests/unit/test_camera.py`:

- Idle→Scanning→Focused succeeds with flat Laplacian; one trigger; lens lock
- `AfState=Failed` retries as hardware failure
- Timeout classified
- Lens-settle fallback accepts improvement and rejects 10→8
- ROI pad keeps center; tiny ROI grows
- Verify accepts Focused + noise; rejects collapse

Run from this workspace after the change: focused 173 passed; full suite **327 passed, 6 skipped**. Hardware: **NOT RUN**.

## Remaining Risks

- Pi production AF + still capture after this patch is **not** executed here.
- A 0.6s grace for leftover `AfState=Focused` may accept preview Continuous AF without a full windowed rescan if the driver never reports Scanning.
- `jpeg_quality` is still not passed to Picamera2 `capture_file`.
- Encoder `CaptureError` still enters **RECOVERY**, not IDLE+re-arm.
- Upload still depends on network/server; this patch only preserves the existing handoff.

## Final validation (this session)

| # | Question | Evidence |
|---|---|---|
| 1 | Does hardware AF move the lens? | Prior Pi log: 1.0→7.83. **This session: NOT RUN.** |
| 2 | Does AfState reach Focused? | Prior Pi log: state 2. **This session: NOT RUN.** Software unit test simulates Idle→Scanning→Focused. |
| 3 | Does document ROI reach AfWindows? | Code: pad + `preview_roi_to_af_window` + live ScalerCrop. **Hardware mapping NOT RUN.** |
| 4 | Does the lens lock after AF? | `pause_autofocus()` on hardware success; unit test asserts `lens_locked`. **Pi: NOT RUN.** |
| 5 | Does automatic mode proceed to still capture? | Orchestrator still FOCUS_VERIFICATION→CAPTURING after AF success. Previously blocked by Laplacian fail. **Pi: NOT RUN.** |
| 6 | Is the still full resolution? | Unchanged still config 9152×6944. **Pi: NOT RUN.** |
| 7 | Is the JPEG valid? | Unchanged CaptureService + ImageValidator. **Pi: NOT RUN.** |
| 8 | Does the image proceed to API upload? | Unchanged `UPLOADING` → `ApiClient.upload_document`. **Pi: NOT RUN.** |

**HARDWARE VALIDATION: NOT RUN**
