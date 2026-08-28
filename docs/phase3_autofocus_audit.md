# Phase 3 autofocus audit

**HEAD at audit:** `b7546e4` (Phase 2D). Working tree clean.  
**Scope:** document-ROI autofocus. Capture/upload/SM/UI are Phase 4+ and are not changed here.

Pi detector work (2B/2C) is **not** an AF hardware PASS.

---

## Answers (required)

### 1. How autofocus currently starts

Production path (Phase 2C/2D):

1. Geometry READY → snapshot `_accepted_geometry` → `FOCUSING` (one-shot).
2. `_focusing_tick` takes a preview frame, `preview_roi = _accepted_preview_roi(frame)`, then `autofocus.run_autofocus(..., preview_roi=preview_roi)`.
3. `AutofocusController` measures preview Laplacian on that ROI, maps ROI via `camera.apply_document_focus_window`, then `_run_hardware_af`.

Preferred hardware path: `trigger_autofocus` + `wait_for_lens_settle` if both exist (CameraManager). Else `autofocus_cycle`, else `AfMode=Auto` + `AfTrigger=Start`.

Preview init also starts **Continuous** AF (`_enable_preview_autofocus`) for live view. Document AF is a later one-shot, not per-frame.

Manual capture skips stability, snapshots last geometry if present, same `FOCUSING` path.

### 2. Camera controls used

| Control | Where | Notes |
|---|---|---|
| `AfMode` | preview Continuous; one-shot Auto; Manual on pause/still | Integers via libcamera enums with fallbacks 0/1/2 |
| `AfTrigger` | Start (0) | One-shot |
| `AfSpeed` | Fast | Preview enable + trigger |
| `AfRange` | Full (2) | With AfWindows |
| `AfMetering` | Windows (1) | With AfWindows |
| `AfWindows` | `[(x,y,w,h)]` | Sensor/ScalerCrop space |
| `LensPosition` | Settle wait; pause; still hold | Metadata, not advertised ranges |
| `AfPause` | `pause_autofocus` if advertised | Else Manual + LensPosition |
| `AfState` | Optional metadata | Often **unadvertised** on this hardware; not required for completion |
| `FocusFoM` | **Unused** | Do not invent |

`set_controls` drops keys not in `camera_controls`.

### 3. Full frame vs ROI

Sharpness: **preview-pixel ROI** (`laplacian_variance(..., roi=)`).  
Hardware AF: **AfWindows** mapped from that preview ROI through **live ScalerCrop**.  
If `preview_roi` is omitted, `_resolve_preview_roi` uses the **centered 70%** search rectangle (`roi_fraction`). That is the dangerous fallback.

### 4. AfWindows configured?

Yes, in `CameraManager.apply_document_focus_window`: clip preview ROI → `map_preview_roi_to_sensor` → `AfWindows=[sensor_xywh]`, `AfMetering=Windows`, `AfRange=Full`.

### 5. AF success meaning

`accept_preview_focus(before, after, absolute_floor)` on **preview** Laplacian:

- Relative: after/before ≥ 1.2 and delta ≥ 2, **or**
- Already sharp: after ≥ `PREVIEW_ABSOLUTE_FLOOR` (28) and after ≥ 0.95×before.

Still `min_sharpness` (80) / `0.6*80=48` is **not** a veto for preview AF. Hardware `AfState==Focused` is **not** required.

### 6. Focus completion detection

Not AfState-first. `wait_for_lens_settle`: LensPosition change < 0.05 for 3 samples, timeout `min(af_timeout_sec, 3.0)`. If AfState is advertised, `_wait_for_af` can use Focused/Failed. Timeouts log `event=autofocus_timeout` and still evaluate sharpness (may fail).

### 7. Lens lock after focus

Yes, on success:

- Controller: `pause_autofocus` after accepted sharpness.
- Orchestrator: `pause_autofocus` again after `run_autofocus`.

`pause_autofocus`: `AfPause` if advertised, else `AfMode=Manual` + current `LensPosition`. Mock camera no-ops. Failure path does **not** call pause. Still capture (Phase 4) separately holds lens — not Phase 3 work.

### 8–9. Preview sharpness / document ROI

Laplacian variance on a gray ROI. Controller uses resolved `preview_roi`. Orchestrator **intends** accepted bbox. Gap: `_accepted_preview_roi` still falls back to `document_preview_roi` → **70% center** if snapshot is missing. `_resolve_preview_roi` also centers if `preview_roi is None`. Invalid/zero bbox is clipped to a 1×1 box rather than rejected.

### 10–11. Timeout / failure

`af_timeout_sec` (default 10), `af_max_retries` (3), `af_verify_max_retries` (2). Timeout of cycle/settle → retry or `AutofocusError`. Orchestrator: `focus_failed`, not SUCCESS. FOCUSING/FOCUS_VERIFICATION state timeouts also `focus_failed`.

### 12. Document movement during AF

**Not** checked during FOCUSING (intentionally: lens motion ≠ page motion). Geometry abort runs in **CAPTURING** (`_check_motion_abort`, read-only `find_document_quad`). Phase 3 can reuse that comparison **after** AF returns, before treating focus as valid. Do not run it during lens settle.

### 13. Multiple AF runs

`FOCUSING` is one-shot per entry. Verify failure can re-enter FOCUSING (`refocus`) up to `af_verify_max_retries`. Not every preview frame. `_idle_tick` does not call AF.

### 14. Minimum changes (Phase 3 only)

1. Deterministic **preview → AfWindows** helper: validate, clamp to preview and to ScalerCrop, reject empty/invalid, **never** substitute center.
2. Orchestrator: if accepted geometry exists, AF/verify **must** use that bbox; invalid snapshot → `AutofocusError`, not search ROI.
3. Controller: provided ROI that is invalid must raise; center fallback only when ROI omitted **and** document ROI not required (legacy unit tests / no-geometry manual).
4. After successful AF, reuse `_geometry_moved_readonly`; if the page moved, fail focus.
5. `event=lens_lock`, duration_ms on result, `event=autofocus_failure` / timeout already partly present — align names.
6. Tests A–L. Do not change capture, geometry thresholds, SM, or camera FPS.

---

## AfWindows coordinate system (do not assume = preview)

| Space | Typical size | Origin | Rectangle |
|---|---|---|---|
| Preview array | Requested 1920×1080 (use **configured** stream size) | Top-left of the preview buffer | `(x, y, w, h)` pixels; `x+w` exclusive |
| Live `ScalerCrop` | e.g. `(0, 898, 9152, 5148)` 16:9 window on ~9152×6944 array | Sensor/active-array pixels | Same xywh |
| `AfWindows` | Same as ScalerCrop | Sensor pixels | **List** of rectangles `[(x, y, w, h)]` |

Mapping (existing, correct idea):

```
sensor_x = scaler_x + round(preview_x * scaler_w / preview_w)
```

(and same for y, w, h). **Not** 70% of `ScalerCropMaximum` independently. **Not** PixelArraySize unless ScalerCrop is the full array.

Inclusive x,y; width/height are extents (libcamera Rectangle). Clamp mapped window **into** the live ScalerCrop so rounding cannot escape the crop. Reject w≤0 or h≤0. Minimum preview side 8 px so AfWindows is not a 1×1 accident.

libcamera does **not** take a quadrilateral for AF; use the accepted **bbox**.

`FocusFoM` is not used (often unadvertised). Completion remains lens settle + ROI sharpness.

---

## Existing config (do not duplicate)

- `camera.af_timeout_sec` (10)
- `camera.af_max_retries` (3)
- `camera.af_verify_max_retries` (2)
- `detection.roi_center_fraction` (0.7) — **search visualization / legacy fallback only**
- `quality.min_sharpness` — still images; preview floor is `PREVIEW_ABSOLUTE_FLOOR=28`

No new knobs unless a gap appears. AfSpeed Fast / AfRange Full stay; no evidence-free retune.

---

## Gaps vs Phase 3 criteria

| Criterion | Today | Phase 3 action |
|---|---|---|
| Accepted geometry as AF ROI | Intended; fallback to 70% if snapshot missing | Fail closed when snapshot exists but ROI invalid; auto path requires snapshot |
| AfWindows conversion tested | Partial (`map_preview_roi_to_sensor`) | Validate + clamp + reject tests |
| One AF per READY | Yes (one-shot) | Keep; regression test |
| Bounded timeout | Yes | Keep |
| Explicit success/failure | Yes | `autofocus_failure` log |
| Sharpness on document ROI | Yes if ROI passed | Enforce; no center when document ROI required |
| Lens lock | `pause_autofocus` | Log `event=lens_lock`; test |
| No center fallback with valid geometry | Incomplete | Fix |
| Movement during AF | Capture-only | Post-AF reuse of read-only compare |
| Capture | Phase 2D already captures after verify | **Do not extend capture** |

---

## Phase 4+ (leave untouched)

Still encode, temp files, upload, SM topology, live viewer architecture, geometry algorithm/thresholds, sensor/preview FPS, `_enable_preview_autofocus` Continuous for idle preview (needed for live view; document AF still one-shot).

---

## Raspberry Pi validation (after unit tests)

Do not claim PASS from pytest. Off-center page: logs `autofocus_roi` / `autofocus_start` must match document bbox, not frame center; `autofocus_result success`; `focus_verification`; `lens_lock`. Positions, distances, already-sharp (must not fail solely because after≈before), forced fail, move-during-AF. Capture remaining production is Phase 4; Phase 3 must not add a new capture path.
