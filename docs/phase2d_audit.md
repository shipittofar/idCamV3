# Phase 2D audit — READY → autofocus → capture

**HEAD at audit:** `50b510b` (Phase 2C).  
**Scope:** describe the existing pipeline and the smallest production-safe changes.  
**Not in scope:** state-machine redesign, AF rewrite, capture rewrite, geometry/threshold changes, upload/API, camera driver/FPS, MJPEG.

Hardware already verified by the developer (Phase 2B/2C detector diagnostic): empty absent, quad present, confidence ~0.72–0.78, hold/removal, 1920×1080 RGB, `detect_hz` ~4.79 (application sample rate, not sensor FPS). This audit does not treat that as a Phase 2D capture PASS.

---

## 1. How autofocus is invoked

`WAITING_FOR_STABILITY` (`_stability_tick`) calls `GeometryStability.update`. When ready, it transitions to `FOCUSING` (existing topology).

`FOCUSING` is a one-shot state (`ONE_SHOT_STATES` + `_claim_state_action`). `_focusing_tick` runs once per entry:

1. Capture a preview frame.
2. Call `detect_geometry(frame)` again.
3. `preview_roi = detector.document_preview_roi(frame)`.
4. `autofocus.run_autofocus(..., preview_roi=preview_roi, roi_fraction=roi_center_fraction)`.
5. On `AutofocusError` → `_fail_transaction_and_wait("focus_failed", ...)`.
6. On success → `FOCUS_VERIFICATION`.

Manual capture skips `WAITING_FOR_STABILITY` and goes `DOCUMENT_DETECTED` → `FOCUSING`. Same AF path.

`AutofocusController.run_autofocus` already:

- Resolves `preview_roi` (or 70% center fallback).
- Maps it through `camera.apply_document_focus_window` (preview → live ScalerCrop → `AfWindows`).
- Measures Laplacian on the **preview** ROI before/after a one-shot hardware trigger (`trigger_autofocus` + `wait_for_lens_settle`).
- Accepts via `accept_preview_focus` (relative improvement vs a preview floor, not still `min_sharpness=80`).
- Retries up to `camera.af_max_retries`.
- Logs `event=autofocus_start` and `event=autofocus_result`.

Focus verification (`verify_focus`) is a second preview sharpness check on the same ROI; failure re-enters `FOCUSING` up to `af_verify_max_retries`, then `focus_failed`.

**AF image:** live 1080p preview arrays, not the 64MP still.

---

## 2. Autofocus success / failure

| Outcome | Meaning |
|---|---|
| Success | `accept_preview_focus` true → `AutofocusResult.success=True`; orchestrator does not capture yet |
| Failure | `AutofocusError` after retries, or verify exhausted → transaction failed, not SUCCESS |
| Timeout | `FOCUSING` / `FOCUS_VERIFICATION` state timeout → `focus_failed` |

A failed AF path does **not** call `CaptureService`. That invariant already holds.

---

## 3. How the still is captured

`CAPTURING` is one-shot. `_capturing_tick`:

1. `_check_motion_abort()` (read-only `find_document_quad` vs last `GeometryStability` corners).
2. Disk reserve check.
3. `output_path = temp_store.create_temp_path()` (UUID `capture-<hex>.jpg`, does not exist yet).
4. `capture_service.capture(camera, output_path)`.

`CaptureService.capture`:

- Writes to an infixed partial (`capture-x.partial.jpg`), never `*.jpg.partial`.
- `camera.capture_still_to_file(partial)` with explicit JPEG format (Phase 1).
- Timeout / exception → cleanup partial, `CaptureError`.
- Missing or empty partial → `CaptureError`.
- `finalize_capture`: fsync + `os.replace` to the UUID final path.

`CameraManager.capture_still_to_file`: stop preview → still config → start → hold `LensPosition` → `capture_metadata` settle → `capture_file(..., format="jpeg")` → restore preview. `_verify_still_written` rejects missing/empty files.

Orchestrator on `CaptureError` currently calls `_enter_recovery` (not `_fail_transaction_and_wait`). That is existing capture-hardware policy, not a silent SUCCESS.

---

## 4. Atomic files / stale frames

| Guarantee | Status |
|---|---|
| Encoder format not inferred from `.partial` | Already true (Phase 1) |
| Unique output path per attempt | UUID `create_temp_path` |
| Partial not accepted as final | Finalize only after non-empty partial |
| Old file at the **final** path treated as this capture | **Gap:** if `output_path` already existed, success could be confused in tests/misuse. Production always uses a new UUID. Add a reject-if-exists guard + regression test. |
| Leftover partial reused | **Gap:** if a leftover `.partial.jpg` existed for the same UUID (practically impossible), it could be finalized. Cleanup leftover partial before encode. |
| Still encoded from a settled still stream | `capture_metadata` before `capture_file` (existing) |

Preview frames are captured at AF/abort time; the still is a separate still-configuration capture, not a saved preview JPEG.

---

## 5. Geometry available at capture

At READY, `detect_geometry` has `present`, `corners`, `bbox`, `confidence`, `reason`. `GeometryStability.corners` is the last accepted quad.

**Gap:** `_focusing_tick` **re-runs** `detect_geometry` and then `document_preview_roi`. If that live detect misses (or hold expires), `document_preview_roi` falls back to the **70% search rectangle** — a second ROI, not the READY quad. Focus verification uses `document_preview_roi()` without a frame, same fallback.

`GeometryStability` is not updated during FOCUSING, so capture-time abort still compares against READY corners. Good. AF ROI can still diverge.

Existing AF API **can** consume a preview-pixel bbox (`preview_roi=`). No AF class refactor required. Adapter = freeze accepted geometry on the orchestrator and pass that bbox.

Do **not** change `geometry_stability_threshold_px` or `geometry_stable_frames_required`.

---

## 6. Capture-time geometry safety

`_check_motion_abort` → `_geometry_moved_readonly`: `find_document_quad` (no `classify`/`detect`, no EMA). Abort if no quad or mean corner travel > existing `geometry_stability_threshold_px`.

Keep this policy. Log `event=capture_geometry_abort` / check-passed. Compare against **frozen READY corners** if present (same threshold).

---

## 7. Observability today vs needed

Already present: `autofocus_start`, `autofocus_result`, `capture_started`, `capture_completed`, `capture_failed`, `pipeline_stage`, `geometry_gate`.

Missing / weak:

- Explicit `event=document_ready` at READY → FOCUSING.
- Capture-time geometry check named `event=capture_geometry_abort` / pass.
- AF ROI logged as the **accepted** bbox (pipeline_stage focusing already logs `preview_roi`).

Do not log arrays or invent sensor timestamps.

---

## 8. Minimum required changes

1. **Orchestrator:** snapshot `_accepted_geometry` when auto READY (and when manual enters FOCUSING, last geometry if present). Use that bbox for AF + verify. Do not re-run `detect_geometry` to choose AF ROI. Clear snapshot on transaction clear/fail/removal.
2. **Orchestrator:** `_geometry_moved_readonly` prefers accepted corners; structured abort/pass logs; `event=document_ready`.
3. **CaptureService:** refuse pre-existing final path; delete leftover partial before encode. Do not rewrite still capture or `manager.py`.
4. **Tests:** READY → AF once; AF failure ≠ capture success; AF success → capture; capture error not SUCCESS; old file not accepted; geometry abort; stable path; re-arm + manual unchanged.
5. **Do not modify:** `camera/manager.py`, `autofocus.py` algorithm, `temp_files` naming, geometry detector, stability thresholds, SM topology, live viewer, upload.

---

## 9. Known limitations (not Phase 2D)

- AF sharpness is preview Laplacian, not 64MP still sharpness. Still quality is `ImageValidator` after capture (existing).
- Capture hardware `CaptureError` still enters **RECOVERY** rather than IDLE+re-arm. Changing that is a policy/state-machine change; out of scope.
- Manual capture with no geometry still uses the search-ROI fallback (existing).
- `AfWindows` mapping is already in `CameraManager`; Phase 2D only feeds the accepted preview bbox into that existing API.

---

## 10. Raspberry Pi validation (after unit tests)

Do not claim PASS from pytest. Stop the service, run the installed app, confirm in logs/files:

1. Empty table → no transaction.
2. Place page → geometry present.
3. Wait until READY (`event=document_ready` / FOCUSING).
4. `event=autofocus_start` then `event=autofocus_result success=True`.
5. New `capture-*.jpg` appears under the temp dir (not a previous UUID).
6. Validation accepts the still (`event=image_validated` or equivalent).
7. Remove page → WAITING_FOR_REMOVAL → IDLE.
8. Place a second page → second capture without restarting.
9. Move the page during CAPTURING → `event=capture_geometry_abort`, not SUCCESS.
10. After removal, auto-arm works again.

Use the existing service and `scripts/document_detect_test.py` only as a detector check; do not change camera_capture_test.py.
