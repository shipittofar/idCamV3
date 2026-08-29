# Phase 6 — State Machine

## Scope

This phase changes **orchestration and state-machine behavior only**.

Touched:

- `src/idcam/app/state_machine.py` (existing transition table; names unchanged)
- `src/idcam/app/orchestrator.py` (ticks, failure routing, stale results, logs)
- `src/idcam/app/transaction.py` (async consume tokens)
- `tests/unit/test_state_machine_phase6.py` and related orchestrator tests
- this document

Not modified as algorithms/implementations:

- `CameraManager` / preview acquisition
- document detection / geometry / `GeometryStability` thresholds
- autofocus algorithm / ROI mapping / lens control
- still capture / JPEG encode
- image validation thresholds
- `ApiClient` / upload payload / upload timeouts
- UI rendering

No new `AppState` values were added.

## Existing State Graph

Single source of truth: `ALLOWED_TRANSITIONS` in `src/idcam/app/state_machine.py`.

Any edge not in that table is **forbidden**. `_transition` logs `event=state_transition_rejected` (`from_state`, `to_state`, `reason`, `capture_id`) and raises `StateMachineError`. State is not mutated.

```
BOOT → INITIALIZING → CAMERA_INITIALIZING → CAMERA_READY → IDLE

Automatic capture:
IDLE
  → DOCUMENT_DETECTED          (geometry present AND GeometryStability ready)
  → WAITING_FOR_STABILITY
  → FOCUSING
  → FOCUS_VERIFICATION
  → CAPTURING
  → IMAGE_VALIDATION
  → UPLOADING
  → WAITING_FOR_SERVER
  → PROCESSING_RESPONSE
  → SUCCESS
  → WAITING_FOR_REMOVAL
  → IDLE                       (removal_frames_required consecutive absent frames)

Manual capture (IDLE or WAITING_FOR_REMOVAL only):
  → DOCUMENT_DETECTED → FOCUSING (skips WAITING_FOR_STABILITY)

Infrastructure:
  camera/preview/watchdog → RECOVERY → IDLE or ERROR
  ERROR → RECOVERY (backoff) or SHUTDOWN
```

`IDLE → FOCUSING` remains **allowed** in the table for compatibility; the live auto path does not use it. Auto capture always goes `IDLE → DOCUMENT_DETECTED → WAITING_FOR_STABILITY → FOCUSING`.

### Entry / exit (production)

| State | Enter | Exit success | Exit failure / timeout | Cancel |
|---|---|---|---|---|
| `IDLE` | Camera ready, or removal confirmed, or failed txn waiting for empty scene then re-arm | `DOCUMENT_DETECTED` when stable (auto) or manual queued | Preview `CameraError` → `RECOVERY` | Shutdown |
| `DOCUMENT_DETECTED` | `_begin_transaction` | Auto: `WAITING_FOR_STABILITY`. Manual: `FOCUSING` | 5 s → abort to `IDLE` | Abort txn |
| `WAITING_FOR_STABILITY` | Auto detect hop | Geometry ready → `FOCUSING` | Consecutive absent (`removal_frames_required`) or 15 s → fail txn | Abort txn |
| `FOCUSING` | Stable (auto) or manual skip | AF accepted → `FOCUS_VERIFICATION` | AF error / 10 s → fail txn (**not** camera restart) | Stale AF ignored |
| `FOCUS_VERIFICATION` | AF accepted | Verified → `CAPTURING` | Retry `FOCUSING` while `af_verify_max_retries`; then fail txn | Stale verify ignored |
| `CAPTURING` | Focus verified | File returned → `IMAGE_VALIDATION` | `CaptureError` / 45 s → fail txn (**not** camera restart) | Stale still ignored |
| `IMAGE_VALIDATION` | Still encoded | Validator ok → `UPLOADING` | Invalid / 10 s → fail txn | Stale validation ignored |
| `UPLOADING` | Validated image | HTTP returned → `WAITING_FOR_SERVER` | Network/protocol / 60 s → fail txn | Stale upload ignored |
| `WAITING_FOR_SERVER` | Upload sent | Immediate → `PROCESSING_RESPONSE` | 60 s → fail txn | |
| `PROCESSING_RESPONSE` | Response object | `ACCEPTED` → `SUCCESS` | Reject/unknown → fail txn | Mismatched `capture_id` ignored |
| `SUCCESS` | API accepted | ~2 s → `WAITING_FOR_REMOVAL` | 5 s timeout same hop | |
| `WAITING_FOR_REMOVAL` | Success or manual resume | Consecutive absent → `IDLE` | No timeout; document stays here | Manual starts new txn |
| `RECOVERY` | Init fail, preview `CameraError`, watchdog | Restart ok → `IDLE` | Attempts exhausted → `ERROR` | |
| `ERROR` | Storage/recovery exhausted | Backoff → `RECOVERY` | Shutdown | |

Timeouts were **not** changed (`StateTimeoutsConfig` defaults; `UPLOADING` remains 60 s).

### Async operations

Owned by the current `ActiveCaptureTransaction.capture_id`:

- autofocus / verify
- still capture
- `validator.validate` (`asyncio.to_thread`)
- `api.upload_document`
- `_pending_response` in `PROCESSING_RESPONSE`

Apply only if: matching `capture_id`, expected state, transaction not terminal, operation not already consumed (`try_consume_async`). Else `event=stale_async_ignored operation=...`.

Error completions use `consume=False` so bounded AF-verify retries still work.

### Removal hysteresis

`WAITING_FOR_REMOVAL` and `WAITING_FOR_STABILITY` (absent) both require **consecutive** empty frames: `DetectionConfig.removal_frames_required` (production default **10**; tests often use **2**). One missing frame is not removal.

Stability uses existing `GeometryStability` (`geometry_stable_frames_required` default **5**, `geometry_stability_threshold_px` default **12**). Detector hold/smoothing unchanged.

### Concurrency protection

- `_dispatch_state_tick`: at most one handler (`_state_tick_running`)
- `ONE_SHOT_STATES`: AF/capture/validate/upload/success run once per state entry
- `_begin_transaction`: rejects a second non-terminal txn (`event=capture_rejected reason=busy`)
- Manual capture busy-rejects outside `IDLE` / `WAITING_FOR_REMOVAL`

## Changes Made

| File | What | Why |
|---|---|---|
| `orchestrator.py` `_capturing_tick` / capture timeout | `CaptureError` and capture timeout → `_fail_transaction_and_wait("still_capture_failed")` | Transaction failure must not restart the camera |
| `orchestrator.py` `_stability_tick` | Consecutive absent frames use `removal_frames_required` | Same hysteresis as removal; one bad frame ≠ gone |
| `orchestrator.py` `_transition` | Reject illegal edges with `from_state` / `to_state`; clear consume tokens on entry | Deterministic table; retry-safe async consume |
| `orchestrator.py` `_stale_async_completion` | Match `capture_id`, state, not terminal, consume-once | Late AF/capture/validate/upload/server cannot move a newer txn |
| `orchestrator.py` `_processing_response_tick` | Ignore response whose `capture_id` ≠ current txn | Stale server result |
| `orchestrator.py` `_begin_transaction` / `request_manual_capture` | `event=capture_rejected reason=busy` | Detection/manual while busy does not start a second txn |
| `orchestrator.py` success/fail | `event=transaction_completed` / `event=transaction_failed` | Reconstruct lifecycle from journalctl |
| `transaction.py` | `applied_async_ops` / `try_consume_async` | One apply per operation name per state entry |
| tests | Phase 6 coverage expanded | Spec items 1–28 (unit, mocked camera/API) |

## Failure Semantics

**Transaction failure** (AF, verify, capture, validation, upload, server reject/timeout, stability timeout, document absent during wait, motion abort):

- Mark txn failed, cleanup temp file
- Auto: `IDLE` + `_awaiting_clear_scene` until consecutive empty frames
- Manual started from removal: may return to `WAITING_FOR_REMOVAL`
- Log `event=transaction_failed`
- **Do not** call `camera.restart` / **do not** enter `RECOVERY`

**Infrastructure failure:**

- Camera init failure
- Preview `CameraError` (disconnect / unrecoverable camera I/O)
- Watchdog stall (no progress in a watched state)
- Recovery attempts exhausted → `ERROR`

Those use `RECOVERY` (bounded `recovery.max_attempts`, then `ERROR`).

**Retry:** Hardware AF retries stay inside `AutofocusController` (`af_max_retries`). Verify may re-enter `FOCUSING` up to `af_verify_max_retries`. After a **failed transaction**, the same page does not auto-retry until the scene is cleared. After **success**, the page stays in `WAITING_FOR_REMOVAL`.

**Removal:** Consecutive absent frames only.

## Concurrency

- Tick serialization: `_state_tick_running` in `_dispatch_state_tick`
- Transaction ownership: one `ActiveCaptureTransaction`; `_begin_transaction` raises after `capture_rejected`
- Stale async: `_stale_async_completion` + response `capture_id` check
- Consume tokens reset on legal `_transition` so verify→refocus can run AF again

A hung upload still occupies the current tick until `ApiClient` returns (timeouts unchanged; not a Phase 6 upload rewrite).

## Manual Capture

Allowed **only** from `IDLE` and `WAITING_FOR_REMOVAL` (existing design).

Busy (returns `status=busy`, logs `capture_rejected` + `manual_capture_rejected`):

`FOCUSING`, `FOCUS_VERIFICATION`, `CAPTURING`, `IMAGE_VALIDATION`, `UPLOADING`, `WAITING_FOR_SERVER`, `PROCESSING_RESPONSE`, and any state with an active non-terminal transaction.

Manual uses a new `capture_id` via `_begin_transaction(trigger="manual")` and skips stability.

## Tests

Command:

```text
python -m pytest tests -q --tb=line
```

Result (this environment, 2026-08-29):

**369 passed, 6 skipped** in 37.95 s.

Phase 6 file: `tests/unit/test_state_machine_phase6.py` (32 tests in that module).

Mapped coverage:

| Spec | Test |
|---|---|
| 1 boot → IDLE | `test_startup_reaches_idle` |
| 2 empty IDLE | `test_idle_empty_scene_does_not_start_pipeline` |
| 3 detect → stability | `test_document_appears_then_moves_then_stabilizes` |
| 4 unstable ≠ AF | `test_unstable_geometry_does_not_trigger_af` |
| 5 one AF / one txn | `test_stable_document_triggers_af_once` |
| 6 SUCCESS → WAITING_FOR_REMOVAL | `test_success_reaches_waiting_for_removal_and_holds` |
| 7 same document no second capture | `test_same_document_does_not_recapture_after_success` |
| 8 removal hysteresis | `test_single_bad_frame_does_not_abandon_stability`, `test_consecutive_absence_abandons_stability` |
| 9 replacement | `test_document_removal_returns_idle_then_replacement_starts` |
| 10–14 txn fail ≠ camera restart | AF/capture/validation/upload/server tests (`camera.restart.assert_not_called`) |
| 15 camera fail → RECOVERY | `test_camera_preview_failure_enters_recovery` |
| 16 illegal transition | `test_invalid_transition_logs_and_raises` |
| 17–21 stale AF/capture/validation/upload/server | `TestStaleAsyncOps` + consume-once |
| 22 concurrent tick | `test_concurrent_tick_is_rejected` |
| 23 concurrent txn | `test_second_transaction_is_rejected` |
| 24 manual IDLE | `test_manual_capture_from_idle_starts_transaction` |
| 25 manual busy | `test_manual_capture_rejected_during_automatic_pipeline` |
| 26 lighting/noise | `test_small_geometry_jitter_does_not_leave_waiting` |
| 27 no auto retry loop | `test_af_failure_is_bounded_and_does_not_loop` |
| 28 one placement ≤ one txn | success hold + `_awaiting_clear_scene` tests |

## Hardware Validation

**NOT RUN**

Pi scenarios A–P from the Phase 6 brief were not executed in this environment.

Required before full Phase 6 acceptance: empty scene, one placement, leave in place, remove, replace, AF/capture/validation failures, slow upload, temporary bad frame, lighting change — confirm one txn per placement, no unexpected `RECOVERY`, no duplicate capture/upload.

## Known Issues

Left unchanged (other phases / existing design):

- Upload wall-clock ~47 s on Pi; `UPLOADING` timeout remains 60 s; `ApiClient` not modified.
- While a tick is blocked in `await upload_document` / still capture, the main loop cannot run watchdog until the await returns.
- `docs/ARCHITECTURE.md` still lists some older failure edges (e.g. capture → `RECOVERY`). Code is authoritative.
- `IDLE → FOCUSING` remains in `ALLOWED_TRANSITIONS` for compatibility; auto path does not use it.
- `ERROR` may sleep 10 s and re-enter `RECOVERY` if storage is healthy (bounded per cycle, process-lifetime latch not added).
- Detector/geometry/AF/validation/upload bugs, if any, are out of scope unless they block SM correctness. None were “fixed” by changing those algorithms.

## Status

**PHASE 6 SOFTWARE COMPLETE — HARDWARE VALIDATION PENDING**
