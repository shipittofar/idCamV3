# Auto-capture stop and WAITING_FOR_REMOVAL delay

Static analysis of the **current** production path in this repository.
No Raspberry Pi runtime dump was available.

| Verification | Status |
|---|---|
| SOFTWARE VERIFIED | Yes — `src/idcam/app/orchestrator.py`, state machine, geometry stability, config loader, systemd unit |
| HARDWARE VERIFIED | **NOT VERIFIED** |
| `/etc/idcam/config.yaml` on device | **NOT VERIFIED** |

`document_confirmation` is **not** on the auto-capture abort path. Do not reintroduce it.

---

## 1. Executive summary

After a document is already a **stable geometry candidate**, auto-capture can still die and land on **IDLE** while the page is still on the platen. The dominant software causes in this codebase are:

1. **Auto-only raw-quad rechecks** (`find_document_quad`) at pre-AF and pre-capture, **without** detector hold. One contour miss → `document_absent` → IDLE + `_awaiting_clear_scene`.
2. **Post-AF motion abort** treating a missing or shifted quad as “document moved” (`displacement=inf` if `find_document_quad` returns `None`). Lens change at 1920×1080 vs a **12 px** threshold makes this easy.
3. **`_awaiting_clear_scene`** after those failures: IDLE is shown, but auto-capture is **blocked until geometry is absent** for `removal_frames_required` frames. Overlay can still show a document. Looks like “detected → no capture → IDLE”.
4. **WAITING_FOR_STABILITY** is a **second** stability window after IDLE already required `GeometryStability.ready`. One miss **resets** the stable counter (`GeometryStability.update`). Wall-clock timeout is **15 s** (`state_timeouts.waiting_for_stability`) and **ignores** `touch_progress`.
5. **WAITING_FOR_REMOVAL is slow by construction**, not by motion/edge factors (those config keys are **unused** on this path):
   - hardcoded **`asyncio.sleep(2.0)`** in `_success_tick` before entering removal
   - `geometry_hold_frames` (default **3**) still reports `present=True` after the page is gone
   - then **`removal_frames_required` (default 10)** consecutive `not geom.present`
   - loop rate ≈ **`1 / preview_fps`** (default **5 Hz**)
   - **no timeout** on WAITING_FOR_REMOVAL; if the finder still sees a quad, IDLE never happens

Minimum time **after physical removal**, schema defaults:  
`2.0 s (SUCCESS sleep) + (3+10)/5 s ≈ 2.6 s geometry + loop overshoot ≈ **~4.6 s+**`.  
If the document is still on the desk: **unbounded**.

---

## 2. Auto-capture lifecycle trace

All transitions go through `ApplicationOrchestrator._transition` → `StateMachine.transition`.

### IDLE → DOCUMENT_DETECTED

| Item | Code |
|---|---|
| File | `src/idcam/app/orchestrator.py` |
| Method | `_idle_tick` → `_auto_document_ready` → `_begin_transaction` → `_transition(DOCUMENT_DETECTED)` |
| Condition | `geom.present and geometry_stability.update(geom)` (`_observe_geometry`) |
| Not used | `confirm_document`, `temporally_confirmed` |
| Blockers | `has_active_transaction`; `disk.can_capture` false (silent return); `_awaiting_clear_scene and geom.present` (re-arm blocked); `frame is None` (preview/recovery) |

`GeometryStability.update` (`src/idcam/vision/geometry_stability.py`):

- First present frame **never** returns ready (seeds corners).
- Later: mean corner L2 ≤ `geometry_stability_threshold_px` increments count; else count **= 0**.
- `not present` or `corners is None` → **`reset()`** (count 0).

At default 5 Hz and `geometry_stable_frames_required=5`: **≥1.0 s** after first present observation (plus first seed frame).

### DOCUMENT_DETECTED → WAITING_FOR_STABILITY (auto)

| Item | Code |
|---|---|
| Method | `_document_detected_tick` |
| Auto | `_transition(WAITING_FOR_STABILITY, reason="auto_wait_stability")` |
| Manual | skips stability → FOCUSING (`manual_capture_skip_stability`) |

DOCUMENT_DETECTED is **one-shot** (`ONE_SHOT_STATES`). It does **not** re-observe geometry.

Timeout: `state_timeouts.document_detected` default **5 s**. Handler `_handle_state_timeout` → **IDLE** (not `_fail_transaction_and_wait`). Should not fire if the one-shot tick runs; it **is** a silent-looking abort if the tick never runs.

### WAITING_FOR_STABILITY → FOCUSING

| Item | Method `_stability_tick` |
|---|---|
| Need | `geom.present` and `geometry_stability` **ready again** |
| Abort | `not geom.present` for `removal_frames_required` consecutive ticks → `document_absent` |
| Timeout | **15 s wall clock from state entry** (`is_timed_out` uses `elapsed()`, not `last_progress_at`) → `stability_timeout` |

Detector **hold** (`geometry_hold_frames`, default 3): `detect_geometry` can still return `present=True` (`reason=geometry_held`) on misses. Stability therefore may **not** reset during hold. After hold expires, `present=False` → stability **reset**.

### FOCUSING → FOCUS_VERIFICATION / abort

| Step | Method | Auto vs manual |
|---|---|---|
| Pre-AF | `_preview_candidate_usable` = `find_document_quad(frame) is not None` | **auto only** |
| ROI | `_accepted_preview_roi(..., allow_search_fallback=not auto)` | auto: frozen bbox only |
| AF | `AutofocusController.run_autofocus` | same algorithm |
| Post-AF | `_geometry_moved_readonly` | both if `_accepted_geometry` set |
| Fail | `_fail_transaction_and_wait` → IDLE + `_awaiting_clear_scene` | |

`_geometry_moved_readonly`: `find_document_quad` **None** → `(True, inf)` → treated as moved.

### FOCUS_VERIFICATION → CAPTURING

`verify_focus`; on failure, refocus if `af_verify_max_retries` remains, else fail.

### CAPTURING

Auto: `_preview_candidate_usable` again, then `_check_motion_abort` (same raw-quad / 12 px test). Then still capture.

### IMAGE_VALIDATION → UPLOADING → WAITING_FOR_SERVER → PROCESSING_RESPONSE → SUCCESS

One-shot ticks. Upload uses existing API client. ACCEPTED → SUCCESS.

### SUCCESS → WAITING_FOR_REMOVAL

`_success_tick`: **`await asyncio.sleep(2.0)`** (hardcoded), then transition. SUCCESS timeout **5 s** is a backup (`_handle_state_timeout` also goes to WAITING_FOR_REMOVAL).

### WAITING_FOR_REMOVAL → IDLE

`_removal_tick`: `empty = not geom.present`; increment `_removal_count`; need `>= removal_frames_required`. **No state timeout** (`NO_TIMEOUT_STATES`).

---

## 3. All paths that return to IDLE

| # | From | Mechanism | File / method | Condition | Classification |
|---|---|---|---|---|---|
| 1 | CAMERA_READY | `_startup` | `_startup` | Camera + storage OK | NORMAL |
| 2 | DOCUMENT_DETECTED | timeout | `_handle_state_timeout` | `elapsed > document_detected` (5 s) | TIMEOUT / POSSIBLE BUG if one-shot did not run |
| 3 | WAITING_FOR_STABILITY | fail | `_fail_transaction_and_wait` | `document_absent` (N consecutive misses) | EXPECTED FAILURE or POSSIBLE BUG if finder flakes |
| 4 | WAITING_FOR_STABILITY | fail | `_handle_state_timeout` | 15 s not ready | TIMEOUT |
| 5 | FOCUSING / FOCUS_VERIFICATION | fail | `_fail_transaction_and_wait` | pre-AF miss; invalid ROI; AF error; post-AF “moved”; verify fail; focus timeout | POSSIBLE BUG for pre-AF raw miss and post-AF 12 px / inf |
| 6 | CAPTURING | fail | same | pre-capture miss; motion abort; capture error; timeout | POSSIBLE BUG same as pre-AF |
| 7 | IMAGE_VALIDATION | fail | same | validation fail / timeout | EXPECTED FAILURE |
| 8 | UPLOAD / SERVER / PROCESSING | fail | same | network/server/reject/timeout | EXPECTED FAILURE |
| 9 | WAITING_FOR_REMOVAL | success path | `_removal_tick` | N consecutive `not geom.present` | NORMAL |
| 10 | RECOVERY | `_recovery_tick` | camera restart OK | `_transition(IDLE)` | EXPECTED FAILURE recovery |
| 11 | Any fail dest IDLE | `_fail_transaction_and_wait` | `_resume_state != WAITING_FOR_REMOVAL` | IDLE + **`_awaiting_clear_scene=True`** | EXPECTED FAILURE **UX trap** |

Watchdog (`src/idcam/system/watchdog.py`): stall > **per-state** timeout (not `watchdog.state_timeout_sec`) in WATCHED_STATES → `_handle_watchdog` → **RECOVERY**, then later IDLE (#10). WAITING_FOR_REMOVAL is **not** watched.

`document_confirmation` does **not** send the machine to IDLE.

Exception: `_safe_preview` `CameraError` → RECOVERY, not direct IDLE.

---

## 4. Root cause of intermittent auto-capture stop

**Primary (code, auto-only):** `_preview_candidate_usable` / `_geometry_moved_readonly` use **raw** `find_document_quad`. The idle/stability gate uses **`detect_geometry`**, which holds presence for `geometry_hold_frames`. A single contour failure after DOCUMENT_DETECTED aborts with `document_absent` even though the detector would still report a held candidate.

**Then:** `_fail_transaction_and_wait(..., dest=IDLE)` sets `_awaiting_clear_scene`. `_idle_tick` will **not** start another auto transaction while `geom.present`. The UI is IDLE (“Ready — Place document”) with the page still there.

**Secondary:** After AF, corner displacement vs frozen READY corners vs **12 preview pixels**. AF changes blur/scale; 12 px at 1920×1080 is tight. `None` quad → inf displacement → `AutofocusError` → same IDLE + clear-scene lock.

**Tertiary:** WAITING_FOR_STABILITY requires stability **again**. `GeometryStability.update` resets on any `present=False` after hold expires, and on corner travel > 12 px. 15 s later → `stability_timeout` → same lock.

**Not the cause (this tree):** semantic confirmation, center-70% as presence, watchdog 300 s (unused as stall threshold), `removal_motion_factor`.

**Manual vs auto:** manual skips WAITING_FOR_STABILITY and skips pre-AF/pre-capture `_preview_candidate_usable`. Manual still does post-AF `_geometry_moved_readonly` if geometry was snapshotted.

---

## 5. WAITING_FOR_REMOVAL lifecycle trace

```
PROCESSING_RESPONSE (ACCEPTED)
  → SUCCESS
  → sleep 2.0 s          # _success_tick, hardcoded
  → WAITING_FOR_REMOVAL
  → each main-loop tick: preview + detect_geometry
       if geom.present: removal_count = 0
       else: removal_count += 1
  → when count >= removal_frames_required:
       reset detector baseline, GeometryStability
       → IDLE
```

**Not used here:** `removal_motion_factor`, `removal_edge_factor`, `DocumentDetector.is_scene_empty`, `is_document_removed` (except the count comparison lives in orchestrator). `baseline_alpha` is **not** consulted in `_removal_tick`. `reset_baseline()` runs **after** removal is confirmed.

If geometry still finds a quad (page still down, or false quad on the table), `_removal_count` stays 0. State stays WAITING_FOR_REMOVAL **forever** (by design: “Please remove document”).

`_log_geometry_status` logs **only when status string changes**. A long `geometry_present` wait produces **one** log, then silence.

---

## 6. Root cause of removal delay

| Cause | Role |
|---|---|
| A. Configuration | `removal_frames_required=10`, `preview_fps=5`, `geometry_hold_frames=3` |
| B. Algorithm | Removal = **geometry absent**, not motion/edge vs baseline |
| C. State machine | Mandatory WAITING_FOR_REMOVAL; **no timeout** |
| D. Frame rate | One observe per loop + `sleep(1/preview_fps)` **after** the tick (effective Hz ≤ configured FPS; slower if preview is slow) |
| E. Baseline | Irrelevant until after confirmed removal |
| F. Lifecycle | **2.0 s SUCCESS sleep** before removal even starts |
| G. Race | Finder false positives reset the counter; hold delays “absent” |

Delay class: **A + B + C + D + F** (and G if the page is gone but a quad remains).

---

## 7. Effective runtime configuration

**Load order** (`src/idcam/config/loader.py`, `src/idcam/app/main.py`, `deploy/idcam.service`):

1. `--config` if passed; else `IDCAM_CONFIG` (systemd: `/etc/idcam/config.yaml`); else first existing of `/etc/idcam/config.yaml`, `./config.yaml`.
2. YAML → env overlays: `IDCAM_AUTH_TOKEN`, `IDCAM_DEVICE_ID`, `IDCAM_SERVER_URL` only (detection/FPS **not** overridable via env).
3. Pydantic defaults for **omitted** keys.
4. `IDCAM_ENV=production` rejects example.com URLs; does not change detection numbers.

**This Windows workspace has no `/etc/idcam/config.yaml`.** Device file is **NOT VERIFIED**.

| Key | Schema default | `config.example.yaml` | `deploy/config.yaml.template` |
|---|---|---|---|
| `geometry_hold_frames` | 3 | 3 | **omitted → 3** |
| `geometry_stable_frames_required` | 5 | 5 | **omitted → 5** |
| `geometry_stability_threshold_px` | 12.0 | 12.0 | **omitted → 12.0** |
| `geometry_smooth_alpha` | 0.40 | 0.40 | **omitted → 0.40** |
| `removal_frames_required` | 10 | 10 | 10 |
| `removal_motion_factor` | 0.5 | 0.5 | 0.5 (**unused** in `_removal_tick`) |
| `removal_edge_factor` | 0.5 | 0.5 | 0.5 (**unused**) |
| `baseline_alpha` | 0.05 | 0.05 | 0.05 (**unused** in removal tick) |
| `preview_fps` | 5.0 | 5 | 5 |
| `state_timeouts.document_detected` | 5 | 5 | 5 |
| `waiting_for_stability` | 15 | 15 | 15 |
| `focusing` / `focus_verification` | 10 / 5 | 10 / 5 | 10 / 5 |
| `success` | 5 | (schema) | (schema) |
| `watchdog.state_timeout_sec` | 300 | 300 | 300 (**not** used as stall limit; stall uses per-state timeout) |
| `watchdog.check_interval_sec` | 1.0 | 1.0 | 1.0 (**not** used; watchdog runs every main-loop iteration) |

**Verify on Pi:**

```bash
sudo -u idcam /opt/idcam/venv/bin/python - <<'PY'
from idcam.config.loader import load_config
c = load_config()
d, cam, t, w = c.detection, c.camera, c.state_timeouts, c.watchdog
print("config", __import__("idcam.config.loader").config.loader._resolve_config_path())
print("hold", d.geometry_hold_frames, "stable_n", d.geometry_stable_frames_required,
      "thr_px", d.geometry_stability_threshold_px, "removal_n", d.removal_frames_required)
print("preview_fps", cam.preview_fps)
print("timeouts", t.document_detected, t.waiting_for_stability, t.focusing, t.success)
print("watchdog", w.check_interval_sec, w.state_timeout_sec)
PY
```

Journal: `event=state_transition`, `event=document_absent`, `event=transaction_failed`, `event=document_removed`, `event=auto_capture_armed`.

---

## 8. Timing / latency calculation

Assume schema defaults, `preview_fps=5`, tick ≈ `work + 0.2 s` sleep.

**Idle → first auto DOCUMENT_DETECTED**

- Seed frame (not ready) + 5 consecutive stable ≈ **6 observations ≈ 1.2 s** after first `present`, plus jitter resets.

**WAITING_FOR_STABILITY** (if already ready): often **1 tick**. If reset: another **5** stable frames. Max **15 s** then fail.

**SUCCESS → removal start:** **2.0 s** fixed.

**Removal after last real quad (page lifted):**

- Hold: up to **3** ticks still `present` (`geometry_held`)
- Then **10** ticks `not present`
- ≈ **13 / 5 = 2.6 s** plus SUCCESS 2 s already spent

**Typical “upload OK → IDLE” if user removes promptly:** **~4.6 s** plus AF/capture/upload (out of scope) and any finder lag.

**If page stays:** IDLE **never** (until geometry gone).

---

## 9. Race conditions / reset conditions

| Race | What happens |
|---|---|
| Detector hold vs pre-AF raw quad | Stability/idle see document; focusing does not |
| AF vs 12 px corners | Appearance change counted as motion |
| `GeometryStability.reset` on miss | WAITING_FOR_STABILITY starts over |
| `_awaiting_clear_scene` | Failures suppress retry while page remains |
| Main loop `sleep(1/fps)` after slow tick | Effective detect Hz < `preview_fps` |
| `_dispatch_state_tick` reentry guard | Nested ticks skipped (AF holds the loop; OK if sequential) |
| `_stale_async_completion` | Late AF/upload ignored; can look like a hung pipeline if timeout then recovery races |
| Watchdog vs `touch_progress` | Watchdog uses **progress**; **state timeout uses entry time**. Stability can `touch_progress` forever until **15 s** state timeout |

---

## 10. Logging gaps

Existing: `event=state_transition` (rich), `event=auto_capture_gate` on **success** only, `event=transaction_failed`, `event=document_absent` (pre-AF/capture), `event=geometry_gate` **on status change only**.

Missing for “detected → idle”:

- `event=transaction_abort` with `state_from`, `failure_key`, `awaiting_clear_scene=1`, `geometry_present`, `raw_quad_present` (if checking `find_document_quad` vs `detect_geometry`)
- Pre-AF/pre-capture: `raw_quad` vs `detector.present` vs `reason=geometry_held`
- Post-AF: `displacement_px`, `threshold_px`, `raw_quad_none`
- WAITING_FOR_STABILITY: periodic `stable_count/required_count` (not only status change)
- Removal: `event=removal_check` each tick or every N: `removal_count`, `required_count`, `geometry_present`, `elapsed_ms` since SUCCESS
- Startup: dump detection + `preview_fps` + timeouts once (`event=config_effective`)

Do not log full frames.

---

## 11. Minimal recommended fix

Do **not** change Phase 6 topology, AF, capture, upload, or confirmation-as-gate.

**A. Align pre-AF / pre-capture / motion abort with detector presence (auto)**

- Usable if `detect_geometry` (or last + hold) still `present`, **or** raw quad exists.
- Abort only when detector reports **not present** after hold, same as WAITING_FOR_STABILITY’s document-absent rule.
- Keep `_check_motion_abort` for **real** corner travel when **both** quads exist; do **not** treat `raw is None` as inf-move if `detect_geometry` is still held.

**B. Post-AF displacement**

- Skip “moved” when raw quad is missing but held geometry exists; **or** widen threshold only for the post-AF sample (prefer A first).
- Do not change AF search algorithm.

**C. `_awaiting_clear_scene`**

- Keep for true user-removed-during-capture.
- For `document_absent` / `stability_timeout` caused by **finder flake** while the page is still there, either: do not set the flag, or re-arm after N frames of continued `present` (retry auto). Smallest: **do not set `_awaiting_clear_scene` for `document_absent` when the following idle preview still has `geom.present`** — only set it when going idle after **success** or when the scene is actually empty. (Success already uses WAITING_FOR_REMOVAL, not this flag.)

Recommended smallest: set `_awaiting_clear_scene` only for failures where retrying immediately would **double-capture the same page after a failed AF** (product choice). If the product wants immediate retry on flake, **clear the flag when `geom.present` stays true**.

**D. Removal latency**

- Remove or reduce `_success_tick` `sleep(2.0)` (e.g. 0–0.3 s) — **largest fixed delay**.
- Optionally lower `removal_frames_required` (e.g. 3–5) in **config**, not a new algorithm.
- Optional: treat `geometry_held` as not blocking removal countdown (count absent when **raw** quad is gone). Speeds “page lifted” without waiting hold+10.

**E. Logs:** `transaction_abort`, `removal_check`, one-shot `config_effective`.

Preserve: empty scene no capture; WAITING_FOR_REMOVAL until gone (or config-faster); manual path.

---

## 12. Exact files and functions to modify

| File | Functions |
|---|---|
| `src/idcam/app/orchestrator.py` | `_preview_candidate_usable`, `_geometry_moved_readonly`, `_focusing_tick`, `_capturing_tick`, `_check_motion_abort`, `_fail_transaction_and_wait` / `_idle_tick` (`_awaiting_clear_scene`), `_success_tick`, `_removal_tick` (logging + optional hold policy) |
| `src/idcam/config/schema.py` / YAML | only if changing `removal_frames_required` or documenting SUCCESS delay |
| Tests | `tests/unit/test_document_confirmation.py` (pre-AF gone vs held), capture lifecycle, removal tick |

Do not modify `document_confirmation.py` as a gate. Do not rewrite `find_document_quad`.

---

## 13. Test plan (software)

- Auto: detector `geometry_held` but raw `find_document_quad` None → **must still start AF** (after A).
- Auto: empty desk pre-AF → still abort, no AF.
- Post-AF: raw None + held present → **must not** `AutofocusError` document moved (after A).
- Post-AF: large real corner travel → still abort.
- Fail `document_absent` with page still present → IDLE then **retry auto** if B/C chosen; or document lock if product keeps clear-scene.
- SUCCESS: no mandatory 2 s if sleep reduced; WAITING_FOR_REMOVAL still entered.
- Removal: 10 (or new N) consecutive `present=False` → IDLE; `present=True` resets count.
- Empty desk idle: no transaction.
- Manual Capture & Send unchanged (skip stability, skip auto pre-AF).
- `pytest tests -q --tb=line`

Unit tests **do not** prove Pi timing.

---

## 14. Hardware validation plan

**HARDWARE: NOT RUN** until deployed.

1. Dump effective config (section 7).
2. `journalctl -u idcam -f` during: place ID → should see `auto_capture_gate` → `FOCUSING` → capture → `upload_accepted` → `WAITING_FOR_REMOVAL` → lift page → `document_removed` → IDLE.
3. Reproduce flake: note `event=document_absent reason=pre_af_recheck` or `capture_geometry_abort` or `stability_timeout` then IDLE with page still down → `_awaiting_clear_scene` (look for `auto_rearm_blocked` / missing `document_detected`).
4. Time SUCCESS → IDLE with stopwatch after lifting the page; compare to ~4.6 s vs unbounded (page left down).
5. Confirm `detect_hz` in `event=preview_tick` vs `preview_fps`.

---

## Summary table

| Problem | Root cause | Evidence | Fix | Hardware test required |
|---|---|---|---|---|
| Detected → no capture → IDLE | Auto pre-AF/pre-capture/`_geometry_moved_readonly` use raw quad; miss → `document_absent` / moved; then `_awaiting_clear_scene` blocks retry | `orchestrator.py` `_preview_candidate_usable`, `_fail_transaction_and_wait`, `_idle_tick` | Use detector presence/hold for “still there”; don’t inf-move on raw None; reconsider clear-scene on flake | **Yes** |
| Detected → WAITING_FOR_STABILITY → IDLE | Second stability gate + 12 px reset + 15 s `elapsed()` timeout | `_stability_tick`, `GeometryStability.update`, `_handle_state_timeout` | Optional: skip re-stability if already ready; or don’t reset on single miss; timeout uses progress (careful) | **Yes** |
| WAITING_FOR_REMOVAL long after success | 2 s SUCCESS sleep + hold 3 + 10 frames at 5 Hz; no timeout; geometry-only empty | `_success_tick`, `_removal_tick`, `NO_TIMEOUT_STATES` | Cut sleep; maybe fewer removal frames; count raw-miss vs hold | **Yes** |
| Removal never if quad remains | `empty = not geom.present` only | `_removal_tick` | Finder issue **or** page still there; do not bring confirmation back | **Yes** |
| Motion/edge removal unused | `_removal_tick` never reads those factors | grep vs `schema.py` | Do not wire them unless product wants old detector empty-scene | No (unless changing algorithm) |
| Runtime config unknown | Device YAML not in repo | `IDCAM_CONFIG=/etc/idcam/config.yaml` | Dump on Pi | **Yes** |
| Confirmation blocking auto | **Not in this tree** | no `confirm_document` in orchestrator | Do not re-add | N/A |

**HARDWARE: NOT VERIFIED**
