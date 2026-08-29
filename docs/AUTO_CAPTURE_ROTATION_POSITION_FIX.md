# Auto-capture: rotation, position, and stability

This note records why rotated or off-center ID cards could be **seen** by the geometry detector and then **never leave** `WAITING_FOR_STABILITY` / `IDLE`, and what was changed so presence + “same page, stable enough” is the only auto-capture gate.

`document_confirmation` is **not** an auto-capture gate. Center ROI, rotation limits, and mandatory border/edge/print/fold checks are **not** used to start FOCUS → CAPTURE.

**HARDWARE: NOT VERIFIED.** Pytest on the development machine is not a Raspberry Pi 5 placement study. After deploy, run the checklist in section 9 and fill in the table.

---

## 1. Exact root cause(s)

Traced path: `IDLE` → `DOCUMENT_DETECTED` → `WAITING_FOR_STABILITY` → `FOCUSING` → …

| Condition | Role | Did it block rotated IDs? |
|-----------|------|---------------------------|
| `geometry.present` / `detect_geometry()` | Idle start | No, if the finder returns a quad (including hold). |
| `confirm_document` / edge support / paper / print / fold | Unused in orchestrator | No (already removed as a gate). |
| `roi_center_fraction` | AF fallback overlay / motion ROI crop, not presence | No. |
| Rotation vs a max angle | Finder stores `rotation_deg` only | No explicit rotation cap. |
| Border contacts | Logged; not a reject in orchestrator | No, unless the finder itself returns no quad. |
| **Ordered-corner L2 at 12 px** | `GeometryStability.update()` | **Yes.** At ~1920 preview, contour/label jitter on a diamond (~45°) often exceeds 12 px even when the card has not moved. |
| **`order_corners` start vertex (`argmin(x+y)`)** | Mean L2 of TL/TR/BR/BL | **Yes.** Near 45° the labeled “top-left” can hop one vertex. Mean displacement looks like a large move → counter reset → never `ready`. |
| Extra `WAITING_FOR_STABILITY` gate `last_displacement <= 12` (historical) | After idle already counted stability | **Yes.** One noisy WAITING frame blocked `FOCUSING` even when idle had latched. |
| Raw `find_document_quad() is None` as `displacement=inf` | Pre-AF / motion (historical) | Could abort; **fixed earlier** to use detector presence/hold. |
| `_awaiting_clear_scene` | After flake failure | Could freeze IDLE until removal; **re-arm while still present** (earlier + kept). |

The product symptom “detector shows a document, state stays WAITING / IDLE” matches **stability matching**, not “rotation is forbidden.”

---

## 2. Functions / files changed

| File | Change |
|------|--------|
| `src/idcam/vision/geometry_stability.py` | `matched_corner_displacement_px` (cyclic roll + reverse winding); relative threshold `max(px, fraction × mean side)`; miss tolerance default 2; `last_same_document` from displacement vs a **looser latch** budget (12% of mean side), **not** AABB IoU (overlapping pages after a slide were too similar). |
| `src/idcam/app/orchestrator.py` | Relative fraction passed into `GeometryStability`; motion check uses matched displacement + effective threshold; WAITING: if latched auto and **not** same document → stay waiting (`document_moved`); if same document → `FOCUSING` without a 12 px extra veto; `event=auto_capture_blocked` (throttled); idle logs blocked when present but not yet stable. |
| `src/idcam/config/schema.py` | `geometry_stability_relative` (0.05), `geometry_stability_miss_tolerance` (2). |
| `config.example.yaml`, `deploy/config.yaml.template` | Same keys. |
| `tests/unit/test_vision.py` | Vertex-flip displacement; relative jitter; miss_tolerance on moving/absent tests. |
| `tests/unit/test_document_confirmation.py` | Placement / rotation / size / border matrices must reach **`FOCUSING`**, not only `DOCUMENT_DETECTED`. Extra angles: −15°, 33°, 75°. |
| `tests/unit/test_config.py` | Default relative / miss / removal frames. |

Not rewritten: `find_document_quad()` ranking (aspect, rectangularity, right-angle). If the finder returns **no** candidate, auto-capture still does not start.

---

## 3. Before / after

**Before:** A still, rotated ID could keep `geometry.present=1` while `GeometryStability` never reached `frames_required`, or WAITING refused `FOCUSING` because labeled-corner travel exceeded 12 px. Overlay looked “detected”; pipeline stayed WAITING or fell back to IDLE.

**After:** The same physical page is matched even if corner **labels** rotate. Threshold scales with document size. Auto WAITING treats “same page” as latch displacement, not pixel-identical corners. Then FOCUS → existing capture/validate/upload.

Empty desk (`present=0`) still does not autofocus, capture, or upload.

---

## 4. Capture-blocking conditions removed or relaxed

Removed / not used as gates:

- Semantic confirmation (`insufficient_edge_support`, `border_occlusion`, paper/print/fold, center bias).
- Center-of-frame / `roi_center_fraction` as presence.
- Max rotation / must-be-axis-aligned.
- Border contact count as reject.
- Extra WAITING `displacement <= 12` after idle already stable.
- Raw-quad miss as infinite motion when detector still present.

Relaxed:

- Corner travel: matched correspondence + `max(12 px, 5% of mean side)`.
- 1–2 frame misses / over-threshold jitter before resetting the stable counter.
- Latch “same document” at 12% of mean side so AF/noise does not look like a new page.

Still blocking (intentional):

- No geometry candidate (`present=False`) after hold expires.
- Meaningful relocate (matched displacement above latch budget) while WAITING.
- Empty scene.
- Later quality/upload failures (unchanged).

---

## 5. Stability changes

- Compare quads with **best cyclic / reverse** labeling.
- Increment counter when matched mean L2 ≤ effective threshold.
- Do not reset on the first 1–2 misses or jitter frames (`miss_tolerance=2`).
- `last_same_document`: matched L2 ≤ the **same** effective increment threshold (`max(12 px, 5% of mean side)`). WAITING after idle may enter FOCUS on the same page without a second full count, but a real slide (above that threshold) stays in WAITING until consecutive frames agree again. AABB IoU is not used (overlapping pages after a slide were too similar).

---

## 6. Re-arm behavior

If a transaction fails while the page is still in frame, `_awaiting_clear_scene` does **not** wait forever for a clear desk. After `max(2, geometry_stable_frames_required)` consecutive present idle ticks: `event=auto_capture_armed reason=document_still_present`. Empty scene after failure still uses consecutive absent frames.

---

## 7. `WAITING_FOR_REMOVAL`

- Success → removal with **no** fixed `asyncio.sleep`.
- Removal uses **raw** `find_document_quad is None` for several frames (default **4**), so detector hold does not delay lift.
- State timeout **20 s** → IDLE (`waiting_for_removal_timeout`) so a stuck false quad cannot hold the device forever.

---

## 8. Test results

Command:

```text
python -m pytest tests -q --tb=line
```

Recorded on this tree:

```text
579 passed, 6 skipped in 79.30s
```

Coverage added/extended:

- Placement: center, left, right, top, bottom, four corners → idle geometry → **FOCUSING** when `present`.
- Rotation: 0, 20, 45, 60, 90, −15, 33, 75° → **FOCUSING** when `present`.
- Border contacts 0–3; sizes small/medium/large.
- Vertex-flip displacement; relative jitter; raw miss + detector present; confirmation reasons do not block; re-arm while present; success without sleep; removal timeout.

Empty scene: no AF / capture / upload (`test_idle_empty_scene_does_not_start_pipeline`).

These tests **do** require `WAITING_FOR_STABILITY` → `FOCUSING` for synthetic cards. They do **not** replace Pi hardware.

---

## 9. Hardware test results

**HARDWARE: NOT VERIFIED**

Deployed `/etc/idcam/config.yaml` may still have older `debug.max_width` (e.g. 1280). Repository defaults are **1920×1080** contain-fit, no crop. Confirm on device.

| # | Case | Detected? | Stable? | AF started? | Capture? | Upload? | State before capture | Reason if fail | Detect→capture time |
|---|------|-----------|---------|-------------|----------|---------|----------------------|----------------|---------------------|
| 1 | ID straight | | | | | | | | |
| 2 | ~20° | | | | | | | | |
| 3 | ~45° | | | | | | | | |
| 4 | ~60° | | | | | | | | |
| 5 | ~90° | | | | | | | | |
| 6 | Left edge | | | | | | | | |
| 7 | Right edge | | | | | | | | |
| 8 | Top/bottom | | | | | | | | |
| 9 | 1–3 border contacts | | | | | | | | |
| 10 | Partially off-frame | | | | | | | | |
| 11 | Remove after upload | | | | | | | | |
| 12 | Place again immediately | | | | | | | | |

Look for `event=auto_capture_blocked` (`stability_not_ready`, `stability_jitter`, `document_moved`) and `event=auto_capture_gate`.

---

## 10. Remaining limitations

- If `find_document_quad` returns nothing (too small, too dark, extreme crop, non-quad), auto-capture will not start. That is a **finder** miss, not a rotation policy.
- Aspect / rectangularity / right-angle **inside the finder** can still drop a candidate; those thresholds were not rewritten.
- Stability is “same page,” not “frozen pixels.” Very fast waving will still wait or abort.
- Debug JPEG size follows **runtime** config on the Pi, not only git defaults.
- **Do not treat this document as complete** until the table in section 9 is filled on a Raspberry Pi 5.
