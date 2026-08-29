# Document false-positive and center-bias audit

**Status: INVESTIGATION ONLY — not a fix. Autofocus/capture/upload of empty scenes is not resolved by this document.**

Scope: production auto-capture path on Raspberry Pi 5 / Arducam 64MP Hawkeye. Code cited is the current tree (`src/idcam/...`). No thresholds were changed.

---

## 1. Executive summary

The auto pipeline **does not require a semantic document**. It requires a **stable quadrilateral** from `find_document_quad`.

`DocumentDetector.classify()` / motion-vs-empty-baseline / `edge_density` **are not on the trigger path**. IDLE uses only `detect_geometry` + `GeometryStability`. Overlay motion/edge numbers are **not** what starts AF.

Once `geom.present` and corner motion ≤ **12 px** for **5 consecutive** geometry frames, Phase 6 starts a transaction. DOCUMENT_DETECTED is a one-tick hop. WAITING_FOR_STABILITY usually succeeds on the next tick if the false quad is still still. AF then runs on that bbox. If libcamera reports `AfState=Focused`, **preview Laplacian verification is skipped**. Still validation uses a **preview-equivalent floor of 6.0** (and 50% of the AF score). Table grain / a placemat / a window can exceed 6. Upload then proceeds.

**Center bias is real but secondary for auto-capture.** Geometry search is full-frame. Center 70% is used for **legacy presence metrics**, **debug overlay**, **AF/sharpness fallback**, and **validation when no preview ROI is passed**. Off-center documents fail more from **axis-aligned merge**, **`order_corners`**, **AABB occupancy**, **aspect of bbox vs quad**, and **border/full-frame gates**, not from “must be in the middle.”

**Rotation/size:** Authoritative representation is the **recovered 4-gon** (and AABB for merge/ROI). Merge is **left/right AABB**. Normal aspect is **width/height of ordered sides**, not rotation-invariant `max/min` except on the large-document path. Occupancy is **contour area / frame area** at working width 960, not center-weighted.

**Geometry detected ≠ document confirmed.** That is the architectural defect. Stability of a table edge is not evidence of ID/passport paper.

---

## 2. Exact false-positive trigger path

```
preview RGB (CameraManager.capture_preview_array)
    → ApplicationOrchestrator._idle_tick
        → _observe_geometry
            → DocumentDetector.detect_geometry
                → find_document_quad          # ANY accepted quad
            → GeometryStability.update        # corner motion only
        → if geom.present and ready:
            _begin_transaction(trigger="auto")
            DOCUMENT_DETECTED
    → _document_detected_tick
        → WAITING_FOR_STABILITY              # no extra evidence
    → _stability_tick
        → _observe_geometry again
        → if present and ready:
            _snapshot_accepted_geometry
            FOCUSING
    → AutofocusController.run_autofocus
        → trigger_autofocus / wait
        → if AfState Focused: Laplacian not required
    → FOCUS_VERIFICATION → CAPTURING
    → ImageValidator.validate
        → sharpness ≥ max(6.0, 0.5 × AF Laplacian)
        → JPEG size/dims/brightness
        → NO “is there printed text / page interior”
    → UPLOADING → ApiClient
```

Manual path (debug Capture & Send): IDLE → DOCUMENT_DETECTED with **`manual_capture_skip_stability`**, AF allowed with **center 70% fallback** if last geometry is empty. That is a second empty-scene upload path if the operator fires capture with no page.

---

## 3. Exact files / functions

| Stage | File | Function | Trigger condition | Evidence required | False-positive risk |
|---|---|---|---|---|---|
| Preview | `src/idcam/camera/manager.py` | `capture_preview_array` | Camera started | RGB frame | N/A |
| Idle poll | `src/idcam/app/orchestrator.py` | `_idle_tick` | Disk OK, no txn | Frame | N/A |
| Quad search | `src/idcam/vision/geometry.py` | `find_document_quad` | Canny/Sobel/paper/large path passes gates | Geometric quad, **not** “paper vs table” | **High** — grain, shadows, mats, windows |
| Presence | `src/idcam/vision/document_detector.py` | `detect_geometry` | `raw is not None` **or** `geometry_held` | Same as finder + 3-frame hold | Hold keeps a false quad |
| Stability | `src/idcam/vision/geometry_stability.py` | `GeometryStability.update` | `displacement ≤ 12` px, count ≥ **5** | Corner motion only | **High** — static background |
| Transaction | `orchestrator.py` | `_begin_transaction` | `present and ready` in IDLE | None beyond geometry | Starts capture_id |
| Detected | `orchestrator.py` | `_document_detected_tick` | Always (auto) | None | Instant hop |
| Ready | `orchestrator.py` | `_stability_tick` | `present and ready` | Same stability | AF starts |
| AF | `src/idcam/camera/autofocus.py` | `run_autofocus` | Enter FOCUSING | ROI from accepted bbox (auto) | Focuses on false ROI |
| AF HW | `manager.py` | `trigger_autofocus` | Called from AF controller | None | Scans whatever AfWindows is |
| Capture | `orchestrator.py` | `_capturing_tick` | Focus marked complete; auto needs `_accepted_geometry` | ROI still present via `_check_motion_abort` | Abort only if **quad disappears or moves** |
| Validate | `src/idcam/camera/quality.py` | `ImageValidator.validate` | JPEG + sharpness floor | File integrity + Laplacian ≥ ~6 | Table texture can pass |
| Upload | `orchestrator.py` | `_uploading_tick` | Validation succeeded | None about document content | Uploads empty-of-ID JPEG |

**Not on the auto trigger path** (despite names):

| File | Function | Role |
|---|---|---|
| `document_detector.py` | `classify` / `detect` / `is_scene_empty` | Center-ROI motion×edge vs **baseline**. Used for **removal-style** helpers and debug overlay metrics, **not** IDLE auto-start. |
| `stability_detector.py` | `StabilityDetector.update` | Center-ROI absdiff. **Not** wired in `_observe_geometry`. |

---

## 4. Detection algorithm audit

`find_document_quad` (`geometry.py`):

1. Resize so width = **960**.
2. Illumination-normalize gray; **Canny (40,120)** and **Sobel mag > 28**; `MORPH_CLOSE` 5×5 ×2; `RETR_LIST` contours.
3. **Paper mask** on unnormalized gray: threshold vs **darkest of four border medians + 28**, ellipse open/close, `RETR_EXTERNAL`, hull area, **forced minAreaRect**, mode `paper_region`.
4. If normal gates fail: **`large_document_outer`** with dedicated occupancy/aspect/edge/fold (does **not** use `MIN_RECTANGULARITY` as a hard gate).
5. Pairwise **AABB two-page merge** (`merged_two_page`).
6. Suppress inner quads inside outer booklet.
7. Rank: confidence + booklet bonus − **flush penalty if `border_distance < 2.5`**.

**Normal accept gates** (`_build_candidate`):  
`MIN_AREA_FRACTION=0.08`, `MAX_AREA_FRACTION=0.97`, `MIN_ASPECT=0.40`, `MAX_ASPECT=3.35` (**side-length w/h**, not AABB), `MIN_RECTANGULARITY=0.72`, `MIN_RIGHT_ANGLE=0.55`, `MIN_CONFIDENCE=0.48`, reject **full-frame AABB** (all four sides).

**Why empty background can become a document**

| Source | Mechanism |
|---|---|
| Table grain + Canny + CLOSE | Closed “rectangles” in wood/tile |
| Paper mask | Any region brighter than the darkest **border strip** by 28 gray levels — a lit patch of table, a placemat, a sheet of light on the desk |
| `minAreaRect` / hull | Forces a 4-gon on an irregular blob |
| Rectangularity 0.72 | A real rectangle (mat, phone, laptop, window) **should** pass — that is by design |
| Confidence 0.48 | Weighted mix; a decent rectangle at ~25–40% occupancy scores well |
| Large-document path | Low raw rectangularity allowed if recovered quad + edge support / fold |
| Merge | Two adjacent grain blobs can become one wide “booklet” |
| `geometry_held` (3 frames) | A one-frame false quad persists |
| No texture / print check | No OCR, no line density, no “page interior vs wood” classifier |
| No empty-baseline on this path | Comment in `geometry.py`: *“Motion vs an empty baseline is not used.”* |

**Can accept (yes/partial):** table edges (long high-aspect may fail MAX_ASPECT 3.35; **square mats pass**), shadows (if they form a bright/dark blob), wall/window rectangles, **not** the camera frame itself if `_is_full_frame_quad`, cables (usually fail rectangularity), lighting transitions (paper mask), arbitrary 4-gons if they look rectangular enough, large blobs (`paper_region` + occupancy lobe ~0.72), partial contours (hull / minAreaRect).

---

## 5. Center-bias audit

| Location | What it does | Class |
|---|---|---|
| `DetectionConfig.roi_center_fraction` default **0.7** (`schema.py`) | Config for center crop | C for auto geometry; A for legacy metrics |
| `DocumentDetector._get_roi` / `roi_rect` | Center **70%** for motion/edge **classify** | C vs auto trigger; **D** if someone assumed this *is* detection |
| `document_preview_roi` fallback | Center 70% if **no** geometry bbox | D for **manual** AF/capture; auto forbids this fallback in FOCUSING |
| `StabilityDetector._roi` | Center 70% absdiff | C — unused on trigger |
| `sharpness.roi_rect` | Default Laplacian 70% center | A when no document ROI; **D** if used instead of bbox |
| `AutofocusController._center_roi` / `_resolve_preview_roi` | Center fallback | D for **manual**; auto uses accepted bbox |
| `autofocus.py` `applier, 0.7` TypeError fallback | Legacy AF window | C |
| `ImageValidator` `laplacian_variance_preview_equivalent_center70` | If `preview_roi is None` | D if ROI missing; auto passes accepted bbox |
| `live_viewer.draw_overlay` | Draws **green center 70%** always | C (visualization) — operators may think detection is center-only |
| `expand_preview_roi` | Grow **document** bbox about **its** center for AfWindows, min **25% of preview** | B — not image-center |
| `geometry._rank_key` flush −0.06 if `border_distance < 2.5` | Prefers inset vs flush-to-frame | B / slight **D** for clipped pages |
| `_area_term` Gaussians at **0.35** and **0.72** occupancy | Prefers mid/large pages | B size, **not** XY center |
| `_aspect_term` peaks **1.25** and **2.55** | ID vs booklet | B |
| Merge `left.bbox[0] <= right.bbox[0]` | Horizontal AABB pairing | D for rotated spreads |
| `order_corners` via `x+y` / `x-y` | Axis-aligned TL labeling | D at large rotation |
| Paper mask **border strips** | Table reference | A; fails if document fills a whole side |
| Manual focus `center_fallback` | Debug only | C for production auto |

**Scoring does not add “distance to image center.”** Occupancy is area fraction of the **whole frame**.

**Why it *feels* center-biased:** overlay paints the search square; AF windows expand toward 25% of frame; tests and fixtures put pages in the middle; merge wants side-by-side AABBs.

---

## 6. Rotation / tilt audit

| Mechanism | Dependence |
|---|---|
| Canny/Sobel | Rotation-agnostic |
| Paper morph ellipse | Better than old horizontal close |
| `_aspect_of_quad` | Ordered **top/bottom vs left/right** — 90° swap can look like portrait ID (`MIN_ASPECT` 0.4) |
| Large path `quad_aspect` = max/min side averages | More rotation-tolerant |
| Merge | **AABB** gap, height ratio, **vertical** overlap — breaks when pages are stacked in image X or rotated ~45° |
| `order_corners` | Fails to label TL at strong rotation → aspect/right-angle noise |
| ROI / AF / validation | **Axis-aligned bbox of the quad** — rotated page gets extra table in the box (hurts Laplacian, not detection per se) |
| Fold evidence | Samples booklet midline in ordered-corner space — assumes a readable TL–TR |

**Authoritative geometry should remain the convex 4-gon** (perspective polygon), not AABB. AABB is a convenience for ROI/AF/JPEG crop. Oriented min-area rect is already a fallback, not the rank key.

**5–20°:** usually OK if outer contour exists. **30–45°:** merge + `order_corners` + AABB ROI are the weak points. Do **not** “fix” by dropping rectangularity.

---

## 7. Size sensitivity audit

| Constant | Kind | Tied to |
|---|---|---|
| `MIN/MAX_AREA_FRACTION` 0.08–0.97 | Fraction of **work image** (scaled 960-wide) | Preview area, **not** center |
| `LARGE_MIN/MAX_AREA` 0.12–0.92 | Recovered **quad** / frame | Same |
| `MIN/MAX_ASPECT` 0.40–3.35 | Quad side w/h | Not pixels |
| Rectangularity / confidence / right-angle | Shape | Absolute gates |
| Border / full-frame | Pixel proximity to image edges | Frame, not center |
| `_area_term` | Soft score vs occupancy 0.35 / 0.72 | Size preference **in score**, not a hard reject if gates pass |

**Small pages** (~8%): hard floor. **Very large / clipped:** full-frame reject; 3+ sides + high occupancy on large path. **Working width 960** makes thresholds resolution-relative for 1080p vs 720p similarly.

Robust scale-invariant behavior: keep **relative occupancy bands**, require **interior texture / print / paper-vs-table**, use **quad** not AABB for aspect, allow clip if **two** sides only.

---

## 8. Empty-scene false-positive analysis

**To `geom.present` / “document detected” in logs**

1. `find_document_quad` returns a candidate (paper/canny/large).
2. `detect_geometry` sets `present=True` (`reason=quad` or `geometry_held`).
3. IDLE logs `event=document_detected` when **also** `GeometryStability` ready.

**To WAITING_FOR_STABILITY:** automatic after DOCUMENT_DETECTED (`auto_wait_stability`). No extra check.

**To FOCUSING:** `_stability_tick` `present and ready` → `document_stable`. A **static false quad** is ready immediately after the first stability window.

**To CAPTURING:** AF success. `AfState=Focused` **skips** Laplacian veto. `_check_motion_abort` only fails if the **same class of quad** vanishes or moves > 12 px.

**To UPLOADING:** validation pass. **No document-content check.** Floor **6.0** Laplacian on whatever ROI was frozen.

**Missing guard:** “stable geometry” ≠ “real document.” A table edge can be stable forever.

**Hold:** `geometry_hold_frames=3` keeps `present=True` across brief finder misses — helps real pages, **also** sticky false positives.

---

## 9. Autofocus guard analysis

`trigger_autofocus()`: `AutofocusController` after FOCUSING, not from a single contour in isolation — but FOCUSING **is** entered because of one persistent quad.

**Can start because of:** stable background geometry **yes**; single frame **no** (5 stable frames + hop); detection confidence **only as finder gate 0.48**; state transition **yes**; manual **yes, skipping stability**; stale geometry **hold + frozen `_accepted_geometry`**; stale async **guarded** by `capture_id` + expected state.

**Minimum evidence before AF (recommended, not implemented):**  
confirmed document (see §15) **and** stability **and** ROI usable **and** not empty-baseline-like. Not FoM. Not “corners didn’t move.”

---

## 10. Capture guard analysis

`_capturing_tick` requires:

1. `focus_completed_at` set  
2. Auto: `_accepted_geometry` not None  
3. `_check_motion_abort` (quad still similar)  
4. Disk  

**Does not re-confirm** “this is paper.” Frozen false ROI is enough. Manual can theoretically capture with weaker ROI rules (`allow_search_fallback`).

---

## 11. Upload guard analysis

Upload after **IMAGE_VALIDATION success** only. Auto uses `require_document_roi=True` with frozen bbox — **maps that bbox to the JPEG**, does not prove an ID is inside.

Can upload: empty-of-document stills that are sharp-enough **table**; weak geometry that passed finder; **manual** captures; retries of a **new** transaction (stale async ignored). Cannot upload without validation except bugs outside this path.

**Automatic pipeline must not upload unconfirmed documents** — currently it **can**, because confirmation **is** the quad.

---

## 12. Phase 6 orchestration analysis

| Verdict | |
|---|---|
| A correctly orchestrating | Transaction, stale-async, removal hysteresis, one-shot ID — **yes** |
| B compensating for weak detection | DOCUMENT_DETECTED / stability **assume** finder truth |
| C AF too early | ~1 s after first stable false quad (5 frames @ 5 Hz + one hop) |
| D stale geometry | `_accepted_geometry` frozen; hold frames |
| E false positive → transaction | **Yes — this is the designed contract** |

Do **not** redesign the SM graph. Change **what may enter DOCUMENT_DETECTED**.

IDLE already allows `FOCUSING` in the transition table; production auto goes DETECTED → STABILITY → FOCUSING. Manual skips stability.

---

## 13. Root causes ranked by severity

1. **`geom.present` treated as document confirmed** (finder is geometric only).  
2. **Stability = low corner motion**, not “object is a page.”  
3. **Still validation Laplacian floor 6.0** does not reject empty desk crops.  
4. **`AfState=Focused` skips Laplacian** — hardware AF on a mat still “succeeds.”  
5. **Paper-region / large-document paths** increase recall of **any** bright rectangle.  
6. **Motion×edge detector unused on auto start** (and it was itself center-biased).  
7. **AABB merge / order_corners / overlay 70%** — false **negatives** and operator confusion, not the empty-scene upload.  
8. **Manual skip-stability + center fallback** — operator empty capture.

---

## 14. Recommended architecture

Keep Phase 6 states. Insert **one confirmation gate** before `_begin_transaction` (auto):

`GEOMETRY_CANDIDATE` (finder) → **`DOCUMENT_CONFIRMED`** (multi-feature) → existing stability → AF.

Do not start a transaction on candidate-only.

---

## 15. Recommended detection evidence model

| Class | Today | Gap |
|---|---|---|
| Geometric | Quad, angles, aspect, area, rectangularity | Keep |
| Edge | Partial (large path); Canny used for **proposal** | Require **four-side support** for confirm |
| Region | Paper brightness vs border | Need **interior vs surround** (not only border median); reject wood-like FFT/grain |
| Temporal | Corner stability, hold | Add **stable confirmation**, not only stable corners |
| Context | Full-frame reject, flush rank penalty | Add **not-empty-table** vs learned/empty baseline **on the quad interior** |

**Strong AF trigger combo (example policy, not implemented):**  
quad gates **and** interior/surround contrast **and** edge support ≥ threshold **and** (print/line energy **or** passport fold) **and** N stable confirmed frames. Lower none of the hard geometry gates to “fix” position.

---

## 16. Recommended AF trigger guard

Auto: `DOCUMENT_CONFIRMED && GeometryStability.ready && usable bbox`.  
Never AF on `geometry_held` alone.  
Do not treat `AfState=Focused` as skip of document Laplacian **unless** Laplacian also in band.  
No center-70% on auto.

---

## 17. Recommended capture guard

Re-run **confirmation** (or a cheap interior check) on a fresh preview immediately before still, not only corner displacement. Keep motion abort.

---

## 18. Recommended upload guard

Auto: reject stills whose **mapped ROI** fails confirmation (or whose Laplacian is “texture without page structure” if you add that metric). Keep file/dimension checks. Do not lower sharpness floor to pass documents; **raise semantic reject** for empty ROI.

---

## 19. Hardware experiment matrix

For each case log: `event=document_geometry` fields, `geometry_gate`, `document_detected`, `document_stable`, AF `preview_roi` / `af_state` / Laplacian, `capture_result`, `image_validation` method/threshold, whether upload started.

| ID | Scene | Expect (after a real fix) |
|---|---|---|
| A | Empty | IDLE, `present=0` |
| B | Table only | IDLE |
| C | Wall/window | IDLE unless a real rectangle is a “document” by policy |
| D | Centered ID | detect → AF → capture |
| E–I | Left/right/top/bottom/corner | same as D |
| J–L | Rotate 10/20/30° | detect; merge may fail at 30° |
| M | Keystone | detect via quad |
| N–O | Small/large | per occupancy gates |
| P | Open passport | one outer quad |
| Q | Near frame edge | detect if not full-frame |
| R | Removed | WAITING_FOR_REMOVAL / IDLE |
| S | Replace | new transaction after removal |
| T | Lighting, no page | IDLE |

Today, **A–C and T can still fire the full pipeline** if a quad locks.

---

## 20. Files that would need modification (later)

- `src/idcam/vision/geometry.py` — confirmation features; not SM  
- `src/idcam/vision/document_detector.py` — expose candidate vs confirmed  
- `src/idcam/app/orchestrator.py` — gate `_begin_transaction` / `_stability_tick`  
- `src/idcam/camera/autofocus.py` — do not skip Laplacian on Focused for auto  
- `src/idcam/camera/quality.py` — empty-ROI / confirmation (not random threshold slash)  
- Tests + debug overlay (stop implying center 70% is the detector)

---

## 21. Things that must remain unchanged

- Phase 1 camera manager / 64MP still path / JPEG encoder contract  
- Phase 4 capture transaction / temp files (except calling extra checks)  
- Phase 5 ApiClient / upload protocol  
- Phase 6 **state set and legal transitions** (no new “always capture on timeout”)  
- Global **lowering** of rectangularity/confidence/area to “see more documents”  
- Using FocusFoM as document presence  

---

## 22. Risks / tradeoffs

- Stricter confirmation → more false **negatives** (plain paper, featureless cards). Mitigate with print/edge, not looser quads.  
- Empty-baseline again → center 70% blindness; if used, compute it **inside the candidate quad**, not the image center.  
- Requiring print lines → blank certificates fail. Product decision.  
- Passport large-path ↑ recall of mats. Confirmation must apply to `paper_region` / `large_document_outer` too.

---

## Logging: what exists vs what is missing

**Exists:** `event=document_geometry` (mode, area, aspect, rectangularity, bbox, quad); `event=geometry_gate`; `event=document_detected`; `event=document_stable`; AF ROI/lens/FoM/Laplacian; `event=capture_result`; validation method/threshold; `event=state_transition`; `event=capture_rejected reason=busy`.

**Insufficient for empty-scene RCA:** no field `document_confirmed` vs `geometry_candidate`; overlay `detected` is **motion×edge**, not the trigger; no interior/surround score; no “why this quad beat empty.”

**Add (later):** `evidence_paper_contrast`, `evidence_edge_support`, `evidence_print`, `trigger=geometry_only`, ROI source `accepted_bbox|center70`.

---

## Why empty scenes are accepted (one paragraph)

A **static rectangular-looking region** in the 1080p preview can pass `find_document_quad`. The orchestrator treats that as a document, waits until its **corners do not move 12 px**, then autofocuses and captures that region. Validation asks whether that crop is **somewhat sharp and a valid JPEG**, not whether an ID is present. Hardware AF “Focused” can skip even the weak Laplacian check.

## Why center / orientation / size feel too sensitive (one paragraph)

Auto search is full-frame; **position** is not a finder weight. Failures off-center/rotated/odd-size come from **AABB merge**, **corner ordering**, **occupancy/aspect gates**, **clipped full-frame reject**, and **AF/validation using AABB** (extra table in the box). The green **70% overlay** and unused motion detector make the system *look* center-locked.

## Minimum safe fix (next implementation phase)

1. Split **candidate vs confirmed**; do not `_begin_transaction` on candidate.  
2. Confirm with **interior vs surround + edge support** (and optional print/fold); apply to all geometry modes.  
3. Auto AF: never center-70%; never skip Laplacian solely on `AfState`.  
4. Validation: reject unconfirmed / empty-desk ROI; do not mass-lower geometry thresholds.  
5. Leave Phase 6 topology, camera, upload, and AF *algorithm* rewrite for after confirmation exists.

**This audit does not claim the problem is fixed.**
