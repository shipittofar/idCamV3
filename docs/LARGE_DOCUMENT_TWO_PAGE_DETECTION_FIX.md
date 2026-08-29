# PHASE 2 — LARGE DOCUMENT / TWO-PAGE DETECTION FIX

Software status: **COMPLETE**  
Hardware status: **PENDING**

Scope: `src/idcam/vision/geometry.py` and detector logging in `src/idcam/vision/document_detector.py`. Camera, AF, capture, validation, upload, and the Phase 6 state machine were not redesigned.

## 1. Root cause

Pipeline: preview RGB → `DocumentDetector.detect_geometry` → `find_document_quad` (`geometry.py`) → temporal hold/smooth in the detector → bbox used as AF ROI.

Opened / large passports failed in **candidate generation and scoring**, not in the state machine.

| Mechanism | Effect |
|---|---|
| `MAX_ASPECT = 2.60` | An opened ICAO passport is ~250×88 mm (**~2.84**). The outer quad was rejected as `aspect_out_of_range`. |
| `_area_term` Gaussian peaked at **35%** of the frame | A half-page (~25–35%) outscored the outer booklet (~50–80%). |
| `RETR_LIST` + Canny on a **center fold** | The seam split the edge map into two page rectangles. Each was a better “ID-like” quad than the outer hull. |
| `_approx_quad` required **exactly 4** vertices | A slightly notched outer contour (fold) never became a quad, so only the inner pages survived. |
| `MAX_AREA_FRACTION = 0.90` | Very large occupancy with a small margin could be dropped as `area_too_large`. |

Failure was **not** temporal hysteresis (`geometry_hold_frames` / `geometry_smooth_alpha`); those still run after a quad is accepted.

There are no labeled Raspberry Pi passport stills in-repo for this fix (`_diagout/preview.png` / a capture JPEG exist but are not two-page fixtures).

## 2. Changes made

### Files

- `src/idcam/vision/geometry.py` — scoring, aspect limits, two-page merge, frame-quad reject, rate-limited reject logs
- `src/idcam/vision/document_detector.py` — `geometry_mode` on `DocumentGeometryResult`; success log fields
- `tests/unit/test_large_two_page_geometry.py` — new regressions
- this document

### Algorithm (kept hard filters; did not globally lower rectangularity / confidence / right-angle)

- **Aspect:** `MAX_ASPECT` 2.60 → **3.35** so a single outer booklet contour can pass. Merge path allows up to **3.45** with extra geometric gates.
- **Area scoring:** keep the mid-size Gaussian (~0.35) **and** a second lobe around **~0.72** so large occupancy can compete with a single inner page without making a *bloated* 55% contour beat a tight 42% page.
- **Two-page merge:** if two passing quads are side-by-side (similar height, high vertical overlap, small gap / slight overlap at the fold, combined aspect 1.45–3.45, combined area ≥ 16%), emit **one** outer quad `geometry_mode=merged_two_page` from left TL + right TR/BR + left BL.
- **Inner suppression:** drop a quad contained in a **merged** (or ≥1.75× area) outer booklet so page L/R are not chosen instead of the passport.
- **minAreaRect fallback:** only for contours with area 20–85% when `approxPolyDP` is not 4-sided (fold notches). Not used for near-full-frame blobs.
- **Frame protection:** reject quads whose axis-aligned box spans **all four** image sides (`frame_aligned`) — the preview rectangle itself (Pi C3). A page that only hits top and bottom of a 16:9 frame is allowed. Ranking still penalizes `border_distance < 2.5`.
- **Paper region (Pi CASE A):** illumination-normalized Canny sees hollow/broken outlines on real preview. Filled blobs are taken from **unnormalized** gray vs left/right table strips (`geometry_mode=paper_region`). Fold close can feed the existing two-page merge. `MIN_RECTANGULARITY` / `MIN_CONFIDENCE` were not lowered.
- **False merge:** two distant rectangles fail the gap test (`MERGE_MAX_GAP_FRAC = 0.12` of the thinner page). They must not become one document.

### Logging

- Success (on present 0→1 only): `geometry_mode`, `document_area_ratio`, `document_bbox`, `document_quad`, `geometry_confidence` (existing `confidence=`).
- Rejects: `event=geometry_candidate_rejected` at **DEBUG**, **≤1 Hz**, with area/aspect/rectangularity/corners/edge score/border distance/`rejection_reason`.

### Downstream

Geometry is still one preview-space quad + bbox. AF ROI continues to use `document_preview_roi` / accepted bbox. No orchestrator changes.

## 3. Tests

```text
python -m pytest tests -q --tb=line
```

**384 passed, 6 skipped** in 39.98 s (this environment, 2026-08-29).

New module `tests/unit/test_large_two_page_geometry.py`:

| Case | Test |
|---|---|
| A Normal single-page | `test_normal_single_page` |
| B Small document | `test_small_document` |
| C Large single-page | `test_large_single_page` |
| D Very large | `test_very_large_document` + occupancy bands 25–82% |
| E Opened two-page | `test_two_page_is_one_document` |
| F Strong center seam | `test_strong_center_seam` |
| G Uneven page brightness | `test_uneven_page_brightness` |
| H Close to frame | `test_close_to_frame_boundaries` |
| Clipped 16:9 preview | `test_vertically_clipped_opened_passport`, `test_clipped_page_without_drawn_border` |
| I Large rect on table | page detected; empty wood not |
| J Two unrelated side-by-side | `test_two_unrelated_rectangles_are_not_merged` (union IoU &lt; 0.55) |
| AF ROI inside passport | `test_bbox_stays_inside_passport` |

Existing `tests/unit/test_vision.py` geometry cases still pass (empty table, hands, shadows, 40% IoU, hold/smooth).

## 4. Hardware status

**PENDING** — Pi debug script (2026-08-29) proved CASE A on live preview (`docs/PASSPORT_DETECTION_RUNTIME_DEBUG.md`). Re-run that script after deploying this paper-region / `frame_aligned` change. Empty table, hands, and two distant rectangles remain unit-tested.

## 5. Status line

**PHASE 2 — LARGE DOCUMENT / TWO-PAGE DETECTION FIX**  
Software status: **COMPLETE**  
Hardware status: **PENDING**
