# PHASE 2 — LARGE DOCUMENT / TWO-PAGE DETECTION FIX

Software status: **UNIT TESTS PASSING — NOT HARDWARE-COMPLETE**  
Hardware status: **PENDING** (do not treat this as proven on the Pi)

Scope: `src/idcam/vision/geometry.py`, debug/trace helpers, unit tests, this document. Camera, AF, capture, validation, upload, UI, and the Phase 6 state machine were not changed.

## 1. Confirmed Pi root cause

Live 1920×1080 preview (2026-08-29), Desktop `geometry.py` with `MAX_ASPECT=3.35`, merge helpers present, service inactive:

| | |
|---|---|
| CASE | **A** |
| `accepted` | 0 |
| `merge_attempts` | 0 |
| `detector_reason` | `no_quadrilateral` |

| ID | What it was | Why it died |
|---|---|---|
| C1 | tiny contour | `area_too_small` (0.044 &lt; 0.08) |
| C2 | `[58,0,1112,1078]`, rect **0.185** | `low_rectangularity` (gate **0.72**). Hollow/broken outline of a **vertically clipped** booklet. |
| C3 | `[0,0,1918,1078]`, aspect **3.77** | `aspect_out_of_range` (normal `MAX_ASPECT` **3.35**) plus a degenerate near-frame quad |

Illumination-normalized Canny/Sobel never produced a **filled** outer 4-gon. Real Hawkeye preview has no synthetic dark page border. Merge never ran because **no candidate was accepted**.

A second bug: treating “≥3 corners on the image border” as `frame_aligned` would also reject a **valid** clipped passport (all four corners on top+bottom).

## 2. Why merge was never reached

`_try_merge_two_page` only pairs **accepted** quads. CASE A had `accepted=0`, so `merge_attempts=0`. This was not a fold-gap / height-ratio miss (CASE B).

## 3. Files changed

- `src/idcam/vision/geometry.py`
- `src/idcam/vision/geometry_debug.py`
- `scripts/passport_detection_debug.py`
- `tests/unit/test_large_two_page_geometry.py`
- `docs/LARGE_DOCUMENT_TWO_PAGE_DETECTION_FIX.md`
- `docs/PASSPORT_DETECTION_RUNTIME_DEBUG.md` (Pi CASE A notes)

## 4. Geometry / reconstruction added

**Normal path unchanged:** `MIN_AREA_FRACTION=0.08`, `MAX_AREA_FRACTION=0.97`, `MIN_ASPECT=0.40`, `MAX_ASPECT=3.35`, `MIN_RECTANGULARITY=0.72`, `MIN_CONFIDENCE=0.48`, `MIN_RIGHT_ANGLE=0.55`.

**Paper region (filled blob):** unnormalized gray vs the **darkest** of four border strips; isotropic morph close so a rotated fold can still join pages. `geometry_mode=paper_region` when that blob already passes **normal** gates (typical filled page).

**Large-document path** (`_build_large_document_candidate`, `geometry_mode=large_document_outer`): used only after the normal gates reject a recovered outer quad. It:

1. Recovers a convex 4-gon (`approxPolyDP` on the hull, else `minAreaRect`).
2. Scores **that quad**, not raw contour area / min-area-rect (Pi C2 hollow rect 0.18 is not a global accept).
3. Measures aspect from **side lengths**:  
   `width_est = avg(top,bottom)`, `height_est = avg(left,right)`,  
   `quad_aspect = max/min` (range 1.12–4.20 on this path only, covering observed 3.768-class recovered geometry without changing `MAX_ASPECT`).
4. Requires edge support along the recovered sides, opposite-side consistency, interior angles, occupancy 0.12–0.92, both spans substantial (rejects thin bars / hands).
5. If raw rectangularity is still &lt; 0.72: needs **fold evidence** or **strong** edge support; without fold, `quad_aspect` must be ≥ 1.55.
6. Existing two-page merge still runs on **accepted** pages; inner quads inside an outer booklet are suppressed (`merged_two_page` / `large_document_outer` / `paper_region`).

## 5. Rotated / skewed passports

Axis-aligned bbox aspect is **not** the large-path accept test. Side-length aspect and `minAreaRect` / hull recovery do not require edges parallel to the image axes. Paper close is elliptical (not a horizontal-only kernel). Unit tests cover ~10° and ~25° rotation and a keystone warp.

## 6. Oversized occupancy

Normal `MAX_AREA_FRACTION` is unchanged. The large path allows recovered-quad occupancy up to **0.92** with evidence; **0.97+** contour blobs still die on the normal early reject. Slightly oversized fixtures (small margins) are unit-tested.

## 7. Border contact

`frame_aligned` means the recovered AABB spans **all four** image sides (the camera rectangle), not “two corners on a border”. One- or two-side clip (including top+bottom of 16:9) is allowed when the quad is still a booklet, not the frame.

## 8. Full-frame false positives

Rejected when: four-side span, or occupancy &gt; 0.92 with ≥3 sides, or a uniform full-frame fill. C3-class degenerates stay invalid.

## 9. Distant rectangles

Merge still requires height ratio, vertical overlap, and gap ≤ 12% of the thinner page. Two far cards fail `horizontal_gap`. Unit test unchanged.

## 10. Why normal ID detection is unchanged

Large-path constants are separate (`LARGE_*`). Landscape/portrait ID fixtures and `test_vision` geometry (empty table, hands, shadows, hold/smooth) still use the original hard filters on the normal path. Hands/shadows were used to tighten large-path span/aspect so they do **not** ride `large_document_outer`.

## 11. Unit-test results

```text
python -m pytest tests -q --tb=line
```

**404 passed, 6 skipped** (this environment, 2026-08-29). New coverage includes rotation, perspective, oversized, one/two-border clip, full-frame reject, random wide / rotated wide bars, fold = one document, low-raw-rect reject vs recovered-quad+fold accept, and frozen normal thresholds.

## 12. Hardware validation results

**Not performed in this environment.** Windows checkout cannot open the Arducam. After copying this `geometry.py` onto the Pi tree the debug script uses:

```bash
sudo systemctl stop idcam
sudo -u idcam /opt/idcam/venv/bin/python \
  ~/Desktop/idCamV3/scripts/passport_detection_debug.py \
  --save-dir /tmp --frames 5
```

Required live cases: A normal open, B rotated, C keystone, D oversized, E one border, F two borders if practical, G empty / random wide. Success is `accepted≥1`, `best_mode` in `{merged_two_page, large_document_outer, paper_region}`, overlay **one** outer quad, `detector_reason != no_quadrilateral`.

## 13. Remaining limitations

- Paper mask still needs a darker table than the page; similar-luminance tables can miss the blob.
- Strong 45°+ rotation can confuse `order_corners` (x+y heuristic); 10–30° is the tested band.
- Merge of two pages is still AABB-based; a split rotated pair may rely on paper close + outer hull rather than merge.
- Overlay bbox is axis-aligned; the **polyline** is the recovered quad.

## Status line

**PHASE 2 — LARGE DOCUMENT / TWO-PAGE DETECTION FIX**  
Software (unit): **PASSING**  
Hardware: **PENDING — not COMPLETE**
