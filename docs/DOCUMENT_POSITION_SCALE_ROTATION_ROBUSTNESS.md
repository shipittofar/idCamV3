# Document detection: position, scale, and rotation robustness

## 1. Current limitations (before this change)

The Phase 2 finder already used a perspective quad, but several gates and
rankers still behaved as if the document were a centered, axis-aligned page
of a preferred size:

- Corner pairing used independent `x+y` / `x-y` extrema (fails near 45°).
- Aspect for scoring used ordered width/height, which flips under rotation.
- Confidence occupancy term peaked at ~35% and ~72% of the frame.
- `minAreaRect` fallback was only allowed for occupancy 20–85%.
- Contours below `MIN_AREA_FRACTION` (8%) were dropped with no extra-evidence path.
- Two-page merge required horizontal AABB adjacency.
- Ranking penalized any quad near the frame border.
- Inner pages were suppressed by a larger `large_document_outer` hypothesis,
  so a table-sized outer could beat a real ID.
- Confirmation edge/surround bands were scaled to the **frame**, not the candidate.

False-positive confirmation and Phase 6 topology were already in place and
are preserved.

## 2. Root causes

| Symptom | Cause |
| --- | --- |
| Off-center / edge / corner miss or wrong box | Border-distance ranking; large outer swallowing the page |
| Small page miss | Occupancy-peaked confidence; no sub-8% evidence path; min-rect band |
| Large page miss | Axis-fraction span gates on the large-document path |
| Rotation / 45° | `x+y` pairing; AABB aspect; AABB-only booklet merge |
| Keystone | Pairing + AABB aspect, not opposite-side geometry |
| Open passport rotated | Horizontal gap / vertical overlap on AABBs only |

## 3. Exact code changes

- `src/idcam/vision/geometry.py`
  - Angular + winding corner order; start vertex is still `argmin(x+y)` **after** cyclic order.
  - Oriented aspect = max(opposite-side means) / min(...). `MIN_ASPECT` / `MAX_ASPECT` / `MIN_RECTANGULARITY` / `MIN_CONFIDENCE` / `MIN_RIGHT_ANGLE` / `MIN_AREA_FRACTION` **unchanged**.
  - Occupancy-neutral `_area_term`; aspect term no longer prefers 1.25 vs 2.55.
  - Small-document path: occupancy in `[SMALL_EVIDENCE_MIN_AREA, MIN_AREA_FRACTION)` only with relative side/diagonal + edge support.
  - `minAreaRect` allowed from the small-evidence floor to 90% occupancy.
  - Large-document axis span gates relaxed; fold-less **≥3 frame borders** still rejected as `frame_aligned`.
  - Oriented two-page merge (centroid projection) after AABB merge.
  - Ranking: do not prefer center/occupancy/border inset; demote non-booklet large outers; do not suppress a real `quad` inside a large outer.
  - Optional higher-res pass (1440 px) only when the 960 px pass is empty **and** a small paper blob hint exists.
- `src/idcam/vision/document_confirmation.py` — edge / interior / surround bands proportional to **side length**.
- `src/idcam/vision/document_detector.py` — logs rotation and border contacts; still full-frame geometry (no center crop).
- `src/idcam/vision/geometry_debug.py` + `scripts/passport_detection_debug.py` — oriented aspect, occupancy, rotation, borders, confirmation block.
- `tests/unit/test_geometry_invariance.py` — position / size / rotation / perspective / border / negatives matrix.
- Orchestrator auto-trigger: **not redesigned** (`present` + confirmation + temporal + stability).

## 4. Why this is invariant

Measurements use the recovered 4-point polygon: side lengths, opposite-side
aspect, centroid-relative merge, and candidate-normal sampling. Image center,
AABB width/height, and preferred occupancy are not gates for a real page.

## 5. False-positive protections preserved

- Confirmation still required for auto-capture (not geometry-only).
- Frozen normal gates (asserted in tests).
- Full-frame quads rejected.
- Fold-less large outer touching ≥3 borders rejected.
- Distant rectangles are not merged into one document.
- Empty desk / mat / grain / window tests still require `document_confirmation` rejected.

## 6. Test matrix and results

`python -m pytest tests -q --tb=line` → **488 passed, 6 skipped**.

## 7. Performance

Default path is still a 960-wide preview. A second 1440-wide pass runs only on
a miss with a small paper-blob hint (not on empty desks with no paper mask blob).
Confirmation stays ≤640-wide gray. No 64MP decode, OCR, or extra ML.

## 8. Hardware validation

**NOT RUN** on Raspberry Pi.

On device (service stopped):

```bash
sudo -u idcam /opt/idcam/venv/bin/python \
  ~/Desktop/idCamV3/scripts/passport_detection_debug.py \
  --save-dir /tmp --frames 10
```

Confirm `document_confirmation.result=accepted` on real documents anywhere in
frame, and `rejected` with no `autofocus_start` on empty/background scenes.

## 9. Remaining limitations

- Detection is still contour/edge/paper-mask geometry, not a semantic detector.
  A high-contrast rectangular placemat can still confirm.
- Extremely tiny cards (below ~3% occupancy and weak edges) are treated as noise.
- A document indistinguishable from the camera frame is rejected on purpose.
- Preview-only; still capture is not used for detection.
- 45° recovery depends on visible sides after rotation (page must remain in frame).

This is **not** unlimited detection. The goal is every **realistically visible**
document independent of position, scale, orientation, and moderate perspective,
while keeping confirmation evidence and rejecting background geometry.
