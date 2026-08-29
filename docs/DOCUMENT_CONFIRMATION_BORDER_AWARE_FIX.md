# Document confirmation: border-aware edge evidence

## 1. Exact root cause of `insufficient_edge_support`

Geometry returned a strong `paper_region` quad (confidence ≈ 0.924, occupancy ≈ 0.76,
finder `edge_support` ≈ 0.61) whose **right, top, and bottom sides lie on the
preview crop**. The only in-frame physical boundary is the **left** edge.

Confirmation still required **three scored sides**. For border-coincident sides it:

- sampled outward along the normal into **pixels that do not exist**, or
- measured Sobel along a **chord of paper** that is not a document outline
  (the true edge is outside the frame).

Those sides were stored as **score = 0.0** (failed), not “cannot evaluate”.
Result: `supported_sides = 1`, `mean = 0.25` (zeros averaged in),
`reason=insufficient_edge_support` — even with `interior_surround = 1.0`.

Occlusion ≠ absence of a document. Zero ≠ missing samples.

## 2. Before / after sampling

**Before:** every side contributed a numeric score; OOB/crop sides counted as
weak edges toward the 3-side gate.

**After:** each side is `fully_observable`, `partially_observable`, or
`border_occluded` from in-frame sample fraction, outward in-frame fraction, and
fraction of the side within 8 px of the image border. Occluded sides are
**unevaluable**. Means and `supported_sides` use **observable sides only**.

Occlusion is not treated as a successful edge.

## 3. Exact code changes

- `src/idcam/vision/document_confirmation.py` — side classification; large
  border-contact path; diagnostics.
- `scripts/passport_detection_debug.py` — per-side observable / occluded dump.
- `tests/unit/test_confirmation_edge_support.py` — Pi-like 3-border quad, gates.
- `docs/DOCUMENT_CONFIRMATION_BORDER_AWARE_FIX.md` — this file.

Unchanged: geometry `MIN_*` constants, Phase 6, AF, capture, upload,
`_MIN_SIDES_NORMAL = 3`, `_MIN_SIDE_ON = 14`.

## 4. Border-aware evidence logic

Normal documents with ≥3 observable sides: same 3-side + paper/print gates.

Two observable sides: existing 2-side + paper + mean-score exception.

**Large border-contact** (does not auto-accept occupancy or `border_contacts`):

- ≥2 sides classified occluded
- ≤2 observable
- ≥1 **supported** observable side (a real in-frame boundary)
- occupancy in [0.28, 0.92] (not full-frame)
- `geom.border_sides >= 2`
- `geom.confidence >= 0.70`
- finder `edge_support >= 0.32` **or** mode in
  `{paper_region, large_document_outer, merged_two_page}`
- interior/surround paper contrast still required (`paper_norm` and `paper_delta`)

Full-frame AABB / area ≥ 0.94 still rejected first.

## 5. False-positive protections

Empty desk, grain, mat, window, table edge, full-frame, and weak 3-border
patches without a real observed edge or without paper contrast still fail.
Occupancy alone is not enough. 1-side fully in-frame IDs still need the normal
multi-side path (they are not “large border-contact”).

## 6. Unit-test results

See pytest in the session report.

## 7. Hardware validation

**NOT RUN** on the Raspberry Pi in this session.

Re-run `scripts/passport_detection_debug.py` on the 3-border document and
confirm `occluded_sides >= 2`, `observed_side_count >= 1`,
`confirmation_result=accepted`, and empty desk still `rejected`.

## 8. Remaining limitations

- A sheet-like object with one sharp in-frame edge, high paper/table contrast,
  high finder confidence, and two crop sides can still confirm (inherent to
  non-OCR confirmation).
- Extreme glare on the only in-frame edge can still fail.
- Overlay/debug JPEG is the preview array, not a second pipeline.
