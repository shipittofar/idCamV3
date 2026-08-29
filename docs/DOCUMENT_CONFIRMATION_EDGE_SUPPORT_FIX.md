# Document confirmation: `insufficient_edge_support` and working resolution

## 1. Root cause of `insufficient_edge_support`

Rejection is `sides < 3` (normal documents) or `sides < 2` (booklet modes), unless a
2-side paper-contrast exception fires.

A side counted only if Sobel **peak in a normal band around that quad side** exceeded
absolute `_MIN_SIDE_ON = 14` **and** beat farther interior/exterior samples.

That failed on valid documents because:

1. **Confirmation ran at 640-wide** (`INTER_AREA` from 1920). Geometry used 960-wide.
   The document boundary was anti-aliased; Sobel peaks often fell below 14.
2. **OOB / border samples were clipped or treated as 0.** Clipping copied the
   image border (often the page itself) into the “outside” band, killing contrast.
   Treating missing outside as 0 could also distort the comparison.
3. **Far interior/exterior offsets were too large** on short or keystone sides,
   so “inside” samples left the page.
4. Geometry and confirmation **did not share a working scale**, so the recovered
   quad sat a few pixels off the edge that confirmation sampled.

This was **boundary contrast**, but measured on a blurry, mismatched raster.

## 2. Current preview resolution (camera)

Camera preview is unchanged: **1920×1080** (`CameraConfig.preview_width/height`).
Numpy preview arrays are that size. Debug JPEGs are that array, not a second decode.

## 3. New working resolution

| Stage | Before | After |
| --- | --- | --- |
| Camera preview | 1920×1080 | 1920×1080 (unchanged) |
| Geometry work | 960×540 | **1280×720** |
| Geometry refine (miss + small blob) | 1440-wide | 1600-wide |
| Confirmation work | 640×360 | **1280×720** (LINEAR if scale > 0.5) |

Not full-preview contour search (1920). 1280 is the smallest step that keeps
Canny/Sobel edges after a 1920 downscale.

## 4. Before/after performance

Measured on the development PC (`tests/unit/test_confirmation_edge_support.py::TestLatencySanity`,
1920×1080 synthetic ID), not on the Pi:

- Combined geometry+confirm for that test finished in **&lt; 1 s** wall time including
  fixture setup; asserts require geometry &lt; 2.5 s and confirm &lt; 0.75 s.
- Pi 5 numbers were **not** collected in this session.

Expect geometry contour work to cost more than 960-wide (≈(1280/960)² ≈ **1.8×**
pixels). Confirmation at 1280 vs 640 is ≈ **4×** Sobel pixels, still a single
preview-sized gray image when a candidate exists.

## 5. Edge-support algorithm change

Per ordered side (TL→TR→BR→BL when upright; still valid under rotation):

- `tangent` = side vector; `inward` / `outward` = ±90° normals (not scanlines).
- Sample **t ∈ [0.15, 0.85]** (drop corners).
- Band width `near = clip(0.03 × side_length, 4, 16)` px.
- Compare peak in ±near vs interior/exterior at `far`, capped by side length and
  8% of the working min-dimension (not a huge fixed band).
- **In-frame samples only** (no clip-to-border). If the exterior band is off-image
  (border contact), use the interior band only; do not score missing pixels as 0.
- Localized peak vs nearby in/out — not mean Sobel over the whole quad.

## 6. Threshold changes

| Constant | Old | New | Why |
| --- | --- | --- | --- |
| `_MIN_SIDE_ON` | 14 | **14** (unchanged) | Higher-res peaks can meet 14; not loosened. |
| `_MIN_SIDE_DELTA_OUT` / `_IN` | 6 / 3 | **unchanged** | |
| `_MIN_SIDES_NORMAL` | 3 | **3** | |
| 2-side exception | paper Δ ≥ 28 only | **also** 2 sides + paper Δ ≥ min_paper + mean score ≥ 0.28 | Weak glare/fold side; still not 1-side. |
| Interior-noisy exterior | both deltas required | **or** strong interior delta and exterior not higher than the peak | Keystone / table texture outside. |

False-positive tests (empty desk, grain, mat, window, full-frame, table edge) still
require confirmation rejected.

## 7. Why rotation is supported

Tangents and normals are taken from the recovered quad sides, not axis-aligned
rows/columns.

## 8. Why scale is supported

`near` / `far` scale with side length and are clamped. Small and large pages
use the same logic. Working width 1280 is shared, so a small ID is not
confirmed on a 640 raster while found on a 960 raster.

## 9. Why off-center / border documents are supported

All samples are in candidate coordinates. Off-frame exterior bands are omitted,
not zero-filled. One or two AABB border contacts are allowed; full-frame quads
are still rejected.

## 10. False-positive protections preserved

- Confirmation still required for auto-capture.
- 1-side rectangles still fail.
- Full-frame still rejected.
- Background fixtures in unit tests still fail confirmation.
- No OCR, no 64MP path, no Phase 3/4/5/6 redesign.

## 11. Unit test results

See the pytest line in the session report (`tests` and `tests/unit`).

## 12. Hardware validation

**NOT RUN** on Raspberry Pi.

```bash
sudo -u idcam /opt/idcam/venv/bin/python \
  ~/Desktop/idCamV3/scripts/passport_detection_debug.py \
  --save-dir /tmp --frames 10
```

Inspect `RESOLUTION_CHAIN`, per-side `edge_support`, and overlay JPEGs.

## 13. Remaining limitations

- Extreme glare on three sides can still fail confirmation.
- A physical sheet-like mat with four sharp boundaries can still confirm.
- Pi latency/CPU was not measured here.
- Camera preview stays 1920×1080; only detector working size increased.
