# Passport Detection Runtime Debug

**Pi evidence: CASE A. Software follow-up is in git; re-run the script on the Pi to prove the camera path.**

This environment is a Windows checkout. Hardware confirmation still requires `passport_detection_debug.py` on the Raspberry Pi after deploying this tree.

## Runtime Code Verification (Pi, 2026-08-29)

Command:

```bash
sudo -u idcam /opt/idcam/venv/bin/python \
  ~/Desktop/idCamV3/scripts/passport_detection_debug.py \
  --save-dir /tmp --frames 5
```

That interpreter **preloads Desktop `src`**, so it is **not** the installed `/opt/idcam` package. Printed `geometry_module` was `/home/moj/Desktop/idCamV3/src/idcam/vision/geometry.py` with `MAX_ASPECT=3.35` and merge helpers present. `idcam.service` was **inactive**. When debugging the **service**, import `idcam.vision.geometry` **without** putting Desktop `src` on `sys.path`.

`PrivateTmp=true`: daemon `/tmp` is not the host `/tmp`.

## Effective Configuration

Unchanged hard filters (not globally lowered):

| Name | Value |
|---|---|
| MIN_AREA_FRACTION | 0.08 |
| MAX_AREA_FRACTION | 0.97 |
| MIN_ASPECT | 0.40 |
| MAX_ASPECT | 3.35 |
| MIN_RECTANGULARITY | 0.72 |
| MIN_CONFIDENCE | 0.48 |
| MIN_RIGHT_ANGLE | 0.55 |

## Observed Candidate Pipeline (live 1920×1080 preview)

`candidate_count_returned=0`, `accepted=0`, `merge_attempts=0`, `detector_reason=no_quadrilateral`, **CASE A**.

| ID | bbox | area_ratio | aspect | rectangularity | reject |
|---|---|---|---|---|---|
| C1 | — | 0.044 | 0 | 0 | `area_too_small` |
| C2 | 58,0,1112,1078 | 0.109 | 0.52 | **0.185** | `low_rectangularity` (touches top+bottom; contour area ≪ bbox) |
| C3 | 0,0,1918,1078 | 0.188 | **3.77** | 0.189 | `aspect_out_of_range` (full-frame degenerate) |

Merge never ran (no accepted quads) → not CASE B.

Saved on the Pi: `/tmp/idcam_passport_debug.jpg`, `/tmp/idcam_passport_debug_overlay.jpg`.

## Exact Failure Stage

**CASE A** on illumination-normalized Canny/Sobel: no filled outer 4-gon. The booklet is **vertically clipped** (C2 `y=0`, `height=1078`). Normalize (`gray/blur*128`) removes paper-vs-table brightness, so only thin/broken edges remain. Lowering `MIN_RECTANGULARITY` would accept C2/C3 garbage.

A second bug: **≥3 corners on the image border** was treated as `frame_aligned`. A clipped passport has all four corners on the top and bottom edges, so a *correct* filled quad would have been rejected even if rectangularity passed.

## Root Cause

1. Edge-only search on normalized gray never yields a filled rectangular passport on Hawkeye preview (no drawn dark border).
2. `frame_aligned` used corner-on-border count instead of “quad spans **both** width and height of the frame” (C3), which also blocked a vertically clipped page (C2 geometry).

## Implemented Fix (minimum change)

In `src/idcam/vision/geometry.py`:

1. **`paper_region` candidates** from **unnormalized** working gray: left/right strips as table reference, bright blobs, morph open/close, `RETR_EXTERNAL`, hull area, `minAreaRect` allowed. Existing two-page merge still applies if the fold splits two blobs.
2. **`frame_aligned`** only if the quad’s AABB spans **left and right and top and bottom** (the image rectangle). Top+bottom clip with left/right table remaining is allowed.

Rectangularity 0.72 and confidence 0.48 are unchanged.

## Re-verify on the Pi

```bash
sudo systemctl stop idcam
sudo -u idcam /opt/idcam/venv/bin/python ~/Desktop/idCamV3/scripts/passport_detection_debug.py \
    --save-dir /tmp --frames 5
```

Expect `accepted≥1`, a bbox that is not full-frame C3, overlay `FINAL`, and `geometry_mode` often `paper_region` or `merged_two_page`. Then restart the service from the **same** tree you intend to run.

## Tests

```text
python -m pytest tests -q --tb=line
```

**389 passed, 6 skipped** (this environment, 2026-08-29). New regressions: `test_vertically_clipped_opened_passport`, `test_clipped_page_without_drawn_border`.
