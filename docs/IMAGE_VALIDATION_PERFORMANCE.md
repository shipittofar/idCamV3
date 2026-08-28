# Image Validation Performance

## Measured Timing

**Raspberry Pi 5 after this patch: NOT RUN.**

The 35–40 s figure is from production on Pi 5 *before* this change. This workspace cannot execute that hardware.

Software analysis of the pre-change path (what the Pi was running):

| Stage | Work | Expected cost on Pi 5 @ 64MP |
|---|---|---|
| Capture JPEG `verify()` + `load()` | Two full decompressions | several seconds |
| Validation `verify()` | Third full decompression | several seconds |
| Validation `load()` + `np.asarray` of entire still | Fourth decode + ~190 MB RGB | several seconds |
| `reconstruct_preview_view` | `cv2.resize` **ScalerCrop (~9152×5148 RGB) → 1920×1080**, `INTER_AREA` | **dominant, tens of seconds** |
| `np.std` of full 64MP array | Extra pass over ~190 MB | seconds |
| Laplacian on ~800×450 document patch | Tiny | milliseconds |
| SHA-256 of JPEG file | ~10–30 MB | well under 1 s |

Validation is **one-shot** (`IMAGE_VALIDATION` in `ONE_SHOT_STATES`). It was **not** looping per AF retry. It **did** run synchronously on the asyncio loop (`validate()` with no `to_thread`).

## Root Cause

The delay was **not** “OpenCV Laplacian is slow.” Laplacian runs on the preview-sized document ROI.

The expensive operation was:

1. **Full-resolution JPEG decoded multiple times** (capture + `verify` + `load`).
2. **`cv2.resize` of the entire live ScalerCrop window** (~47 megapixels RGB, area-average) down to 1080p, only to crop the document afterward.

That resize is O(still pixels). On a Pi 5 it readily accounts for **most of a 35–40 s** validation.

## Why It Took 35-40 Seconds

Preview-equivalent sharpness was implemented as:

```
decode full 9152×6944
  → resize full 16:9 ScalerCrop to 1920×1080
  → Laplacian on document ROI (~800×450)
```

AF already had the document ROI. Only that region is needed for the metric. Resizing the rest of the sensor does not change the Laplacian of that ROI (aside from tiny filter-window differences at the ROI edge).

## Optimization

Unchanged: threshold, document ROI mapping via ScalerCrop, preview-equivalent Laplacian, reject blur / accept sharp, upload only after success.

Changed:

- **One** JPEG `load()` in validation (no extra `verify()` decode). Capture likewise uses a single `load()`.
- Map preview ROI → still pixels, **PIL crop**, `INTER_AREA` resize of **that crop only** to preview ROI size.
- Brightness / uniform checks on the preview-equivalent **patch**, not the full 64MP array.
- Per-stage `event=image_validation_timing` (wall + CPU).
- `asyncio.to_thread` so validation does not block the event loop.

Resize input is now ~document still size (order 4k×2k), not ~9k×5k.

JPEG decode of 64MP is still required once (baseline JPEG is not ROI-decodable cheaply in PIL). That should be seconds, not tens of seconds.

## Before/After

| | Before (Pi, reported) | After (this code) |
|---|---|---|
| Validation wall time | ~35–40 s | **Pi not measured** |
| JPEG decodes in validation | 2 (`verify`+`load`) | 1 |
| Resize source | Full ScalerCrop (~47 MP) | Document still crop only |
| `np.std` | Full still | Preview-equivalent patch |
| Event loop | Blocked for entire validate | Worker thread |

Do not treat the right-hand column as a Pi benchmark.

## Validation Correctness

- File exists, size > 0, JPEG loads, dimensions checked.
- Document ROI still mapped through ScalerCrop; Laplacian still `INTER_AREA` onto preview ROI pixels.
- Threshold still `max(6.0, 0.5 × AF preview sharpness)`.
- Genuinely blurred document still fails; sharp checkerboard tests still pass.
- Upload still only after `IMAGE_VALIDATION` succeeds.

## Tests

- Existing still-ROI / blur / scale tests.
- Resize spy: OpenCV resize source is the document crop, not the full still.
- One `validation_complete` log per call, `decode_ops=1`, `resize_ops=1`.

Full suite this session: **337 passed, 6 skipped**.

## Hardware Smoke Test

**NOT RUN.**

Do not claim these Pi numbers:

- capture duration
- JPEG decode duration
- resize duration
- ROI duration
- Laplacian duration
- total validation duration
- upload duration
- capture-to-upload duration

On the device, grep logs for `event=image_validation_timing stage=`.
