# Capture Validation Fix

## Root Cause

Hardware AF and still capture succeeded. Upload was skipped because **image validation compared the wrong region, on the wrong colour layout, to a threshold from a different scale**.

`ImageValidator.validate()` did all of:

1. `cv2.imread` → **BGR**, then `laplacian_variance` assumed **RGB**.
2. `laplacian_variance(frame)` with **no document ROI** → default **centre 70% of the entire still**.
3. Compared that score to `quality.min_sharpness = 80`.

Autofocus measures Laplacian variance on the **document ROI of the 1920×1080 RGB preview** (live: 10.2 → 10.7). The validator measured a huge centre crop of the 64MP JPEG (including desk/letterbox). That score was **9.0**. Nine and ten are the same operator on similar content; **80 is not**.

`min=80` was never calibrated on this camera’s still JPEGs. Comments assumed “full-res Laplacian will be much higher than 1080p.” The Pi run disproved that: native 64MP Laplacian of this scene is ~10, like preview.

This was **not** fixed by changing 80 → 10. Production still validation now rebuilds the **preview pixel grid** (ScalerCrop → resize to preview size) and runs the **same Laplacian on the same document ROI**. The gate is:

```
threshold = max(6.0, 0.5 × AF preview sharpness)
```

Config `min_sharpness=80` is unchanged. It is used only on the legacy native 70% path (unit tests / callers that omit `preview_size`).

## Evidence From Hardware Run

| Stage | Result |
|---|---|
| Camera | healthy |
| Detection | detected, ROI 797×457, preview array `1080×1920×3` (H×W×C → 1920×1080) |
| AF | `hardware_trigger` 10.2 → 10.7, lens 7.98 → 7.99 |
| Focus ROI | `757,414,795,467` (preview pixels) |
| Capture | invoked (`capture=yes`) |
| Validation | `sharpness=9.0 min=80.0` → `image_invalid` |
| Upload | skipped |

AF was not the failure.

## Captured Image Dimensions

CaptureService still requires JPEG dimensions matching `capture_width` × `capture_height` (9152×6944, ±64) **before** quality validation. The live run reached `image_invalid` from blur, not from a dimension error, so the file was accepted as a production still at encode time.

This session did **not** re-read that JPEG on the Pi. Dimensions in the table below are therefore **not hardware-reverified here**.

## ROI Mapping

Preview document bbox is **not** copied onto the still.

1. Take live `ScalerCrop` (16:9 window on the 4:3 sensor).
2. Crop that window from the still JPEG.
3. Resize to preview size (1920×1080).
4. Apply the **same** preview document ROI used by AF.

`preview_roi_to_still_roi` logs the sensor/still rectangle for diagnostics. Numpy preview shape `1080x1920x3` is height×width, consistent with config 1920×1080.

## Sharpness Metric

| Path | Image | ROI | Typical focused score |
|---|---|---|---|
| AF | 1080p RGB preview | document bbox | ~10–40 (live 10.7) |
| Old validator | BGR still, native 64MP | centre 70% of full sensor | live **9.0** |
| New validator | RGB still → ScalerCrop → 1080p | document bbox | same scale as AF |

Same function: `cv2.Laplacian` variance on uint8 gray.

## Threshold Calibration

- **Do not** treat 9.0 vs 80 as “the still is 9× too soft.”
- 80 assumed a different (unmeasured) still-native scale.
- Preview-equivalent gate: floor 6.0 and 50% of the AF score (10.7 → 5.35, so floor 6). Live 9–10 **would pass**; a Gaussian-blurred document **fails**.

## Changes

| File | Change |
|---|---|
| `src/idcam/camera/quality.py` | RGB decode; preview-equivalent Laplacian + ROI; structured `event=image_validation` logs |
| `src/idcam/vision/sharpness.py` | `reconstruct_preview_view` |
| `src/idcam/camera/manager.py` | `preview_roi_to_still_roi` |
| `src/idcam/app/orchestrator.py` | Pass accepted ROI, preview size, ScalerCrop, AF sharpness into `validate` |
| Tests | Still ROI/scale/blur/upload gating |

Autofocus code was not changed.

## Validation Flow

```
JPEG exists, size > 0, PIL decode
    → dimensions vs 9152×6944 (if required)
    → ScalerCrop + preview size + document ROI
    → Laplacian on preview-equivalent document patch
    → brightness / uniform checks
    → UPLOADING → existing ApiClient
```

Invalid still → `image_invalid`, no upload.

## Tests

- Sharp still + document ROI passes with `min_sharpness=80` still in config
- Wrong dimensions fail
- Empty ROI fails
- Preview ROI applied on reconstructed preview (not raw still pixels)
- Blurred document fails
- Preview-equivalent threshold is not 80
- Centre-70% misses off-centre document; ROI path passes
- Orchestrator: validation error does not call `upload_document`

## Hardware Smoke Test

**NOT RUN** after this patch.

Do not claim:

1. captured dimensions  
2. validation ROI on device  
3. sharpness / threshold on device  
4. transition to UPLOADING  
5. ApiClient called  
6. upload success  

until a Pi run is recorded.

## Remaining Risks

- Mock `current_scaler_crop` is preview-sized; production depends on live ScalerCrop matching the still.
- Preview-equivalent resize is INTER_AREA, not a bit-exact copy of the ISP 1080p stream; scores can differ slightly from AF 10.7.
- `min_sharpness=80` remains in config for legacy native validation only; operators must not assume it gates production stills.
