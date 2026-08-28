# Phase 4 capture audit

**HEAD at audit:** `2b2797b` (Phase 3). Working tree clean.  
**Out of scope:** upload/API, SM redesign, AF redesign, geometry thresholds, camera FPS, UI.

Phase 3 hardware AF was not verified. This audit does not claim a capture hardware PASS.

---

## Answers (required)

### 1. How capture currently starts

`FOCUS_VERIFICATION` success → `CAPTURING` (one-shot). `_capturing_tick`:

1. `_check_motion_abort()` — read-only `find_document_quad` vs frozen accepted corners.
2. Disk reserve (`capture_reserve_mb`).
3. `output_path = temp_store.create_temp_path()` (UUID `capture-<hex>.jpg`).
4. `capture_service.capture(camera, output_path)`.
5. On `CaptureError` → `_enter_recovery` (not IDLE+re-arm). That is existing hardware-failure policy (Phase 8 / SM, not Phase 4 redesign).
6. On success → `IMAGE_VALIDATION` → existing `ImageValidator` → `UPLOADING` (Phase 5).

`CAPTURING` is in `ONE_SHOT_STATES`. One READY transaction cannot capture every preview tick.

### 2–5. Stream, resolution, format

Not the 1920×1080 preview. `CameraManager.capture_still_to_file`:

- Stop preview, `configure(_still_config)`, start.
- Still config: `create_still_configuration(main={size: (capture_width, capture_height), format: RGB888}, buffer_count=1)`.
- Defaults: **9152×6944** (`CameraConfig`).
- Hold `LensPosition` (Phase 3 lock) if metadata provides it.
- `capture_metadata()` settle, then `capture_file(path, format="jpeg")` (`STILL_ENCODER_FORMAT`).
- Restore preview.

libcamera may align requested size; `stream_geometry()["still"]` is the configured still. Mock `_mock_still` writes JPEG at `config.capture_width × capture_height`.

**Explicit JPEG:** yes, `format="jpeg"`. `jpeg_quality` applies to mock/PIL only, not Picamera2 `capture_file` (no quality kwarg today). Do not retune the driver in Phase 4.

### 6–8. Paths, partials, existing final

- Unique UUID final path. Collision is astronomically rare; if the path exists, CaptureService **raises** (`already exists`) and does **not** overwrite.
- Partial: `capture-<id>.partial.jpg` (infix). Leftover partial is deleted before encode.
- Finalize: fsync + `os.replace` partial → final.
- Empty/missing partial → `CaptureError`, cleanup.

### 9–10. Encode / write failures

Timeout → cleanup partial, `CaptureError`. Other exceptions from `capture_still_to_file` → cleanup, `CaptureError`. `_verify_still_written` rejects missing/empty **partial**. `os.replace` errors would propagate as `CaptureError` via the generic except only if they occur inside the wait_for block; **finalize is outside the try** — `FileNotFoundError` from finalize is **not** wrapped today. Gap: wrap finalize failures as `CaptureError` and cleanup.

### 11–13. Validity vs success

`CaptureService` currently returns after **exists + size > 0 + rename**. It does **not** decode JPEG or check dimensions.

Orchestrator then logs `event=capture_result success=1` **before** `IMAGE_VALIDATION`.

`ImageValidator` (existing, do not duplicate) checks: exists, size vs min/max bytes, PIL `verify` + `load`, OpenCV decode, dimension min, optional expected capture WxH ± `dimension_tolerance`, sharpness, brightness, non-uniform. That is the **upload-quality** gate.

**Gap:** a truncated JPEG with SOI and a few bytes can pass CaptureService (`size > 0`) and be logged as capture success before validator runs. User requirement: do not report capture success before decode + dimension check. Zero-byte is already rejected.

### 14. Crash during capture

Partial may remain (`*.partial.jpg`). Startup `cleanup_stale` removes old capture-/partial files by mtime. Crash after `os.replace` leaves a final file that was not validated — next transaction uses a new UUID, so it is not reused as “this” capture. Full power-loss durability is Phase 8.

### 15–16. Stale partial / existing final

Stale partial: deleted then encode; if encode no-ops, missing partial → fail, no final. Existing final: reject, leave bytes unchanged. Already tested.

### 17. Minimum Phase 4 changes

1. After finalize, **structurally** validate JPEG (PIL open/verify/load, format JPEG, non-zero, dimensions vs configured still ± existing tolerance) **before** CaptureService returns. On fail: delete final, `CaptureError` with a clear reason (`validation_failure` / `invalid_dimensions`). Reuse PIL like `ImageValidator`; do **not** run sharpness inside CaptureService (that stays `IMAGE_VALIDATION`).
2. Orchestrator: log `capture_result success=1` only after `ImageValidator` (or at least after structural check + keep IMAGE_VALIDATION for quality). Capturing tick: `success` only if we treat structural check as encode-complete; user test F wants success after validation — move `success=1` to `_validation_tick`, capturing logs encode path/size/duration.
3. Capturing gate: require `focus_completed_at`; auto require `_accepted_geometry`; keep geometry abort. No AF rerun. No `detect()`/`classify()`.
4. Wrap finalize/write errors as `CaptureError`; duration_ms in logs.
5. Tests A–O. Do not change upload, SM topology, geometry, AF algorithm, `scripts/camera_capture_test.py` (Phase 1 diagnostic).

---

## Chosen collision policy

UUID unique path. If it exists → **fail** (`CaptureError`), never overwrite, never treat as this capture.

---

## CaptureError mapping (existing type, no new hierarchy)

| Situation | Existing / Phase 4 |
|---|---|
| Encoder/timeout | `CaptureError` (encode/timeout) |
| Disk/replace | `CaptureError` (write) |
| Decode/truncated | `CaptureError` (validation) |
| Wrong WxH | `CaptureError` (invalid dimensions) |
| Path exists | `CaptureError` (path) |
| Geometry abort | existing `stability_timeout` fail, no final |
| Focus not verified | `focus_failed`, no capture |

`ImageValidationError` remains the IMAGE_VALIDATION quality gate (sharpness/brightness). Phase 4 does not skip that state.

---

## Already correct (preserve)

- Still vs preview split; explicit `format="jpeg"`; `.partial.jpg` infix; UUID finals; reject existing final; delete stale partial; one-shot CAPTURING; geometry abort; lens hold on still; `_verify_still_written`.

---

## Gaps

| Requirement | Status |
|---|---|
| Decode before capture success log | Missing |
| Dimension check in CaptureService | Missing (only later ImageValidator) |
| Focus-verified guard on capturing tick | Implicit via SM only |
| Auto accepted-geometry guard | Missing on capturing tick |
| finalize errors as CaptureError | Weak |
| duration_ms | Missing |
| `jpeg_quality` on Picamera2 | Not applied (document; don’t fake) |

---

## Config (do not duplicate)

`capture_width/height`, `jpeg_quality`, `capture_timeout_sec`, `quality.min_file_bytes`, `min_width/height`, `require_expected_dimensions`, `dimension_tolerance`. No new knobs.

---

## Phase 5+ leave untouched

Upload, checksum, API, UPLOADING transitions beyond existing, SM redesign, UI, 24/7 fsync-directory durability.

---

## Pi validation (after unit tests)

Not claimed from pytest. Place page → READY → AF → still JPEG at production WxH, decode, no leftover partial; second document new UUID; off-center; move-during-capture abort; focus fail → no auto capture. Do not upload.
