# Document confirmation — empty-scene auto-capture fix

## 1. Root cause

The auto-capture path treated **a stable geometry candidate as a confirmed document**.

`IDLE` called `detect_geometry()` → `GeometryStability.update()`. When `geom.present` and `stability.ready`, it started a transaction, then autofocus, capture, validation, and upload.

A static table edge, mat-like blob, or other quadrilateral can remain geometrically stable for many frames. Stability is **corner motion**, not **document evidence**. Validation only checks JPEG sharpness/size, not “is there a document.”

## 2. Architectural change

No new Phase 6 states.

```
geometry candidate (unchanged finder)
        ↓
confirm_document(preview, candidate)   # candidate-relative, not center-70%
        ↓
temporal confirmation (same window as GeometryStability)
        ↓
existing IDLE → DOCUMENT_DETECTED → WAITING_FOR_STABILITY → FOCUSING → …
```

`geom.present` is now a **candidate**. Auto AF/capture/upload require **document_confirmed** (single-frame evidence) **and** **temporally_confirmed** (held for `geometry_stable_frames_required` frames) **and** geometric stability.

Manual Capture & Send still bypasses confirmation.

## 3. Confirmation evidence model

Implemented in `src/idcam/vision/document_confirmation.py`. All samples use the **recovered quadrilateral**, anywhere in the frame (left/right/top/bottom/corner/rotated).

| Signal | Role |
| --- | --- |
| Existing geometry gates | Candidate must already be a plausible quad (normal constants unchanged). |
| Four-side localized Sobel | Peak edge in a normal band around each side vs farther inside/outside. Need ≥3 sides (booklets ≥2 with fold or strong paper). |
| Interior vs surround | Brightness mean/std in parametric interior vs outward surround. Not frame-border paper mask. |
| Print / line / fold | Supporting; blank paper can pass on edges + paper contrast. |
| Full-frame | Rejected (`full_frame`). |

Large-document path: extra reject when a **fold-less** outer quad touches **≥3 image borders** and occupancy ≥ 0.45 (frame-swallowing false outer). Normal `MIN_*` / `MAX_ASPECT` constants are unchanged.

## 4. Why empty scenes no longer trigger AF

Empty desk / grain typically: no candidate, **or** a candidate that fails edge localization and/or paper contrast. IDLE never calls `_begin_transaction(trigger="auto")` without `temporally_confirmed`.

A **stable false rectangle** can still be `geom.present`; it will not start a transaction.

## 5. Why off-center documents still work

Confirmation never uses image-center distance or the 70% search ROI. ROI is the candidate bbox/quad. Debug overlay still draws a faint legacy search rectangle; document box is the candidate/confirmed polygon AABB.

## 6. Why rotated documents still work

Side sampling uses **edge tangents and outward normals** of the ordered quad, not axis-aligned width/height. Moderate rotation/perspective is the same code path as a centered ID.

## 7. Why open passports still work

`merged_two_page` / `large_document_outer` remain geometry modes. Confirmation treats those as booklets (2-side + fold **or** 3-side + paper). One outer quad is still one document.

## 8. Performance

Confirmation runs only when a geometry candidate is present, on a ≤640-wide gray working image (same order as the finder). No 64MP decode, no OCR, no extra asyncio thread (same as existing `detect_geometry` on the tick).

## 9. Files changed

- `src/idcam/vision/document_confirmation.py` (new)
- `src/idcam/vision/geometry.py` (rank + 3-border large-outer reject)
- `src/idcam/app/orchestrator.py` (gates, pre-AF / pre-capture recheck)
- `src/idcam/app/transaction.py` (confirmation metadata)
- `src/idcam/preview/live_viewer.py` (candidate vs confirmed)
- `tests/unit/test_document_confirmation.py` (new)
- `tests/unit/test_capture_lifecycle.py`, `tests/unit/test_state_machine_phase6.py` (preview on capture recheck)

## 10. Unit-test results

`python -m pytest tests/unit`: **433 passed** (this checkout). Software only.

## 11. Hardware-test results

**NOT RUN** on Raspberry Pi in this change. Do not treat software tests as a Pi empty-scene or off-center proof.

Required on device (production venv, `idcam` stopped):

1. Empty desk — `document_confirmed=false`, no AF/capture/upload  
2. Table only  
3. Centered ID  
4. Left / right ID  
5. Rotated ID  
6. Open passport / rotated passport  
7. Oversized document  
8. Remove document  

Look for `event=document_confirmation result=accepted|rejected` and `event=autofocus_start`.

## 12. Remaining limitations

- A physical object that **looks like** a sheet (sharp four-sided contrast, paper-like interior vs surround) can still confirm — e.g. a high-contrast placemat with a hard border. That is inherent to a non-OCR confirmer.
- Geometry can still pick a bad outer quad; confirmation then **rejects** (no AF). Some placements may need the finder to propose the real page before confirmation can accept.
- Pre-AF/pre-capture recheck is cheap confirmation on the **frozen** quad, not a full re-detect.
- Image validation is unchanged (sharpness/file), not a second semantic detector.
- Center-70% remains for **manual** AF fallback only.

## Must remain unchanged

- Phase 1 camera acquisition, Phase 3 AF **algorithm**, Phase 4 capture, Phase 5 upload protocol  
- Phase 6 state topology  
- Normal geometry constants: `MIN_RECTANGULARITY`, `MIN_CONFIDENCE`, `MIN_RIGHT_ANGLE`, `MIN_AREA_FRACTION`, `MAX_AREA_FRACTION`, `MAX_ASPECT`  
- No FocusFoM as presence, no center-distance detector  
