# Passport Detection Runtime Debug

**Status: ROOT CAUSE NOT YET PROVEN ON THE PI — FIX NOT IMPLEMENTED**

This environment is a Windows checkout. It cannot inspect `systemd` on the Raspberry Pi, capture the live preview the service sees, or confirm `/opt/idcam/venv` matches git. No live opened-passport JPEG was available in the repo (`_diagout/preview.png` is gone).

What this change **does** provide: a diagnostic path that uses the **same** `find_document_quad` accept/reject rules, dumps **effective** constants from the imported module, saves the **exact preview array**, and labels CASE A–G from that frame.

Do not treat synthetic unit-test passports as Pi proof. Those already pass; the Pi failure is still unexplained until the script below is run on device.

## Runtime Code Verification

**This workstation:** not the idcam service.

**What the Pi service is *supposed* to run** (`deploy/idcam.service`):

| Item | Value |
|---|---|
| Unit | `idcam.service` |
| `ExecStart` | `/opt/idcam/venv/bin/idcam` |
| `Environment` | `IDCAM_CONFIG=/etc/idcam/config.yaml` |
| WorkingDirectory | `/var/lib/idcam` |
| `PrivateTmp` | **true** — the service’s `/tmp` is **not** the host `/tmp` |
| `ProtectHome` | true |

If the device was not redeployed after the large-document patch, the service can still be running **old** `geometry.py` (`MAX_ASPECT=2.60`, no `merged_two_page`). That would explain “384 passed in git” vs “passport not detected on Pi”. **Prove it on the Pi:**

```bash
systemctl show idcam -p FragmentPath -p ExecStart -p MainPID -p ActiveState
readlink -f /proc/$(systemctl show -p MainPID --value idcam)/exe
/opt/idcam/venv/bin/python -c "import idcam.vision.geometry as g; print(g.__file__); print('MAX_ASPECT', g.MAX_ASPECT); print('merged', 'merged_two_page' in open(g.__file__, encoding='utf-8').read())"
```

If `MAX_ASPECT` is not `3.35` or `merged` is `False`, the previous fix is **not** the running code. Stop there and deploy; do not retune thresholds.

## Effective Configuration

Printed by the imported module (git checkout values). **Pi must print the same from `/opt/idcam/venv`.**

Geometry constants (`src/idcam/vision/geometry.py`, not YAML):

| Name | Git value |
|---|---|
| WORKING_WIDTH | 960 |
| MIN_AREA_FRACTION | 0.08 |
| MAX_AREA_FRACTION | 0.97 |
| MIN_ASPECT | 0.40 |
| MAX_ASPECT | 3.35 |
| MIN_RECTANGULARITY | 0.72 |
| MIN_CONFIDENCE | 0.48 |
| MIN_RIGHT_ANGLE | 0.55 |
| MERGE_MIN_HEIGHT_RATIO | 0.82 |
| MERGE_MIN_VERTICAL_OVERLAP | 0.78 |
| MERGE_MAX_GAP_FRAC | 0.12 |
| MERGE_MIN_GAP_FRAC | -0.08 |
| MERGE_MIN_ASPECT | 1.45 |
| MERGE_MAX_ASPECT | 3.45 |
| MERGE_MIN_AREA | 0.16 |

DetectionConfig defaults (overridable in `/etc/idcam/config.yaml`):

| Name | Default |
|---|---|
| geometry_hold_frames | 3 |
| geometry_smooth_alpha | 0.40 |
| geometry_stability_threshold_px | 12.0 |
| geometry_stable_frames_required | 5 |
| removal_frames_required | 10 |

These geometry **filters were not changed** in this debugging task.

## Observed Candidate Pipeline

Not available from a real Pi passport frame.

Synthetic opened-passport fixtures in `tests/unit/test_large_two_page_geometry.py` **are detected** in unit tests. That only proves the algorithm can work on drawings, not on Hawkeye preview lighting/fold/texture.

## Exact Failure Stage

**Unknown on hardware (CASE not assigned).**

How the Pi script assigns a case from **one saved preview**:

| Case | Meaning |
|---|---|
| A | No accepted quad; a large/wide hull was rejected (`not_quadrilateral`, rectangularity, angles, …) |
| B | ≥2 page-like accepted quads; merge missing or only `horizontal_gap` / `height_ratio` / `vertical_overlap` |
| C | Merge geometry built then rejected (`merge_geometry_rejected` / aspect / confidence) |
| D | Finder has a final quad; `detect_geometry` `present=0` (hold/smooth) |
| E | Not classified by this detector-only tool (orchestrator would need logs) |
| F | No usable contours (area/perimeter) — preview may lack outer edges |
| G | Success, or leftover / unclassified |

## Debug Frame

**Not captured here.**

On the Pi, **stop idcam first** (camera exclusive), then:

```bash
sudo systemctl stop idcam
sudo -u idcam /opt/idcam/venv/bin/python /opt/idcam/scripts/passport_detection_debug.py \
    --save-dir /tmp --frames 5
```

Writes:

- `/tmp/idcam_passport_debug.jpg` — copy of the preview array the finder used
- `/tmp/idcam_passport_debug_overlay.jpg` — labeled C1 REJECT / ACCEPT, MERGE, FINAL

To dump from the **running** service (host `/tmp` will **not** see service writes because of `PrivateTmp`):

```bash
sudo mkdir -p /var/lib/idcam/tmp
# add to /etc/idcam/environment:
# IDCAM_GEOMETRY_DEBUG=1
# IDCAM_GEOMETRY_DEBUG_DIR=/var/lib/idcam/tmp
sudo systemctl restart idcam
# place passport, wait ≥2s, then:
ls -l /var/lib/idcam/tmp/idcam_passport_debug.jpg
journalctl -u idcam | grep document_detection_debug
```

Rate limit: **2 seconds**. Event: `event=document_detection_debug`.

## Root Cause

**Not identified on the Pi.** Fixing thresholds without the overlay + `MAX_ASPECT` printout would be speculation.

Hypotheses **to confirm**, not conclusions:

1. Service still serving pre-merge `geometry.py`.
2. CASE B: pages detected, fold gap larger than `0.12 * min(page_width)` so merge never runs.
3. CASE F: Canny/Sobel on real preview never forms page or outer contours.
4. CASE A: outer contour exists but `approxPolyDP` ≠ 4 and min-rect fallback gated off (area not in 20–85%).

## Recommended Fix

**Do not implement a detection fix yet.**

1. Run `passport_detection_debug.py` on the Pi with an opened passport under the camera.
2. Confirm venv `MAX_ASPECT` and `merged_two_page` in the **same** interpreter as `ExecStart`.
3. Attach `/tmp/idcam_passport_debug.jpg`, overlay, printed CASE, candidate JSON, merge JSON.
4. Only then apply a **minimum** change aimed at that CASE (e.g. merge gap if B and measured `horizontal_gap` is just above the limit).

Negative case (two far rectangles) remains covered by unit tests; merge records `horizontal_gap` when pairs are considered.

## Tests (this change)

```text
python -m pytest tests -q --tb=line
```

**388 passed, 6 skipped** in 40.46 s (this environment, 2026-08-29). New tests: `tests/unit/test_geometry_debug.py`.
