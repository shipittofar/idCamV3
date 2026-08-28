"""Image sharpness measurement."""

from __future__ import annotations

import cv2
import numpy as np


def roi_rect(frame_shape: tuple[int, ...], roi_fraction: float = 0.7) -> tuple[int, int, int, int]:
    """Centered ROI as (x, y, width, height) in frame pixels."""
    h, w = int(frame_shape[0]), int(frame_shape[1])
    frac = min(max(float(roi_fraction), 0.05), 1.0)
    rh, rw = max(1, int(h * frac)), max(1, int(w * frac))
    y0, x0 = (h - rh) // 2, (w - rw) // 2
    return int(x0), int(y0), int(rw), int(rh)


def clip_roi(roi: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    x, y, rw, rh = (int(v) for v in roi)
    x = max(0, min(x, max(0, w - 1)))
    y = max(0, min(y, max(0, h - 1)))
    rw = max(1, min(rw, w - x))
    rh = max(1, min(rh, h - y))
    return x, y, rw, rh


def laplacian_variance(
    frame: np.ndarray,
    roi_fraction: float = 0.7,
    roi: tuple[int, int, int, int] | None = None,
) -> float:
    """Laplacian variance on a preview-pixel ROI. Higher is sharper."""
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame

    if roi is None:
        x0, y0, rw, rh = roi_rect(gray.shape, roi_fraction)
    else:
        x0, y0, rw, rh = clip_roi(roi, gray.shape)
    patch = gray[y0 : y0 + rh, x0 : x0 + rw]
    laplacian = cv2.Laplacian(patch, cv2.CV_64F)
    return float(laplacian.var())


def accept_preview_focus(
    *,
    before: float,
    after: float,
    absolute_floor: float,
    min_ratio: float = 1.2,
    min_delta: float = 2.0,
) -> bool:
    """Accept preview AF from relative improvement; absolute floor is not a veto.

    10 → 30 succeeds even when floor is 48.
    10 → 8 fails.
    """
    if after < 0:
        return False
    if before <= 0:
        return after >= max(absolute_floor, min_delta)
    ratio = after / before
    if ratio >= min_ratio and (after - before) >= min_delta:
        return True
    if after >= absolute_floor and after >= before * 0.95:
        return True
    return False


def mean_brightness(frame: np.ndarray) -> float:
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    return float(np.mean(gray))


def reconstruct_preview_view(
    still_rgb: np.ndarray,
    scaler_crop: tuple[int, int, int, int],
    preview_size: tuple[int, int],
) -> np.ndarray:
    """Map the still JPEG through ScalerCrop and resize to the preview pixel grid.

    Production validation crops the document ROI first and does not call this
    on the full 64MP frame. Kept for tests and diagnostics.
    """
    fw, fh = int(preview_size[0]), int(preview_size[1])
    if fw < 1 or fh < 1:
        raise ValueError("invalid_preview_size")
    sx, sy, sw, sh = clip_roi(scaler_crop, still_rgb.shape)
    window = still_rgb[sy : sy + sh, sx : sx + sw]
    if window.size == 0:
        raise ValueError("invalid_scaler_crop")
    return cv2.resize(window, (fw, fh), interpolation=cv2.INTER_AREA)


def resize_patch_to_preview(
    patch_rgb: np.ndarray,
    preview_wh: tuple[int, int],
) -> np.ndarray:
    """INTER_AREA downsample a still crop onto the preview-pixel ROI size."""
    pw, ph = int(preview_wh[0]), int(preview_wh[1])
    if pw < 1 or ph < 1:
        raise ValueError("invalid_preview_size")
    if patch_rgb.size == 0:
        raise ValueError("invalid_roi")
    if patch_rgb.shape[1] == pw and patch_rgb.shape[0] == ph:
        return patch_rgb
    return cv2.resize(patch_rgb, (pw, ph), interpolation=cv2.INTER_AREA)
