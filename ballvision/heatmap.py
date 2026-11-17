"""Gaussian heatmaps and peak finding (numpy only)."""

from __future__ import annotations

import numpy as np

from ballvision.constants import HEATMAP_SIGMA, PATCH_SIZE


def gaussian_heatmap(
    height: int,
    width: int,
    cx: float,
    cy: float,
    sigma: float = HEATMAP_SIGMA,
) -> np.ndarray:
    """Unit-peak Gaussian blob at (cx, cy) in pixel coordinates. Float32 [0, 1]."""
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return np.zeros((height, width), dtype=np.float32)
    ys = np.arange(height, dtype=np.float64)[:, None]
    xs = np.arange(width, dtype=np.float64)[None, :]
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma * sigma))
    return g.astype(np.float32)


def peak_xy(heatmap: np.ndarray) -> tuple:
    """Argmax peak as (x, y) float and peak value."""
    h = np.asarray(heatmap, dtype=np.float64)
    flat = int(np.argmax(h))
    y, x = divmod(flat, h.shape[1])
    return float(x), float(y), float(h[y, x])


def crop_patch(
    image: np.ndarray,
    cx: float,
    cy: float,
    size: int = PATCH_SIZE,
) -> tuple:
    """Crop a size×size patch centred on (cx, cy); pad with edge values if needed.

    Returns (patch uint8 HxWxC, origin_x, origin_y) where origin is the top-left
    of the patch in full-image coordinates (may be negative before clamping).
    """
    half = size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    h, w = image.shape[:2]
    # Destination patch, filled by safe copy.
    if image.ndim == 2:
        patch = np.zeros((size, size), dtype=image.dtype)
    else:
        patch = np.zeros((size, size) + image.shape[2:], dtype=image.dtype)

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(w, x0 + size)
    src_y1 = min(h, y0 + size)
    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    if src_x1 > src_x0 and src_y1 > src_y0:
        patch[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    # Edge pad empty borders.
    if dst_y0 > 0 and dst_y1 > dst_y0:
        patch[:dst_y0] = patch[dst_y0:dst_y0 + 1]
    if dst_y1 < size and dst_y1 > dst_y0:
        patch[dst_y1:] = patch[dst_y1 - 1:dst_y1]
    if dst_x0 > 0 and dst_x1 > dst_x0:
        patch[:, :dst_x0] = patch[:, dst_x0:dst_x0 + 1]
    if dst_x1 < size and dst_x1 > dst_x0:
        patch[:, dst_x1:] = patch[:, dst_x1 - 1:dst_x1]
    return patch, float(x0), float(y0)


def stack_three_rgb(frames_rgb: list) -> np.ndarray:
    """Stack three HxWx3 uint8 frames into 9xHxW float32 in [0, 1]."""
    if len(frames_rgb) != 3:
        raise ValueError("expected 3 frames, got {}".format(len(frames_rgb)))
    chans = []
    for img in frames_rgb:
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("each frame must be HxWx3")
        chans.append(np.transpose(arr, (2, 0, 1)))
    return np.concatenate(chans, axis=0).astype(np.float32)
