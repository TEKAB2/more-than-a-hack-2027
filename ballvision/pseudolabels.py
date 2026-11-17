"""ATD ball → pixel pseudo-labels via homography; train/val split; patch index.

No human labels. Positions come only from the ATD ball channel projected with
``starter.load_homography.project``. Patches are written under ``cache/ballvision/``
(gitignored). This module never prints frame-level coordinates.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ballvision.constants import (
    DEFAULT_CACHE_DIRNAME,
    DEFAULT_PITCH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    INDEX_NAME,
    PATCH_SIZE,
    SEED,
    VAL_FRACTION,
)
from ballvision.frames import FrameSource
from ballvision.heatmap import crop_patch, gaussian_heatmap, stack_three_rgb


def metres_to_norm(ball_m: np.ndarray, pitch: Tuple[float, float]) -> np.ndarray:
    """(n, 2) metres → normalised 0-1 pitch coordinates (ATD convention)."""
    length, width = pitch
    out = ball_m.astype(np.float64).copy()
    out[:, 0] /= length
    out[:, 1] /= width
    return out


def project_ball_pixels(
    ball_m: np.ndarray,
    matrices: np.ndarray,
    valid: np.ndarray,
    pitch: Tuple[float, float] = DEFAULT_PITCH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project ATD ball (metres) to pixels.

    Returns
    -------
    pixels : (n, 2) float64 — NaN where ball missing or H invalid
    usable : (n,) bool — True when a finite pixel position exists
    """
    from starter.load_homography import project

    n = len(ball_m)
    pixels = np.full((n, 2), np.nan, dtype=np.float64)
    ball_n = metres_to_norm(ball_m, pitch)
    seen = np.isfinite(ball_m[:, 0]) & np.isfinite(ball_m[:, 1])
    usable_h = valid.astype(bool) & seen
    for t in np.flatnonzero(usable_h):
        px, py = project(matrices[t], ball_n[t, 0], ball_n[t, 1])
        if np.isfinite(px) and np.isfinite(py):
            if 0.0 <= px < IMAGE_WIDTH and 0.0 <= py < IMAGE_HEIGHT:
                pixels[t, 0] = px
                pixels[t, 1] = py
    usable = np.isfinite(pixels[:, 0]) & np.isfinite(pixels[:, 1])
    return pixels, usable


def temporal_train_val_split(
    usable: np.ndarray,
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Chronological split of usable frames with a fixed-seed boundary jitter.

    Takes the last ``val_fraction`` of chronological usable indices as val,
    after a small deterministic jitter of the cut (±1 % of the span) so the
    boundary is not always on a round number — still fully reproducible.
    """
    idx = np.flatnonzero(usable)
    if len(idx) < 10:
        raise ValueError("need at least 10 usable frames for a train/val split")
    rng = np.random.default_rng(seed)
    cut = int(round(len(idx) * (1.0 - val_fraction)))
    jitter = int(rng.integers(-max(1, len(idx) // 100), max(2, len(idx) // 100 + 1)))
    cut = int(np.clip(cut + jitter, max(1, len(idx) // 5), len(idx) - max(1, len(idx) // 10)))
    train_idx = idx[:cut]
    val_idx = idx[cut:]
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def cache_root(repo_root: str, dirname: str = DEFAULT_CACHE_DIRNAME) -> str:
    path = os.path.join(repo_root, "cache", dirname)
    os.makedirs(path, exist_ok=True)
    return path


def build_index(
    ball_m: np.ndarray,
    matrices: np.ndarray,
    valid: np.ndarray,
    pitch: Tuple[float, float] = DEFAULT_PITCH,
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
) -> Dict[str, np.ndarray]:
    """Build the pseudo-label index (aggregate-safe metadata only)."""
    pixels, usable = project_ball_pixels(ball_m, matrices, valid, pitch)
    train_idx, val_idx = temporal_train_val_split(usable, val_fraction, seed)
    split = np.full(len(ball_m), -1, dtype=np.int8)  # -1 unused, 0 train, 1 val
    split[train_idx] = 0
    split[val_idx] = 1
    return {
        "pixels": pixels.astype(np.float32),
        "usable": usable.astype(np.bool_),
        "split": split,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }


def save_index(
    index: Dict[str, np.ndarray],
    out_dir: str,
    *,
    source_game: str = "unknown",
    for_training: bool = False,
) -> str:
    """Persist index + metadata. Provided-match indexes are tagged not-for-training."""
    from ballvision.policy import mark_index_meta_training_allowed

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, INDEX_NAME)
    np.savez_compressed(
        path,
        pixels=index["pixels"],
        usable=index["usable"],
        split=index["split"],
        train_idx=index["train_idx"],
        val_idx=index["val_idx"],
    )
    meta = {
        "n_frames": int(len(index["usable"])),
        "n_usable": int(index["usable"].sum()),
        "n_train": int(len(index["train_idx"])),
        "n_val": int(len(index["val_idx"])),
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "patch_size": PATCH_SIZE,
    }
    meta = mark_index_meta_training_allowed(
        meta, source_game=source_game, for_training=for_training
    )
    with open(os.path.join(out_dir, "index_meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def load_index(out_dir: str) -> Dict[str, np.ndarray]:
    path = os.path.join(out_dir, INDEX_NAME)
    data = np.load(path)
    return {key: data[key] for key in data.files}


def extract_triplet_sample(
    source: FrameSource,
    frame: int,
    centre_xy: Tuple[float, float],
    patch_size: int = PATCH_SIZE,
    positive: bool = True,
    sigma: float = 4.0,
) -> Dict[str, np.ndarray]:
    """Extract a 3-frame stacked patch + heatmap + presence label around centre_xy.

    For negatives, centre_xy is the crop centre and the heatmap is empty.
    Never logs pixel values.
    """
    n = source.n_frames()
    f0 = max(0, min(n - 1, frame - 1))
    f1 = max(0, min(n - 1, frame))
    f2 = max(0, min(n - 1, frame + 1))
    imgs = [source.get_rgb(f0), source.get_rgb(f1), source.get_rgb(f2)]
    # Crop each frame at the same centre so motion is relative inside the patch.
    patches = []
    origin = None
    for img in imgs:
        patch, x0, y0 = crop_patch(img, centre_xy[0], centre_xy[1], patch_size)
        patches.append(patch)
        origin = (x0, y0)
    stacked = stack_three_rgb(patches)
    if positive:
        lx = centre_xy[0] - origin[0]
        ly = centre_xy[1] - origin[1]
        heat = gaussian_heatmap(patch_size, patch_size, lx, ly, sigma=sigma)
        presence = np.array([1.0], dtype=np.float32)
    else:
        heat = np.zeros((patch_size, patch_size), dtype=np.float32)
        presence = np.array([0.0], dtype=np.float32)
    return {
        "input": stacked,
        "heatmap": heat,
        "presence": presence,
        "frame": np.array([frame], dtype=np.int64),
    }


def write_patch_dataset(
    source: FrameSource,
    index: Dict[str, np.ndarray],
    out_dir: str,
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    neg_ratio: float = 0.5,
    seed: int = SEED,
    patch_size: int = PATCH_SIZE,
) -> Dict[str, int]:
    """Materialise train/val .npz patches under out_dir/{train,val}/.

    Returns aggregate counts only. Negatives are random crop centres far from
    the ball (≥ 2 × patch diagonal in image space when possible).
    """
    rng = np.random.default_rng(seed)
    pixels = index["pixels"]
    counts = {"train_pos": 0, "train_neg": 0, "val_pos": 0, "val_neg": 0}

    for split_name, key, max_n in (
        ("train", "train_idx", max_train),
        ("val", "val_idx", max_val),
    ):
        frames = np.asarray(index[key], dtype=np.int64)
        if max_n is not None and len(frames) > max_n:
            frames = rng.choice(frames, size=max_n, replace=False)
            frames.sort()
        split_dir = os.path.join(out_dir, "patches", split_name)
        os.makedirs(split_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            cx, cy = float(pixels[frame, 0]), float(pixels[frame, 1])
            sample = extract_triplet_sample(
                source, int(frame), (cx, cy), patch_size=patch_size, positive=True
            )
            path = os.path.join(split_dir, "pos_{:08d}.npz".format(int(frame)))
            np.savez_compressed(path, **sample)
            counts["{}_pos".format(split_name)] += 1

            if rng.random() < neg_ratio:
                # Negative: shift far from the ball.
                angle = float(rng.uniform(0, 2 * np.pi))
                dist = float(rng.uniform(patch_size * 1.5, patch_size * 4.0))
                nx = float(np.clip(cx + dist * np.cos(angle), 0, IMAGE_WIDTH - 1))
                ny = float(np.clip(cy + dist * np.sin(angle), 0, IMAGE_HEIGHT - 1))
                neg = extract_triplet_sample(
                    source, int(frame), (nx, ny), patch_size=patch_size, positive=False
                )
                npath = os.path.join(split_dir, "neg_{:08d}.npz".format(int(frame)))
                np.savez_compressed(npath, **neg)
                counts["{}_neg".format(split_name)] += 1

            if (i + 1) % 500 == 0:
                # Aggregate progress only — no coordinates.
                print("  {} patches written: {}".format(split_name, i + 1))

    with open(os.path.join(out_dir, "patch_counts.json"), "w") as handle:
        json.dump(counts, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return counts


def summarise_index(index: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Aggregate-only summary safe to print / log."""
    usable = index["usable"]
    return {
        "n_frames": float(len(usable)),
        "n_usable": float(usable.sum()),
        "usable_rate": float(usable.mean()) if len(usable) else 0.0,
        "n_train": float(len(index["train_idx"])),
        "n_val": float(len(index["val_idx"])),
    }
