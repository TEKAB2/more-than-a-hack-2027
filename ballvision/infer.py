"""Fill ATD ball gaps with TrackNet detections (deterministic cache).

Runs only where the ball is still missing after an optional broadcast fusion
step. Search windows come from a kinematic prior (bridge / nearby players),
pixels are mapped back with ``unproject``, then a speed filter rejects teleports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ballvision.constants import (  # noqa: E402
    DEFAULT_CACHE_DIRNAME,
    DEFAULT_PITCH,
    FILL_CACHE_NAME,
    FPS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAX_BALL_SPEED_M_S,
    MIN_HEATMAP_PEAK,
    MIN_PRESENCE,
    PATCH_SIZE,
    SEARCH_RADIUS_M,
    SEED,
    WEIGHTS_NAME,
)
from ballvision.frames import FrameSource, open_video  # noqa: E402
from ballvision.heatmap import crop_patch, peak_xy, stack_three_rgb  # noqa: E402
from ballvision.kinematics import filter_teleports, search_centre_metres  # noqa: E402


def _weights_fingerprint(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def default_weights_path(repo_root: str = ROOT) -> str:
    return os.path.join(repo_root, "cache", DEFAULT_CACHE_DIRNAME, WEIGHTS_NAME)


def default_fill_cache_path(game: str, repo_root: str = ROOT) -> str:
    return os.path.join(
        repo_root, "cache", DEFAULT_CACHE_DIRNAME, FILL_CACHE_NAME.format(game=game)
    )


def metres_to_pixels(
    xy_m: np.ndarray,
    matrix: np.ndarray,
    pitch: Tuple[float, float],
) -> Tuple[float, float]:
    from starter.load_homography import project
    length, width = pitch
    nx, ny = float(xy_m[0]) / length, float(xy_m[1]) / width
    px, py = project(matrix, nx, ny)
    return float(px), float(py)


def pixels_to_metres(
    px: float,
    py: float,
    matrix: np.ndarray,
    pitch: Tuple[float, float],
) -> np.ndarray:
    from starter.load_homography import unproject
    nx, ny = unproject(matrix, px, py)
    length, width = pitch
    return np.asarray([float(nx) * length, float(ny) * width], dtype=np.float64)


def detect_on_frame(
    model,
    device,
    source: FrameSource,
    frame: int,
    centre_px: Tuple[float, float],
    patch_size: int = PATCH_SIZE,
    min_peak: float = MIN_HEATMAP_PEAK,
    min_presence: float = MIN_PRESENCE,
) -> Optional[Tuple[float, float, float, float]]:
    """Return (px, py, peak, presence) in full-image coords, or None."""
    import torch

    n = source.n_frames()
    f0 = max(0, min(n - 1, frame - 1))
    f1 = max(0, min(n - 1, frame))
    f2 = max(0, min(n - 1, frame + 1))
    patches = []
    origin = None
    for f in (f0, f1, f2):
        img = source.get_rgb(f)
        patch, x0, y0 = crop_patch(img, centre_px[0], centre_px[1], patch_size)
        patches.append(patch)
        origin = (x0, y0)
    stacked = stack_three_rgb(patches)
    tensor = torch.from_numpy(stacked).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        heat, pres = model(tensor)
    heat_np = heat.squeeze().cpu().numpy()
    presence = float(torch.sigmoid(pres).item())
    lx, ly, peak = peak_xy(heat_np)
    if peak < min_peak or presence < min_presence:
        return None
    return origin[0] + lx, origin[1] + ly, peak, presence


def try_load_fill_cache(
    data: Dict[str, np.ndarray],
    cache_path: str,
    weights_fp: Optional[str] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Numpy-only load of a previously written fill cache. No torch / no pickle.

    If ``weights_fp`` is given, it must match the cache fingerprint; if omitted,
    any shape-compatible cache is accepted (submission path after ``infer --write-cache``).
    """
    if not cache_path or not os.path.isfile(cache_path):
        return None
    ball = np.asarray(data["ball"], dtype=np.float32)
    cached = np.load(cache_path)
    if "ball" not in cached.files or cached["ball"].shape != ball.shape:
        return None
    if weights_fp is not None and "weights_fp" in cached.files:
        if str(cached["weights_fp"]) != str(weights_fp):
            return None
    out = dict(data)
    out["ball"] = cached["ball"].astype(np.float32)
    mask = cached["filled_mask"] if "filled_mask" in cached.files else np.zeros(len(ball), bool)
    out["ball_vision_filled"] = mask.astype(np.bool_)
    n_missing = int((~np.isfinite(ball[:, 0])).sum())
    print("ballvision: loaded fill cache (numpy)  filled={}/{} missing".format(
        int(mask.sum()), n_missing))
    return out


def fill_missing_ball(
    data: Dict[str, np.ndarray],
    *,
    matrices: np.ndarray,
    valid: np.ndarray,
    source: FrameSource,
    weights_path: str,
    pitch: Tuple[float, float] = DEFAULT_PITCH,
    blocks=None,
    cache_path: Optional[str] = None,
    device: Optional[str] = None,
    max_frames: Optional[int] = None,
    seed: int = SEED,
) -> Dict[str, np.ndarray]:
    """Run the detector on missing frames (requires torch). Prefer ``try_load_fill_cache`` for submissions.

    Writes a deterministic fill cache keyed by weights fingerprint when ``cache_path`` is set.
    """
    import torch
    from ballvision.model import load_checkpoint

    if not weights_path or not os.path.isfile(weights_path):
        return data

    fp = _weights_fingerprint(weights_path)
    hit = try_load_fill_cache(data, cache_path or "", weights_fp=fp)
    if hit is not None:
        return hit

    ball = np.asarray(data["ball"], dtype=np.float32).copy()
    players = np.asarray(data["players"], dtype=np.float32)
    missing = ~np.isfinite(ball[:, 0])

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_checkpoint(weights_path, map_location=device)
    model.to(device)
    patch_size = int(meta.get("patch_size", PATCH_SIZE))

    filled_mask = np.zeros(len(ball), dtype=np.bool_)
    candidates = np.flatnonzero(missing & valid.astype(bool))
    if max_frames is not None:
        rng = np.random.default_rng(seed)
        if len(candidates) > max_frames:
            candidates = np.sort(rng.choice(candidates, size=max_frames, replace=False))

    n_attempt = 0
    n_detect = 0
    for t in candidates:
        prior_m = search_centre_metres(ball, players, int(t), blocks)
        try:
            cx, cy = metres_to_pixels(prior_m, matrices[t], pitch)
        except Exception:
            continue
        if not (np.isfinite(cx) and np.isfinite(cy)):
            continue
        # Keep the search window on-screen.
        cx = float(np.clip(cx, 0, IMAGE_WIDTH - 1))
        cy = float(np.clip(cy, 0, IMAGE_HEIGHT - 1))
        n_attempt += 1
        det = detect_on_frame(
            model, device, source, int(t), (cx, cy), patch_size=patch_size
        )
        if det is None:
            continue
        px, py, peak, presence = det
        try:
            xy = pixels_to_metres(px, py, matrices[t], pitch)
        except Exception:
            continue
        if not (np.isfinite(xy[0]) and np.isfinite(xy[1])):
            continue
        # Soft search-radius gate in metres.
        if np.hypot(xy[0] - prior_m[0], xy[1] - prior_m[1]) > SEARCH_RADIUS_M * 2.5:
            continue
        ball[t] = xy.astype(np.float32)
        filled_mask[t] = True
        n_detect += 1
        _ = (peak, presence)  # used only for the gate above

    ball, kept = filter_teleports(ball, filled_mask, max_speed=MAX_BALL_SPEED_M_S, fps=FPS)
    filled_mask = kept
    n_kept = int(filled_mask.sum())

    if cache_path:
        folder = os.path.dirname(cache_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        np.savez_compressed(
            cache_path,
            ball=ball,
            filled_mask=filled_mask,
            weights_fp=np.asarray(fp),
        )

    print(
        "ballvision: attempted={}  detected={}  kept={}  still_missing={}".format(
            n_attempt, n_detect, n_kept, int((~np.isfinite(ball[:, 0])).sum())
        )
    )

    out = dict(data)
    out["ball"] = ball
    out["ball_vision_filled"] = filled_mask
    return out


def apply_ballvision(
    data: Dict[str, np.ndarray],
    cfg: Optional[Dict] = None,
) -> Dict[str, np.ndarray]:
    """ATD hook: prefer numpy fill-cache; torch detector only if ``run_detector=True``.

    Submission-safe path (numpy + pandas only): set ``cache_path`` to an existing
    ``fill_*.npz`` produced by ``python -m ballvision infer --write-cache``.
    Regenerating that cache is a local GPU step and is not part of the default CSV path.
    """
    cfg = dict(cfg or {})
    repo_root = cfg.get("repo_root", ROOT)
    cache_path = cfg.get("cache_path") or default_fill_cache_path(
        cfg.get("game") or "BAR-ATH", repo_root
    )
    weights = cfg.get("weights_path") or default_weights_path(repo_root)
    weights_fp = None
    if weights and os.path.isfile(weights):
        weights_fp = _weights_fingerprint(weights)

    # 1) Numpy-only cache hit — no torch import.
    # If weights are on disk, require fingerprint match; otherwise accept any
    # shape-compatible fill_*.npz (release asset / organiser download).
    hit = try_load_fill_cache(data, cache_path, weights_fp=weights_fp)
    if hit is not None:
        return hit

    if not cfg.get("run_detector", False):
        print(
            "ballvision: no fill cache at {} — skip "
            "(run `python -m ballvision infer --write-cache` on GPU, or pass run_detector)".format(
                cache_path
            )
        )
        return data

    if not weights or not os.path.isfile(weights):
        print("ballvision: no weights at {} — skip".format(weights))
        return data
    video_path = cfg.get("video_path")
    matrices = cfg.get("matrices")
    valid = cfg.get("valid")
    if matrices is None or valid is None:
        print("ballvision: missing homography in cfg — skip")
        return data
    source = cfg.get("source")
    if source is None:
        if not video_path or not os.path.isfile(video_path):
            print("ballvision: no video / frame source — skip")
            return data
        source = open_video(
            video_path, expect_size=(IMAGE_WIDTH, IMAGE_HEIGHT)
        )
        owns_source = True
    else:
        owns_source = False
    try:
        return fill_missing_ball(
            data,
            matrices=matrices,
            valid=valid,
            source=source,
            weights_path=weights,
            pitch=tuple(cfg.get("pitch") or DEFAULT_PITCH),
            blocks=cfg.get("blocks"),
            cache_path=cache_path,
            device=cfg.get("device"),
            max_frames=cfg.get("max_frames"),
            seed=int(cfg.get("seed", SEED)),
        )
    finally:
        if owns_source:
            source.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fill ATD ball gaps with ballvision")
    parser.add_argument("--game", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--write-cache", action="store_true")
    args = parser.parse_args(argv)

    from games import GAME, GAMES, asset_paths, half_video_blocks, require, source_frames
    from possession import atd as atd_io
    from starter import load_homography

    game = args.game or GAME
    meta, raw, video = require(["atd_meta", "atd_raw", "video"], game, args.data_dir)
    n = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]
    cache = os.path.join(ROOT, "cache", "{}_atd.npz".format(game))
    data = atd_io.load(meta, raw, n, pitch, cache_path=cache)
    matrices, _, valid = load_homography.load(game=game, data_dir=args.data_dir)
    weights = args.weights or default_weights_path()
    fill_cache = default_fill_cache_path(game) if args.write_cache else None
    before = float(np.isfinite(data["ball"][:, 0]).mean())
    out = fill_missing_ball(
        data,
        matrices=matrices,
        valid=valid,
        source=open_video(video, expect_size=(IMAGE_WIDTH, IMAGE_HEIGHT)),
        weights_path=weights,
        pitch=pitch,
        blocks=half_video_blocks(game),
        cache_path=fill_cache,
        device=args.device,
        max_frames=args.max_frames,
    )
    after = float(np.isfinite(out["ball"][:, 0]).mean())
    print("ball_seen_rate  before={:.4f}  after={:.4f}".format(before, after))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
