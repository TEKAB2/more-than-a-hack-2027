"""CLI: python -m ballvision {build-index|extract|train|infer|smoke}"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def cmd_build_index(args):
    import numpy as np
    from games import GAME, GAMES, require, source_frames
    from possession import atd as atd_io
    from starter import load_homography
    from ballvision.pseudolabels import (
        build_index,
        cache_root,
        save_index,
        summarise_index,
    )

    game = args.game or GAME
    meta, raw = require(["atd_meta", "atd_raw"], game, args.data_dir)[:2]
    n = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]
    cache = os.path.join(ROOT, "cache", "{}_atd.npz".format(game))
    data = atd_io.load(meta, raw, n, pitch, cache_path=cache)
    matrices, _, valid = load_homography.load(game=game, data_dir=args.data_dir)
    index = build_index(data["ball"], matrices, valid, pitch=pitch, seed=args.seed)
    out_dir = args.cache_dir or cache_root(ROOT)
    save_index(index, out_dir)
    summary = summarise_index(index)
    print("index -> {}".format(out_dir))
    print("  usable={n_usable:.0f}/{n_frames:.0f} ({usable_rate:.1%})  "
          "train={n_train:.0f}  val={n_val:.0f}".format(**summary))


def cmd_extract(args):
    from games import GAME, require
    from ballvision.frames import open_video
    from ballvision.pseudolabels import cache_root, load_index, write_patch_dataset
    from ballvision.constants import IMAGE_HEIGHT, IMAGE_WIDTH

    game = args.game or GAME
    video = require(["video"], game, args.data_dir)[0]
    out_dir = args.cache_dir or cache_root(ROOT)
    index = load_index(out_dir)
    source = open_video(video, expect_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    try:
        counts = write_patch_dataset(
            source,
            index,
            out_dir,
            max_train=args.max_train,
            max_val=args.max_val,
            neg_ratio=args.neg_ratio,
            seed=args.seed,
        )
    finally:
        source.close()
    print("patches:", counts)


def cmd_train(args):
    from ballvision.train import main as train_main
    argv = []
    if args.cache_dir:
        argv += ["--cache-dir", args.cache_dir]
    if args.out:
        argv += ["--out", args.out]
    argv += ["--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
             "--seed", str(args.seed)]
    if args.no_amp:
        argv.append("--no-amp")
    if args.device:
        argv += ["--device", args.device]
    return train_main(argv)


def cmd_infer(args):
    from ballvision.infer import main as infer_main
    argv = []
    if args.game:
        argv += ["--game", args.game]
    if args.data_dir:
        argv += ["--data-dir", args.data_dir]
    if args.weights:
        argv += ["--weights", args.weights]
    if args.device:
        argv += ["--device", args.device]
    if args.max_frames is not None:
        argv += ["--max-frames", str(args.max_frames)]
    if args.write_cache:
        argv.append("--write-cache")
    return infer_main(argv)


def cmd_smoke(args):
    """Synthetic end-to-end without real video (aggregate prints only)."""
    import numpy as np
    import torch
    from ballvision.constants import IMAGE_HEIGHT, IMAGE_WIDTH, PATCH_SIZE, SEED
    from ballvision.frames import SyntheticFrameSource
    from ballvision.model import TrackNetUNet, heatmap_presence_loss, save_checkpoint
    from ballvision.pseudolabels import (
        build_index,
        extract_triplet_sample,
        project_ball_pixels,
        temporal_train_val_split,
        write_patch_dataset,
    )
    from ballvision.train import PatchDataset, evaluate_aggregate, set_seed, _loader
    from ballvision.infer import fill_missing_ball
    from starter.load_homography import project

    set_seed(SEED)
    n = 120
    pitch = (105.0, 68.0)
    # Fake identity-ish homography: norm -> pixels roughly linear.
    H = np.array([[IMAGE_WIDTH, 0, 0], [0, IMAGE_HEIGHT, 0], [0, 0, 1]], dtype=np.float64)
    matrices = np.broadcast_to(H, (n, 3, 3)).copy()
    valid = np.ones(n, dtype=bool)
    # Ball path in metres.
    t = np.arange(n, dtype=np.float64)
    ball = np.column_stack([
        20 + 40 * (t / n),
        34 + 10 * np.sin(2 * np.pi * t / 40),
    ]).astype(np.float32)
    # Punch holes for "missing" frames.
    missing = (t % 17 == 0) & (t > 10)
    ball_obs = ball.copy()
    ball_obs[missing] = np.nan

    pixels, usable = project_ball_pixels(ball_obs, matrices, valid, pitch)
    train_idx, val_idx = temporal_train_val_split(usable, seed=SEED)
    print("smoke usable={} train={} val={}".format(
        int(usable.sum()), len(train_idx), len(val_idx)))

    # Pixel positions for the synthetic renderer (full trajectory).
    pix_full = np.full((n, 2), np.nan)
    for i in range(n):
        nx, ny = ball[i, 0] / pitch[0], ball[i, 1] / pitch[1]
        pix_full[i, 0], pix_full[i, 1] = project(H, nx, ny)

    source = SyntheticFrameSource(n, pix_full, seed=SEED)
    index = {
        "pixels": pixels.astype(np.float32),
        "usable": usable,
        "split": np.full(n, -1, dtype=np.int8),
        "train_idx": train_idx,
        "val_idx": val_idx,
    }
    index["split"][train_idx] = 0
    index["split"][val_idx] = 1

    out_dir = args.cache_dir or os.path.join(ROOT, "cache", "ballvision_smoke")
    os.makedirs(out_dir, exist_ok=True)
    counts = write_patch_dataset(
        source, index, out_dir, max_train=40, max_val=12, neg_ratio=0.4, seed=SEED
    )
    print("smoke patches", counts)

    device = "cpu"
    model = TrackNetUNet(base=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_ds = PatchDataset(os.path.join(out_dir, "patches", "train"))
    val_ds = PatchDataset(os.path.join(out_dir, "patches", "val"))
    train_loader = _loader(train_ds, batch_size=4, shuffle=True, seed=SEED)
    val_loader = _loader(val_ds, batch_size=4, shuffle=False, seed=SEED)
    for epoch in range(1, 4):
        model.train()
        for batch in train_loader:
            x = batch["input"].to(device)
            heat_t = batch["heatmap"].to(device)
            pres_t = batch["presence"].to(device)
            opt.zero_grad()
            heat_p, pres_p = model(x)
            loss = heatmap_presence_loss(heat_p, pres_p, heat_t, pres_t)
            loss.backward()
            opt.step()
        metrics = evaluate_aggregate(model, val_loader, device)
        print("smoke epoch {}  recall={recall:.3f}  median_px={median_px_error:.2f}".format(
            epoch, **metrics))

    weights = os.path.join(out_dir, "smoke_weights.pt")
    save_checkpoint(weights, model, meta={"patch_size": PATCH_SIZE, "base": 16, "seed": SEED})

    players = np.full((n, 22, 3), np.nan, dtype=np.float32)
    players[:, :, 0] = 52.0
    players[:, :, 1] = 34.0
    data = {"players": players, "ball": ball_obs.copy(), "ball_extra": np.zeros((n, 0), np.float32),
            "team": np.zeros(22, dtype=np.int8)}
    before = float(np.isfinite(data["ball"][:, 0]).mean())
    filled = fill_missing_ball(
        data,
        matrices=matrices,
        valid=valid,
        source=source,
        weights_path=weights,
        pitch=pitch,
        cache_path=os.path.join(out_dir, "smoke_fill.npz"),
        device="cpu",
    )
    after = float(np.isfinite(filled["ball"][:, 0]).mean())
    print("smoke ball_seen  before={:.3f}  after={:.3f}".format(before, after))
    # Determinism: second fill from cache must match.
    filled2 = fill_missing_ball(
        data,
        matrices=matrices,
        valid=valid,
        source=source,
        weights_path=weights,
        pitch=pitch,
        cache_path=os.path.join(out_dir, "smoke_fill.npz"),
        device="cpu",
    )
    assert np.array_equal(
        np.nan_to_num(filled["ball"]), np.nan_to_num(filled2["ball"])
    )
    print("smoke OK (deterministic fill cache)")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m ballvision")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-index", help="ATD→pixel index + train/val split")
    p.add_argument("--game", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("extract", help="Write patch .npz under cache/ballvision/")
    p.add_argument("--game", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-val", type=int, default=None)
    p.add_argument("--neg-ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("train", help="Train TrackNetUNet (local GPU)")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default=None)

    p = sub.add_parser("infer", help="Fill missing ATD ball from video")
    p.add_argument("--game", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--weights", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--write-cache", action="store_true")

    p = sub.add_parser("smoke", help="Synthetic unit smoke (no real mp4)")
    p.add_argument("--cache-dir", default=None)

    args = parser.parse_args(argv)
    if args.cmd == "build-index":
        cmd_build_index(args)
        return 0
    if args.cmd == "extract":
        cmd_extract(args)
        return 0
    if args.cmd == "train":
        return cmd_train(args) or 0
    if args.cmd == "infer":
        return cmd_infer(args) or 0
    if args.cmd == "smoke":
        return cmd_smoke(args)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
