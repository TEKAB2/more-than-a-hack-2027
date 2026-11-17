"""Train TrackNetUNet on ATD-projected pseudo-labels (local GPU).

Cloud agents must not train on the real video. This module is the local entry
point for Joy Boy's RTX 4070. Aggregate metrics only (loss, recall, median px error).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ballvision.constants import (  # noqa: E402
    DEFAULT_CACHE_DIRNAME,
    PATCH_SIZE,
    SEED,
    WEIGHTS_NAME,
)
from ballvision.heatmap import peak_xy  # noqa: E402
from ballvision.model import (  # noqa: E402
    TrackNetUNet,
    heatmap_presence_loss,
    load_checkpoint,
    save_checkpoint,
)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PatchDataset:
    """Loads .npz patches written by ``pseudolabels.write_patch_dataset``."""

    def __init__(self, folder: str):
        self.paths = sorted(
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if name.endswith(".npz")
        )
        if not self.paths:
            raise FileNotFoundError("no .npz patches in {}".format(folder))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        data = np.load(self.paths[i])
        return {
            "input": data["input"].astype(np.float32),
            "heatmap": data["heatmap"].astype(np.float32),
            "presence": data["presence"].astype(np.float32).reshape(-1)[0],
        }


def _collate(batch: List[Dict]) -> Dict:
    import torch
    return {
        "input": torch.from_numpy(np.stack([b["input"] for b in batch])),
        "heatmap": torch.from_numpy(np.stack([b["heatmap"] for b in batch])),
        "presence": torch.from_numpy(np.asarray([b["presence"] for b in batch],
                                                dtype=np.float32)),
    }


def _loader(dataset: PatchDataset, batch_size: int, shuffle: bool, seed: int):
    import torch
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=_collate,
        generator=gen if shuffle else None,
    )


def evaluate_aggregate(
    model,
    loader,
    device,
    peak_thresh: float = 0.35,
) -> Dict[str, float]:
    """Recall + median pixel error on positive samples only (aggregate)."""
    import torch
    model.eval()
    errors = []
    n_pos = 0
    n_hit = 0
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            heat_t = batch["heatmap"].to(device)
            pres_t = batch["presence"].to(device)
            heat_p, pres_p = model(x)
            loss = heatmap_presence_loss(heat_p, pres_p, heat_t, pres_t)
            total_loss += float(loss.item())
            n_batches += 1
            heat_np = heat_p.squeeze(1).cpu().numpy()
            tgt = heat_t.cpu().numpy()
            if tgt.ndim == 4:
                tgt = tgt[:, 0]
            for i in range(heat_np.shape[0]):
                if float(pres_t[i].item()) < 0.5:
                    continue
                n_pos += 1
                px, py, peak = peak_xy(heat_np[i])
                tx, ty, _ = peak_xy(tgt[i])
                if peak >= peak_thresh:
                    n_hit += 1
                    errors.append(float(np.hypot(px - tx, py - ty)))
    return {
        "loss": total_loss / max(n_batches, 1),
        "n_pos": float(n_pos),
        "recall": float(n_hit / n_pos) if n_pos else 0.0,
        "median_px_error": float(np.median(errors)) if errors else float("nan"),
    }


def train_loop(
    train_dir: str,
    val_dir: str,
    out_path: str,
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    base: int = 32,
    seed: int = SEED,
    amp: bool = True,
    device: Optional[str] = None,
) -> Dict[str, float]:
    import torch
    from torch import amp as torch_amp

    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    print("device={}  cuda={}".format(device, torch.cuda.is_available()))
    if device == "cuda":
        print("gpu={}".format(torch.cuda.get_device_name(0)))

    train_ds = PatchDataset(train_dir)
    val_ds = PatchDataset(val_dir)
    train_loader = _loader(train_ds, batch_size, shuffle=True, seed=seed)
    val_loader = _loader(val_ds, batch_size, shuffle=False, seed=seed)

    model = TrackNetUNet(base=base).to(device_t)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    use_amp = bool(amp and device_t.type == "cuda")
    scaler = torch_amp.GradScaler("cuda", enabled=use_amp)

    best = {"recall": -1.0, "median_px_error": float("inf"), "loss": float("inf")}
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_b = 0
        for batch in train_loader:
            x = batch["input"].to(device_t)
            heat_t = batch["heatmap"].to(device_t)
            pres_t = batch["presence"].to(device_t)
            opt.zero_grad(set_to_none=True)
            with torch_amp.autocast("cuda", enabled=use_amp):
                heat_p, pres_p = model(x)
                loss = heatmap_presence_loss(heat_p, pres_p, heat_t, pres_t)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.item())
            n_b += 1
        train_loss = running / max(n_b, 1)
        metrics = evaluate_aggregate(model, val_loader, device_t)
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(row)
        print(
            "epoch {epoch:03d}  train_loss={train_loss:.4f}  val_loss={loss:.4f}  "
            "recall={recall:.3f}  median_px={median_px_error:.2f}  n_pos={n_pos:.0f}".format(**row)
        )
        score = (metrics["recall"], -metrics["median_px_error"]
                 if np.isfinite(metrics["median_px_error"]) else -1e9)
        best_score = (best["recall"], -best["median_px_error"]
                      if np.isfinite(best["median_px_error"]) else -1e9)
        if score > best_score:
            best = dict(metrics)
            save_checkpoint(
                out_path,
                model,
                meta={
                    "seed": seed,
                    "epoch": epoch,
                    "val_recall": metrics["recall"],
                    "val_median_px_error": metrics["median_px_error"],
                    "patch_size": PATCH_SIZE,
                    "base": base,
                },
            )
            print("  saved {}".format(out_path))

    with open(out_path + ".history.json", "w") as handle:
        json.dump({"history": history, "best": best}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return best


def metres_error_from_px(median_px: float, metres_per_px: float = 0.05) -> float:
    """Rough conversion for logs; tactical H varies — local diagnose uses unproject."""
    return median_px * metres_per_px


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train ballvision TrackNetUNet on EXTERNAL or synthetic patches only. "
            "Never on BAR-ATH video / ATD-derived patches (organizer rule)."
        )
    )
    parser.add_argument("--cache-dir", default=os.path.join(ROOT, "cache", DEFAULT_CACHE_DIRNAME))
    parser.add_argument("--out", default=None, help="weights path (default: cache/ballvision/tracknet_unet.pt)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    from ballvision.policy import refuse_train_cache_from_provided
    refuse_train_cache_from_provided(args.cache_dir)

    train_dir = os.path.join(args.cache_dir, "patches", "train")
    val_dir = os.path.join(args.cache_dir, "patches", "val")
    out = args.out or os.path.join(args.cache_dir, WEIGHTS_NAME)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    best = train_loop(
        train_dir, val_dir, out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        base=args.base,
        seed=args.seed,
        amp=not args.no_amp,
        device=args.device,
    )
    print("best_val  recall={recall:.3f}  median_px={median_px_error:.2f}".format(**best))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
