"""Synthetic unit tests for ballvision/ — no real mp4, no gated data.

    python -m pytest tests/test_ballvision.py -q
    # or without pytest:
    python tests/test_ballvision.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ballvision.constants import IMAGE_HEIGHT, IMAGE_WIDTH, PATCH_SIZE, SEED  # noqa: E402
from ballvision.frames import SyntheticFrameSource  # noqa: E402
from ballvision.heatmap import crop_patch, gaussian_heatmap, peak_xy, stack_three_rgb  # noqa: E402
from ballvision.kinematics import filter_teleports, search_centre_metres  # noqa: E402
from ballvision.pseudolabels import (  # noqa: E402
    build_index,
    extract_triplet_sample,
    metres_to_norm,
    project_ball_pixels,
    summarise_index,
    temporal_train_val_split,
)


def _identity_homography(n):
    H = np.array(
        [[IMAGE_WIDTH, 0, 0], [0, IMAGE_HEIGHT, 0], [0, 0, 1]], dtype=np.float64
    )
    return np.broadcast_to(H, (n, 3, 3)).copy(), np.ones(n, dtype=bool)


def test_gaussian_peak_at_centre():
    heat = gaussian_heatmap(64, 64, 32, 20, sigma=3.0)
    x, y, v = peak_xy(heat)
    assert abs(x - 32) <= 1 and abs(y - 20) <= 1
    assert v > 0.99


def test_crop_patch_shape():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[40, 80] = (255, 0, 0)
    patch, x0, y0 = crop_patch(img, 80, 40, size=32)
    assert patch.shape == (32, 32, 3)
    # Ball pixel should land near patch centre.
    assert patch[16, 16, 0] == 255


def test_stack_three_rgb():
    a = np.zeros((16, 16, 3), dtype=np.uint8)
    stacked = stack_three_rgb([a, a, a])
    assert stacked.shape == (9, 16, 16)
    assert stacked.dtype == np.float32


def test_project_and_split_deterministic():
    n = 80
    pitch = (105.0, 68.0)
    matrices, valid = _identity_homography(n)
    ball = np.column_stack([
        np.linspace(10, 90, n),
        np.full(n, 34.0),
    ]).astype(np.float32)
    ball[::11, :] = np.nan
    pixels, usable = project_ball_pixels(ball, matrices, valid, pitch)
    assert usable.sum() > 20
    # Projected x roughly = norm_x * IMAGE_WIDTH
    for t in np.flatnonzero(usable)[:5]:
        nx = ball[t, 0] / pitch[0]
        assert abs(pixels[t, 0] - nx * IMAGE_WIDTH) < 1e-5

    a1, b1 = temporal_train_val_split(usable, seed=SEED)
    a2, b2 = temporal_train_val_split(usable, seed=SEED)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    assert len(a1) + len(b1) == int(usable.sum())


def test_build_index_summary():
    n = 100
    pitch = (105.0, 68.0)
    matrices, valid = _identity_homography(n)
    ball = np.column_stack([np.linspace(5, 100, n), np.full(n, 30.0)]).astype(np.float32)
    index = build_index(ball, matrices, valid, pitch=pitch, seed=SEED)
    summary = summarise_index(index)
    assert summary["n_usable"] == float(n)
    assert summary["n_train"] + summary["n_val"] == float(n)


def test_synthetic_triplet_and_model_forward():
    import torch
    from ballvision.model import TrackNetUNet, heatmap_presence_loss

    n = 30
    pos = np.column_stack([
        np.linspace(100, 500, n),
        np.full(n, 200.0),
    ])
    source = SyntheticFrameSource(n, pos, height=240, width=320, seed=0)
    sample = extract_triplet_sample(source, 10, (float(pos[10, 0]), float(pos[10, 1])),
                                    patch_size=64)
    assert sample["input"].shape == (9, 64, 64)
    assert sample["presence"][0] == 1.0
    x, y, _ = peak_xy(sample["heatmap"])
    assert abs(x - 32) <= 2 and abs(y - 32) <= 2

    model = TrackNetUNet(base=8)
    x_t = torch.from_numpy(sample["input"]).unsqueeze(0)
    heat, pres = model(x_t)
    assert heat.shape == (1, 1, 64, 64)
    assert pres.shape == (1, 1)
    loss = heatmap_presence_loss(
        heat, pres,
        torch.from_numpy(sample["heatmap"]).unsqueeze(0),
        torch.from_numpy(sample["presence"]),
    )
    assert torch.isfinite(loss)


def test_filter_teleports():
    ball = np.full((10, 2), np.nan, dtype=np.float32)
    ball[0] = (10.0, 10.0)
    ball[1] = (10.5, 10.0)  # filled, ok
    ball[2] = (80.0, 10.0)  # filled, teleport
    filled = np.zeros(10, dtype=bool)
    filled[1] = True
    filled[2] = True
    out, kept = filter_teleports(ball, filled, max_speed=40.0, fps=25.0)
    assert kept[1]
    assert not kept[2]
    assert not np.isfinite(out[2, 0])


def test_search_centre_bridge():
    ball = np.full((5, 2), np.nan, dtype=np.float32)
    ball[0] = (0.0, 0.0)
    ball[4] = (40.0, 0.0)
    players = np.zeros((5, 4, 3), dtype=np.float32)
    mid = search_centre_metres(ball, players, 2, blocks=None)
    assert abs(mid[0] - 20.0) < 1e-5


def test_atd_hook_noop_without_weights():
    from possession.atd import apply_ballvision_fill

    data = {
        "ball": np.zeros((5, 2), dtype=np.float32),
        "players": np.zeros((5, 2, 3), dtype=np.float32),
        "ball_extra": np.zeros((5, 0), dtype=np.float32),
        "team": np.zeros(2, dtype=np.int8),
    }
    out = apply_ballvision_fill(data, {
        "weights_path": "/nonexistent/tracknet.pt",
        "cache_path": "/nonexistent/fill.npz",
        "run_detector": False,
    })
    assert out is data or np.array_equal(out["ball"], data["ball"])


def test_fill_deterministic_cache():
    import torch
    from ballvision.model import TrackNetUNet, save_checkpoint
    from ballvision.infer import fill_missing_ball
    from starter.load_homography import project

    n = 40
    pitch = (105.0, 68.0)
    matrices, valid = _identity_homography(n)
    t = np.arange(n, dtype=np.float64)
    ball_true = np.column_stack([20 + 0.5 * t, np.full(n, 34.0)]).astype(np.float32)
    ball = ball_true.copy()
    ball[5:8] = np.nan
    pix = np.full((n, 2), np.nan)
    for i in range(n):
        nx, ny = ball_true[i, 0] / pitch[0], ball_true[i, 1] / pitch[1]
        pix[i, 0], pix[i, 1] = project(matrices[i], nx, ny)
    source = SyntheticFrameSource(n, pix, seed=0)
    players = np.zeros((n, 8, 3), dtype=np.float32)
    players[:, :, 0] = 50
    players[:, :, 1] = 34

    with tempfile.TemporaryDirectory() as tmp:
        weights = os.path.join(tmp, "w.pt")
        model = TrackNetUNet(base=8)
        # Tiny train on one positive sample so the head is not random noise only.
        sample = extract_triplet_sample(
            source, 10, (float(pix[10, 0]), float(pix[10, 1])), patch_size=PATCH_SIZE
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        x = torch.from_numpy(sample["input"]).unsqueeze(0)
        ht = torch.from_numpy(sample["heatmap"])
        pt = torch.from_numpy(sample["presence"])
        from ballvision.model import heatmap_presence_loss
        for _ in range(5):
            opt.zero_grad()
            heat, pres = model(x)
            heatmap_presence_loss(heat, pres, ht, pt).backward()
            opt.step()
        save_checkpoint(weights, model, meta={"patch_size": PATCH_SIZE, "base": 8})

        data = {
            "ball": ball.copy(),
            "players": players,
            "ball_extra": np.zeros((n, 0), np.float32),
            "team": np.zeros(8, np.int8),
        }
        cache = os.path.join(tmp, "fill.npz")
        a = fill_missing_ball(
            data, matrices=matrices, valid=valid, source=source,
            weights_path=weights, pitch=pitch, cache_path=cache, device="cpu",
        )
        b = fill_missing_ball(
            data, matrices=matrices, valid=valid, source=source,
            weights_path=weights, pitch=pitch, cache_path=cache, device="cpu",
        )
        assert np.array_equal(np.nan_to_num(a["ball"]), np.nan_to_num(b["ball"]))


def test_metres_to_norm():
    out = metres_to_norm(np.array([[105.0, 34.0]], dtype=np.float32), (105.0, 68.0))
    assert abs(out[0, 0] - 1.0) < 1e-6
    assert abs(out[0, 1] - 0.5) < 1e-6


def run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print("ALL {} PASSED".format(len(tests)))


if __name__ == "__main__":
    run_all()
