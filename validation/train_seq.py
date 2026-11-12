"""Train and evaluate the two-stage MS-TCN possession model on external matches (10 bench games + PFF FC).

Strictly follows the Fiche F protocol with explicit kinematic duel/deflection and sub-second
boundary channels (78 channels by default, or 58 legacy channels with `--no-kinematic`):
  - Architecture: 2-stage MS-TCN (78 input channels: dead_features[stat] + team_features[duel_past]
    + direct signals from features.compute + 20 kinematic duel/deflection & gap-bridge channels;
    NO set-piece tag channels in inputs).
  - Kernel size 5, Stage 1: 6 dilated blocks x 64 channels (dilations 1..32, +/- 10 s),
    Stage 2: 4 dilated blocks x 48 channels (dilations 1..8).
  - Targets: possession-weighted H/A/D cross-entropy + multi-scale team-change head (sigma=2.0/4.5)
    + dead/restart head + boundary-aware temporal smoothing loss (unpenalized at true transitions).
  - Augmentation: 5 degradation seeds on bench games (+ degraded PFF FC World Cup 2022 games when
    `--use-pff` is enabled) + H <-> A symmetry swap.
  - Evaluation: 2 degradation seeds (0, 1) on held-out bench folds, comparing:
      * GBDT v7 reference,
      * MS-TCN alone (without and with `DirGated + KinSnap` / `switch_schedule`),
      * GBDT v7 + MS-TCN ensemble (`possession/ensemble.py` with `boundary_gate_trees=True`).
  - Pure NumPy inference verified against PyTorch (max abs diff < 1e-5).
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from possession import decode, emissions, ensemble, features, model, seqmodel  # noqa: E402
from validation import external as E  # noqa: E402
from validation import train as T  # noqa: E402

FPS = 25

# Focused cost grid for fast 20-run evaluation
SEQ_GRID = {
    "switch": (16.0, 22.0, 28.0),
    "to_dead": (60.0, 75.0),
    "restart": (60.0, 75.0),
    "t_team": (0.5, 0.7, 1.0),
    "t_dead": (0.5, 0.7),
}
CHANGE_BONUS_WEIGHTS = (3.0, 6.0, 9.0, 12.0)

# Standard 5-fold benchmark partition covering all 10 bench games (2 games per fold, 4 runs per fold)
FIVE_FOLDS = (
    ("idsse_J03WMX", "sample1"),
    ("idsse_J03WN1", "sample2"),
    ("idsse_J03WOH", "sample3"),
    ("idsse_J03WOY", "idsse_J03WPY"),
    ("idsse_J03WQQ", "idsse_J03WR9"),
)


# ------------------------------------------------------------------------- PyTorch MS-TCN
class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.15):
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.conv_dilated = nn.Conv1d(
            channels, channels, kernel_size, padding=pad, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        z = F.relu(self.conv_dilated(x))
        z = self.drop(z)
        z = self.conv_1x1(z)
        return x + z


class TCNStage(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dilations, kernel_size=5, dropout=0.15):
        super().__init__()
        self.in_proj = nn.Conv1d(in_channels, hidden_channels, 1)
        self.blocks = nn.ModuleList([
            DilatedResidualBlock(hidden_channels, kernel_size, d, dropout=dropout)
            for d in dilations
        ])
        self.out_proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x):
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.out_proj(h)


class MSTCN(nn.Module):
    def __init__(
        self,
        in_channels=78,
        out_heads=6,
        s1_channels=seqmodel.STAGE1_CHANNELS,
        s1_dilations=seqmodel.STAGE1_DILATIONS,
        s2_channels=seqmodel.STAGE2_CHANNELS,
        s2_dilations=seqmodel.STAGE2_DILATIONS,
        kernel_size=seqmodel.KERNEL_SIZE,
        dropout=0.15,
    ):
        super().__init__()
        self.out_heads = out_heads
        self.stage1 = TCNStage(in_channels, s1_channels, out_heads, s1_dilations, kernel_size, dropout)
        self.stage2 = TCNStage(out_heads, s2_channels, out_heads, s2_dilations, kernel_size, dropout)

    def forward(self, x):
        # x: (B, C, T)
        logits1 = self.stage1(x)
        probs_had = F.softmax(logits1[:, :3, :], dim=1)
        prob_chg = torch.sigmoid(logits1[:, 3:4, :])
        prob_de = torch.sigmoid(logits1[:, 4:5, :])
        if self.out_heads >= 6:
            off1 = torch.tanh(logits1[:, 5:6, :])
            s2_in = torch.cat([probs_had, prob_chg, prob_de, off1], dim=1)
        else:
            s2_in = torch.cat([probs_had, prob_chg, prob_de], dim=1)
        logits2 = self.stage2(s2_in)
        return logits1, logits2


def export_numpy_params(net, norm_mu, norm_sd):
    """Export PyTorch MSTCN state dict + input normalisation to a plain numpy dict."""
    params = {
        "norm_mu": np.asarray(norm_mu, dtype=np.float32),
        "norm_sd": np.asarray(norm_sd, dtype=np.float32),
    }
    sd = net.state_dict()
    for k, v in sd.items():
        params[k] = v.detach().cpu().numpy().astype(np.float32)
    return params


def verify_numpy_vs_torch(net, params, X_sample):
    """Verify that pure NumPy forward matches PyTorch eval forward within 1e-5."""
    net.eval()
    mu = params["norm_mu"]
    sd = params["norm_sd"]
    x_norm = ((X_sample.astype(np.float32) - mu[None, :]) / sd[None, :]).T  # (C, T)
    with torch.no_grad():
        xt = torch.from_numpy(x_norm[None, :, :])
        l1_t, l2_t = net(xt)
        l1_t = l1_t[0].T.cpu().numpy()
        l2_t = l2_t[0].T.cpu().numpy()
    l1_np, l2_np = seqmodel.forward_block_numpy(X_sample, params)
    diff1 = float(np.max(np.abs(l1_t - l1_np)))
    diff2 = float(np.max(np.abs(l2_t - l2_np)))
    max_diff = max(diff1, diff2)
    if max_diff >= 1e-5:
        raise AssertionError(f"NumPy vs Torch mismatch: max abs diff = {max_diff:.3e} >= 1e-5")
    return max_diff


# ------------------------------------------------------------------------- Targets & Data
def _gaussian_targets(events_idx, length, sigma=4.0, radius=12):
    """Smooth boundary target in [0, 1] peaking at 1.0 on transition frames."""
    y = np.zeros(length, dtype=np.float32)
    if not len(events_idx):
        return y
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / float(sigma)) ** 2).astype(np.float32)
    for idx in events_idx:
        for off, val in zip(offsets, kernel):
            t = idx + int(off)
            if 0 <= t < length and val > y[t]:
                y[t] = val
    return y


def make_block_targets(labels_seg, sharp_change=True, max_offset=15):
    """Build per-frame targets for one contiguous half block:
      - y_had: (T,) int64 in {0: H, 1: A, 2: D, -1: X}
      - w_had: (T,) float32 possession-duration weights (short possessions weighted up)
      - y_chg: (T,) float32 soft label in [0, 1] around H <-> A switches
      - y_de:  (T,) float32 soft label in [0, 1] around alive <-> D transitions
      - y_off: (T,) float32 signed offset `(t_nearest_chg - t) / max_offset` in [-1, 1]
      - m_off: (T,) float32 Gaussian mask around H <-> A transitions for offset regression
    """
    n = len(labels_seg)
    y_had = np.full(n, -1, dtype=np.int64)
    y_had[labels_seg == "H"] = 0
    y_had[labels_seg == "A"] = 1
    y_had[labels_seg == "D"] = 2

    w_had = T.spell_weights(labels_seg).astype(np.float32)
    alive = (y_had == 0) | (y_had == 1)
    if alive.any():
        w_had[alive] /= float(w_had[alive].mean())
    w_had[y_had == 2] = 1.0
    w_had[y_had < 0] = 0.0

    changes = np.flatnonzero(labels_seg[1:] != labels_seg[:-1]) + 1
    team_chg, dead_edge = [], []
    for i in changes:
        a, b = labels_seg[i - 1], labels_seg[i]
        if "X" in (a, b):
            continue
        if {a, b} == {"H", "A"}:
            team_chg.append(int(i))
        elif "D" in (a, b):
            dead_edge.append(int(i))

    if sharp_change:
        y_chg = (
            0.70 * _gaussian_targets(team_chg, n, sigma=2.0, radius=8)
            + 0.30 * _gaussian_targets(team_chg, n, sigma=4.5, radius=14)
        ).astype(np.float32)
    else:
        y_chg = _gaussian_targets(team_chg, n, sigma=4.0, radius=12)
    y_de = _gaussian_targets(dead_edge, n, sigma=5.0, radius=15)

    y_off = np.zeros(n, dtype=np.float32)
    m_off = np.zeros(n, dtype=np.float32)
    if team_chg:
        tc_arr = np.asarray(team_chg, dtype=np.int64)
        idx = np.arange(n, dtype=np.int64)
        pos = np.searchsorted(tc_arr, idx)
        left = np.clip(pos - 1, 0, len(tc_arr) - 1)
        right = np.clip(pos, 0, len(tc_arr) - 1)
        d_left = tc_arr[left] - idx
        d_right = tc_arr[right] - idx
        delta = np.where(np.abs(d_left) <= np.abs(d_right), d_left, d_right)
        near = np.abs(delta) <= max_offset
        y_off[near] = np.clip(delta[near] / float(max_offset), -1.0, 1.0)
        m_off[near] = np.exp(-0.5 * (delta[near] / 8.0) ** 2).astype(np.float32)

    return y_had, w_had, y_chg, y_de, y_off, m_off


def kinematic_snap_signal(f, blocks, pitch):
    """Sub-second physical contact / deflection / gap-transfer / 22p-interception cue in [0, 1.5] per frame."""
    sk, _ = seqmodel.kinematic_duel_and_boundary_channels(f, blocks, pitch)
    sc, _ = seqmodel.carrier_and_imputation_channels(f)
    ctrl_edge = np.zeros(len(f["control"]), dtype=np.float32)
    ctrl_edge[1:] = (f["control"][1:] != f["control"][:-1]).astype(np.float32)
    # sc[:, 7] is interception_sym_05, sc[:, 5] is same_player_continuity, sc[:, 9] is carrier_decel_sym
    inter_cue = sc[:, 7].astype(np.float32) * np.clip(1.0 - sc[:, 5].astype(np.float32), 0.2, 1.0)
    return np.clip(
        0.55 * (f["touch"] != 0).astype(np.float32)
        + 0.25 * ctrl_edge
        + 0.45 * sk[:, 1].astype(np.float32)
        + 0.25 * np.tanh(sk[:, 3].astype(np.float32) / 2.0)
        + 0.35 * sk[:, 7].astype(np.float32)
        + 0.45 * inter_cue
        + 0.20 * sc[:, 9].astype(np.float32),
        0.0,
        1.5,
    ).astype(np.float32)


def _channel_tag(include_kinematic=True, include_carrier=True):
    if not include_kinematic:
        return "58ch"
    return "102ch" if include_carrier else "78ch"


def build_seq_rows(samples, seeds, cache_dir=None, games_filter=None, include_kinematic=True,
                   include_carrier=True):
    """Build feature matrices and targets for each (game, seed)."""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    tag = _channel_tag(include_kinematic, include_carrier)
    rows = []
    for seed in seeds:
        prepared = E.prepare(samples, seed=seed)
        for name, atd, blocks, labels, events in prepared:
            if games_filter and name not in games_filter:
                continue
            cache_file = (
                os.path.join(cache_dir, f"seq_{tag}_{name}_seed{seed}.npz")
                if cache_dir else None
            )
            f = features.compute(atd, blocks, E.PITCH)
            pieces = E.fake_set_piece_tags(events, np.random.default_rng(seed + 100))
            if cache_file and os.path.exists(cache_file):
                with np.load(cache_file) as z:
                    X = z["X"].astype(np.float32)
                    sk_snap = z["sk_snap"].astype(np.float32) if "sk_snap" in z else kinematic_snap_signal(f, blocks, E.PITCH)
            elif not include_kinematic and cache_dir and os.path.exists(os.path.join(cache_dir, f"seq_{name}_seed{seed}.npz")):
                with np.load(os.path.join(cache_dir, f"seq_{name}_seed{seed}.npz")) as z:
                    X = z["X"].astype(np.float32)
                sk_snap = kinematic_snap_signal(f, blocks, E.PITCH)
            else:
                ev = emissions.ball_team_evidence(f, emissions.PARAMS) / emissions.PARAMS["w_ball"]
                X = seqmodel.seq_features(
                    f,
                    blocks,
                    E.PITCH,
                    ev,
                    "stat",
                    "duel_past",
                    include_kinematic=include_kinematic,
                    include_carrier=include_carrier,
                )
                sk_snap = kinematic_snap_signal(f, blocks, E.PITCH)
                if cache_file:
                    np.savez(cache_file, X=X, sk_snap=sk_snap)
            block_targets = [
                make_block_targets(labels[lo:hi + 1], sharp_change=include_kinematic)
                for _, lo, hi in blocks
            ]
            rows.append({
                "name": name,
                "seed": seed,
                "blocks": blocks,
                "labels": labels,
                "events": events,
                "pieces": pieces,
                "f": f,
                "X": X,
                "sk_snap": sk_snap,
                "block_targets": block_targets,
                "is_pff": False,
            })
    return rows


def build_pff_rows(pff_dir=E.IDSSE_DIR, seeds=(0, 1), cache_dir=None, max_games=None,
                   include_kinematic=True, include_carrier=True):
    """Load converted PFF FC World Cup 2022 `.npz` files (`pff_*.npz`), degrade them like the ATD,
    and compute/cache the 102-channel (or 78/58-channel) MS-TCN inputs and targets."""
    if not pff_dir or not os.path.isdir(pff_dir):
        return []
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    tag = _channel_tag(include_kinematic, include_carrier)
    pff_files = sorted(
        f for f in os.listdir(pff_dir) if f.startswith("pff_") and f.endswith(".npz")
    )
    if max_games is not None:
        pff_files = pff_files[:max_games]
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed + 500)
        for fname in pff_files:
            name = fname[:-4]
            path = os.path.join(pff_dir, fname)
            cache_file = (
                os.path.join(cache_dir, f"seq_{tag}_{name}_seed{seed}.npz")
                if cache_dir else None
            )
            (players, ball), team, blocks, labels, events = E.load_idsse(path, merge_s=0.0)
            E.mark_openings(events, blocks)
            if cache_file and os.path.exists(cache_file):
                with np.load(cache_file) as z:
                    X = z["X"].astype(np.float32)
            else:
                players_deg, ball_deg = E.degrade(players, ball, rng, labels)
                atd = {"players": E.speeds(players_deg), "ball": ball_deg, "team": team}
                f = features.compute(atd, blocks, E.PITCH)
                ev = emissions.ball_team_evidence(f, emissions.PARAMS) / emissions.PARAMS["w_ball"]
                X = seqmodel.seq_features(
                    f,
                    blocks,
                    E.PITCH,
                    ev,
                    "stat",
                    "duel_past",
                    include_kinematic=include_kinematic,
                    include_carrier=include_carrier,
                )
                if cache_file:
                    np.savez(cache_file, X=X)
            block_targets = [
                make_block_targets(labels[lo:hi + 1], sharp_change=include_kinematic)
                for _, lo, hi in blocks
            ]
            rows.append({
                "name": name,
                "seed": seed,
                "blocks": blocks,
                "labels": labels,
                "events": events,
                "pieces": [],
                "f": None,
                "X": X,
                "sk_snap": None,
                "block_targets": block_targets,
                "is_pff": True,
            })
    return rows


# ------------------------------------------------------------------------- Loss & Training
def stage_loss(logits, y_had, w_had, y_chg, y_de, y_off=None, m_off=None,
               pos_weight_chg=10.0, pos_weight_de=8.0, smooth_w=0.18,
               edge_aware_smooth=True, use_segment_iou=False):
    """Combined multi-task loss for one MS-TCN stage.
    When `use_segment_iou=True`, adds:
      - Windowed Soft-IoU (Jaccard) loss on H, A, D probabilities (`loss_iou`),
      - Continuous Boundary-Offset Regression loss (`SmoothL1(tanh(logit_off), y_off)`),
      - Intra-Possession Anti-Fragmentation Margin penalty (`loss_frag`).
    """
    logits_had = logits[:, :3, :]  # (B, 3, T)
    logit_chg = logits[:, 3, :]    # (B, T)
    logit_de = logits[:, 4, :]     # (B, T)

    valid = (y_had >= 0).float()
    y_clamped = torch.clamp(y_had, min=0)
    ce = F.cross_entropy(logits_had, y_clamped, reduction="none")  # (B, T)
    w_eff = torch.clamp(w_had, max=2.5) if use_segment_iou else w_had
    loss_had = (ce * w_eff * valid).sum() / torch.clamp((w_eff * valid).sum(), min=1.0)

    pw_c = torch.tensor(pos_weight_chg, dtype=logits.dtype, device=logits.device)
    pw_d = torch.tensor(pos_weight_de, dtype=logits.dtype, device=logits.device)
    bce_chg = F.binary_cross_entropy_with_logits(logit_chg, y_chg, pos_weight=pw_c, reduction="none")
    bce_de = F.binary_cross_entropy_with_logits(logit_de, y_de, pos_weight=pw_d, reduction="none")
    loss_chg = (bce_chg * valid).sum() / torch.clamp(valid.sum(), min=1.0)
    loss_de = (bce_de * valid).sum() / torch.clamp(valid.sum(), min=1.0)

    log_p = F.log_softmax(logits_had, dim=1)
    diff_sq = torch.clamp((log_p[:, :, 1:] - log_p[:, :, :-1]).pow(2), min=0.0, max=16.0)
    if edge_aware_smooth:
        no_edge = (1.0 - torch.maximum(y_chg[:, 1:], y_de[:, 1:])).unsqueeze(1)
        smooth = (diff_sq * no_edge).mean()
        w_c = 0.45
    else:
        smooth = diff_sq.mean()
        w_c = 0.35

    total = loss_had + w_c * loss_chg + 0.25 * loss_de + smooth_w * smooth

    if use_segment_iou:
        probs = F.softmax(logits_had, dim=1)
        one_hot = F.one_hot(y_clamped, num_classes=3).permute(0, 2, 1).float() * valid.unsqueeze(1)
        B_sz, _, T_len = probs.shape
        n_win = 6
        w_len = T_len // n_win
        p_w = probs[:, :3, :n_win * w_len].reshape(B_sz, 3, n_win, w_len)
        g_w = one_hot[:, :3, :n_win * w_len].reshape(B_sz, 3, n_win, w_len)
        inter = (p_w * g_w).sum(dim=-1)
        union = p_w.sum(dim=-1) + g_w.sum(dim=-1) - inter
        valid_win = (g_w.sum(dim=-1) > 5.0).float()
        soft_iou = (inter + 1.0) / (union + 1.0)
        loss_iou = ((1.0 - soft_iou) * valid_win).sum() / torch.clamp(valid_win.sum(), min=1.0)

        interior_h = ((y_had == 0) & (y_chg < 0.05)).float()
        interior_a = ((y_had == 1) & (y_chg < 0.05)).float()
        s_team = logits_had[:, 0, :] - logits_had[:, 1, :]
        frag_pen = interior_h * F.relu(0.5 - s_team).pow(2) + interior_a * F.relu(0.5 + s_team).pow(2)
        loss_frag = frag_pen.sum() / torch.clamp((interior_h + interior_a).sum(), min=1.0)
        total = total + 0.30 * loss_iou + 0.15 * loss_frag

        if logits.shape[1] >= 6 and y_off is not None and m_off is not None:
            pred_off = torch.tanh(logits[:, 5, :])
            off_err = F.smooth_l1_loss(pred_off, y_off, reduction="none", beta=0.1)
            loss_off = (off_err * m_off * valid).sum() / torch.clamp((m_off * valid).sum(), min=1.0)
            total = total + 0.50 * loss_off

    return total


def train_mstcn(train_rows, steps=3000, batch_size=4, seq_len=1200, lr=1.5e-3, seed=42,
                swap_aug=True, edge_aware_smooth=True, use_segment_iou=False):
    """Fit the 2-stage MS-TCN on `train_rows` with fixed random seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    segments = []
    all_x = []
    for r in train_rows:
        for (_, lo, hi), bt in zip(r["blocks"], r["block_targets"]):
            x_seg = r["X"][lo:hi + 1]
            yo = bt[4] if len(bt) >= 6 else np.zeros(len(x_seg), dtype=np.float32)
            mo = bt[5] if len(bt) >= 6 else np.zeros(len(x_seg), dtype=np.float32)
            segments.append((x_seg, bt[0], bt[1], bt[2], bt[3], yo, mo))
            all_x.append(x_seg[::10])
    cat_x = np.vstack(all_x).astype(np.float64)
    norm_mu = cat_x.mean(axis=0).astype(np.float32)
    norm_sd = (cat_x.std(axis=0) + 1e-6).astype(np.float32)

    n_channels = int(norm_mu.shape[0])
    anti_mask = seqmodel.antisym_mask(n_channels)
    norm_mu[anti_mask] = 0.0

    norm_segments = []
    for x_seg, yh, wh, yc, yd, yo, mo in segments:
        xn = ((x_seg - norm_mu[None, :]) / norm_sd[None, :]).astype(np.float32)
        norm_segments.append((xn, yh, wh, yc, yd, yo, mo))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_heads = 6 if use_segment_iou else 5
    net = MSTCN(in_channels=n_channels, out_heads=out_heads).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-4)

    anti_idx = np.flatnonzero(anti_mask)
    seg_weights = np.array([len(s[0]) for s in norm_segments], dtype=np.float64)
    seg_weights /= seg_weights.sum()

    net.train()
    t0 = time.time()
    running_loss = 0.0
    for step in range(1, steps + 1):
        bx = np.empty((batch_size, n_channels, seq_len), dtype=np.float32)
        by_had = np.empty((batch_size, seq_len), dtype=np.int64)
        bw_had = np.empty((batch_size, seq_len), dtype=np.float32)
        by_chg = np.empty((batch_size, seq_len), dtype=np.float32)
        by_de = np.empty((batch_size, seq_len), dtype=np.float32)
        by_off = np.empty((batch_size, seq_len), dtype=np.float32)
        bm_off = np.empty((batch_size, seq_len), dtype=np.float32)

        chosen = rng.choice(len(norm_segments), size=batch_size, p=seg_weights)
        for b_i, s_i in enumerate(chosen):
            xn, yh, wh, yc, yd, yo, mo = norm_segments[s_i]
            m = len(xn)
            if m > seq_len:
                start = int(rng.integers(0, m - seq_len))
                sl = slice(start, start + seq_len)
                x_crop = xn[sl].T.copy()
                yh_crop = yh[sl].copy()
                wh_crop = wh[sl].copy()
                yc_crop = yc[sl].copy()
                yd_crop = yd[sl].copy()
                yo_crop = yo[sl].copy()
                mo_crop = mo[sl].copy()
            else:
                x_crop = np.zeros((n_channels, seq_len), dtype=np.float32)
                yh_crop = np.full(seq_len, -1, dtype=np.int64)
                wh_crop = np.zeros(seq_len, dtype=np.float32)
                yc_crop = np.zeros(seq_len, dtype=np.float32)
                yd_crop = np.zeros(seq_len, dtype=np.float32)
                yo_crop = np.zeros(seq_len, dtype=np.float32)
                mo_crop = np.zeros(seq_len, dtype=np.float32)
                x_crop[:, :m] = xn.T
                yh_crop[:m] = yh
                wh_crop[:m] = wh
                yc_crop[:m] = yc
                yd_crop[:m] = yd
                yo_crop[:m] = yo
                mo_crop[:m] = mo

            if swap_aug and rng.random() < 0.5:
                x_crop[anti_idx, :] = -x_crop[anti_idx, :]
                h_mask = yh_crop == 0
                a_mask = yh_crop == 1
                yh_crop[h_mask] = 1
                yh_crop[a_mask] = 0

            bx[b_i] = x_crop
            by_had[b_i] = yh_crop
            bw_had[b_i] = wh_crop
            by_chg[b_i] = yc_crop
            by_de[b_i] = yd_crop
            by_off[b_i] = yo_crop
            bm_off[b_i] = mo_crop

        tx = torch.from_numpy(bx).to(device)
        ty_had = torch.from_numpy(by_had).to(device)
        tw_had = torch.from_numpy(bw_had).to(device)
        ty_chg = torch.from_numpy(by_chg).to(device)
        ty_de = torch.from_numpy(by_de).to(device)
        ty_off = torch.from_numpy(by_off).to(device)
        tm_off = torch.from_numpy(bm_off).to(device)

        optimizer.zero_grad()
        l1, l2 = net(tx)
        loss1 = stage_loss(
            l1, ty_had, tw_had, ty_chg, ty_de, ty_off, tm_off,
            edge_aware_smooth=edge_aware_smooth, use_segment_iou=use_segment_iou,
        )
        loss2 = stage_loss(
            l2, ty_had, tw_had, ty_chg, ty_de, ty_off, tm_off,
            edge_aware_smooth=edge_aware_smooth, use_segment_iou=use_segment_iou,
        )
        loss = loss1 + loss2
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        optimizer.step()
        scheduler.step()

        running_loss += float(loss.item())
        if step % 500 == 0 or step == steps:
            elapsed = time.time() - t0
            print(
                f"  step {step:4d}/{steps}  loss {running_loss / 500 if step >= 500 else running_loss / step:.4f}  "
                f"elapsed {elapsed:.1f}s ({elapsed / step * 1000:.1f} ms/step)",
                flush=True,
            )
            running_loss = 0.0

    params = export_numpy_params(net, norm_mu, norm_sd)
    max_diff = verify_numpy_vs_torch(net.cpu(), params, segments[0][0][:2000])
    print(f"  numpy vs torch max abs diff: {max_diff:.2e} (< 1e-5 OK)", flush=True)
    return params


# ------------------------------------------------------------------------- Evaluation
def precompute_seq_outputs(rows, params, tta_swap=True):
    """Run pure NumPy MS-TCN inference on `rows`, optionally averaging with H <-> A swap TTA."""
    n_channels = int(params["norm_mu"].shape[0])
    anti_mask = seqmodel.antisym_mask(n_channels)
    outputs = []
    for r in rows:
        out = seqmodel.predict_game_numpy(r["X"], r["blocks"], params)
        if tta_swap:
            X_swap = r["X"].copy()
            X_swap[:, anti_mask] = -X_swap[:, anti_mask]
            out_swap = seqmodel.predict_game_numpy(X_swap, r["blocks"], params)
            out = {
                "s_team_raw": 0.5 * (out["s_team_raw"] - out_swap["s_team_raw"]),
                "s_dead_raw": 0.5 * (out["s_dead_raw"] + out_swap["s_dead_raw"]),
                "p_change": 0.5 * (out["p_change"] + out_swap["p_change"]),
                "p_dead_edge": 0.5 * (out["p_dead_edge"] + out_swap["p_dead_edge"]),
                "delta_t": 0.5 * (out["delta_t"] + out_swap["delta_t"]),
            }
        outputs.append(out)
    return outputs


def evaluate_seq(rows, seq_outs, setting, change_weight=0.0, use_dir_gate=False, use_kin_snap=False):
    """Evaluate MS-TCN outputs on `rows` with given temperatures, costs, and optional change head."""
    p = dict(emissions.PARAMS, pitch=E.PITCH)
    costs = {k: setting[k] for k in ("switch", "to_dead", "restart")}
    if use_dir_gate:
        costs.update(
            change_alpha=float(change_weight),
            change_guard=4.0,
            switch_min=4.5,
            change_thresh=0.28,
            change_nms=15,
            directional_gate=True,
        )
    scores = []
    for r, out in zip(rows, seq_outs):
        s_team = (out["s_team_raw"] * setting["t_team"]).astype(np.float32)
        s_dead = (out["s_dead_raw"] * setting["t_dead"]).astype(np.float32)
        s_team, s_dead, _ = emissions.add_restarts(
            r["f"], r["blocks"], s_team, s_dead, r["pieces"], [], p
        )
        pc = out["p_change"]
        if use_kin_snap and r.get("sk_snap") is not None:
            pc = np.clip(pc * (0.70 + 0.55 * r["sk_snap"]), 0.0, 1.0)
        if use_dir_gate:
            pred = decode.decode(s_team, s_dead, r["blocks"], len(r["labels"]), costs, p_change=pc)
        else:
            cb = (change_weight * pc) if change_weight > 0.0 else None
            pred = decode.decode(s_team, s_dead, r["blocks"], len(r["labels"]), costs, change_bonus=cb)
        res = pm.evaluate(pm.label_codes(r["labels"]), pm.label_codes(pred), bootstrap_n=0)
        scores.append(
            (res["spell_f1_mean"], res["macro_f1"], res["n_gt_spells"], res["n_pred_spells"])
            + T.timing(r["labels"], pred)
            + (res["spell_f1"][pm.SPELL_IOU_HEADLINE]["n_spurious"],)
            + T.team_change_hits(r["labels"], pred)
        )
    return scores


def evaluate_ensemble_rows(rows, tree_logits, seq_outs, ens_cfg, cost_cfg, use_kin_snap=False):
    """Evaluate GBDT + MS-TCN ensemble via `possession.ensemble.ensemble_and_decode`."""
    p = dict(emissions.PARAMS, pitch=E.PITCH)
    scores = []
    for r, (lt, ld), out in zip(rows, tree_logits, seq_outs):
        pc = out["p_change"]
        if use_kin_snap and r.get("sk_snap") is not None:
            pc = np.clip(pc * (0.70 + 0.55 * r["sk_snap"]), 0.0, 1.0)
        s_out = {
            "s_team": out["s_team_raw"],
            "s_dead": out["s_dead_raw"],
            "p_change": pc,
        }
        res_dec = ensemble.ensemble_and_decode(
            r["f"],
            r["blocks"],
            len(r["labels"]),
            E.PITCH,
            tree_out=(lt, ld),
            seq_out=s_out,
            set_pieces=r["pieces"],
            params=p,
            ensemble_cfg=ens_cfg,
            costs=cost_cfg,
        )
        pred = res_dec["labels"]
        res = pm.evaluate(pm.label_codes(r["labels"]), pm.label_codes(pred), bootstrap_n=0)
        scores.append(
            (res["spell_f1_mean"], res["macro_f1"], res["n_gt_spells"], res["n_pred_spells"])
            + T.timing(r["labels"], pred)
            + (res["spell_f1"][pm.SPELL_IOU_HEADLINE]["n_spurious"],)
            + T.team_change_hits(r["labels"], pred)
        )
    return scores


def perfect_team_bound(rows, seq_outs, setting):
    """Upper bound on spellF1 if team attribution is oracle-perfect given our dead-ball predictions."""
    p = dict(emissions.PARAMS, pitch=E.PITCH)
    costs = {k: setting[k] for k in ("switch", "to_dead", "restart")}
    f1s = []
    for r, out in zip(rows, seq_outs):
        s_team = (out["s_team_raw"] * setting["t_team"]).astype(np.float32)
        s_dead = (out["s_dead_raw"] * setting["t_dead"]).astype(np.float32)
        s_team, s_dead, _ = emissions.add_restarts(
            r["f"], r["blocks"], s_team, s_dead, r["pieces"], [], p
        )
        pred = decode.decode(s_team, s_dead, r["blocks"], len(r["labels"]), costs)
        oracle = pred.copy()
        gt = r["labels"]
        alive_pred = np.isin(oracle, ["H", "A"])
        for _, lo, hi in r["blocks"]:
            last_team = "H"
            for t in range(lo, hi + 1):
                if gt[t] in ("H", "A"):
                    last_team = gt[t]
                if alive_pred[t]:
                    oracle[t] = last_team
        res = pm.evaluate(pm.label_codes(gt), pm.label_codes(oracle), bootstrap_n=0)
        f1s.append(res["spell_f1_mean"])
    return float(np.mean(f1s))


def summarize_scores(scores, rows=None):
    f1 = float(np.mean([s[0] for s in scores]))
    macro = float(np.mean([s[1] for s in scores]))
    gt_n = int(np.sum([s[2] for s in scores]))
    pred_n = int(np.sum([s[3] for s in scores]))
    spurious = int(np.sum([s[7] for s in scores]))
    tc = np.nanmedian(np.array([s[4:7] for s in scores], dtype=float), axis=0)
    short_mean = float(np.nanmean([s[6] for s in scores]))
    n_true, n_fired, hit_half, hit_one = np.sum([s[8:12] for s in scores], axis=0)
    out = {
        "spell_f1": f1,
        "macro_f1": macro,
        "gt_spells": gt_n,
        "pred_spells": pred_n,
        "spurious": spurious,
        "short_recall": short_mean,
        "short_recall_med": float(tc[2]),
        "delay_team": float(tc[0]),
        "delay_restart": float(tc[1]),
        "chg_gt": int(n_true),
        "chg_pred": int(n_fired),
        "chg_05s": float(hit_half / max(n_true, 1)),
        "chg_05s_prec": float(hit_half / max(n_fired, 1)),
        "chg_10s": float(hit_one / max(n_true, 1)),
    }
    if rows is not None:
        m_f1 = [s[0] for r, s in zip(rows, scores) if r["name"].startswith("sample")]
        d_f1 = [s[0] for r, s in zip(rows, scores) if not r["name"].startswith("sample")]
        out["metrica_f1"] = float(np.mean(m_f1)) if m_f1 else float("nan")
        out["dfl_f1"] = float(np.mean(d_f1)) if d_f1 else float("nan")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=os.environ.get("MTAH_SAMPLES"))
    parser.add_argument("--fold", default=None, help="Run a single held-out match fold (e.g. idsse_J03WMX)")
    parser.add_argument("--five-folds", action="store_true", help="Run the 5-fold benchmark across all 10 games")
    parser.add_argument("--val-game", default="idsse_J03WN1", help="Internal validation match for cost tuning")
    parser.add_argument("--exclude-val-from-train", action="store_true", help="Exclude val-game from training")
    parser.add_argument("--use-pff", action="store_true", help="Include PFF FC World Cup 2022 games in training")
    parser.add_argument("--pff-seeds", type=int, default=1, help="Degradation seeds for PFF training games")
    parser.add_argument("--pff-max-games", type=int, default=None, help="Max number of PFF games to load")
    parser.add_argument("--train-seeds", type=int, default=5, help="Number of degradation seeds for bench training")
    parser.add_argument("--eval-seeds", type=int, default=2, help="Number of degradation seeds for evaluation")
    parser.add_argument("--steps", type=int, default=3000, help="Training steps per fold")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1200)
    parser.add_argument("--no-kinematic", action="store_true", help="Use legacy 58-channel inputs instead of kinematic channels")
    parser.add_argument("--no-carrier", action="store_true", help="Use 78-channel inputs without 22-player carrier/imputation channels")
    parser.add_argument("--segment-iou", action="store_true", help="Train 6-head MS-TCN with Segment Soft-IoU, Anti-Fragmentation, and Boundary-Offset Regression")
    parser.add_argument("--reuse-checkpoints", action="store_true", help="Reuse cached fold model checkpoints if present")
    parser.add_argument("--cache-dir", default=os.path.join(ROOT, "cache", "seq_features"))
    parser.add_argument("--save-oof", default=None, help="Save out-of-fold MS-TCN predictions to .npz")
    parser.add_argument("--write-weights", action="store_true", help="Fit final model on all games and write possession/weights*.json")
    args = parser.parse_args()

    if not args.samples:
        parser.error("--samples is required (or set MTAH_SAMPLES)")

    torch.set_num_threads(max(1, os.cpu_count() or 4))

    include_kin = not args.no_kinematic
    include_car = include_kin and (not args.no_carrier)
    ch_tag = _channel_tag(include_kin, include_car)
    n_seeds = max(args.train_seeds, args.eval_seeds)
    print(
        f"Building/loading degraded bench features for {n_seeds} seeds "
        f"(channels={ch_tag}, kinematic={include_kin}, carrier={include_car})...",
        flush=True,
    )
    t_build0 = time.time()
    all_rows = build_seq_rows(
        args.samples,
        tuple(range(n_seeds)),
        cache_dir=args.cache_dir,
        include_kinematic=include_kin,
        include_carrier=include_car,
    )
    print(f"Bench features ready in {time.time() - t_build0:.1f}s ({len(all_rows)} game-seed rows).", flush=True)

    pff_rows = []
    if args.use_pff:
        t_pff0 = time.time()
        pff_rows = build_pff_rows(
            E.IDSSE_DIR,
            seeds=tuple(range(args.pff_seeds)),
            cache_dir=args.cache_dir,
            max_games=args.pff_max_games,
            include_kinematic=include_kin,
            include_carrier=include_car,
        )
        print(
            f"PFF FC features ready in {time.time() - t_pff0:.1f}s "
            f"({len(pff_rows)} game-seed rows from {len({r['name'] for r in pff_rows})} WC2022 matches).",
            flush=True,
        )

    names = sorted({r["name"] for r in all_rows})
    if args.five_folds:
        fold_groups = [list(g) for g in FIVE_FOLDS]
    elif args.fold:
        fold_groups = [[g.strip() for g in args.fold.split(",")]]
    else:
        fold_groups = [[n] for n in names]

    combos = [
        dict(zip(SEQ_GRID.keys(), c))
        for c in itertools.product(*SEQ_GRID.values())
        if c[1] == c[2]  # TIED: to_dead == restart
    ]

    v7_weights = model.load_weights()
    T.LEARNER = "gbdt"
    T.SPELL_WEIGHT = True

    eval_rows_all = [r for r in all_rows if r["seed"] < args.eval_seeds]
    v7_all_rows = []
    for r in eval_rows_all:
        ev = emissions.ball_team_evidence(r["f"], emissions.PARAMS) / emissions.PARAMS["w_ball"]
        v7_all_rows.append(dict(
            r,
            Xd=model.dead_features(r["f"], r["blocks"], E.PITCH, v7_weights.get("dead_variant", "stat")),
            Xt=model.team_features(r["f"], r["blocks"], E.PITCH, ev, v7_weights.get("team_variant", "duel_past")),
        ))

    oof_seq_map = {}
    oof_tree_map = {}
    ch_tag = _channel_tag(include_kin, include_car)
    pff_tag = "_pff" if args.use_pff else ""

    for fold_idx, test_names in enumerate(fold_groups, 1):
        fold_label = "+".join(test_names)
        print(f"\n==================== FOLD {fold_idx}/{len(fold_groups)}: {fold_label} ====================", flush=True)

        test_set = set(test_names)
        test_rows_eval = [r for r in eval_rows_all if r["name"] in test_set]

        # 1. Fit/evaluate GBDT v7 held-out on this fold
        v7_train = [r for r in v7_all_rows if r["name"] not in test_set]
        v7_test = [r for r in v7_all_rows if r["name"] in test_set]
        v7_base = T.fit(v7_train)
        v7_fold_w = dict(v7_base, t_team=v7_weights.get("t_team", 1.0), t_dead=v7_weights.get("t_dead", 0.7))
        v7_base_costs = {"switch": 22.0, "to_dead": 75.0, "restart": 75.0}
        v7_scores = T.evaluate(v7_test, v7_fold_w, v7_base_costs)
        v7_sum = summarize_scores(v7_scores, test_rows_eval)
        for r_v7 in v7_test:
            lt = model.logit_any(v7_base["team"], r_v7["Xt"])
            ld = model.logit_any(v7_base["dead"], r_v7["Xd"])
            oof_tree_map[(r_v7["name"], r_v7["seed"])] = (lt, ld)

        print(
            f"[v7 held-out ref on {fold_label}] spellF1 {v7_sum['spell_f1']:.4f}  macroF1 {v7_sum['macro_f1']:.4f}  "
            f"poss gt {v7_sum['gt_spells']} pred {v7_sum['pred_spells']} spurious {v7_sum['spurious']}  "
            f"short {v7_sum['short_recall']:.1%}  chg@0.5s {v7_sum['chg_05s']:.1%} chg@1s {v7_sum['chg_10s']:.1%}",
            flush=True,
        )

        # 2. Train MS-TCN (or load cached checkpoint when --reuse-checkpoints is set)
        fold_npz = os.path.join(args.cache_dir, f"mstcn_{ch_tag}{pff_tag}_{fold_label}.npz")
        if args.reuse_checkpoints and os.path.exists(fold_npz):
            params = seqmodel.load_seq_params(fold_npz)
            print(f"Loaded cached fold checkpoint -> {os.path.relpath(fold_npz, ROOT)}", flush=True)
        else:
            train_rows = [
                r for r in all_rows
                if r["name"] not in test_set
                and (not args.exclude_val_from_train or not args.val_game or args.val_game in test_set or r["name"] != args.val_game)
                and r["seed"] < args.train_seeds
            ] + pff_rows

            t_train0 = time.time()
            params = train_mstcn(
                train_rows,
                steps=args.steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                seed=42 + fold_idx,
                swap_aug=True,
                edge_aware_smooth=include_kin,
                use_segment_iou=args.segment_iou,
            )
            train_dur = time.time() - t_train0
            np.savez(fold_npz, **params)
            print(
                f"Fold {fold_label} training ({len(train_rows)} game-seeds) completed in "
                f"{train_dur:.1f}s ({train_dur / 60.0:.2f} min) -> {os.path.relpath(fold_npz, ROOT)}",
                flush=True,
            )

        # 3. Precompute MS-TCN outputs on test fold
        seq_outs_test = precompute_seq_outputs(test_rows_eval, params, tta_swap=True)
        for r_t, out_t in zip(test_rows_eval, seq_outs_test):
            oof_seq_map[(r_t["name"], r_t["seed"])] = out_t

        # 4. Evaluate MS-TCN alone (without and with DirGated + KinSnap)
        std_setting = {"switch": 22.0, "to_dead": 60.0, "restart": 60.0, "t_team": 0.7, "t_dead": 0.7}
        sc_seq_no = evaluate_seq(test_rows_eval, seq_outs_test, std_setting)
        sum_seq_no = summarize_scores(sc_seq_no, test_rows_eval)

        sc_seq_dg = evaluate_seq(
            test_rows_eval, seq_outs_test, std_setting,
            change_weight=10.0, use_dir_gate=True, use_kin_snap=include_kin
        )
        sum_seq_dg = summarize_scores(sc_seq_dg, test_rows_eval)

        # 5. Evaluate GBDT v7 + MS-TCN Ensemble (with boundary_gate_trees=True)
        tree_logits_fold = [oof_tree_map[(r_t["name"], r_t["seed"])] for r_t in test_rows_eval]
        ens_cfg = dict(
            ensemble.DEFAULT_ENSEMBLE,
            w_tree_team=0.25,
            w_seq_team=0.75,
            w_tree_dead=0.30,
            w_seq_dead=0.70,
            scale_tree_team=1.0,
            scale_seq_team=0.7,
            scale_tree_dead=0.7,
            scale_seq_dead=0.7,
            t_team=1.0,
            t_dead=1.0,
            boundary_gate_trees=True,
            boundary_band=1.5,
        )
        ens_costs_no = {"switch": 22.0, "to_dead": 60.0, "restart": 60.0, "change_alpha": 0.0}
        ens_costs_dg = dict(
            ens_costs_no,
            change_alpha=10.0,
            change_guard=4.0,
            switch_min=4.5,
            change_thresh=0.28,
            change_nms=15,
            directional_gate=True,
        )

        sc_ens_no = evaluate_ensemble_rows(
            test_rows_eval, tree_logits_fold, seq_outs_test, ens_cfg, ens_costs_no, use_kin_snap=include_kin
        )
        sum_ens_no = summarize_scores(sc_ens_no, test_rows_eval)

        sc_ens_dg = evaluate_ensemble_rows(
            test_rows_eval, tree_logits_fold, seq_outs_test, ens_cfg, ens_costs_dg, use_kin_snap=include_kin
        )
        sum_ens_dg = summarize_scores(sc_ens_dg, test_rows_eval)

        print(
            f"[MS-TCN {ch_tag} seul (sans tête chg) on {fold_label}] "
            f"spellF1 {sum_seq_no['spell_f1']:.4f}  macroF1 {sum_seq_no['macro_f1']:.4f}  "
            f"short {sum_seq_no['short_recall']:.1%}  chg@0.5s {sum_seq_no['chg_05s']:.1%} (prec {sum_seq_no['chg_05s_prec']:.1%})  spurious {sum_seq_no['spurious']}",
            flush=True,
        )
        print(
            f"[MS-TCN {ch_tag} seul (+ DirGated+KinSnap) on {fold_label}] "
            f"spellF1 {sum_seq_dg['spell_f1']:.4f}  macroF1 {sum_seq_dg['macro_f1']:.4f}  "
            f"short {sum_seq_dg['short_recall']:.1%}  chg@0.5s {sum_seq_dg['chg_05s']:.1%} (prec {sum_seq_dg['chg_05s_prec']:.1%})  spurious {sum_seq_dg['spurious']}",
            flush=True,
        )
        print(
            f"[Ensemble GBDT+MS-TCN {ch_tag} (sans tête chg) on {fold_label}] "
            f"spellF1 {sum_ens_no['spell_f1']:.4f}  macroF1 {sum_ens_no['macro_f1']:.4f}  "
            f"short {sum_ens_no['short_recall']:.1%}  chg@0.5s {sum_ens_no['chg_05s']:.1%} (prec {sum_ens_no['chg_05s_prec']:.1%})  spurious {sum_ens_no['spurious']}",
            flush=True,
        )
        print(
            f"[Ensemble GBDT+MS-TCN {ch_tag} (+ DirGated+KinSnap) on {fold_label}] "
            f"spellF1 {sum_ens_dg['spell_f1']:.4f}  macroF1 {sum_ens_dg['macro_f1']:.4f}  "
            f"short {sum_ens_dg['short_recall']:.1%}  chg@0.5s {sum_ens_dg['chg_05s']:.1%} (prec {sum_ens_dg['chg_05s_prec']:.1%})  spurious {sum_ens_dg['spurious']}",
            flush=True,
        )

    # Global summary across all evaluated folds
    evaluated_rows = [r for r in eval_rows_all if (r["name"], r["seed"]) in oof_seq_map]
    if len(evaluated_rows) > 0:
        print(f"\n==================== GLOBAL HELD-OUT SUMMARY ({len(evaluated_rows)} runs) ====================", flush=True)
        oof_seq_list = [oof_seq_map[(r["name"], r["seed"])] for r in evaluated_rows]
        oof_tree_list = [oof_tree_map[(r["name"], r["seed"])] for r in evaluated_rows]

        if args.save_oof:
            save_dict = {}
            for r, out, (lt, ld) in zip(evaluated_rows, oof_seq_list, oof_tree_list):
                prefix = f"{r['name']}_seed{r['seed']}"
                save_dict[f"{prefix}_s_team"] = out["s_team_raw"]
                save_dict[f"{prefix}_s_dead"] = out["s_dead_raw"]
                save_dict[f"{prefix}_p_change"] = out["p_change"]
                save_dict[f"{prefix}_p_dead_edge"] = out["p_dead_edge"]
                save_dict[f"{prefix}_tree_team"] = lt.astype(np.float32)
                save_dict[f"{prefix}_tree_dead"] = ld.astype(np.float32)
            np.savez_compressed(args.save_oof, **save_dict)
            print(f"Saved OOF MS-TCN + GBDT predictions -> {args.save_oof}", flush=True)

        std_setting = {"switch": 22.0, "to_dead": 60.0, "restart": 60.0, "t_team": 0.7, "t_dead": 0.7}
        s1 = summarize_scores(evaluate_seq(evaluated_rows, oof_seq_list, std_setting), evaluated_rows)
        s2 = summarize_scores(
            evaluate_seq(evaluated_rows, oof_seq_list, std_setting, change_weight=10.0, use_dir_gate=True, use_kin_snap=include_kin),
            evaluated_rows,
        )
        ens_cfg = dict(
            ensemble.DEFAULT_ENSEMBLE,
            w_tree_team=0.25,
            w_seq_team=0.75,
            w_tree_dead=0.30,
            w_seq_dead=0.70,
            scale_tree_team=1.0,
            scale_seq_team=0.7,
            scale_tree_dead=0.7,
            scale_seq_dead=0.7,
            boundary_gate_trees=True,
            boundary_band=1.5,
        )
        ens_costs_no = {"switch": 22.0, "to_dead": 60.0, "restart": 60.0, "change_alpha": 0.0}
        ens_costs_dg = dict(
            ens_costs_no,
            change_alpha=10.0,
            change_guard=4.0,
            switch_min=4.5,
            change_thresh=0.28,
            change_nms=15,
            directional_gate=True,
        )
        s3 = summarize_scores(
            evaluate_ensemble_rows(evaluated_rows, oof_tree_list, oof_seq_list, ens_cfg, ens_costs_no, use_kin_snap=include_kin),
            evaluated_rows,
        )
        s4 = summarize_scores(
            evaluate_ensemble_rows(evaluated_rows, oof_tree_list, oof_seq_list, ens_cfg, ens_costs_dg, use_kin_snap=include_kin),
            evaluated_rows,
        )

        for label, s in [
            (f"1. MS-TCN {ch_tag} seul (sans tête chg)", s1),
            (f"2. MS-TCN {ch_tag} seul (+ DirGated+KinSnap)", s2),
            (f"3. Ensemble GBDT v7 + MS-TCN {ch_tag} (sans tête chg)", s3),
            (f"4. Ensemble GBDT v7 + MS-TCN {ch_tag} (+ DirGated+KinSnap)", s4),
        ]:
            print(
                f"{label}:\n"
                f"   spellF1 {s['spell_f1']:.4f} (Metrica {s['metrica_f1']:.4f}, DFL {s['dfl_f1']:.4f})  "
                f"macroF1 {s['macro_f1']:.4f}  short {s['short_recall']:.1%}  "
                f"chg@0.5s {s['chg_05s']:.1%} (prec {s['chg_05s_prec']:.1%})  "
                f"chg@1s {s['chg_10s']:.1%}  spurious {s['spurious']}",
                flush=True,
            )

        if args.write_weights:
            print("\nFitting final full MS-TCN model on all bench + PFF games...", flush=True)
            full_train_rows = [r for r in all_rows if r["seed"] < args.train_seeds] + pff_rows
            final_params = train_mstcn(
                full_train_rows,
                steps=args.steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                seed=100,
                swap_aug=True,
                edge_aware_smooth=include_kin,
                use_segment_iou=args.segment_iou,
            )
            seq_npz_path = os.path.join(os.path.dirname(model.WEIGHTS), "weights_seq.npz")
            np.savez_compressed(seq_npz_path, **final_params)

            best_s = max([s1, s2, s3, s4], key=lambda x: x["spell_f1"])
            new_weights = {
                "kind": "ensemble",
                "tree_model_file": "weights_trees.json",
                "seq_model_file": "weights_seq.npz",
                "team_variant": "duel_past",
                "dead_variant": "stat",
                "t_team": 1.0,
                "t_dead": 1.0,
                "ensemble": ens_cfg,
                "costs": ens_costs_no if s3["spell_f1"] >= s4["spell_f1"] else ens_costs_dg,
                "held_out_spell_f1": float(best_s["spell_f1"]),
                "trained_on": (
                    "metrica-sports/sample-data games 1-3 + IDSSE open Bundesliga matches (7 games) "
                    "+ PFF FC FIFA World Cup 2022 open dataset, degraded"
                ),
            }
            with open(model.WEIGHTS, "w") as handle:
                json.dump(new_weights, handle, indent=1)
            print(
                f"Wrote updated ensemble weights -> {os.path.relpath(model.WEIGHTS, ROOT)} "
                f"and {os.path.relpath(seq_npz_path, ROOT)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
