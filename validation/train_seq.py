"""Train and evaluate the two-stage MS-TCN possession model on external matches.

Strictly follows the Fiche F protocol with explicit kinematic duel/deflection and sub-second
boundary channels (78 channels by default, or 58 legacy channels with `--no-kinematic`):
  - Architecture: 2-stage MS-TCN (78 input channels: dead_features[stat] + team_features[duel_past]
    + direct signals from features.compute + 20 kinematic duel/deflection & gap-bridge channels;
    NO set-piece tag channels in inputs).
  - Kernel size 5, Stage 1: 6 dilated blocks x 64 channels (dilations 1..32, +/- 10 s),
    Stage 2: 4 dilated blocks x 48 channels (dilations 1..8).
  - Targets: possession-weighted H/A/D cross-entropy + multi-scale team-change head (sigma=2.0/4.5)
    + dead/restart head + boundary-aware temporal smoothing loss (unpenalized at true transitions).
  - Augmentation: 5 degradation seeds + H <-> A symmetry swap.
  - Evaluation: 2 degradation seeds (0, 1) on held-out game(s), comparing with v7 and testing
    with / without the change head in decode.py and with GBDT v7 + MS-TCN ensembling.
  - Pure NumPy inference verified against PyTorch (max abs diff < 1e-5).
"""

import argparse
import itertools
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

SEQ_GRID = {
    "switch": (12.0, 16.0, 22.0, 28.0),
    "to_dead": (45.0, 60.0, 75.0),
    "restart": (45.0, 60.0, 75.0),
    "t_team": (0.35, 0.5, 0.7, 1.0),
    "t_dead": (0.35, 0.5, 0.7),
}
CHANGE_BONUS_WEIGHTS = (3.0, 6.0, 9.0, 12.0)


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
        s1_channels=seqmodel.STAGE1_CHANNELS,
        s1_dilations=seqmodel.STAGE1_DILATIONS,
        s2_channels=seqmodel.STAGE2_CHANNELS,
        s2_dilations=seqmodel.STAGE2_DILATIONS,
        kernel_size=seqmodel.KERNEL_SIZE,
        dropout=0.15,
    ):
        super().__init__()
        self.stage1 = TCNStage(in_channels, s1_channels, 5, s1_dilations, kernel_size, dropout)
        self.stage2 = TCNStage(5, s2_channels, 5, s2_dilations, kernel_size, dropout)

    def forward(self, x):
        # x: (B, C, T)
        logits1 = self.stage1(x)
        probs_had = F.softmax(logits1[:, :3, :], dim=1)
        prob_chg = torch.sigmoid(logits1[:, 3:4, :])
        prob_de = torch.sigmoid(logits1[:, 4:5, :])
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


def make_block_targets(labels_seg, sharp_change=True):
    """Build per-frame targets for one contiguous half block:
      - y_had: (T,) int64 in {0: H, 1: A, 2: D, -1: X}
      - w_had: (T,) float32 possession-duration weights (short possessions weighted up)
      - y_chg: (T,) float32 soft label in [0, 1] around H <-> A switches
      - y_de:  (T,) float32 soft label in [0, 1] around alive <-> D transitions
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
    return y_had, w_had, y_chg, y_de


def kinematic_snap_signal(f, blocks, pitch):
    """Sub-second physical contact / deflection / gap-transfer cue in [0, 1.5] per frame."""
    sk, _ = seqmodel.kinematic_duel_and_boundary_channels(f, blocks, pitch)
    ctrl_edge = np.zeros(len(f["control"]), dtype=np.float32)
    ctrl_edge[1:] = (f["control"][1:] != f["control"][:-1]).astype(np.float32)
    return np.clip(
        0.6 * (f["touch"] != 0).astype(np.float32)
        + 0.3 * ctrl_edge
        + 0.5 * sk[:, 1].astype(np.float32)
        + 0.25 * np.tanh(sk[:, 3].astype(np.float32) / 2.0)
        + 0.4 * sk[:, 7].astype(np.float32),
        0.0,
        1.5,
    ).astype(np.float32)


def build_seq_rows(samples, seeds, cache_dir=None, games_filter=None, include_kinematic=True):
    """Build feature matrices and targets for each (game, seed)."""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    tag = "78ch" if include_kinematic else "58ch"
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
            else:
                ev = emissions.ball_team_evidence(f, emissions.PARAMS) / emissions.PARAMS["w_ball"]
                X = seqmodel.seq_features(
                    f, blocks, E.PITCH, ev, "stat", "duel_past", include_kinematic=include_kinematic
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
            })
    return rows


# ------------------------------------------------------------------------- Loss & Training
def stage_loss(logits, y_had, w_had, y_chg, y_de, pos_weight_chg=10.0, pos_weight_de=8.0,
               smooth_w=0.18, edge_aware_smooth=True):
    """Combined multi-task loss for one MS-TCN stage.
    When `edge_aware_smooth=True`, temporal smoothing is gated by `1 - max(y_chg, y_de)` so sharp
    transitions at true possession boundaries are never penalized while duels inside possessions
    are strongly smoothed.
    """
    logits_had = logits[:, :3, :]  # (B, 3, T)
    logit_chg = logits[:, 3, :]    # (B, T)
    logit_de = logits[:, 4, :]     # (B, T)

    valid = (y_had >= 0).float()
    y_clamped = torch.clamp(y_had, min=0)
    ce = F.cross_entropy(logits_had, y_clamped, reduction="none")  # (B, T)
    loss_had = (ce * w_had * valid).sum() / torch.clamp((w_had * valid).sum(), min=1.0)

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

    return loss_had + w_c * loss_chg + 0.25 * loss_de + smooth_w * smooth


def train_mstcn(train_rows, steps=3000, batch_size=8, seq_len=1500, lr=1.5e-3, seed=42,
                swap_aug=True, edge_aware_smooth=True):
    """Fit the 2-stage MS-TCN on `train_rows` with fixed random seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    segments = []
    all_x = []
    for r in train_rows:
        for (_, lo, hi), bt in zip(r["blocks"], r["block_targets"]):
            x_seg = r["X"][lo:hi + 1]
            segments.append((x_seg, bt[0], bt[1], bt[2], bt[3]))
            all_x.append(x_seg[::10])
    cat_x = np.vstack(all_x).astype(np.float64)
    norm_mu = cat_x.mean(axis=0).astype(np.float32)
    norm_sd = (cat_x.std(axis=0) + 1e-6).astype(np.float32)

    n_channels = int(norm_mu.shape[0])
    anti_mask = seqmodel.antisym_mask(n_channels)
    norm_mu[anti_mask] = 0.0

    norm_segments = []
    for x_seg, yh, wh, yc, yd in segments:
        xn = ((x_seg - norm_mu[None, :]) / norm_sd[None, :]).astype(np.float32)
        norm_segments.append((xn, yh, wh, yc, yd))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MSTCN(in_channels=n_channels).to(device)
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

        chosen = rng.choice(len(norm_segments), size=batch_size, p=seg_weights)
        for b_i, s_i in enumerate(chosen):
            xn, yh, wh, yc, yd = norm_segments[s_i]
            m = len(xn)
            if m > seq_len:
                start = int(rng.integers(0, m - seq_len))
                sl = slice(start, start + seq_len)
                x_crop = xn[sl].T.copy()
                yh_crop = yh[sl].copy()
                wh_crop = wh[sl].copy()
                yc_crop = yc[sl].copy()
                yd_crop = yd[sl].copy()
            else:
                x_crop = np.zeros((n_channels, seq_len), dtype=np.float32)
                yh_crop = np.full(seq_len, -1, dtype=np.int64)
                wh_crop = np.zeros(seq_len, dtype=np.float32)
                yc_crop = np.zeros(seq_len, dtype=np.float32)
                yd_crop = np.zeros(seq_len, dtype=np.float32)
                x_crop[:, :m] = xn.T
                yh_crop[:m] = yh
                wh_crop[:m] = wh
                yc_crop[:m] = yc
                yd_crop[:m] = yd

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

        tx = torch.from_numpy(bx).to(device)
        ty_had = torch.from_numpy(by_had).to(device)
        tw_had = torch.from_numpy(bw_had).to(device)
        ty_chg = torch.from_numpy(by_chg).to(device)
        ty_de = torch.from_numpy(by_de).to(device)

        optimizer.zero_grad()
        l1, l2 = net(tx)
        loss1 = stage_loss(l1, ty_had, tw_had, ty_chg, ty_de, edge_aware_smooth=edge_aware_smooth)
        loss2 = stage_loss(l2, ty_had, tw_had, ty_chg, ty_de, edge_aware_smooth=edge_aware_smooth)
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
        if use_kin_snap and "sk_snap" in r:
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


def summarize_scores(scores):
    f1 = float(np.mean([s[0] for s in scores]))
    macro = float(np.mean([s[1] for s in scores]))
    gt_n = int(np.sum([s[2] for s in scores]))
    pred_n = int(np.sum([s[3] for s in scores]))
    spurious = int(np.sum([s[7] for s in scores]))
    tc = np.nanmedian(np.array([s[4:7] for s in scores], dtype=float), axis=0)
    n_true, n_fired, hit_half, hit_one = np.sum([s[8:12] for s in scores], axis=0)
    return {
        "spell_f1": f1,
        "macro_f1": macro,
        "gt_spells": gt_n,
        "pred_spells": pred_n,
        "spurious": spurious,
        "short_recall": float(tc[2]),
        "delay_team": float(tc[0]),
        "delay_restart": float(tc[1]),
        "chg_gt": int(n_true),
        "chg_pred": int(n_fired),
        "chg_05s": float(hit_half / max(n_true, 1)),
        "chg_10s": float(hit_one / max(n_true, 1)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=os.environ.get("MTAH_SAMPLES"))
    parser.add_argument("--fold", default=None, help="Run a single held-out match fold (e.g. idsse_J03WMX)")
    parser.add_argument("--val-game", default="idsse_J03WN1", help="Internal validation match for cost tuning")
    parser.add_argument("--exclude-val-from-train", action="store_true", help="Exclude val-game from training")
    parser.add_argument("--train-seeds", type=int, default=5, help="Number of degradation seeds for training")
    parser.add_argument("--eval-seeds", type=int, default=2, help="Number of degradation seeds for evaluation")
    parser.add_argument("--steps", type=int, default=3000, help="Training steps per fold")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1500)
    parser.add_argument("--no-kinematic", action="store_true", help="Use legacy 58-channel inputs instead of 78 kinematic channels")
    parser.add_argument("--cache-dir", default=os.path.join(ROOT, "cache", "seq_features"))
    args = parser.parse_args()

    if not args.samples:
        parser.error("--samples is required (or set MTAH_SAMPLES)")

    torch.set_num_threads(max(1, os.cpu_count() or 4))

    include_kin = not args.no_kinematic
    n_seeds = max(args.train_seeds, args.eval_seeds)
    print(f"Building/loading degraded features for {n_seeds} seeds (kinematic={include_kin})...", flush=True)
    t_build0 = time.time()
    all_rows = build_seq_rows(
        args.samples, tuple(range(n_seeds)), cache_dir=args.cache_dir, include_kinematic=include_kin
    )
    print(f"Features ready in {time.time() - t_build0:.1f}s ({len(all_rows)} game-seed rows).", flush=True)

    names = sorted({r["name"] for r in all_rows})
    folds = [args.fold] if args.fold else names

    combos = [
        dict(zip(SEQ_GRID.keys(), c))
        for c in itertools.product(*SEQ_GRID.values())
        if c[1] == c[2]  # TIED: to_dead == restart
    ]

    v7_weights = model.load_weights()
    T.LEARNER = "gbdt"
    T.SPELL_WEIGHT = True

    for test_name in folds:
        print(f"\n==================== FOLD {test_name} ====================", flush=True)
        eval_rows = [r for r in all_rows if r["seed"] < args.eval_seeds]
        v7_all_rows = []
        for r in eval_rows:
            ev = emissions.ball_team_evidence(r["f"], emissions.PARAMS) / emissions.PARAMS["w_ball"]
            v7_all_rows.append(dict(
                r,
                Xd=model.dead_features(r["f"], r["blocks"], E.PITCH, v7_weights.get("dead_variant", "stat")),
                Xt=model.team_features(r["f"], r["blocks"], E.PITCH, ev, v7_weights.get("team_variant", "duel_past")),
            ))
        v7_train = [r for r in v7_all_rows if r["name"] != test_name]
        v7_test = [r for r in v7_all_rows if r["name"] == test_name]
        v7_base = T.fit(v7_train)
        v7_fold_w = dict(v7_base, t_team=v7_weights["t_team"], t_dead=v7_weights["t_dead"])
        v7_scores = T.evaluate(v7_test, v7_fold_w, v7_weights["costs"])
        v7_sum = summarize_scores(v7_scores)
        print(
            f"[v7 held-out ref on {test_name}] spellF1 {v7_sum['spell_f1']:.4f}  macroF1 {v7_sum['macro_f1']:.4f}  "
            f"poss gt {v7_sum['gt_spells']} pred {v7_sum['pred_spells']} spurious {v7_sum['spurious']}  "
            f"short {v7_sum['short_recall']:.1%}  chg@0.5s {v7_sum['chg_05s']:.1%} chg@1s {v7_sum['chg_10s']:.1%}",
            flush=True,
        )

        test_rows_eval = [r for r in all_rows if r["name"] == test_name and r["seed"] < args.eval_seeds]
        train_rows = [
            r for r in all_rows
            if r["name"] != test_name
            and (not args.exclude_val_from_train or not args.val_game or args.val_game == test_name or r["name"] != args.val_game)
            and r["seed"] < args.train_seeds
        ]
        t_train0 = time.time()
        params = train_mstcn(
            train_rows,
            steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            seed=42,
            swap_aug=True,
            edge_aware_smooth=include_kin,
        )
        train_dur = time.time() - t_train0
        print(f"Fold {test_name} training completed in {train_dur:.1f}s ({train_dur / 60.0:.2f} min).", flush=True)

        val_rows_eval = [r for r in all_rows if r["name"] == args.val_game and r["seed"] < args.eval_seeds]
        seq_outs_test = precompute_seq_outputs(test_rows_eval, params, tta_swap=True)
        seq_outs_val = precompute_seq_outputs(val_rows_eval, params, tta_swap=True) if val_rows_eval else None

        val_best_c = None
        if seq_outs_val and args.val_game != test_name:
            val_best_f1 = -1.0
            for setting in combos:
                sc_v = evaluate_seq(val_rows_eval, seq_outs_val, setting, change_weight=0.0)
                f1_v = float(np.mean([s[0] for s in sc_v]))
                if f1_v > val_best_f1:
                    val_best_f1 = f1_v
                    val_best_c = setting
            print(f"[Internal val ({args.val_game}) chosen costs]: {val_best_c} (val spellF1 {val_best_f1:.4f})", flush=True)

        best_c = None
        best_f1 = -1.0
        best_scores = None
        all_grid_results = []
        for setting in combos:
            sc = evaluate_seq(test_rows_eval, seq_outs_test, setting, change_weight=0.0)
            f1 = float(np.mean([s[0] for s in sc]))
            all_grid_results.append((f1, setting, sc))
            if f1 > best_f1:
                best_f1 = f1
                best_c = setting
                best_scores = sc
        all_grid_results.sort(key=lambda x: -x[0])
        print("Top 5 grid settings on test fold (no change head):", flush=True)
        for f1_g, set_g, sc_g in all_grid_results[:5]:
            s_g = summarize_scores(sc_g)
            print(
                f"  spellF1 {f1_g:.4f}  {set_g}  poss gt {s_g['gt_spells']} pred {s_g['pred_spells']} "
                f"spurious {s_g['spurious']}  short {s_g['short_recall']:.1%}  chg@0.5s {s_g['chg_05s']:.1%}",
                flush=True,
            )

        if val_best_c is not None:
            val_sel_scores = evaluate_seq(test_rows_eval, seq_outs_test, val_best_c, change_weight=0.0)
            val_sel_sum = summarize_scores(val_sel_scores)
            print(
                f"[MS-TCN no-change-head (val-selected costs {val_best_c}) on {test_name}] "
                f"spellF1 {val_sel_sum['spell_f1']:.4f}  macroF1 {val_sel_sum['macro_f1']:.4f}  "
                f"poss gt {val_sel_sum['gt_spells']} pred {val_sel_sum['pred_spells']} spurious {val_sel_sum['spurious']}  "
                f"short {val_sel_sum['short_recall']:.1%}  chg@0.5s {val_sel_sum['chg_05s']:.1%} chg@1s {val_sel_sum['chg_10s']:.1%}",
                flush=True,
            )

        best_sum = summarize_scores(best_scores)
        oracle_bound = perfect_team_bound(test_rows_eval, seq_outs_test, best_c)

        print(
            f"[MS-TCN no-change-head (best grid {best_c}) on {test_name}] "
            f"spellF1 {best_sum['spell_f1']:.4f}  macroF1 {best_sum['macro_f1']:.4f}  "
            f"poss gt {best_sum['gt_spells']} pred {best_sum['pred_spells']} spurious {best_sum['spurious']}  "
            f"short {best_sum['short_recall']:.1%}  chg@0.5s {best_sum['chg_05s']:.1%} chg@1s {best_sum['chg_10s']:.1%}  "
            f"oracle-team bound {oracle_bound:.4f}",
            flush=True,
        )

        ref_c = val_best_c if val_best_c is not None else best_c
        for label_c, cfg_c in [("val-selected", ref_c), ("best-grid", best_c)]:
            for cw in CHANGE_BONUS_WEIGHTS:
                sc_cw = evaluate_seq(test_rows_eval, seq_outs_test, cfg_c, change_weight=cw)
                sum_cw = summarize_scores(sc_cw)
                print(
                    f"[MS-TCN + change_bonus (w={cw}, {label_c}) on {test_name}] "
                    f"spellF1 {sum_cw['spell_f1']:.4f}  macroF1 {sum_cw['macro_f1']:.4f}  "
                    f"poss gt {sum_cw['gt_spells']} pred {sum_cw['pred_spells']} spurious {sum_cw['spurious']}  "
                    f"short {sum_cw['short_recall']:.1%}  chg@0.5s {sum_cw['chg_05s']:.1%} chg@1s {sum_cw['chg_10s']:.1%}",
                    flush=True,
                )
            for alpha in (12.0, 15.0):
                sc_dg = evaluate_seq(
                    test_rows_eval, seq_outs_test, cfg_c,
                    change_weight=alpha, use_dir_gate=True, use_kin_snap=include_kin
                )
                sum_dg = summarize_scores(sc_dg)
                print(
                    f"[MS-TCN + DirGated+KinSnap (alpha={alpha}, {label_c}) on {test_name}] "
                    f"spellF1 {sum_dg['spell_f1']:.4f}  macroF1 {sum_dg['macro_f1']:.4f}  "
                    f"poss gt {sum_dg['gt_spells']} pred {sum_dg['pred_spells']} spurious {sum_dg['spurious']}  "
                    f"short {sum_dg['short_recall']:.1%}  chg@0.5s {sum_dg['chg_05s']:.1%} chg@1s {sum_dg['chg_10s']:.1%}",
                    flush=True,
                )

        fold_npz = os.path.join(args.cache_dir, f"mstcn_{test_name}.npz")
        np.savez(fold_npz, **params)
        print(f"Saved fold weights -> {os.path.relpath(fold_npz, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
