"""Two-stage Dilated Residual Temporal Convolutional Network (MS-TCN) for possession emissions.

Pure NumPy inference (no PyTorch dependency at inference time, Python >= 3.8 compatible).
Weights are stored in `.npz` format alongside `weights.json` (with `kind: "seq"`).

Architecture:
  - Inputs (~58 channels):
      * dead_features(..., variant="stat")          [16 cols]
      * team_features(..., variant="duel_past")     [26 cols]
      * direct frame signals from features.compute  [16 cols]
    Split into symmetric channels (dead + unsigned frame signals) and antisymmetric
    channels (team + signed frame signals, negated under H <-> A swap).
  - Stage 1: 1x1 projection -> 6 dilated residual blocks (64 channels, kernel 5, dilations 1..32,
             receptive field +/- 10 s at 25 fps) -> 1x1 head producing 5 logits per frame:
             [logit_H, logit_A, logit_D, logit_change, logit_dead_edge].
  - Stage 2: takes Concatenate([softmax(H, A, D), sigmoid(change), sigmoid(dead_edge)]) (5 channels)
             -> 1x1 projection -> 4 dilated residual blocks (48 channels, kernel 5, dilations 1..8)
             -> 1x1 head producing refined 5 logits per frame.
"""

import os
import numpy as np

from possession import emissions as em_mod
from possession import model as lin_model

FPS = 25
STAGE1_CHANNELS = 64
STAGE1_DILATIONS = (1, 2, 4, 8, 16, 32)
STAGE2_CHANNELS = 48
STAGE2_DILATIONS = (1, 2, 4, 8)
KERNEL_SIZE = 5


def _fill(x, val=0.0):
    return np.where(np.isfinite(x), x, val)


def raw_signal_columns(f, blocks, pitch):
    """16 per-frame signals directly from features.compute(atd, blocks, pitch).

    Returns (sym_cols, anti_cols):
      - sym_cols (8 columns): invariant to H <-> A swap
      - anti_cols (8 columns): negated under H <-> A swap
    """
    n = len(f["ball_seen"])
    length, width = pitch
    seen = f["ball_seen"].astype(np.float64)
    ok = f["ball_ok"].astype(np.float64)
    out = f["ball_out"].astype(np.float64)
    stat = f["stationary"].astype(np.float64) if "stationary" in f else np.zeros(n)
    speed = np.log1p(np.clip(_fill(f["ball_speed"].astype(np.float64), 0.0), 0.0, 45.0))
    disp = np.log1p(np.clip(_fill(f["ball_disp"].astype(np.float64), 0.0), 0.0, 30.0))

    d_h = f["d_home"].astype(np.float64)
    d_a = f["d_away"].astype(np.float64)
    dmin = np.minimum(d_h, d_a)
    dmin_log = np.where(f["ball_ok"] & np.isfinite(dmin), np.log1p(np.clip(dmin, 0.0, 30.0)), np.log1p(10.0))

    nh = f["n_home_near"].astype(np.float64)
    na = f["n_away_near"].astype(np.float64)
    crowd_tot = np.where(f["ball_ok"], (nh + na) / 6.0, 0.0)

    sym_cols = [
        seen,
        ok,
        out,
        stat,
        speed,
        disp,
        dmin_log,
        crowd_tot,
    ]

    # Antisymmetric signals (flip sign when swapping Home <-> Away)
    ctrl = f["control"].astype(np.float64)
    t1 = f["touch"].astype(np.float64)
    t2 = f["touch2"].astype(np.float64)
    with np.errstate(invalid="ignore"):
        d_diff = np.where(
            f["ball_ok"] & np.isfinite(d_h) & np.isfinite(d_a),
            np.tanh((d_a - d_h) / 3.0),
            0.0,
        )
    near_diff = np.where(f["ball_ok"], (nh - na) / 4.0, 0.0)
    shape = np.tanh(_fill(f["shape_adv"].astype(np.float64), 0.0) / 8.0)

    sign = f["attack_sign_home"].astype(np.float64)
    bx = f["ball"][:, 0].astype(np.float64)
    terr = np.where(f["ball_ok"], ((bx - length / 2.0) / (length / 2.0)) * sign, 0.0)

    # Signed ball velocity along attacking axis
    vx_signed = np.zeros(n, dtype=np.float64)
    for _, lo, hi in blocks:
        x = bx[lo:hi + 1]
        if len(x) > 4:
            v = np.zeros(len(x), dtype=np.float64)
            v[2:-2] = (x[4:] - x[:-4]) * (FPS / 4.0)
            vx_signed[lo:hi + 1] = np.tanh(_fill(v, 0.0) * sign[lo:hi + 1] / 10.0)

    anti_cols = [
        ctrl,
        t1,
        t2,
        d_diff,
        near_diff,
        shape,
        _fill(terr, 0.0),
        vx_signed,
    ]
    return np.column_stack(sym_cols), np.column_stack(anti_cols)


def seq_features(f, blocks, pitch, ball_evidence=None, dead_variant="stat", team_variant="duel_past"):
    """Build the full 58-channel feature matrix (n_frames, 58) in float32, plus antisymmetric mask.

    Never uses set-piece tags (to avoid bench leakage).
    """
    if ball_evidence is None:
        ball_evidence = em_mod.ball_team_evidence(f, em_mod.PARAMS) / em_mod.PARAMS["w_ball"]
    Xd = lin_model.dead_features(f, blocks, pitch, dead_variant)        # 16 symmetric cols
    Xt = lin_model.team_features(f, blocks, pitch, ball_evidence, team_variant)  # 26 antisymmetric cols
    X_sym, X_anti = raw_signal_columns(f, blocks, pitch)                # 8 sym + 8 anti cols

    X = np.column_stack([Xd, X_sym, Xt, X_anti]).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def antisym_mask(n_dead=16, n_sym_raw=8, n_team=26, n_anti_raw=8):
    """Boolean mask (58,) True on channels that negate when swapping Home <-> Away."""
    mask = np.zeros(n_dead + n_sym_raw + n_team + n_anti_raw, dtype=bool)
    mask[n_dead + n_sym_raw:] = True
    return mask


# ------------------------------------------------------------------------- Pure NumPy MS-TCN
def _relu(x):
    return np.maximum(x, 0.0)


def _sigmoid(x):
    x_clip = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def _softmax3(logits3):
    """logits3: (3, T) -> (3, T) probabilities."""
    m = logits3.max(axis=0, keepdims=True)
    e = np.exp(logits3 - m)
    return e / e.sum(axis=0, keepdims=True)


def _conv1d_1x1(x, weight, bias):
    """x: (C_in, T), weight: (C_out, C_in, 1) or (C_out, C_in), bias: (C_out,)."""
    w = weight.reshape(weight.shape[0], weight.shape[1])
    return w @ x + bias[:, None]


def _conv1d_dilated(x, weight, bias, dilation):
    """1D dilated convolution with 'same' zero-padding.
    x: (C_in, T) float32
    weight: (C_out, C_in, K) float32
    bias: (C_out,) float32
    Returns: (C_out, T) float32
    """
    c_out, c_in, k = weight.shape
    _, length = x.shape
    pad = (k // 2) * dilation
    out = np.broadcast_to(bias[:, None], (c_out, length)).astype(np.float32).copy()
    for j in range(k):
        shift = (j - k // 2) * dilation
        w_j = weight[:, :, j]  # (C_out, C_in)
        if shift == 0:
            out += w_j @ x
        elif shift < 0:
            # source index t + shift in [0, length + shift) -> writes to [-shift : length]
            s = -shift
            if s < length:
                out[:, s:] += w_j @ x[:, :length - s]
        else:
            # source index t + shift in [shift, length) -> writes to [0 : length - shift]
            if shift < length:
                out[:, :length - shift] += w_j @ x[:, shift:]
    return out


def _forward_stage(x, params, prefix, dilations):
    """Forward one TCN stage in pure NumPy.
    x: (C_in, T) float32
    Returns logits: (5, T) float32
    """
    h = _conv1d_1x1(x, params[f"{prefix}.in_proj.weight"], params[f"{prefix}.in_proj.bias"])
    for i, d in enumerate(dilations):
        z = _conv1d_dilated(
            h,
            params[f"{prefix}.blocks.{i}.conv_dilated.weight"],
            params[f"{prefix}.blocks.{i}.conv_dilated.bias"],
            d,
        )
        z = _relu(z)
        z = _conv1d_1x1(
            z,
            params[f"{prefix}.blocks.{i}.conv_1x1.weight"],
            params[f"{prefix}.blocks.{i}.conv_1x1.bias"],
        )
        h = h + z
    logits = _conv1d_1x1(h, params[f"{prefix}.out_proj.weight"], params[f"{prefix}.out_proj.bias"])
    return logits


def forward_block_numpy(X_block, params):
    """Run the 2-stage MS-TCN on a single contiguous half block.
    X_block: (T, C_in) float32
    Returns (stage1_logits, stage2_logits), each (T, 5) float32:
      cols = [logit_H, logit_A, logit_D, logit_change, logit_dead_edge]
    """
    mu = params["norm_mu"]   # (C_in,)
    sd = params["norm_sd"]   # (C_in,)
    x_norm = ((X_block.astype(np.float32) - mu[None, :]) / sd[None, :]).T  # (C_in, T)

    logits1 = _forward_stage(x_norm, params, "stage1", STAGE1_DILATIONS)   # (5, T)
    probs_had = _softmax3(logits1[:3])                                     # (3, T)
    prob_chg = _sigmoid(logits1[3:4])                                      # (1, T)
    prob_de = _sigmoid(logits1[4:5])                                       # (1, T)
    stage2_in = np.concatenate([probs_had, prob_chg, prob_de], axis=0).astype(np.float32)
    logits2 = _forward_stage(stage2_in, params, "stage2", STAGE2_DILATIONS)  # (5, T)
    return logits1.T, logits2.T


def predict_game_numpy(X, blocks, params):
    """Predict logits across all graded blocks of a game.
    Returns dict with:
      - s_team_raw: (N,) float32 = logit_H - logit_A
      - s_dead_raw: (N,) float32 = logit_D - logaddexp(logit_H, logit_A)
      - p_change:   (N,) float32 = sigmoid(logit_change)
      - p_dead_edge:(N,) float32 = sigmoid(logit_dead_edge)
    """
    n = len(X)
    s_team_raw = np.zeros(n, dtype=np.float32)
    s_dead_raw = np.zeros(n, dtype=np.float32)
    p_change = np.zeros(n, dtype=np.float32)
    p_dead_edge = np.zeros(n, dtype=np.float32)

    for _, lo, hi in blocks:
        _, l2 = forward_block_numpy(X[lo:hi + 1], params)
        lh, la, ld, lchg, lde = l2[:, 0], l2[:, 1], l2[:, 2], l2[:, 3], l2[:, 4]
        s_team_raw[lo:hi + 1] = lh - la
        s_dead_raw[lo:hi + 1] = ld - np.logaddexp(lh, la)
        p_change[lo:hi + 1] = _sigmoid(lchg)
        p_dead_edge[lo:hi + 1] = _sigmoid(lde)

    return {
        "s_team_raw": s_team_raw,
        "s_dead_raw": s_dead_raw,
        "p_change": p_change,
        "p_dead_edge": p_dead_edge,
    }


def load_seq_params(npz_path):
    with np.load(npz_path) as data:
        return {k: data[k].astype(np.float32) for k in data.files}


def emissions_from_weights(f, blocks, pitch, ball_evidence, weights):
    """Compute (s_team, s_dead) and optional dynamic change bonus from `weights` with kind=='seq'."""
    params = weights["seq_params"]
    X = seq_features(
        f,
        blocks,
        pitch,
        ball_evidence,
        dead_variant=weights.get("dead_variant", "stat"),
        team_variant=weights.get("team_variant", "duel_past"),
    )
    out = predict_game_numpy(X, blocks, params)
    s_team = (out["s_team_raw"] * float(weights.get("t_team", 1.0))).astype(np.float32)
    s_dead = (out["s_dead_raw"] * float(weights.get("t_dead", 1.0))).astype(np.float32)
    return s_team, s_dead, out["p_change"]
