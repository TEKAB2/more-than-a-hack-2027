"""Two-stage Dilated Residual Temporal Convolutional Network (MS-TCN) for possession emissions.

Pure NumPy inference (no PyTorch dependency at inference time, Python >= 3.8 compatible).
Weights are stored in `.npz` format alongside `weights.json` (with `kind: "seq"`).

Architecture:
  - Inputs (78 channels by default, or 58 legacy channels when `include_kinematic=False`):
      * dead_features(..., variant="stat")                     [16 symmetric cols]
      * raw_signal_columns(f, blocks, pitch)[0]                [ 8 symmetric cols]
      * kinematic_duel_and_boundary_channels(...)[0]           [ 8 symmetric cols]
      * team_features(..., variant="duel_past")                [26 antisymmetric cols]
      * raw_signal_columns(f, blocks, pitch)[1]                [ 8 antisymmetric cols]
      * kinematic_duel_and_boundary_channels(...)[1]           [12 antisymmetric cols]
    Split into 32 symmetric channels (invariant to H <-> A swap) and 46 antisymmetric
    channels (negated under H <-> A swap). Zero set-piece tag channels (no bench leakage).
  - Stage 1: 1x1 projection -> 6 dilated residual blocks (64 channels, kernel 5, dilations 1..32,
             receptive field +/- 10 s at 25 fps) -> 1x1 head producing 5 logits per frame:
             [logit_H, logit_A, logit_D, logit_change, logit_dead_edge].
  - Stage 2: takes Concatenate([softmax(H, A, D), sigmoid(change), sigmoid(dead_edge)]) (5 channels)
             -> 1x1 projection -> 4 dilated residual blocks (48 channels, kernel 5, dilations 1..8)
             -> 1x1 head producing refined 5 logits per frame.
"""

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


def kinematic_duel_and_boundary_channels(f, blocks, pitch):
    """20 explicit kinematic duel/deflection and sub-second boundary channels (0 set-piece leakage).

    Returns (sym_kin, anti_kin):
      - sym_kin (8 columns, invariant under H <-> A swap):
          0. deflect_norm    : arccos(cos_theta_ball) / pi in [0, 1] over +/- 0.24 s (6 frames)
          1. cos_contact     : 0.5 * (1 - cos_theta_ball) * contact_proximity in [0, 1]
          2. impulse         : log1p(||v_out - v_in||) over +/- 0.16 s (4 frames)
          3. impulse_peak    : impulse / (local_mean_1s + 0.5) * contact_proximity
          4. speed_ratio     : tanh(log((||v_out|| + 0.3) / (||v_in|| + 0.3)))
          5. gap_start       : exp(-age_since_ball_lost / 10) inside missing-ball gaps
          6. gap_end         : exp(-frames_until_ball_seen / 10) inside missing-ball gaps
          7. transfer_bridge : rate of change of interpolated team ownership inside missing-ball gaps
      - anti_kin (12 columns, negated under H <-> A swap):
          0. nt_05           : mean touch sign in [t+1, t+12] (0.5 s ahead)
          1. nt_10           : mean touch sign in [t+1, t+25] (1.0 s ahead)
          2. nt_20           : mean touch sign in [t+1, t+50] (2.0 s ahead)
          3. sign_n          : exponentially decayed sign of the exact first touch strictly after t
          4. nc_05           : mean control sign in [t+1, t+12] (0.5 s ahead)
          5. nc_10           : mean control sign in [t+1, t+25] (1.0 s ahead)
          6. pc_05           : mean control sign in [t-12, t] (0.5 s behind)
          7. ctrl_diff       : nc_05 - pc_05 (sharp step at reception/recovery)
          8. sign_l          : exponentially decayed sign of the last touch at or before t
          9. m_fut05         : mean of tanh((d_away - d_home) / 2.0) in [t+4, t+14] (~0.36 s ahead)
         10. m_fut08         : mean of tanh((d_away - d_home) / 2.0) in [t+12, t+24] (~0.72 s ahead)
         11. bridge_team     : linearly interpolated active ownership across missing-ball gaps
    """
    n = len(f["ball_seen"])
    seen = f["ball_seen"]
    ok = f["ball_ok"]
    b = f["ball"].astype(np.float64)
    d_h = f["d_home"].astype(np.float64)
    d_a = f["d_away"].astype(np.float64)
    dmin = np.minimum(d_h, d_a)
    prox = np.where(ok & np.isfinite(dmin), np.clip((3.0 - dmin) / 3.0, 0.0, 1.0), 0.0)

    deflect_norm = np.zeros(n, dtype=np.float64)
    cos_contact = np.zeros(n, dtype=np.float64)
    impulse = np.zeros(n, dtype=np.float64)
    impulse_peak = np.zeros(n, dtype=np.float64)
    speed_ratio = np.zeros(n, dtype=np.float64)
    gap_start = np.zeros(n, dtype=np.float64)
    gap_end = np.zeros(n, dtype=np.float64)
    transfer_bridge = np.zeros(n, dtype=np.float64)
    bridge_team = np.zeros(n, dtype=np.float64)

    with np.errstate(invalid="ignore"):
        margin = np.where(ok & np.isfinite(d_h) & np.isfinite(d_a), np.tanh((d_a - d_h) / 2.0), 0.0)

    c = f["control"].astype(np.float64)
    t1 = f["touch"].astype(np.float64)

    for _, lo, hi in blocks:
        sl = slice(lo, hi + 1)
        bs = b[sl]
        m = len(bs)
        pr = prox[sl]

        k = 6
        if m > 2 * k:
            vin = bs[k:m - k] - bs[:m - 2 * k]
            vout = bs[2 * k:] - bs[k:m - k]
            n_in = np.hypot(vin[:, 0], vin[:, 1])
            n_out = np.hypot(vout[:, 0], vout[:, 1])
            valid = (n_in > 0.25) & (n_out > 0.25) & np.isfinite(n_in) & np.isfinite(n_out)
            with np.errstate(invalid="ignore", divide="ignore"):
                cos_a = np.where(
                    valid,
                    np.clip(
                        (vin[:, 0] * vout[:, 0] + vin[:, 1] * vout[:, 1])
                        / np.maximum(n_in * n_out, 1e-6),
                        -1.0,
                        1.0,
                    ),
                    1.0,
                )
            ang = np.arccos(cos_a) / np.pi
            deflect_norm[lo + k:hi + 1 - k] = ang
            cos_contact[lo + k:hi + 1 - k] = 0.5 * (1.0 - cos_a) * pr[k:m - k]
            sr = np.where(
                np.isfinite(n_in) & np.isfinite(n_out),
                np.tanh(np.log((n_out + 0.3) / (n_in + 0.3))),
                0.0,
            )
            speed_ratio[lo + k:hi + 1 - k] = sr

        ki = 4
        if m > 2 * ki:
            vi = (bs[ki:m - ki] - bs[:m - 2 * ki]) * (FPS / ki)
            vo = (bs[2 * ki:] - bs[ki:m - ki]) * (FPS / ki)
            dv = np.hypot(vo[:, 0] - vi[:, 0], vo[:, 1] - vi[:, 1])
            dv = np.nan_to_num(dv, nan=0.0, posinf=0.0, neginf=0.0)
            imp = np.log1p(np.clip(dv, 0.0, 60.0))
            impulse[lo + ki:hi + 1 - ki] = imp
            imp_full = impulse[sl]
            mean_imp = np.convolve(imp_full, np.ones(31) / 31.0, mode="same")
            impulse_peak[sl] = (imp_full / (mean_imp + 0.5)) * pr

        seen_seg = seen[sl]
        if not np.all(seen_seg):
            idx_m = np.arange(m)
            last_s = np.maximum.accumulate(np.where(seen_seg, idx_m, -1))
            next_s = np.minimum.accumulate(np.where(seen_seg, idx_m, m)[::-1])[::-1]
            in_gap = ~seen_seg
            age_lost = np.where(in_gap & (last_s >= 0), idx_m - last_s, 999)
            rem_lost = np.where(in_gap & (next_s < m), next_s - idx_m, 999)
            gap_start[sl] = np.where(in_gap, np.exp(-age_lost / 10.0), 0.0)
            gap_end[sl] = np.where(in_gap, np.exp(-rem_lost / 10.0), 0.0)

        active_sig = np.where(
            seen_seg, np.clip(0.5 * c[sl] + 0.5 * t1[sl] + 0.5 * margin[sl], -1.0, 1.0), np.nan
        )
        valid_sig = np.isfinite(active_sig) & (np.abs(active_sig) > 0.05)
        if valid_sig.sum() >= 2:
            idx_m = np.arange(m)
            interp_sig = np.interp(idx_m, idx_m[valid_sig], active_sig[valid_sig])
            bridge_team[sl] = interp_sig
            d_bridge = np.zeros(m, dtype=np.float64)
            if m > 10:
                d_bridge[5:-5] = np.abs(interp_sig[10:] - interp_sig[:-10])
            transfer_bridge[sl] = np.where(~seen_seg, d_bridge, 0.0)

    nt_05 = lin_model._fill(lin_model._shift_mean(t1, 1, 13, blocks, n), 0.0) * 12.0
    nt_10 = lin_model._fill(lin_model._shift_mean(t1, 1, 26, blocks, n), 0.0) * 12.0
    nt_20 = lin_model._fill(lin_model._shift_mean(t1, 1, 51, blocks, n), 0.0) * 12.0

    is_t = t1 != 0
    ni = lin_model._event_index(is_t, blocks, forward=True)
    li = lin_model._event_index(is_t, blocks, forward=False)
    idx_all = np.arange(n)
    age_n = np.where(ni >= 0, (ni - idx_all) / float(FPS), 8.0)
    age_l = np.where(li >= 0, (idx_all - li) / float(FPS), 8.0)
    sign_n = np.where((ni >= 0) & (age_n < 6.0), t1[np.maximum(ni, 0)] * np.exp(-age_n / 1.5), 0.0)
    sign_l = np.where((li >= 0) & (age_l < 6.0), t1[np.maximum(li, 0)] * np.exp(-age_l / 1.5), 0.0)

    nc_05 = lin_model._fill(lin_model._shift_mean(c, 1, 13, blocks, n), 0.0)
    nc_10 = lin_model._fill(lin_model._shift_mean(c, 1, 26, blocks, n), 0.0)
    pc_05 = lin_model._fill(lin_model._shift_mean(c, -12, 1, blocks, n), 0.0)
    ctrl_diff = nc_05 - pc_05

    m_fut05 = lin_model._fill(lin_model._shift_mean(margin, 4, 15, blocks, n), 0.0)
    m_fut08 = lin_model._fill(lin_model._shift_mean(margin, 12, 25, blocks, n), 0.0)

    sym_kin = np.column_stack([
        deflect_norm,
        cos_contact,
        impulse,
        impulse_peak,
        speed_ratio,
        gap_start,
        gap_end,
        transfer_bridge,
    ])
    anti_kin = np.column_stack([
        nt_05,
        nt_10,
        nt_20,
        sign_n,
        nc_05,
        nc_10,
        pc_05,
        ctrl_diff,
        sign_l,
        m_fut05,
        m_fut08,
        bridge_team,
    ])
    return sym_kin, anti_kin


def carrier_and_imputation_channels(f):
    """24 explicit channels from physics-reconstructed missing ball and 22-player carrier attribution:
      - sym_carrier (10 columns, invariant under H <-> A swap):
          0. sigma_b_norm           : log1p(sigma_b(t)) / log1p(15.0) in [0, 1]
          1. speed_rec_log          : log1p(||v_ball_rec(t)||) continuous across gaps
          2. tti_min_sym            : log1p(min(TTI_h1, TTI_a1)) along projected ball trajectory
          3. tti_conflict           : interception duel intensity when both teams arrive simultaneously
          4. carrier_max_prob       : max_j alpha_j(t) across all 22 players
          5. same_player_continuity : sum_j alpha_past_j * alpha_fut_j (+/- 0.5 s)
          6. teammate_pass_sym      : intra-team inter-player pass signature (j != k in same team)
          7. interception_sym_05    : inter-team turnover signature over +/- 0.5 s
          8. interception_sym_12    : stabilized inter-team turnover signature over +/- 1.2 s
          9. carrier_decel_sym      : control braking intensity of the active carrier
      - anti_carrier (14 columns, negated under H <-> A swap):
          0. d_rec_diff_1           : tanh((d_a1_rec - d_h1_rec) / 2.5) * w_sigma
          1. d_rec_diff_2           : tanh((d_a2_rec - d_h2_rec) / 3.5) * w_sigma (2nd support player)
          2. tti_diff_1             : tanh((TTI_a1 - TTI_h1) / 0.45) * w_sigma
          3. tti_diff_2             : tanh((TTI_a2 - TTI_h2) / 0.60) * w_sigma
          4. tti_fut_04             : prospective mean of tti_diff_1 over [t+1, t+11] (+0.4 s)
          5. carrier_prob_diff      : P_carrier(Home, t) - P_carrier(Away, t) bridged across gaps
          6. carrier_vel_align_diff : velocity synchronization difference with reconstructed ball
          7. closing_speed_diff     : approach speed difference towards ball
          8. ctrl_decel_diff        : signed control braking difference near ball
          9. same_player_signed     : signed individual carrier continuity (Home - Away)
         10. teammate_pass_anti     : signed intra-team pass signature (pass_H - pass_A)
         11. interception_anti_05   : signed turnover signature over +/- 0.5 s (A->H minus H->A)
         12. interception_anti_12   : signed turnover signature over +/- 1.2 s (A->H minus H->A)
         13. gap_carrier_own        : smoothed 22-player carrier + TTI ownership across gaps
    """
    n = len(f["ball_seen"])
    if "carrier_sym" in f and "carrier_anti" in f:
        return f["carrier_sym"].astype(np.float64), f["carrier_anti"].astype(np.float64)
    return np.zeros((n, 10), dtype=np.float64), np.zeros((n, 14), dtype=np.float64)


def seq_features(f, blocks, pitch, ball_evidence=None, dead_variant="stat", team_variant="duel_past",
                 include_kinematic=True, include_carrier=True):
    """Build the feature matrix (n_frames, 102) in float32 (or 78 when `include_carrier=False`,
    or 58 when `include_kinematic=False`).

    Never uses set-piece tags (to avoid bench leakage).
    """
    if ball_evidence is None:
        ball_evidence = em_mod.ball_team_evidence(f, em_mod.PARAMS) / em_mod.PARAMS["w_ball"]
    Xd = lin_model.dead_features(f, blocks, pitch, dead_variant)        # 16 symmetric cols
    Xt = lin_model.team_features(f, blocks, pitch, ball_evidence, team_variant)  # 26 antisymmetric cols
    X_sym, X_anti = raw_signal_columns(f, blocks, pitch)                # 8 sym + 8 anti cols

    if include_kinematic:
        sk, ak = kinematic_duel_and_boundary_channels(f, blocks, pitch)  # 8 sym + 12 anti cols
        if include_carrier:
            sc, ac = carrier_and_imputation_channels(f)                  # 10 sym + 14 anti cols
            X = np.column_stack([Xd, X_sym, sk, sc, Xt, X_anti, ak, ac]).astype(np.float32)
        else:
            X = np.column_stack([Xd, X_sym, sk, Xt, X_anti, ak]).astype(np.float32)
    else:
        X = np.column_stack([Xd, X_sym, Xt, X_anti]).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def antisym_mask(n_channels=102):
    """Boolean mask (C,) True on channels that negate when swapping Home <-> Away."""
    if n_channels == 58:
        n_sym = 16 + 8
    elif n_channels == 78:
        n_sym = 16 + 8 + 8
    elif n_channels == 102:
        n_sym = 16 + 8 + 8 + 10
    else:
        raise ValueError(f"Unsupported channel count for antisym_mask: {n_channels}")
    mask = np.zeros(n_channels, dtype=bool)
    mask[n_sym:] = True
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
    out = np.broadcast_to(bias[:, None], (c_out, length)).astype(np.float32).copy()
    for j in range(k):
        shift = (j - k // 2) * dilation
        w_j = weight[:, :, j]  # (C_out, C_in)
        if shift == 0:
            out += w_j @ x
        elif shift < 0:
            s = -shift
            if s < length:
                out[:, s:] += w_j @ x[:, :length - s]
        else:
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
    """Compute (s_team, s_dead, p_change) from `weights` with kind=='seq'."""
    params = weights["seq_params"]
    n_ch = int(params["norm_mu"].shape[0])
    X = seq_features(
        f,
        blocks,
        pitch,
        ball_evidence,
        dead_variant=weights.get("dead_variant", "stat"),
        team_variant=weights.get("team_variant", "duel_past"),
        include_kinematic=(n_ch >= 78),
        include_carrier=(n_ch == 102),
    )
    out = predict_game_numpy(X, blocks, params)
    if weights.get("tta_swap", True):
        anti = antisym_mask(n_ch)
        X_swap = X.copy()
        X_swap[:, anti] = -X_swap[:, anti]
        out_swap = predict_game_numpy(X_swap, blocks, params)
        out = {
            "s_team_raw": 0.5 * (out["s_team_raw"] - out_swap["s_team_raw"]),
            "s_dead_raw": 0.5 * (out["s_dead_raw"] + out_swap["s_dead_raw"]),
            "p_change": 0.5 * (out["p_change"] + out_swap["p_change"]),
            "p_dead_edge": 0.5 * (out["p_dead_edge"] + out_swap["p_dead_edge"]),
        }
    s_team = (out["s_team_raw"] * float(weights.get("t_team", 1.0))).astype(np.float32)
    s_dead = (out["s_dead_raw"] * float(weights.get("t_dead", 1.0))).astype(np.float32)
    return s_team, s_dead, out["p_change"]
