"""Logit calibration, multi-seed MS-TCN pooling, and GBDT + MS-TCN ensembling before decoding.

Inference uses numpy only (no torch, no scikit-learn) and is strictly deterministic.

Why combine in calibrated log-odds space:
  * GBDT (v7 trees) and MS-TCN have very different raw logit dispersions (`std(logit)`). Naive
    probability averaging lets the over-confident model dominate; temperature scaling (`logit / T`)
    puts every model on true log-likelihood-ratio units (nats) before taking the convex combination.
  * For the change-point head `p_change(t)`, multi-seed / multi-model probabilities are pooled
    with a sharpened power mean (or weighted log-odds) and optionally augmented by a fast kinematic
    turnover cue derived from the fused `s_team` derivative and touch/control events.
"""

import numpy as np

from possession import decode, emissions, model

LOGIT_CLIP = 9.21  # log(p / (1 - p)) for p in [1e-4, 1 - 1e-4]

DEFAULT_ENSEMBLE = {
    # Relative weights of the tree model (v7 GBDT) vs the sequential model (MS-TCN multi-seed)
    "w_tree_team": 0.30,
    "w_seq_team": 0.70,
    "w_tree_dead": 0.35,
    "w_seq_dead": 0.65,
    # Per-family inverse-temperature (scaling factor applied to each model's raw logits)
    "scale_tree_team": 1.0,
    "scale_seq_team": 1.0,
    "scale_tree_dead": 0.7,
    "scale_seq_dead": 1.0,
    # Global post-fusion temperature and prior bias before Viterbi decoding
    "t_team": 1.0,
    "t_dead": 1.0,
    "b_team": 0.0,
    "b_dead": 0.0,
    #Change-head pooling (across MS-TCN seeds and optional kinematic change proxy)
    "change_temp": 1.0,
    "change_bias": 0.0,
    "change_power": 1.5,       # power-mean exponent > 1 preserves sharp peaks across seeds
    "w_kinematic_change": 0.0, # optional weight for kinematic turnover cue when seq p_change is absent/blended
    # Optional frequency-separated fusion: near boundaries (|s_seq| < boundary_band or p_change high),
    # suppress causal tree logits so they cannot lag or shift the MS-TCN zero-crossing!
    "boundary_gate_trees": True,
    "boundary_band": 1.5,
}


def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit_from_prob(p, eps=1e-4):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def calibrate_logits(raw_logits, scale=1.0, bias=0.0, clip=LOGIT_CLIP):
    """Affine calibration in log-odds space: `s_cal = clip(scale * raw_logits + bias)`."""
    s = np.asarray(raw_logits, dtype=np.float64) * float(scale) + float(bias)
    return np.clip(s, -clip, clip)


def fit_temperature(raw_logits, y_binary, sample_weight=None, fit_bias=False, l2=1e-3, iters=25):
    """Fit post-hoc Platt / temperature scaling `(scale, bias)` by Newton-Raphson on NLL.

    Used only during validation (`validation/train.py` or `validation/train_seq.py`) on held-out
    or out-of-fold logits so that each model family is calibrated to true log-odds before fusion.
    Deterministic.
    """
    z = np.asarray(raw_logits, dtype=np.float64).ravel()
    y = np.asarray(y_binary, dtype=np.float64).ravel()
    sw = np.ones(len(z), dtype=np.float64) if sample_weight is None else \
        np.asarray(sample_weight, dtype=np.float64).ravel()
    sw = sw / np.maximum(sw.mean(), 1e-12)

    if not fit_bias:
        scale = 1.0
        for _ in range(iters):
            p = sigmoid(scale * z)
            grad = np.mean(sw * (p - y) * z) + l2 * (scale - 1.0)
            hess = np.mean(sw * p * (1.0 - p) * (z ** 2)) + l2
            step = grad / np.maximum(hess, 1e-9)
            scale -= step
            if abs(step) < 1e-6:
                break
        return float(np.clip(scale, 0.05, 10.0)), 0.0

    params = np.array([1.0, 0.0], dtype=np.float64)
    X = np.column_stack([z, np.ones(len(z), dtype=np.float64)])
    reg = np.diag([l2, l2 * 0.1])
    target_prior = np.array([1.0, 0.0], dtype=np.float64)
    for _ in range(iters):
        p = sigmoid(X @ params)
        grad = (X.T @ (sw * (p - y))) / len(z) + reg @ (params - target_prior)
        w_hess = sw * p * (1.0 - p)
        hess = ((X * w_hess[:, None]).T @ X) / len(z) + reg
        step = np.linalg.solve(hess, grad)
        params -= step
        if np.abs(step).max() < 1e-6:
            break
    return float(np.clip(params[0], 0.05, 10.0)), float(np.clip(params[1], -3.0, 3.0))


def pool_seed_logits(seed_logits, weights=None):
    """Pool multi-seed MS-TCN logits `(K, n)` -> `(n,)`, with optional seed weights."""
    arr = np.asarray(seed_logits, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if weights is None:
        return np.mean(arr, axis=0)
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.maximum(w.sum(), 1e-12)
    return np.tensordot(w, arr, axes=(0, 0))


def pool_change_probs(seed_probs, temp=1.0, bias=0.0, power=1.5):
    """Pool multi-seed change-head probabilities `(K, n)` or `(n,)` into a sharp `(n,)` signal.

    A power mean (`power > 1`) prevents a 1-2 frame alignment jitter across MS-TCN seeds from
    flattening the peak height at a true turnover.
    """
    p = np.clip(np.asarray(seed_probs, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    if temp != 1.0 or bias != 0.0:
        p = sigmoid(logit_from_prob(p) * float(temp) + float(bias))
    if p.ndim == 1:
        return p
    gamma = max(float(power), 1.0)
    return np.mean(p ** gamma, axis=0) ** (1.0 / gamma)


def kinematic_change_proxy(f, s_team, blocks, window=15):
    """Fallback / complementary turnover probability `p_change(t)` in [0, 1] from tracking + logits.

    Fires when the short-term team logit flips sign across `t` AND a ball touch or control event
    occurs within +/- 2 frames. Useful when evaluating tree ensembles alone or blending with MS-TCN.
    """
    st = np.asarray(s_team, dtype=np.float64)
    n = len(st)
    out = np.zeros(n, dtype=np.float64)
    for _, lo, hi in blocks:
        seg = st[lo:hi + 1]
        m = len(seg)
        cs = np.concatenate(([0.0], np.cumsum(seg)))
        idx = np.arange(m)
        p_lo = np.maximum(0, idx - window)
        f_hi = np.minimum(m, idx + window)
        past_m = (cs[idx + 1] - cs[p_lo]) / np.maximum(1, idx + 1 - p_lo)
        fut_m = (cs[f_hi] - cs[idx]) / np.maximum(1, f_hi - idx)
        flip = np.maximum(0.0, -past_m * fut_m)
        has_ev = ((f["touch"][lo:hi + 1] != 0) | (f["control"][lo:hi + 1] != 0)).astype(np.float64)
        near_ev = (np.convolve(has_ev, np.ones(5) / 5.0, mode="same") > 0).astype(np.float64)
        out[lo:hi + 1] = np.tanh(flip / 2.5) * (0.4 + 0.6 * near_ev)
    return out.astype(np.float32)


def combine_emissions(tree_out=None, seq_out=None, f=None, blocks=(), cfg=None):
    """Combine tree (`(s_team, s_dead)` or dict) and sequential (`(s_team, s_dead[, p_change])`
    or dict, single-model or multi-seed list) outputs into `(s_team, s_dead, p_change)`.

    Parameters
    ----------
    tree_out : tuple `(s_team, s_dead)` or dict `{'s_team': ..., 's_dead': ..., 'p_change': ...}`
    seq_out  : tuple/dict or list of tuples/dicts (one per MS-TCN seed)
    f        : feature dict from `features.compute` (optional, for `kinematic_change_proxy`)
    blocks   : list of `(half, first, last)` graded blocks
    cfg      : dict overriding `DEFAULT_ENSEMBLE`
    """
    c = dict(DEFAULT_ENSEMBLE, **(cfg or {}))

    def _unpack_one(item):
        if item is None:
            return None, None, None
        if isinstance(item, dict):
            return item.get("s_team"), item.get("s_dead"), item.get("p_change")
        if len(item) == 2:
            return item[0], item[1], None
        return item[0], item[1], item[2]

    t_st, t_sd, t_pc = _unpack_one(tree_out)

    s_st, s_sd, s_pc = None, None, None
    if seq_out is not None:
        if isinstance(seq_out, (list, tuple)) and len(seq_out) > 0 and isinstance(
            seq_out[0], (dict, list, tuple)
        ):
            unpacked = [_unpack_one(x) for x in seq_out]
            seq_w = c.get("seq_model_weights")
            s_st = pool_seed_logits([u[0] for u in unpacked if u[0] is not None], weights=seq_w)
            s_sd = pool_seed_logits([u[1] for u in unpacked if u[1] is not None], weights=seq_w)
            pcs = [u[2] for u in unpacked if u[2] is not None]
            if pcs:
                s_pc = pool_change_probs(
                    pcs, temp=c["change_temp"], bias=c["change_bias"], power=c["change_power"]
                )
        else:
            s_st, s_sd, s_pc = _unpack_one(seq_out)
            if s_pc is not None:
                s_pc = pool_change_probs(
                    s_pc, temp=c["change_temp"], bias=c["change_bias"], power=c["change_power"]
                )

    if t_st is not None:
        t_st = calibrate_logits(t_st, scale=c["scale_tree_team"])
        t_sd = calibrate_logits(t_sd, scale=c["scale_tree_dead"])
    if s_st is not None:
        s_st = calibrate_logits(s_st, scale=c["scale_seq_team"])
        s_sd = calibrate_logits(s_sd, scale=c["scale_seq_dead"])

    if t_st is not None and s_st is not None:
        wt = float(c["w_tree_team"])
        ws = float(c["w_seq_team"])
        norm_t = max(wt + ws, 1e-9)
        wt_rel = wt / norm_t
        if c.get("boundary_gate_trees", True):
            band = max(float(c.get("boundary_band", 1.5)), 0.1)
            far = np.clip((np.abs(s_st) - 0.8) / band, 0.0, 1.0)
            if s_pc is not None:
                far = far * np.clip((0.25 - s_pc) / 0.25, 0.0, 1.0)
            w_t_frame = wt_rel * far
            s_team = (1.0 - w_t_frame) * s_st + w_t_frame * t_st
        else:
            s_team = (wt * t_st + ws * s_st) / norm_t

        wd_t = float(c["w_tree_dead"])
        wd_s = float(c["w_seq_dead"])
        norm_d = max(wd_t + wd_s, 1e-9)
        s_dead = (wd_t * t_sd + wd_s * s_sd) / norm_d
    elif s_st is not None:
        s_team, s_dead = s_st, s_sd
    elif t_st is not None:
        s_team, s_dead = t_st, t_sd
    else:
        raise ValueError("combine_emissions requires at least one of tree_out or seq_out")

    s_team = calibrate_logits(s_team, scale=c["t_team"], bias=c["b_team"]).astype(np.float32)
    s_dead = calibrate_logits(s_dead, scale=c["t_dead"], bias=c["b_dead"]).astype(np.float32)

    p_change = s_pc if s_pc is not None else t_pc
    w_kin = float(c.get("w_kinematic_change", 0.0))
    if w_kin > 0.0 and f is not None and len(blocks) > 0:
        p_kin = kinematic_change_proxy(f, s_team, blocks)
        if p_change is None:
            p_change = p_kin
        else:
            p_change = (1.0 - w_kin) * np.asarray(p_change, dtype=np.float32) + w_kin * p_kin
    if p_change is not None:
        p_change = np.clip(np.asarray(p_change, dtype=np.float32), 0.0, 1.0)

    return s_team, s_dead, p_change


def ensemble_and_decode(f, blocks, n_frames, pitch, tree_out=None, seq_out=None,
                        tag_prior=None, set_pieces=(), goals=(), params=None,
                        ensemble_cfg=None, costs=None):
    """Full inference bridge: fuse tree + MS-TCN logits, apply Smart Tagging restarts, and run
    Viterbi with optional `p_change` dynamic switch modulation."""
    p = dict(emissions.PARAMS, **(params or {}))
    p["pitch"] = pitch

    s_team, s_dead, p_change = combine_emissions(
        tree_out=tree_out, seq_out=seq_out, f=f, blocks=blocks, cfg=ensemble_cfg
    )
    s_team = s_team.copy()
    s_dead = s_dead.copy()
    if tag_prior is not None:
        s_team += p["w_tag"] * np.clip(tag_prior, -2, 2)
    s_team, s_dead, restarts = emissions.add_restarts(
        f, blocks, s_team, s_dead, set_pieces, goals, p
    )
    labels = decode.decode(s_team, s_dead, blocks, n_frames, costs=costs, p_change=p_change)
    return {
        "labels": labels,
        "s_team": s_team,
        "s_dead": s_dead,
        "p_change": p_change,
        "restarts": restarts,
    }
