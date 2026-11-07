"""Learned emissions: two logistic regressions trained on OTHER games (Metrica public samples).

    dead model : P(ball dead | window features)          -> s_dead = logit
    team model : P(home in possession | alive, features)  -> s_team = logit

Weights live in possession/weights.json, produced by `python -m validation.train` from the public
sample data with a fixed seed, so anyone can regenerate them. Nothing here ever sees a label of
the provided game.
"""

import json
import os

import numpy as np

FPS = 25
WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.json")


def _win_mean(x, half_width, blocks, n):
    """NaN-aware centred mean inside each block."""
    out = np.full(n, np.nan, dtype=np.float64)
    for _, lo, hi in blocks:
        seg = x[lo:hi + 1].astype(np.float64)
        valid = ~np.isnan(seg)
        cs = np.concatenate(([0.0], np.cumsum(np.where(valid, seg, 0.0))))
        cn = np.concatenate(([0], np.cumsum(valid)))
        i = np.arange(len(seg))
        a = np.maximum(0, i - half_width)
        b = np.minimum(len(seg), i + half_width + 1)
        cnt = cn[b] - cn[a]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[lo:hi + 1] = np.where(cnt > 0, (cs[b] - cs[a]) / np.maximum(cnt, 1), np.nan)
    return out


def _shift_mean(x, start, stop, blocks, n):
    """Mean of x over [t+start, t+stop) inside each block (one-sided windows)."""
    out = np.full(n, np.nan)
    for _, lo, hi in blocks:
        seg = x[lo:hi + 1].astype(np.float64)
        valid = ~np.isnan(seg)
        cs = np.concatenate(([0.0], np.cumsum(np.where(valid, seg, 0.0))))
        cn = np.concatenate(([0], np.cumsum(valid)))
        i = np.arange(len(seg))
        a = np.clip(i + start, 0, len(seg))
        b = np.clip(i + stop, 0, len(seg))
        cnt = cn[b] - cn[a]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[lo:hi + 1] = np.where(cnt > 0, (cs[b] - cs[a]) / np.maximum(cnt, 1), np.nan)
    return out


def _fill(x, value):
    return np.where(np.isfinite(x), x, value)


def dead_features(f, blocks, pitch, variant=None):
    n = len(f["ball_seen"])
    length, width = pitch
    act = f["activity"].astype(np.float64)
    ratio = np.full(n, np.nan)
    for _, lo, hi in blocks:
        med = np.nanmedian(act[lo:hi + 1])
        if np.isfinite(med) and med > 0:
            ratio[lo:hi + 1] = act[lo:hi + 1] / med
    ratio = _fill(ratio, 1.0)
    past = _fill(_shift_mean(ratio, -3 * FPS, 0, blocks, n), 1.0)
    future = _fill(_shift_mean(ratio, 1, 3 * FPS, blocks, n), 1.0)

    seen = f["ball_seen"].astype(np.float64)
    speed = np.where(f["ball_seen"], f["ball_speed"], np.nan)
    x, y = f["ball"][:, 0].astype(np.float64), f["ball"][:, 1].astype(np.float64)
    boundary = np.minimum.reduce([x, length - x, y, width - y])
    boundary = np.where(f["ball_ok"], np.clip(boundary, -3, 15), np.nan)
    dmin = np.minimum(f["d_home"], f["d_away"]).astype(np.float64)
    dmin = np.where(f["ball_ok"] & np.isfinite(dmin), np.log(np.clip(dmin, 0, 30) + 0.5), np.nan)

    cols = [
        np.log(np.clip(ratio, 0.1, 5)),
        np.log(np.clip(_fill(_win_mean(ratio, 3 * FPS, blocks, n), 1.0), 0.1, 5)),
        np.log(np.clip(future, 0.1, 5)) - np.log(np.clip(past, 0.1, 5)),
        _fill(_win_mean(seen, FPS, blocks, n), 0.0),
        _fill(_win_mean(seen, 5 * FPS, blocks, n), 0.0),
        np.log1p(_fill(_win_mean(speed, FPS // 2, blocks, n), 5.0)),
        np.log1p(_fill(_win_mean(speed, 2 * FPS, blocks, n), 5.0)),
        _fill(_win_mean(boundary, FPS // 2, blocks, n), 15.0) / 15.0,
        np.isnan(_win_mean(boundary, FPS // 2, blocks, n)).astype(np.float64),
        _fill(_win_mean(dmin, FPS // 2, blocks, n), np.log(3.5)),
        _fill(_win_mean(f["ball_out"].astype(np.float64), FPS // 2, blocks, n), 0.0),
        np.log1p(f["gap_len"] / float(FPS)),
    ]
    if "stationary" in f and variant != "v2":
        st = f["stationary"].astype(np.float64)
        cols += [
            _fill(_win_mean(st, FPS, blocks, n), 0.0),
            _fill(_win_mean(st, 3 * FPS, blocks, n), 0.0),
            _fill(_shift_mean(st, 1, 3 * FPS, blocks, n), 0.0),   # ball about to be placed/struck
            _fill(_shift_mean(st, -3 * FPS, 0, blocks, n), 0.0),  # ball was resting just before
        ]
    return np.column_stack(cols)


def _carry(sign, max_age, blocks, n):
    """Last non-zero value of `sign` within the past `max_age` frames (0 if none), per block."""
    out = np.zeros(n)
    for _, lo, hi in blocks:
        seg = sign[lo:hi + 1]
        idx = np.where(seg != 0, np.arange(len(seg)), -1)
        last = np.maximum.accumulate(idx)
        age = np.arange(len(seg)) - last
        val = np.where(last >= 0, seg[np.maximum(last, 0)], 0)
        out[lo:hi + 1] = np.where((last >= 0) & (age <= max_age), val, 0)
    return out


TEAM_VARIANT = "ctrl_causal"


def team_features(f, blocks, pitch, ball_evidence, variant=None):
    variant = variant or TEAM_VARIANT
    if variant == "v2":
        return _team_v2(f, blocks, pitch, ball_evidence)
    n = len(f["ball_seen"])
    ev = ball_evidence.astype(np.float64)
    evn = np.where(ev != 0, ev, np.nan)
    c = f["control"].astype(np.float64)
    base = _team_v2(f, blocks, pitch, ball_evidence)
    if variant == "ctrl_causal":
        # drop the two windows that look ahead (centred 2 s, next 2 s); keep past and present
        base = np.delete(base, [2, 4], axis=1)
        base = np.column_stack([base, _fill(_shift_mean(evn, -4 * FPS, -2 * FPS, blocks, n), 0.0)])
    cols = [
        _carry(f["control"], 2 * FPS, blocks, n),
        _carry(f["control"], 6 * FPS, blocks, n),
        _fill(_win_mean(c, 5, blocks, n), 0.0),
        _fill(_shift_mean(c, -FPS, 0, blocks, n), 0.0),
    ]
    if variant == "ctrl":
        cols.append(_fill(_shift_mean(c, 1, FPS, blocks, n), 0.0))
    return np.column_stack([base] + cols)


def _team_v2(f, blocks, pitch, ball_evidence):
    n = len(f["ball_seen"])
    length = pitch[0]
    ev = ball_evidence.astype(np.float64)
    near = (f["n_home_near"].astype(np.float64) - f["n_away_near"])
    near = np.where(f["ball_ok"], near, np.nan)
    terr = np.where(f["ball_ok"], (f["ball"][:, 0] - length / 2) / (length / 2) *
                    f["attack_sign_home"], np.nan)
    cols = [
        ev,
        _fill(_win_mean(np.where(ev != 0, ev, np.nan), FPS // 2, blocks, n), 0.0),
        _fill(_win_mean(np.where(ev != 0, ev, np.nan), 2 * FPS, blocks, n), 0.0),
        _fill(_shift_mean(np.where(ev != 0, ev, np.nan), -2 * FPS, 0, blocks, n), 0.0),
        _fill(_shift_mean(np.where(ev != 0, ev, np.nan), 1, 2 * FPS, blocks, n), 0.0),
        _fill(_win_mean(near, FPS // 2, blocks, n), 0.0),
        np.tanh(_fill(f["shape_adv"].astype(np.float64), 0.0) / 8.0),
        _fill(_win_mean(terr, 2 * FPS, blocks, n), 0.0),
    ]
    return np.column_stack(cols)


# ------------------------------------------------------------------------- logistic regression
def fit_logistic(X, y, l2=1.0, iters=30):
    """IRLS with L2; features standardised internally. Deterministic."""
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    reg = np.eye(Z.shape[1]) * l2
    reg[0, 0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - y) + reg @ w
        Hm = (Z * (p * (1 - p))[:, None]).T @ Z + reg
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return {"mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist()}


def logit(model, X):
    mu, sd, w = (np.asarray(model[k]) for k in ("mu", "sd", "w"))
    return w[0] + ((X - mu) / sd) @ w[1:]


def load_weights(path=WEIGHTS):
    with open(path) as handle:
        return json.load(handle)


def emissions(f, blocks, pitch, ball_evidence, weights):
    """(s_team, s_dead) log-odds from the learned models, scaled by the stored temperatures."""
    s_dead = logit(weights["dead"], dead_features(f, blocks, pitch, weights.get("dead_variant", "v2"))) \
        * weights.get("t_dead", 1.0)
    s_team = logit(weights["team"], team_features(f, blocks, pitch, ball_evidence,
                                                  weights.get("team_variant", "v2"))) \
        * weights.get("t_team", 1.0)
    return s_team.astype(np.float32), s_dead.astype(np.float32)
