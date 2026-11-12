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
    if variant in ("ctrl", "touch", "duel", "duel_past"):
        cols.append(_fill(_shift_mean(c, 1, FPS, blocks, n), 0.0))
    if variant in ("touch", "duel", "duel_past"):
        t1 = f["touch"].astype(np.float64)
        t2 = f["touch2"].astype(np.float64)
        cols += [
            _carry(f["touch"], 2 * FPS, blocks, n),
            _carry(f["touch2"], 3 * FPS, blocks, n),
            _carry(f["touch2"], 8 * FPS, blocks, n),
            _fill(_shift_mean(t1, -FPS, 1, blocks, n), 0.0) * FPS,
            _fill(_shift_mean(t1, 1, FPS, blocks, n), 0.0) * FPS,
        ]
    if variant == "duel":
        cols += _duel_features(f, blocks, n)
    elif variant == "duel_past":
        # same, without the three indices that look at the NEXT touch (team, delay, streak):
        # looking ahead can make the decoder switch before the ball is actually won.
        # The positions 2, 3, 5 follow the order of the list returned by _duel_features.
        duel = _duel_features(f, blocks, n)
        cols += [col for k, col in enumerate(duel) if k not in (2, 3, 5)]
    return np.column_stack([base] + cols)


DUEL_HORIZON_S = 8.0     # a touch older (or further ahead) than this says nothing about now
DUEL_STREAK_GAP_S = 4.0  # same rule as touch2: two touches of one team more than 4 s apart
#                          are not one sequence


def _event_index(event, blocks, forward=False):
    """Frame index of the last True at or before each frame (or, with `forward`, of the first
    True strictly after it), inside each block; -1 when there is none."""
    out = np.full(len(event), -1, dtype=np.int64)
    for _, lo, hi in blocks:
        seg = event[lo:hi + 1]
        m = len(seg)
        i = np.arange(m)
        if forward:
            at_or_after = np.minimum.accumulate(np.where(seg, i, m)[::-1])[::-1]
            nxt = np.concatenate((at_or_after[1:], [m]))
            out[lo:hi + 1] = np.where(nxt < m, nxt + lo, -1)
        else:
            last = np.maximum.accumulate(np.where(seg, i, -1))
            out[lo:hi + 1] = np.where(last >= 0, last + lo, -1)
    return out


def _duel_features(f, blocks, n):
    """Indices that tell a recovery from a mere contact in a duel.

    A short possession starts with a touch by the other team. What separates a real recovery from
    a deflection is what surrounds that touch: was it followed by a second touch of the same team
    (streak), how crowded was the ball (duel or free reception), and where did the ball go next
    (forward for the team that touched it, or straight back). All signed home = +, away = -.
    """
    touch = f["touch"].astype(np.float64)
    is_touch = touch != 0
    t = np.arange(n)

    # run length of consecutive touches by one team, signed, written on the touch frames
    streak = np.zeros(n)
    max_gap = DUEL_STREAK_GAP_S * FPS
    for _, lo, hi in blocks:
        last_sign, last_t, run = 0.0, -10 ** 9, 0
        for i in np.flatnonzero(is_touch[lo:hi + 1]) + lo:
            run = run + 1 if (touch[i] == last_sign and i - last_t < max_gap) else 1
            streak[i] = touch[i] * min(run, 4)
            last_sign, last_t = touch[i], i

    def around(event):
        last, nxt = _event_index(event, blocks), _event_index(event, blocks, forward=True)
        age_last = np.where(last >= 0, (t - last) / float(FPS), DUEL_HORIZON_S)
        age_next = np.where(nxt >= 0, (nxt - t) / float(FPS), DUEL_HORIZON_S)
        ok_last = age_last < DUEL_HORIZON_S
        ok_next = age_next < DUEL_HORIZON_S
        return (np.maximum(last, 0), ok_last, np.minimum(age_last, DUEL_HORIZON_S),
                np.maximum(nxt, 0), ok_next, np.minimum(age_next, DUEL_HORIZON_S))

    li, ok_l, age_l, ni, ok_n, age_n = around(is_touch)
    _, _, age_home, _, _, _ = around(touch > 0)
    _, _, age_away, _, _, _ = around(touch < 0)

    # crowd around the ball: signed balance and how contested it is (both teams close)
    nh, na = f["n_home_near"].astype(np.float64), f["n_away_near"].astype(np.float64)
    balance = nh - na
    contested = np.where(f["ball_ok"], np.minimum(nh, na), np.nan)

    # where the ball goes in the 0.6 s after a frame, towards the away goal = +, and how fast
    ahead = 15
    forward = np.full(n, np.nan)
    for _, lo, hi in blocks:
        x = f["ball"][lo:hi + 1, 0].astype(np.float64)
        if len(x) > ahead:
            forward[lo:hi + 1 - ahead] = (x[ahead:] - x[:-ahead]) \
                * f["attack_sign_home"][lo:hi + 1 - ahead]
    speed = np.where(f["ball_seen"], f["ball_speed"], np.nan).astype(np.float64)
    speed_after = _shift_mean(speed, 1, 11, blocks, n)

    return [
        np.where(ok_l, touch[li], 0.0),                    # team of the last touch
        age_l,
        np.where(ok_n, touch[ni], 0.0),                    # team of the next touch
        age_n,
        np.where(ok_l, streak[li], 0.0),                   # 1 = lone contact, 2+ = sequence
        np.where(ok_n, streak[ni], 0.0),                   # does the next touch extend it?
        age_away - age_home,                               # > 0: home touched more recently
        np.where(ok_l, balance[li], 0.0),                  # crowd balance at the last touch
        _fill(_win_mean(contested, FPS // 2, blocks, n), 0.0),
        np.where(ok_l, _fill(forward[li], 0.0), 0.0),      # ball direction after the last touch
        np.log1p(np.where(ok_l, _fill(speed_after[li], 5.0), 5.0)),
    ]


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
def fit_logistic(X, y, l2=1.0, iters=30, sample_weight=None):
    """IRLS with L2; features standardised internally. Deterministic."""
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    reg = np.eye(Z.shape[1]) * l2
    reg[0, 0] = 0.0
    sw = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (sw * (p - y)) + reg @ w
        Hm = (Z * (sw * p * (1 - p))[:, None]).T @ Z + reg
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
        w = json.load(handle)
    if w.get("kind") == "gbdt":
        with open(os.path.join(os.path.dirname(path), w["model_file"])) as handle:
            models = json.load(handle)
        w["dead"], w["team"] = models["dead"], models["team"]
    elif w.get("kind") == "seq":
        from possession import seqmodel
        npz_path = os.path.join(os.path.dirname(path), w["model_file"])
        w["seq_params"] = seqmodel.load_seq_params(npz_path)
    return w


def export_trees(clf):
    """A fitted scikit-learn HistGradientBoostingClassifier as plain lists, so the pipeline can
    predict with numpy alone (no scikit-learn, any Python >= 3.8)."""
    if getattr(clf, "is_categorical_", None) is not None and np.any(clf.is_categorical_):
        raise ValueError("categorical features are not supported by the numpy predictor")
    trees = []
    for (pred,) in clf._predictors:
        nd = pred.nodes
        trees.append({
            "feature": nd["feature_idx"].astype(int).tolist(),
            "threshold": nd["num_threshold"].astype(float).tolist(),
            "missing_left": nd["missing_go_to_left"].astype(int).tolist(),
            "left": nd["left"].astype(int).tolist(),
            "right": nd["right"].astype(int).tolist(),
            "leaf": nd["is_leaf"].astype(int).tolist(),
            "value": nd["value"].astype(float).tolist(),
        })
    return {"baseline": float(np.ravel(clf._baseline_prediction)[0]), "trees": trees,
            "n_features": int(clf.n_features_in_)}


def predict_trees(m, X):
    """Raw score (log-odds) of an exported tree ensemble. Same arithmetic as scikit-learn:
    go left when x <= threshold, or when x is NaN and missing_left."""
    X = np.asarray(X, dtype=np.float64)
    out = np.full(len(X), m["baseline"])
    rows = np.arange(len(X))
    for t in m["trees"]:
        feat, thr = np.asarray(t["feature"]), np.asarray(t["threshold"])
        left, right = np.asarray(t["left"]), np.asarray(t["right"])
        leaf, miss = np.asarray(t["leaf"], bool), np.asarray(t["missing_left"], bool)
        value = np.asarray(t["value"])
        node = np.zeros(len(X), dtype=np.int64)
        active = ~leaf[node]
        while active.any():
            r, nd = rows[active], node[active]
            x = X[r, feat[nd]]
            go_left = np.where(np.isnan(x), miss[nd], x <= thr[nd])
            node[r] = np.where(go_left, left[nd], right[nd])
            active = ~leaf[node]
        out += value[node]
    return out


def fit_gbdt(X, y, seed=0, sample_weight=None):
    """Gradient-boosted trees (scikit-learn), fixed seed so the fit is reproducible."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
                                         min_samples_leaf=200, l2_regularization=1.0,
                                         early_stopping=False, random_state=seed)
    clf.fit(X, y, sample_weight=sample_weight)
    return clf


def logit_any(m, X):
    if isinstance(m, dict) and "trees" in m:
        return np.clip(predict_trees(m, X), -9.21, 9.21)   # = log(p/(1-p)) with p in [1e-4, 1-1e-4]
    if isinstance(m, dict):
        return logit(m, X)
    p = np.clip(m.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def emissions(f, blocks, pitch, ball_evidence, weights):
    """(s_team, s_dead) log-odds from the learned models, scaled by the stored temperatures."""
    s_dead = logit_any(weights["dead"], dead_features(f, blocks, pitch, weights.get("dead_variant", "v2"))) \
        * weights.get("t_dead", 1.0)
    s_team = logit_any(weights["team"], team_features(f, blocks, pitch, ball_evidence,
                                                  weights.get("team_variant", "v2"))) \
        * weights.get("t_team", 1.0)
    return s_team.astype(np.float32), s_dead.astype(np.float32)
