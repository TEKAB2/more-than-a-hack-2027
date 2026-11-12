"""Per-frame features from the tracking. Everything in metres and m/s, indexed by video frame.

Design rule: a feature never looks across halftime. Every temporal operation runs inside one
half block at a time (the two graded halves published in DATA.md / games.py).
"""

import numpy as np

FPS = 25


def _runs(mask):
    """[(start, stop)] of True runs in a 1-D boolean array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _moving_mean(x, half_width):
    """Centred NaN-aware moving mean over +/- half_width frames."""
    valid = ~np.isnan(x)
    values = np.where(valid, x, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(values)))
    cn = np.concatenate(([0], np.cumsum(valid)))
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - half_width)
    hi = np.minimum(len(x), idx + half_width + 1)
    count = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (cs[hi] - cs[lo]) / np.maximum(count, 1), np.nan)


def interpolate_ball(ball, max_gap):
    """Linear interpolation over missing stretches of at most max_gap frames.

    Returns (ball_interp, ok, gap_len): ok is True where the ball is observed or interpolated,
    gap_len is the length of the missing stretch each frame belongs to (0 if observed).
    """
    n = len(ball)
    out = ball.copy()
    seen = ~np.isnan(ball[:, 0])
    gap_len = np.zeros(n, dtype=np.int32)
    for start, stop in _runs(~seen):
        gap_len[start:stop] = stop - start
        if stop - start > max_gap or start == 0 or stop == n:
            continue
        a, b = ball[start - 1], ball[stop]
        t = (np.arange(start, stop) - (start - 1)) / float(stop - (start - 1))
        out[start:stop] = a[None, :] + t[:, None] * (b - a)[None, :]
    ok = ~np.isnan(out[:, 0])
    return out, ok, gap_len


def compute(atd, blocks, pitch, max_interp=12, near_radius=3.0):
    """Feature dict. `blocks` = [(half, first_frame, last_frame_inclusive)]."""
    players, ball, team = atd["players"], atd["ball"], atd["team"]
    n = len(ball)
    length, width = pitch
    home_slots, away_slots = team == 0, team == 1

    f = {
        "ball_seen": ~np.isnan(ball[:, 0]),
        "ball": np.full((n, 2), np.nan, dtype=np.float32),
        "ball_ok": np.zeros(n, dtype=bool),
        "gap_len": np.zeros(n, dtype=np.int32),
        "ball_speed": np.full(n, np.nan, dtype=np.float32),
        "attack_sign_home": np.zeros(n, dtype=np.float32),
        "activity": np.full(n, np.nan, dtype=np.float32),
    }

    for _, first, last in blocks:
        sl = slice(first, last + 1)
        b, ok, gap = interpolate_ball(ball[sl], max_interp)
        f["ball"][sl], f["ball_ok"][sl], f["gap_len"][sl] = b, ok, gap

        # Ball speed: central difference over +/-2 frames, then a short median-ish clamp.
        v = np.full(len(b), np.nan, dtype=np.float32)
        if len(b) > 4:
            d = np.hypot(b[4:, 0] - b[:-4, 0], b[4:, 1] - b[:-4, 1]) / (4.0 / FPS)
            v[2:-2] = d
        f["ball_speed"][sl] = np.clip(v, 0, 45)

        # Player activity: mean speed of tracked players, smoothed over +/-1 s.
        speed = players[sl, :, 2]
        if np.all(np.isnan(speed)):
            xy = players[sl, :, :2]
            speed = np.full(speed.shape, np.nan, dtype=np.float32)
            speed[1:] = np.hypot(*(xy[1:] - xy[:-1]).transpose(2, 0, 1)) * FPS
        with np.errstate(invalid="ignore"), _null():
            raw = np.nanmean(speed, axis=1)
        f["activity"][sl] = _moving_mean(raw, FPS)

        # Direction of play: the team whose mean x is lower defends the left goal.
        hx = np.nanmean(players[sl][:, home_slots, 0])
        ax = np.nanmean(players[sl][:, away_slots, 0])
        f["attack_sign_home"][sl] = 1.0 if hx < ax else -1.0

    # Distances ball -> nearest home / away player (metres). inf where unknown.
    bx, by = f["ball"][:, 0:1], f["ball"][:, 1:2]
    dist = np.hypot(players[:, :, 0] - bx, players[:, :, 1] - by)
    dist = np.where(np.isnan(dist), np.inf, dist)
    f["d_home"] = dist[:, home_slots].min(axis=1).astype(np.float32)
    f["d_away"] = dist[:, away_slots].min(axis=1).astype(np.float32)
    f["n_home_near"] = (dist[:, home_slots] < near_radius).sum(axis=1).astype(np.int8)
    f["n_away_near"] = (dist[:, away_slots] < near_radius).sum(axis=1).astype(np.int8)

    # Ball out of the pitch (observed or interpolated position), with a 0.3 m tolerance.
    m = 0.3
    x, y = f["ball"][:, 0], f["ball"][:, 1]
    with np.errstate(invalid="ignore"):
        f["ball_out"] = f["ball_ok"] & ((x < -m) | (x > length + m) | (y < -m) | (y > width + m))

    # Team shape: how far each block is pushed towards the opponent goal (metres from halfway).
    with np.errstate(invalid="ignore"), _null():
        cx_home = np.nanmean(np.where(home_slots[None, :], players[:, :, 0], np.nan), axis=1)
        cx_away = np.nanmean(np.where(away_slots[None, :], players[:, :, 0], np.nan), axis=1)
    sign = f["attack_sign_home"]
    f["shape_adv"] = ((cx_home - length / 2) * sign - (cx_away - length / 2) * (-sign)) \
        .astype(np.float32)
    f["n_players"] = (~np.isnan(players[:, :, 0])).sum(axis=1).astype(np.int8)

    control_features(f, players, team, blocks, dist)
    reconstruct_ball(f, ball, players, team, blocks)
    carrier_features(f, players, team, blocks)
    return f


def _velocity(xy, k=3):
    """Centred velocity (m/s) over +/-k frames. xy (n, ..., 2) -> (n, ..., 2), NaN at the edges."""
    v = np.full(xy.shape, np.nan, dtype=np.float32)
    v[k:-k] = (xy[2 * k:] - xy[:-2 * k]) * (FPS / (2.0 * k))
    return v


def control_features(f, players, team, blocks, dist):
    """Who is *controlling* the ball, not just who is nearest to it.

    A player controls the ball when it is at his feet (< 1.5 m) and travels with him (relative
    speed < 2.5 m/s). During a pass or a deflection the nearest player is not controlling it, which
    is exactly what the challenge's possession rule asks for and what nearest-player gets wrong:
    the evidence for the receiving team must start at the reception, not while the ball flies
    towards him.

    Also flags a *stationary free ball* (seen, not moving for ~1 s, nobody on it): the ball placed
    for a free kick, a corner, a goal kick -- a strong sign the ball is dead.
    """
    n = len(f["ball_seen"])
    f["control"] = np.zeros(n, dtype=np.int8)        # +1 home controls, -1 away, 0 nobody
    f["stationary"] = np.zeros(n, dtype=bool)
    f["ball_disp"] = np.full(n, np.nan, dtype=np.float32)  # metres moved over the last 0.4 s
    f["touch"] = np.zeros(n, dtype=np.int8)     # +1 / -1 on the frame a home / away player touches
    f["touch2"] = np.zeros(n, dtype=np.int8)    # same, only for a touch that follows a team-mate's
    nearest = np.argmin(dist, axis=1)
    dmin = dist[np.arange(n), nearest]
    for _, lo, hi in blocks:
        sl = slice(lo, hi + 1)
        b = f["ball"][sl]
        vb = _velocity(b)
        p = players[sl, :, :2]
        idx = nearest[sl]
        vp = _velocity(p)[np.arange(hi + 1 - lo), idx]
        rel = np.hypot(vb[:, 0] - vp[:, 0], vb[:, 1] - vp[:, 1])
        seen = f["ball_seen"][sl]
        with np.errstate(invalid="ignore"):
            ctrl = seen & (dmin[sl] < 1.5) & (np.nan_to_num(rel, nan=99) < 2.5)
        sign = np.where(team[idx] == 0, 1, -1).astype(np.int8)
        f["control"][sl] = np.where(ctrl, sign, 0)

        f["touch"][sl], f["touch2"][sl] = detect_touches(b, seen, p, team, dmin[sl], idx)

        disp = np.full(len(b), np.nan, dtype=np.float32)
        disp[10:] = np.hypot(b[10:, 0] - b[:-10, 0], b[10:, 1] - b[:-10, 1])
        f["ball_disp"][sl] = disp
        still = seen & (np.nan_to_num(_moving_mean(disp, FPS // 2), nan=9) < 0.3)
        with np.errstate(invalid="ignore"):
            f["stationary"][sl] = still & (dmin[sl] > 0.8)


class _null:
    """Context manager that silences 'mean of empty slice' warnings."""

    def __enter__(self):
        import warnings
        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


def detect_touches(b, seen, p, team, dmin, nearest, min_acc=10.0, min_turn=25.0, radius=2.0):
    """Ball touches, after Link & Hoernig (2017) and ELASTIC (Kim et al., 2025): a touch is a local
    peak of the ball's acceleration (or a sharp change of direction) while a player is within
    `radius` metres. Returns (touch, touch2) per frame: +1/-1 for a touch by home/away on its frame,
    and touch2 marking only touches that follow a touch by a team-mate -- two touches in a row by
    one team is the simplest observable proxy for the challenge's "controlled possession".
    """
    n = len(b)
    touch = np.zeros(n, dtype=np.int8)
    touch2 = np.zeros(n, dtype=np.int8)
    if n < 20:
        return touch, touch2
    # light smoothing of the (interpolated) ball before differentiating
    k = 2
    sm = np.full(b.shape, np.nan, dtype=np.float64)
    cs = np.cumsum(np.nan_to_num(b, nan=0.0), axis=0)
    sm[k + 1:n - k] = (cs[2 * k + 1:] - cs[:n - 2 * k - 1]) / (2 * k + 1)
    v = np.full(b.shape, np.nan)
    v[1:-1] = (sm[2:] - sm[:-2]) * (FPS / 2.0)
    acc = np.full(n, np.nan)
    acc[1:-1] = np.hypot(*(v[2:] - v[:-2]).T) * (FPS / 2.0)
    vin = np.full(b.shape, np.nan)
    vout = np.full(b.shape, np.nan)
    vin[3:] = sm[3:] - sm[:-3]
    vout[:-3] = sm[3:] - sm[:-3]
    dot = (vin * vout).sum(axis=1)
    norm = np.hypot(*vin.T) * np.hypot(*vout.T)
    with np.errstate(invalid="ignore", divide="ignore"):
        turn = np.degrees(np.arccos(np.clip(dot / norm, -1, 1)))
        moving = (np.hypot(*vin.T) > 0.15) | (np.hypot(*vout.T) > 0.15)   # > ~1.2 m/s
    a = np.nan_to_num(acc, nan=0.0)
    peak = np.zeros(n, dtype=bool)
    peak[3:-3] = (a[3:-3] >= np.maximum.reduce([a[i:n - 6 + i] for i in range(7)]))
    seen_ok = np.convolve(seen.astype(float), np.ones(5) / 5, mode="same") >= 0.6
    with np.errstate(invalid="ignore"):
        cand = peak & seen_ok & (dmin < radius) & (
            (a >= min_acc) | (np.nan_to_num(turn) >= min_turn) & moving)
    last_team, last_t = 0, -10 ** 9
    for t in np.flatnonzero(cand):
        sign = 1 if team[nearest[t]] == 0 else -1
        touch[t] = sign
        if last_team == sign and t - last_t < 4 * FPS:
            touch2[t] = sign
        last_team, last_t = sign, t
    return touch, touch2


def reconstruct_ball_block(b_obs, p_xy, team):
    """Physics + kinematic 22-player trajectory reconstructor for missing-ball stretches in one half block.

    Combines:
      1. Damped cubic Hermite interpolation constrained by entry/exit positions (p_in, p_out) and
         velocities (v_in, v_out) with characteristic velocity damping time tau_v = 0.55 s.
      2. Same-carrier dribble tracking when the same player j0 == j1 is within 2.2 m at both ends.
      3. Endpoint carrier forward/backward propagation + collective 22-player block centroid shift.
      4. Multi-player convergence hotspot inside the kinematic feasibility ellipse (v_req < 26 m/s):
         local player crowding (< 3.5 m), opponent duel proximity (< 4.0 m), velocity convergence
         of surrounding players towards the candidate carrier, and braking/acceleration magnitude.

    Returns (b_rec, v_rec, sigma, regime):
      - b_rec  : (m, 2) float32 reconstructed ball coordinates (m)
      - v_rec  : (m, 2) float32 reconstructed ball velocity (m/s)
      - sigma  : (m,)   float32 calibrated positional uncertainty sigma_b(t) in metres
      - regime : (m,)   int8    0 = observed, 1 = same-carrier dribble, 2 = ballistic short gap,
                                3 = multi-player convergence / duel gap
    """
    m = len(b_obs)
    b_rec = b_obs.astype(np.float64).copy()
    seen = ~np.isnan(b_rec[:, 0])
    sigma = np.where(seen, 0.10, 15.0).astype(np.float32)
    regime = np.zeros(m, dtype=np.int8)

    p = p_xy.astype(np.float64)
    vp = np.nan_to_num(_velocity(p, k=3), nan=0.0)
    ap_mag = np.zeros((m, p.shape[1]), dtype=np.float64)
    if m > 4:
        dvp = vp[4:] - vp[:-4]
        ap_mag[2:-2] = np.hypot(dvp[:, :, 0], dvp[:, :, 1]) * (FPS / 4.0)

    with np.errstate(invalid="ignore"), _null():
        centroid = np.nanmean(p, axis=1)

    vb_in = np.zeros((m, 2), dtype=np.float64)
    vb_out = np.zeros((m, 2), dtype=np.float64)
    for k in (4, 3, 2, 1):
        vf = np.zeros(m, dtype=bool)
        vf[k:] = seen[k:] & seen[:-k]
        idx_f = np.flatnonzero(vf & (vb_in[:, 0] == 0) & (vb_in[:, 1] == 0))
        vb_in[idx_f] = (b_rec[idx_f] - b_rec[idx_f - k]) * (FPS / float(k))

        vb = np.zeros(m, dtype=bool)
        vb[:-k] = seen[:-k] & seen[k:]
        idx_b = np.flatnonzero(vb & (vb_out[:, 0] == 0) & (vb_out[:, 1] == 0))
        vb_out[idx_b] = (b_rec[idx_b + k] - b_rec[idx_b]) * (FPS / float(k))

    opp_pair = (team[:, None] != team[None, :])[None, :, :]

    for start, stop in _runs(~seen):
        if start == 0 or stop == m:
            if start == 0 and stop < m:
                b_rec[start:stop] = b_rec[stop]
                sigma[start:stop] = 10.0
            elif stop == m and start > 0:
                b_rec[start:stop] = b_rec[start - 1]
                sigma[start:stop] = 10.0
            continue

        L = stop - start
        dt = (L + 1) / float(FPS)
        p0 = b_rec[start - 1]
        p1 = b_rec[stop]
        v0 = np.clip(vb_in[start - 1], -26.0, 26.0)
        v1 = np.clip(vb_out[stop], -26.0, 26.0)

        u = (np.arange(start, stop) - (start - 1)) / float(L + 1)
        t_in_s = np.arange(1, L + 1) / float(FPS)
        t_out_s = np.arange(L, 0, -1) / float(FPS)
        tau_edge = np.minimum(t_in_s, t_out_s)

        tau = 0.55
        t_eff = tau * (1.0 - np.exp(-dt / tau))
        m0, m1 = v0 * t_eff, v1 * t_eff
        h00 = 2.0 * u ** 3 - 3.0 * u ** 2 + 1.0
        h10 = u ** 3 - 2.0 * u ** 2 + u
        h01 = -2.0 * u ** 3 + 3.0 * u ** 2
        h11 = u ** 3 - u ** 2
        x_herm = (
            h00[:, None] * p0
            + h10[:, None] * m0
            + h01[:, None] * p1
            + h11[:, None] * m1
        )

        d0_now = np.hypot(p[start - 1, :, 0] - p0[0], p[start - 1, :, 1] - p0[1])
        d1_now = np.hypot(p[stop, :, 0] - p1[0], p[stop, :, 1] - p1[1])
        j0 = int(np.nanargmin(np.where(np.isnan(d0_now), 999.0, d0_now)))
        j1 = int(np.nanargmin(np.where(np.isnan(d1_now), 999.0, d1_now)))

        p_gap = p[start:stop]
        if j0 == j1 and d0_now[j0] < 2.2 and d1_now[j1] < 2.2 and dt <= 6.0:
            pj = p_gap[:, j0]
            okj = ~np.isnan(pj[:, 0])
            if np.all(okj):
                off_lin = (
                    (1.0 - u)[:, None] * (p0 - p[start - 1, j0])
                    + u[:, None] * (p1 - p[stop, j0])
                )
                x_drib = pj + off_lin
                w_drib = np.clip(np.sin(np.pi * u) * 0.85, 0.0, 0.85)
                b_rec[start:stop] = (1.0 - w_drib[:, None]) * x_herm + w_drib[:, None] * x_drib
                sigma[start:stop] = (0.4 + 0.65 * tau_edge).astype(np.float32)
                regime[start:stop] = 1
                continue

        if dt <= 1.6:
            b_rec[start:stop] = x_herm
            sigma[start:stop] = (0.35 + 1.15 * tau_edge).astype(np.float32)
            regime[start:stop] = 2
            continue

        x_fwd = x_herm.copy()
        w_fwd = np.zeros(L, dtype=np.float64)
        if d0_now[j0] < 3.2:
            pj0 = p_gap[:, j0]
            ok0 = ~np.isnan(pj0[:, 0])
            off0 = (p0 - p[start - 1, j0]) * np.exp(-t_in_s / 0.8)[:, None]
            x_fwd = np.where(ok0[:, None], pj0 + off0, x_herm)
            w_fwd = np.where(ok0, np.exp(-d0_now[j0] / 2.0) * np.exp(-t_in_s / 1.8), 0.0)

        x_bwd = x_herm.copy()
        w_bwd = np.zeros(L, dtype=np.float64)
        if d1_now[j1] < 3.2:
            pj1 = p_gap[:, j1]
            ok1 = ~np.isnan(pj1[:, 0])
            off1 = (p1 - p[stop, j1]) * np.exp(-t_out_s / 0.8)[:, None]
            x_bwd = np.where(ok1[:, None], pj1 + off1, x_herm)
            w_bwd = np.where(ok1, np.exp(-d1_now[j1] / 2.0) * np.exp(-t_out_s / 1.8), 0.0)

        w_h_cb = np.maximum(1.0 - w_fwd - w_bwd, 0.20)
        w_tot_cb = w_fwd + w_bwd + w_h_cb
        x_cb = (
            w_fwd[:, None] * x_fwd
            + w_bwd[:, None] * x_bwd
            + w_h_cb[:, None] * x_herm
        ) / w_tot_cb[:, None]

        c0, c1 = centroid[start - 1], centroid[stop]
        c_lin = c0[None, :] + u[:, None] * (c1 - c0)[None, :]
        c_shift = np.nan_to_num(centroid[start:stop] - c_lin, nan=0.0)
        w_shift = np.clip((tau_edge - 0.5) / 2.0, 0.0, 0.65)
        x_ref = x_cb + w_shift[:, None] * c_shift

        sub_idx = np.arange(0, L, 3)
        if sub_idx[-1] != L - 1:
            sub_idx = np.append(sub_idx, L - 1)
        p_sub = p_gap[sub_idx]
        vp_sub = vp[start:stop][sub_idx]
        ap_sub = ap_mag[start:stop][sub_idx]
        xref_sub = x_ref[sub_idx]
        tin_sub = t_in_s[sub_idx]
        tout_sub = t_out_s[sub_idx]

        dist_from_p0 = np.hypot(p_sub[:, :, 0] - p0[0], p_sub[:, :, 1] - p0[1])
        dist_to_p1 = np.hypot(p_sub[:, :, 0] - p1[0], p_sub[:, :, 1] - p1[1])
        dist_ref = np.hypot(p_sub[:, :, 0] - xref_sub[:, 0:1], p_sub[:, :, 1] - xref_sub[:, 1:2])

        diff_p = p_sub[:, :, None, :] - p_sub[:, None, :, :]
        pdist = np.hypot(diff_p[..., 0], diff_p[..., 1])
        near_cnt = (np.nan_to_num(pdist, nan=99.0) < 3.5).sum(axis=2) - 1

        d_opp = np.where(opp_pair & ~np.isnan(pdist), pdist, 99.0).min(axis=2)
        duel_prox = np.clip((4.0 - d_opp) / 4.0, 0.0, 1.0)

        unit_kj = diff_p / np.maximum(pdist[..., None], 0.5)
        v_towards = np.sum(vp_sub[:, None, :, :] * unit_kj, axis=-1)
        valid_k = (pdist > 0.1) & (pdist < 12.0) & ~np.isnan(pdist)
        conv_score = np.sum(
            np.where(valid_k, np.clip(v_towards, -1.0, 6.0) * np.exp(-pdist / 4.5), 0.0),
            axis=2,
        )

        v_req_in = dist_from_p0 / tin_sub[:, None]
        v_req_out = dist_to_p1 / tout_sub[:, None]
        feasible = (v_req_in < 26.0) & (v_req_out < 26.0) & ~np.isnan(dist_ref)

        bonus_j = np.zeros_like(dist_ref)
        if d0_now[j0] < 3.0:
            bonus_j[:, j0] += 2.5 * np.exp(-tin_sub / 1.5)
        if d1_now[j1] < 3.0:
            bonus_j[:, j1] += 2.5 * np.exp(-tout_sub / 1.5)

        log_w = (
            -0.22 * dist_ref
            + 0.45 * np.minimum(near_cnt, 5)
            + 0.85 * duel_prox
            + 0.20 * np.clip(conv_score, 0.0, 10.0)
            + 0.12 * np.clip(ap_sub, 0.0, 8.0)
            + bonus_j
        )
        log_w = np.where(feasible, log_w, -1e9)
        max_lw_sub = log_w.max(axis=1, keepdims=True)
        has_feas = max_lw_sub[:, 0] > -1e8
        w_p = np.where(feasible, np.exp(np.clip(log_w - max_lw_sub, -30.0, 0.0)), 0.0)
        w_sum = np.maximum(w_p.sum(axis=1, keepdims=True), 1e-9)
        x_conv_sub = (
            np.nansum(np.where(feasible[:, :, None], w_p[:, :, None] * p_sub, 0.0), axis=1)
            / w_sum
        )
        x_conv_sub = np.where(has_feas[:, None], x_conv_sub, xref_sub)

        full_idx = np.arange(L)
        x_conv = np.column_stack([
            np.interp(full_idx, sub_idx, x_conv_sub[:, 0]),
            np.interp(full_idx, sub_idx, x_conv_sub[:, 1]),
        ])
        max_lw = np.interp(full_idx, sub_idx, max_lw_sub[:, 0])

        if L >= 5:
            k_sm = min(9, (L // 2) * 2 + 1)
            kern = np.ones(k_sm) / float(k_sm)
            for dim in (0, 1):
                pad = np.pad(x_conv[:, dim], k_sm // 2, mode="edge")
                x_conv[:, dim] = np.convolve(pad, kern, mode="valid")

        w_c = np.clip((tau_edge - 0.45) / 1.8, 0.0, 0.85)
        x_hyb = (1.0 - w_c[:, None]) * x_ref + w_c[:, None] * x_conv
        b_rec[start:stop] = x_hyb

        conf_crowd = np.clip((max_lw + 1.0) / 5.0, 0.2, 1.0)
        sigma[start:stop] = np.clip(
            0.5 + 1.8 * tau_edge / np.sqrt(conf_crowd), 0.4, 14.0
        ).astype(np.float32)
        regime[start:stop] = 3

    v_rec = np.zeros((m, 2), dtype=np.float32)
    if m > 4:
        v_rec[2:-2] = ((b_rec[4:] - b_rec[:-4]) * (FPS / 4.0)).astype(np.float32)
        v_rec[:2] = v_rec[2]
        v_rec[-2:] = v_rec[-3]
    return b_rec.astype(np.float32), v_rec, sigma, regime


def reconstruct_ball(f, ball, players, team, blocks):
    """Populate `f` with continuous reconstructed ball positions, velocities, uncertainty, and gap regime."""
    n = len(ball)
    f["ball_rec"] = np.full((n, 2), np.nan, dtype=np.float32)
    f["ball_vel_rec"] = np.zeros((n, 2), dtype=np.float32)
    f["ball_sigma"] = np.full(n, 15.0, dtype=np.float32)
    f["ball_gap_regime"] = np.zeros(n, dtype=np.int8)

    for _, lo, hi in blocks:
        sl = slice(lo, hi + 1)
        b_r, v_r, sig, reg = reconstruct_ball_block(ball[sl], players[sl, :, :2], team)
        f["ball_rec"][sl] = b_r
        f["ball_vel_rec"][sl] = v_r
        f["ball_sigma"][sl] = sig
        f["ball_gap_regime"][sl] = reg


def carrier_features(f, players, team, blocks):
    """Compute vectorized 22-player carrier attribution, physics Time-To-Intercept (TTI),
    top-2 H/A distances/kinematics, and inter-player transfer signatures across all graded blocks.

    Stores `f["carrier_sym"]` (n, 10) and `f["carrier_anti"]` (n, 14) in float32.
    """
    n = len(f["ball_seen"])
    home_mask = team == 0
    away_mask = team == 1
    home_slots = np.flatnonzero(home_mask)
    away_slots = np.flatnonzero(away_mask)

    sym_out = np.zeros((n, 10), dtype=np.float32)
    anti_out = np.zeros((n, 14), dtype=np.float32)

    b_rec = f["ball_rec"]
    v_rec = f["ball_vel_rec"]
    sigma = f["ball_sigma"]
    seen = f["ball_seen"]

    for _, lo, hi in blocks:
        sl = slice(lo, hi + 1)
        m = hi + 1 - lo
        b = b_rec[sl].astype(np.float32)
        vb = v_rec[sl].astype(np.float32)
        sig = sigma[sl].astype(np.float32)
        s_seg = seen[sl]
        p = players[sl, :, :2].astype(np.float32)

        vp = np.nan_to_num(_velocity(p, k=3), nan=0.0)
        sp_p = np.hypot(vp[:, :, 0], vp[:, :, 1])
        decel = np.zeros_like(sp_p)
        if m > 4:
            decel[2:-2] = np.clip((sp_p[:-4] - sp_p[4:]) * (FPS / 4.0), -8.0, 10.0)

        dx = p[:, :, 0] - b[:, 0:1]
        dy = p[:, :, 1] - b[:, 1:2]
        dist = np.hypot(dx, dy)
        valid_p = ~np.isnan(dist)
        dist_clean = np.where(valid_p, dist, 99.0)

        d_home_all = dist_clean[:, home_mask]
        d_away_all = dist_clean[:, away_mask]
        rows_m = np.arange(m)[:, None]
        kh = min(1, d_home_all.shape[1] - 1)
        ka = min(1, d_away_all.shape[1] - 1)
        h_idx2 = np.argpartition(d_home_all, kh, axis=1)[:, :2]
        a_idx2 = np.argpartition(d_away_all, ka, axis=1)[:, :2]
        dh_2 = np.sort(d_home_all[rows_m, h_idx2], axis=1)
        da_2 = np.sort(d_away_all[rows_m, a_idx2], axis=1)
        dh1, dh2 = dh_2[:, 0], dh_2[:, 1]
        da1, da2 = da_2[:, 0], da_2[:, 1]

        h1_global = home_slots[np.argmin(d_home_all, axis=1)]
        a1_global = away_slots[np.argmin(d_away_all, axis=1)]

        # Physics Time-To-Intercept (TTI) along projected ball trajectory (tau = 0.0s, 0.32s, 0.64s)
        tau_react = 0.25
        v_max = 7.5
        px_react = p[:, :, 0] + vp[:, :, 0] * tau_react
        py_react = p[:, :, 1] + vp[:, :, 1] * tau_react

        tti_min_p = np.full((m, p.shape[1]), 15.0, dtype=np.float32)
        for tau_proj in (0.0, 0.32, 0.64):
            bx_proj = b[:, 0:1] + vb[:, 0:1] * (tau_proj * 0.85)
            by_proj = b[:, 1:2] + vb[:, 1:2] * (tau_proj * 0.85)
            d_proj = np.hypot(px_react - bx_proj, py_react - by_proj)
            tti_tau = tau_react + np.where(valid_p, d_proj / v_max, 15.0)
            tti_min_p = np.minimum(tti_min_p, tti_tau)

        tti_h_all = tti_min_p[:, home_mask]
        tti_a_all = tti_min_p[:, away_mask]
        th_2 = np.sort(np.partition(tti_h_all, kh, axis=1)[:, :2], axis=1)
        ta_2 = np.sort(np.partition(tti_a_all, ka, axis=1)[:, :2], axis=1)
        tti_h1, tti_h2 = th_2[:, 0], th_2[:, 1]
        tti_a1, tti_a2 = ta_2[:, 0], ta_2[:, 1]

        rel_v = np.hypot(vp[:, :, 0] - vb[:, 0:1], vp[:, :, 1] - vb[:, 1:2])
        u_pb_x = -dx / np.maximum(dist_clean, 0.3)
        u_pb_y = -dy / np.maximum(dist_clean, 0.3)
        v_close = np.where(
            valid_p, np.clip(vp[:, :, 0] * u_pb_x + vp[:, :, 1] * u_pb_y, -6.0, 9.0), 0.0
        )

        idx_m = np.arange(m)
        vh1_rel = rel_v[idx_m, h1_global]
        va1_rel = rel_v[idx_m, a1_global]
        ch1_close = v_close[idx_m, h1_global]
        ca1_close = v_close[idx_m, a1_global]
        decel_h1 = np.clip(decel[idx_m, h1_global], 0.0, 8.0) * np.exp(-dh1 / 2.5)
        decel_a1 = np.clip(decel[idx_m, a1_global], 0.0, 8.0) * np.exp(-da1 / 2.5)

        logits_p = (
            -1.15 * dist_clean
            - 1.60 * (tti_min_p - tau_react)
            - 0.25 * np.clip(rel_v, 0.0, 12.0)
            + 0.20 * np.clip(decel, 0.0, 6.0)
        )
        logits_p = np.where(valid_p, logits_p, -1e4)
        max_l = np.maximum(logits_p.max(axis=1, keepdims=True), -4.2)
        exp_p = np.where(valid_p, np.exp(np.clip(logits_p - max_l, -25.0, 0.0)), 0.0)
        exp_free = np.exp(np.clip(-4.2 - max_l[:, 0], -25.0, 0.0))
        denom = np.maximum(exp_p.sum(axis=1) + exp_free, 1e-9)
        alpha = (exp_p / denom[:, None]).astype(np.float64)

        # Bridge the 22-player attention distribution across missing-ball gaps
        for start, stop in _runs(~s_seg):
            if start == 0 or stop == m:
                continue
            L = stop - start
            w_in = slice(max(0, start - 6), start)
            w_out = slice(stop, min(m, stop + 6))
            a_in = alpha[w_in].mean(axis=0)
            a_out = alpha[w_out].mean(axis=0)
            u = ((np.arange(L) + 1) / float(L + 1))[:, None]
            a_bridge = (1.0 - u) * a_in[None, :] + u * a_out[None, :]
            t_edge = np.minimum(np.arange(1, L + 1), np.arange(L, 0, -1)) / float(FPS)
            w_loc = np.exp(-((t_edge / 0.55) ** 2))[:, None]
            alpha[start:stop] = w_loc * alpha[start:stop] + (1.0 - w_loc) * a_bridge

        prob_h = alpha[:, home_mask].sum(axis=1).astype(np.float32)
        prob_a = alpha[:, away_mask].sum(axis=1).astype(np.float32)
        carrier_max = alpha.max(axis=1).astype(np.float32)

        cs_alpha = np.vstack([np.zeros((1, p.shape[1]), dtype=np.float64), np.cumsum(alpha, axis=0)])

        def past_fut_alpha(win):
            p_lo = np.maximum(0, idx_m - win)
            p_hi = idx_m + 1
            f_lo = np.minimum(m, idx_m + 1)
            f_hi = np.minimum(m, idx_m + 1 + win)
            a_past = (cs_alpha[p_hi] - cs_alpha[p_lo]) / np.maximum(1, (p_hi - p_lo))[:, None]
            a_fut = (cs_alpha[f_hi] - cs_alpha[f_lo]) / np.maximum(1, (f_hi - f_lo))[:, None]
            return a_past.astype(np.float32), a_fut.astype(np.float32)

        ap_05, af_05 = past_fut_alpha(12)
        ap_12, af_12 = past_fut_alpha(30)

        same_p_all = ap_05 * af_05
        same_p_h = same_p_all[:, home_mask].sum(axis=1)
        same_p_a = same_p_all[:, away_mask].sum(axis=1)
        same_player_continuity = same_p_h + same_p_a
        same_player_signed = same_p_h - same_p_a

        ph_past_05 = ap_05[:, home_mask].sum(axis=1)
        pa_past_05 = ap_05[:, away_mask].sum(axis=1)
        ph_fut_05 = af_05[:, home_mask].sum(axis=1)
        pa_fut_05 = af_05[:, away_mask].sum(axis=1)

        ph_past_12 = ap_12[:, home_mask].sum(axis=1)
        pa_past_12 = ap_12[:, away_mask].sum(axis=1)
        ph_fut_12 = af_12[:, home_mask].sum(axis=1)
        pa_fut_12 = af_12[:, away_mask].sum(axis=1)
        same_p12_h = (ap_12[:, home_mask] * af_12[:, home_mask]).sum(axis=1)
        same_p12_a = (ap_12[:, away_mask] * af_12[:, away_mask]).sum(axis=1)

        pass_h = np.maximum(0.0, ph_past_12 * ph_fut_12 - same_p12_h)
        pass_a = np.maximum(0.0, pa_past_12 * pa_fut_12 - same_p12_a)
        teammate_pass_sym = pass_h + pass_a
        teammate_pass_anti = pass_h - pass_a

        inter_a2h_05 = pa_past_05 * ph_fut_05
        inter_h2a_05 = ph_past_05 * pa_fut_05
        inter_sym_05 = inter_a2h_05 + inter_h2a_05
        inter_anti_05 = inter_a2h_05 - inter_h2a_05

        inter_a2h_12 = pa_past_12 * ph_fut_12
        inter_h2a_12 = ph_past_12 * pa_fut_12
        inter_sym_12 = inter_a2h_12 + inter_h2a_12
        inter_anti_12 = inter_a2h_12 - inter_h2a_12

        w_sig = 1.0 / (1.0 + (sig / 2.2) ** 2)

        vb_speed = np.hypot(vb[:, 0], vb[:, 1])
        tti_min_both = np.minimum(tti_h1, tti_a1)
        tti_conflict = (
            np.exp(-(tti_h1 + tti_a1 - 2.0 * tau_react) / 1.2)
            * np.exp(-np.abs(tti_h1 - tti_a1) / 0.35)
        )
        decel_sym = np.maximum(decel_h1 * prob_h, decel_a1 * prob_a)

        sym_out[sl, 0] = np.log1p(np.clip(sig, 0.0, 15.0)) / np.log1p(15.0)
        sym_out[sl, 1] = np.log1p(np.clip(vb_speed, 0.0, 40.0))
        sym_out[sl, 2] = np.log1p(np.clip(tti_min_both, 0.0, 10.0))
        sym_out[sl, 3] = tti_conflict
        sym_out[sl, 4] = carrier_max
        sym_out[sl, 5] = same_player_continuity
        sym_out[sl, 6] = teammate_pass_sym
        sym_out[sl, 7] = inter_sym_05
        sym_out[sl, 8] = inter_sym_12
        sym_out[sl, 9] = np.tanh(decel_sym / 3.0)

        d_rec_diff_1 = np.tanh((da1 - dh1) / 2.5) * w_sig
        d_rec_diff_2 = np.tanh((da2 - dh2) / 3.5) * w_sig
        tti_diff_1 = np.tanh((tti_a1 - tti_h1) / 0.45) * w_sig
        tti_diff_2 = np.tanh((tti_a2 - tti_h2) / 0.60) * w_sig

        cs_tti1 = np.concatenate(([0.0], np.cumsum(tti_diff_1.astype(np.float64))))
        f_lo10 = np.minimum(m, idx_m + 1)
        f_hi10 = np.minimum(m, idx_m + 11)
        tti_fut_04 = (
            (cs_tti1[f_hi10] - cs_tti1[f_lo10]) / np.maximum(1, f_hi10 - f_lo10)
        ).astype(np.float32)

        carrier_prob_diff = prob_h - prob_a
        align_h = np.exp(-vh1_rel / 3.0) * np.exp(-dh1 / 2.2)
        align_a = np.exp(-va1_rel / 3.0) * np.exp(-da1 / 2.2)
        vel_align_diff = (align_h - align_a) * w_sig

        close_diff = np.tanh(
            (ch1_close * np.exp(-dh1 / 6.0) - ca1_close * np.exp(-da1 / 6.0)) / 3.0
        ) * w_sig
        decel_diff = np.tanh((decel_h1 - decel_a1) / 3.0) * w_sig

        raw_own = 0.70 * carrier_prob_diff + 0.30 * tti_diff_1
        pad_own = np.pad(raw_own, 8, mode="edge")
        gap_carrier_own = np.convolve(pad_own, np.ones(17) / 17.0, mode="valid").astype(np.float32)

        anti_out[sl, 0] = d_rec_diff_1
        anti_out[sl, 1] = d_rec_diff_2
        anti_out[sl, 2] = tti_diff_1
        anti_out[sl, 3] = tti_diff_2
        anti_out[sl, 4] = tti_fut_04
        anti_out[sl, 5] = carrier_prob_diff
        anti_out[sl, 6] = vel_align_diff
        anti_out[sl, 7] = close_diff
        anti_out[sl, 8] = decel_diff
        anti_out[sl, 9] = same_player_signed
        anti_out[sl, 10] = teammate_pass_anti
        anti_out[sl, 11] = inter_anti_05
        anti_out[sl, 12] = inter_anti_12
        anti_out[sl, 13] = gap_carrier_own

    f["carrier_sym"] = sym_out
    f["carrier_anti"] = anti_out

