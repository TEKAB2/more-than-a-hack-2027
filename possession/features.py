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
    return f


class _null:
    """Context manager that silences 'mean of empty slice' warnings."""

    def __enter__(self):
        import warnings
        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)
