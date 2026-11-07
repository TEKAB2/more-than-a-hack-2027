"""Per-frame evidence -> log-emission scores for the three states H, A, D.

Two log-odds per frame:
    s_dead : log P(dead) / P(alive)
    s_team : log P(home) / P(away), given alive

Every weight lives in PARAMS. They are set from football/physics reasoning, NOT tuned on the
public 5 minutes (that is forbidden and, as EVALUATION.md shows, picks the wrong answer anyway).
v2 replaces these hand weights by a model trained on OTHER games (see PLAN.md).
"""

import numpy as np

FPS = 25

PARAMS = {
    # --- team evidence from the ball ---
    "w_ball": 3.0,            # max |s_team| from one frame of clear control
    "control_radius": 3.0,    # m: beyond this the nearest player is not "on" the ball
    "margin_scale": 1.5,      # m: how much closer one team must be to be convincing
    "slow_ball": 6.0,         # m/s: below this the ball can be controlled
    "fast_ball_factor": 0.25, # evidence kept when the ball is travelling (pass, deflection)
    # --- weak priors ---
    "w_shape": 0.15,          # team block pushed up -> in possession
    "shape_scale": 8.0,       # m
    "w_tag": 0.3,             # Smart Tagging phase team
    # --- dead-ball evidence ---
    "dead_base": -2.0,        # alive is the default state
    "w_out": 5.0,             # ball observed outside the pitch
    "w_activity": 1.5,        # players clearly slower than usual for this half -> stoppage
    "activity_low": 0.7,      # fraction of the half's median activity below which the cue starts
    "activity_scale": 0.25,   # fraction of the median for one unit of evidence
    "w_fast_alive": 2.0,      # ball travelling fast inside the pitch -> alive
    "fast_alive_speed": 8.0,  # m/s
    "w_long_gap": 0.5,        # ball lost for long -> slightly more likely dead
    "long_gap": 3 * FPS,
    "w_setpiece_dead": 3.0,   # before a detected restart
    "w_setpiece_alive": 2.5,  # just after a detected restart
    "w_restart_team": 2.0,    # restart awarded to the tagged team
    "restart_alive_frames": FPS,
    "setpiece_pre": 2 * FPS,  # frames before a set-piece window we still treat as dead
    "w_goal_dead": 2.5,
    "w_kickoff_dead": 8.0,    # before a half's opening kick-off nothing is alive
}


def ball_team_evidence(f, p):
    d_home, d_away = f["d_home"], f["d_away"]
    dmin = np.minimum(d_home, d_away)
    finite = np.isfinite(dmin) & f["ball_seen"]
    with np.errstate(invalid="ignore"):
        margin = np.where(finite, np.clip(np.nan_to_num(d_away - d_home), -10, 10), 0.0)
    closeness = np.where(finite, np.clip((p["control_radius"] - dmin) / p["control_radius"], 0, 1),
                         0.0)
    speed = np.nan_to_num(f["ball_speed"], nan=p["slow_ball"])
    pace = np.where(speed < p["slow_ball"], 1.0, p["fast_ball_factor"])
    return (p["w_ball"] * np.tanh(margin / p["margin_scale"]) * closeness * pace).astype(np.float32)


def find_restart(f, start, end, lo, hi, centre=None):
    """First frame in [start-2s, end+1s] where a resting ball starts to move. None if not found.

    "Resting" = seen and within 0.3 m of the same spot for the previous 8 frames; "moves" = more
    than 0.6 m away 3 frames later. Position-based rather than speed-based, so a soft kick-off pass
    is caught on the frame it is struck, not a second later when it has picked up speed.
    For a kick-off, `centre` = (x, y) of the centre spot: the ball must rest there.
    """
    b, seen = f["ball"], f["ball_seen"]
    # The rest window may be shorter than 8 frames at the very start of a half: the first graded
    # frame of the first half IS the kick-off, so requiring 8 frames before it would push the
    # detected strike a second late.
    a, z = max(lo + 1, start - 2 * FPS), min(hi - 4, end + FPS)
    for t in range(a, z):
        w0 = max(lo, t - 8)
        if not (seen[t - 1] and seen[w0]) or np.isnan(b[t + 3, 0]):
            continue
        rest = b[t - 1]
        if np.hypot(*(b[w0:t] - rest).T).max() > 0.3:
            continue
        if np.hypot(*(b[t + 3] - rest)) < 0.6:
            continue
        if centre is not None and np.hypot(rest[0] - centre[0], rest[1] - centre[1]) > 3.0:
            continue
        return t
    return None


def compute(f, blocks, tag_prior=None, set_pieces=(), goals=(), params=None, weights=None,
            pitch=None):
    """(s_team, s_dead, restarts). With `weights` (possession/weights.json) the frame evidence comes
    from the models trained on other games; otherwise from the hand-set PARAMS (v1)."""
    p = dict(PARAMS, **(params or {}))
    if weights is not None:
        p.update(weights.get("params", {}))
    p["pitch"] = pitch
    n = len(f["ball_seen"])

    if weights is not None:
        from possession import model
        s_team, s_dead = model.emissions(f, blocks, pitch, ball_team_evidence(f, p) / p["w_ball"],
                                         weights)
        s_team = s_team.copy()
        s_dead = s_dead.copy()
    else:
        s_team, s_dead = hand_evidence(f, blocks, p)
    if tag_prior is not None:
        s_team += p["w_tag"] * np.clip(tag_prior, -2, 2)
    return add_restarts(f, blocks, s_team, s_dead, set_pieces, goals, p)


def hand_evidence(f, blocks, p):
    n = len(f["ball_seen"])
    s_team = ball_team_evidence(f, p)
    s_team += p["w_shape"] * np.tanh(np.nan_to_num(f["shape_adv"]) / p["shape_scale"])

    s_dead = np.full(n, p["dead_base"], dtype=np.float32)
    s_dead += p["w_out"] * f["ball_out"]
    # Activity relative to the half's own median: self-calibrating, no absolute speed assumed.
    for _, first, last in blocks:
        act = f["activity"][first:last + 1]
        med = np.nanmedian(act) if np.any(~np.isnan(act)) else np.nan
        if not np.isfinite(med) or med <= 0:
            continue
        act = np.nan_to_num(act, nan=med)
        low = np.clip((p["activity_low"] * med - act) / (p["activity_scale"] * med), 0.0, 2.0)
        s_dead[first:last + 1] += p["w_activity"] * low
    fast_inside = f["ball_seen"] & ~f["ball_out"] & \
        (np.nan_to_num(f["ball_speed"]) > p["fast_alive_speed"])
    s_dead -= p["w_fast_alive"] * fast_inside
    s_dead += p["w_long_gap"] * (f["gap_len"] > p["long_gap"])
    return s_team, s_dead


def add_restarts(f, blocks, s_team, s_dead, set_pieces, goals, p):
    n = len(s_dead)
    block_of = np.full(n, -1)
    for i, (_, first, last) in enumerate(blocks):
        block_of[first:last + 1] = i

    restarts = []
    for start, end, kind, side in set_pieces:
        inside = [t for t in (start, (start + end) // 2, end) if 0 <= t < n and block_of[t] >= 0]
        if not inside:
            continue
        _, lo, hi = blocks[block_of[inside[0]]]
        start, end = max(start, lo), min(end, hi)
        centre = (p["pitch"][0] / 2, p["pitch"][1] / 2) if "kick off" in kind and p.get("pitch") \
            else None
        restart = find_restart(f, start, end, lo, hi + 1, centre)
        if restart is None:
            restart = (start + end) // 2
        # A half's opening kick-off: dead from the start of the graded half. Only if the tag
        # really sits near the start of the half, so a mis-tagged kick-off cannot kill a half.
        opening = "kick off (start)" in kind and restart - lo < 5 * 60 * FPS
        dead_from = lo if opening else max(lo, start - p["setpiece_pre"])
        w_dead = p["w_kickoff_dead"] if opening else p["w_setpiece_dead"]
        s_dead[dead_from:restart] += w_dead
        alive_to = min(hi + 1, restart + p["restart_alive_frames"])
        s_dead[restart:alive_to] -= p["w_setpiece_alive"]
        if side is not None:
            s_team[restart:alive_to] += p["w_restart_team"] * (1.0 if side == 0 else -1.0)
        restarts.append((restart, kind))

    kickoff_restarts = sorted(r for r, kind in restarts if "kick off" in kind)
    for start, end, _ in goals:
        following = [r for r in kickoff_restarts if r > end]
        if following:
            s_dead[end:following[0]] += p["w_goal_dead"]

    return s_team, s_dead, restarts
