"""Possession as a hidden state: Viterbi over {H, A, D} with switch costs.

Why a state model: spellF1 scores whole possessions, so the enemy is flicker. A switch cost means
the decoder changes team only when the accumulated evidence for the other team exceeds the cost
-- a deflection or a lost duel (a few frames of weak evidence) does not pay for it; a controlled
reception (many frames of strong evidence) does. That is the challenge's own definition of a
change of possession, expressed as an optimisation.
"""

import numpy as np

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])

COSTS = {
    "switch": 12.0,    # H <-> A
    "to_dead": 10.0,    # alive -> D
    "restart": 10.0,    # D -> alive
    # Optional dynamic switch modulation by p_change(t) (disabled by default: change_alpha=0.0)
    "change_alpha": 0.0,         # max switch cost discount at a detected change peak
    "change_guard": 0.0,         # extra switch cost added during quiet/duel regions (p_change < change_thresh)
    "switch_min": 4.0,           # floor on switch cost at a change peak
    "change_thresh": 0.35,       # minimum p_change to trigger a peak discount
    "change_nms": 18,            # non-maximum suppression radius (frames) between peaks
    "directional_gate": True,    # only discount H->A (or A->H) when s_team shifts towards the destination team
}


def log_emissions(s_team, s_dead):
    """(n, 3) log-probabilities from the two log-odds."""
    log_dead = -np.logaddexp(0.0, -s_dead)     # log sigmoid(s_dead)
    log_alive = -np.logaddexp(0.0, s_dead)     # log (1 - sigmoid)
    log_home = -np.logaddexp(0.0, -s_team)
    log_away = -np.logaddexp(0.0, s_team)
    return np.stack([log_alive + log_home, log_alive + log_away, log_dead], axis=1)


def nms_change_peaks(p_change, radius=18, min_prob=0.35):
    """Isolate local maxima of `p_change` over +/- `radius` frames above `min_prob`, with a
    1-frame shoulder (0.85x) so Viterbi has a 3-frame landing zone at the exact turnover."""
    p = np.clip(np.asarray(p_change, dtype=np.float64), 0.0, 1.0)
    n = len(p)
    if n < 2 * radius + 1 or radius <= 0:
        return np.where(p >= min_prob, p, 0.0)
    padded = np.pad(p, radius, mode="edge")
    max_win = np.maximum.reduce([padded[i:i + n] for i in range(2 * radius + 1)])
    is_peak = (p >= max_win - 1e-9) & (p >= min_prob)
    out = np.where(is_peak, p, 0.0)
    if n >= 3:
        out_dil = out.copy()
        out_dil[1:] = np.maximum(out_dil[1:], 0.85 * out[:-1])
        out_dil[:-1] = np.maximum(out_dil[:-1], 0.85 * out[1:])
        return out_dil
    return out


def switch_schedule(n, costs=None, p_change=None, s_team=None):
    """Per-frame (sw_h2a, sw_a2h) positive switch costs for H->A and A->H transitions.

    When `p_change` is None or `change_alpha == 0`, returns constant arrays equal to `c['switch']`.
    When active:
      * outside change peaks (`p_change < change_thresh`), switch cost is raised by `change_guard`
        to absorb deflections and lost duels without inventing false possessions;
      * on NMS-isolated peaks (`p_change >= change_thresh`), switch cost drops down to `switch_min`
        (by up to `change_alpha`), allowing short possessions to survive and snapping boundaries
        onto the exact turnover frame;
      * if `directional_gate` is True and `s_team` is given, H->A is only discounted when the
        post-frame team logit shifts towards Away (and vice versa for A->H).
    """
    c = dict(COSTS, **(costs or {}))
    base = float(c["switch"])
    alpha = float(c.get("change_alpha", 0.0))
    if p_change is None or alpha == 0.0:
        sw = np.full(n, base, dtype=np.float64)
        return sw, sw

    guard = float(c.get("change_guard", 0.0))
    c_min = float(c.get("switch_min", 4.0))
    thr = float(c.get("change_thresh", 0.35))
    nms_r = int(c.get("change_nms", 18))

    pc_raw = np.clip(np.asarray(p_change, dtype=np.float64), 0.0, 1.0)
    pc_peak = nms_change_peaks(pc_raw, radius=nms_r, min_prob=thr)
    act = np.where(pc_peak >= thr, (pc_peak - thr) / max(1.0 - thr, 1e-6), 0.0)
    quiet = np.clip(1.0 - pc_raw / max(thr, 1e-6), 0.0, 1.0)
    sw_quiet = base + guard * quiet
    sw_active = np.clip(sw_quiet - alpha * act, c_min, base + guard)

    if s_team is not None and c.get("directional_gate", True):
        st = np.asarray(s_team, dtype=np.float64)
        w = 12
        cs = np.concatenate(([0.0], np.cumsum(st)))
        idx = np.arange(n)
        p_lo = np.maximum(0, idx - w)
        f_hi = np.minimum(n, idx + w)
        past_m = (cs[idx + 1] - cs[p_lo]) / np.maximum(1, idx + 1 - p_lo)
        fut_m = (cs[f_hi] - cs[idx]) / np.maximum(1, f_hi - idx)
        ok_h2a = (fut_m < past_m) & (fut_m < 0.5)
        ok_a2h = (fut_m > past_m) & (fut_m > -0.5)
        return np.where(ok_h2a, sw_active, sw_quiet), np.where(ok_a2h, sw_active, sw_quiet)
    return sw_active, sw_active


def viterbi(log_em, costs=None, change_bonus=None, p_change=None, s_team=None):
    c = dict(COSTS, **(costs or {}))
    n = len(log_em)
    back = np.zeros((n, 3), dtype=np.int8)
    score = log_em[0].astype(np.float64).copy()

    use_schedule = (p_change is not None) and (float(c.get("change_alpha", 0.0)) > 0.0)
    if not use_schedule and change_bonus is None:
        trans = np.array([
            [0.0, -c["switch"], -c["to_dead"]],
            [-c["switch"], 0.0, -c["to_dead"]],
            [-c["restart"], -c["restart"], 0.0],
        ])
        for t in range(1, n):
            cand = score[:, None] + trans          # cand[i, j]: come from i into j
            best = cand.argmax(axis=0)
            back[t] = best
            score = cand[best, np.arange(3)] + log_em[t]
    elif use_schedule:
        sw_h2a, sw_a2h = switch_schedule(n, c, p_change=p_change, s_team=s_team)
        if change_bonus is not None:
            bonus = np.asarray(change_bonus, dtype=np.float64)
            sw_h2a = np.maximum(0.0, sw_h2a - bonus)
            sw_a2h = np.maximum(0.0, sw_a2h - bonus)
        to_dead = float(c["to_dead"])
        restart = float(c["restart"])
        trans_all = np.zeros((n, 3, 3), dtype=np.float64)
        trans_all[:, 0, 1] = -sw_h2a
        trans_all[:, 1, 0] = -sw_a2h
        trans_all[:, 0, 2] = -to_dead
        trans_all[:, 1, 2] = -to_dead
        trans_all[:, 2, 0] = -restart
        trans_all[:, 2, 1] = -restart
        idx3 = np.arange(3)
        for t in range(1, n):
            cand = score[:, None] + trans_all[t]
            best = cand.argmax(axis=0)
            back[t] = best
            score = cand[best, idx3] + log_em[t]
    else:
        bonus = np.asarray(change_bonus, dtype=np.float64)
        to_dead = float(c["to_dead"])
        restart = float(c["restart"])
        sw = np.maximum(0.0, float(c["switch"]) - bonus)
        trans_all = np.zeros((n, 3, 3), dtype=np.float64)
        trans_all[:, 0, 1] = -sw
        trans_all[:, 1, 0] = -sw
        trans_all[:, 0, 2] = -to_dead
        trans_all[:, 1, 2] = -to_dead
        trans_all[:, 2, 0] = -restart
        trans_all[:, 2, 1] = -restart
        idx3 = np.arange(3)
        for t in range(1, n):
            cand = score[:, None] + trans_all[t]
            best = cand.argmax(axis=0)
            back[t] = best
            score = cand[best, idx3] + log_em[t]

    path = np.empty(n, dtype=np.int8)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(s_team, s_dead, blocks, n_frames, costs=None, change_bonus=None, p_change=None,
           p_dead_edge=None, delta_t=None, sk_snap=None):
    """Label array (n_frames,) of 'H'/'A'/'D', 'X' outside the graded half blocks.
    When `costs.get('decoder') == 'semi_markov'`, delegates to `segment_decode.decode_segments`."""
    if isinstance(costs, dict) and costs.get("decoder") == "semi_markov":
        from possession import segment_decode
        return segment_decode.decode_segments(
            s_team, s_dead, blocks, n_frames, costs=costs, p_change=p_change,
            p_dead_edge=p_dead_edge, delta_t=delta_t, sk_snap=sk_snap,
        )
    labels = np.full(n_frames, "X", dtype="<U1")
    st64 = s_team.astype(np.float64)
    log_em = log_emissions(st64, s_dead.astype(np.float64))
    for _, first, last in blocks:
        cb = None if change_bonus is None else change_bonus[first:last + 1]
        pc = None if p_change is None else p_change[first:last + 1]
        path = viterbi(
            log_em[first:last + 1],
            costs,
            change_bonus=cb,
            p_change=pc,
            s_team=st64[first:last + 1],
        )
        labels[first:last + 1] = LETTERS[path]
    return labels


def insert_control_spells(labels, control, blocks, min_len=10, gap=3, tail=12):
    """Second pass: give back short possessions the Viterbi swallowed.

    Inside a possession of one team, a run of >= `min_len` frames where the OTHER team controls
    the ball (control sign, gaps of <= `gap` frames bridged) is a recovery: label it for that team
    from the start of the run until `tail` frames after its end, or until the original team
    controls the ball again. Dead-ball frames are never touched.
    """
    out = labels.copy()
    for _, lo, hi in blocks:
        c = control[lo:hi + 1]
        lab = out[lo:hi + 1]
        for team_letter, sign in (("H", 1), ("A", -1)):
            mine = c == sign
            # bridge small gaps
            idx = np.flatnonzero(mine)
            if not len(idx):
                continue
            runs, start, prev = [], idx[0], idx[0]
            for i in idx[1:]:
                if i - prev > gap + 1:
                    runs.append((start, prev + 1))
                    start = i
                prev = i
            runs.append((start, prev + 1))
            for a, b in runs:
                if b - a < min_len:
                    continue
                seg = lab[a:b]
                other = "A" if team_letter == "H" else "H"
                if not np.any(seg == other):
                    continue
                end = min(len(lab), b + tail)
                back = np.flatnonzero(c[b:end] == -sign)
                if len(back):
                    end = b + back[0]
                span = slice(a, end)
                lab[span] = np.where(lab[span] == "D", "D", team_letter)
        out[lo:hi + 1] = lab
    return out
