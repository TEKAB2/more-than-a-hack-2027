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
}


def log_emissions(s_team, s_dead):
    """(n, 3) log-probabilities from the two log-odds."""
    log_dead = -np.logaddexp(0.0, -s_dead)     # log sigmoid(s_dead)
    log_alive = -np.logaddexp(0.0, s_dead)     # log (1 - sigmoid)
    log_home = -np.logaddexp(0.0, -s_team)
    log_away = -np.logaddexp(0.0, s_team)
    return np.stack([log_alive + log_home, log_alive + log_away, log_dead], axis=1)


def viterbi(log_em, costs=None, change_bonus=None):
    c = dict(COSTS, **(costs or {}))
    trans_base = np.array([
        [0.0, -c["switch"], -c["to_dead"]],
        [-c["switch"], 0.0, -c["to_dead"]],
        [-c["restart"], -c["restart"], 0.0],
    ])
    n = len(log_em)
    back = np.zeros((n, 3), dtype=np.int8)
    score = log_em[0].astype(np.float64).copy()
    if change_bonus is None:
        for t in range(1, n):
            cand = score[:, None] + trans_base          # cand[i, j]: come from i into j
            best = cand.argmax(axis=0)
            back[t] = best
            score = cand[best, np.arange(3)] + log_em[t]
    else:
        bonus = np.asarray(change_bonus, dtype=np.float64)
        for t in range(1, n):
            trans = trans_base.copy()
            sw = max(0.0, c["switch"] - bonus[t])
            trans[0, 1] = -sw
            trans[1, 0] = -sw
            cand = score[:, None] + trans
            best = cand.argmax(axis=0)
            back[t] = best
            score = cand[best, np.arange(3)] + log_em[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(s_team, s_dead, blocks, n_frames, costs=None, change_bonus=None):
    """Label array (n_frames,) of 'H'/'A'/'D', 'X' outside the graded half blocks."""
    labels = np.full(n_frames, "X", dtype="<U1")
    log_em = log_emissions(s_team.astype(np.float64), s_dead.astype(np.float64))
    for _, first, last in blocks:
        cb = None if change_bonus is None else change_bonus[first:last + 1]
        path = viterbi(log_em[first:last + 1], costs, change_bonus=cb)
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
