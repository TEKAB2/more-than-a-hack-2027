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


def viterbi(log_em, costs=None):
    c = dict(COSTS, **(costs or {}))
    trans = np.array([
        [0.0, -c["switch"], -c["to_dead"]],
        [-c["switch"], 0.0, -c["to_dead"]],
        [-c["restart"], -c["restart"], 0.0],
    ])
    n = len(log_em)
    back = np.zeros((n, 3), dtype=np.int8)
    score = log_em[0].astype(np.float64).copy()
    for t in range(1, n):
        cand = score[:, None] + trans          # cand[i, j]: come from i into j
        best = cand.argmax(axis=0)
        back[t] = best
        score = cand[best, np.arange(3)] + log_em[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(s_team, s_dead, blocks, n_frames, costs=None):
    """Label array (n_frames,) of 'H'/'A'/'D', 'X' outside the graded half blocks."""
    labels = np.full(n_frames, "X", dtype="<U1")
    log_em = log_emissions(s_team.astype(np.float64), s_dead.astype(np.float64))
    for _, first, last in blocks:
        path = viterbi(log_em[first:last + 1], costs)
        labels[first:last + 1] = LETTERS[path]
    return labels
