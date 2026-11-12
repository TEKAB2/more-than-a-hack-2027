"""Semi-Markov CRF / Boundary-IoU Segment Decoder in pure NumPy (`possession/segment_decode.py`).

Why a Semi-Markov Segment Decoder over 1st-order Viterbi:
  1. `spellF1` scores whole possession intervals `[a, b)` by temporal IoU >= {0.3, 0.4, 0.5, 0.6, 0.7}.
     A 1st-order Markov decoder (Viterbi) has zero memory of segment duration `L = b - a`, segment
     peak conviction `max_{t in [a, b)} s_team(t)`, or whether both boundaries `a` and `b` are
     anchored on sharp change-point peaks `p_change(a), p_change(b)`.
  2. In `semi_markov_viterbi`, we construct a sparse Candidate Boundary Lattice
     `0 = b_0 < b_1 < ... < b_K = N` containing:
       * all exact zero-crossings of `s_team(t)` and `s_dead(t)`,
       * all NMS local maxima of `p_change(t)` and `p_dead_edge(t)`,
       * all offset-regressed boundary candidates `round(t + delta_t[t])` from the neural
         Boundary-Offset Regression head (`delta_t`).
     Because the candidate set preserves the exact zero-crossings AND adds the offset-regressed
     and peak-aligned candidates, dynamic programming over `[b_i, b_j)` is a strict superset of
     Viterbi that evaluates segment-level potentials:
       * **Anti-Flicker Duration Barrier (`duel_len`, `duel_pen`)**: penalizes micro-excursions
         (`L < duel_len`, e.g. `< 1.2 s`) that lack extreme peak conviction, eliminating false
         duel/deflection splits without raising the switch cost on real turnovers.
       * **Short-Possession Recovery Potential (`rec_min_len`, `rec_max_len`, `w_peak_rec`)**:
         rewards brief controlled possessions (`1.2 s <= L <= 3.2 s`) whose interior reaches strong
         peak conviction (`max s_q > 1.0`) and whose boundaries coincide with change-head peaks
         `0.5 * (p_change(b_i) + p_change(b_j))` and physical contact cues `sk_snap`.
       * **Directional Boundary Entry Discount (`w_bound_enter`, `w_snap_enter`, `w_offset_align`)**:
         discounts the switch cost at candidate node `b_i` when `p_change(b_i)`, `sk_snap(b_i)`, and
         the continuous offset regressor `t + delta_t[t] -> b_i` agree on an exact turnover at `b_i`.
"""

import numpy as np

from possession import decode

DEFAULT_SEGMENT_COSTS = {
    "switch": 21.0,
    "to_dead": 60.0,
    "restart": 60.0,
    # Semi-Markov duration & segment profile potentials
    "duel_len": 32,             # frames (1.28 s at 25 fps): micro-spells below this pay a duel penalty
    "duel_pen": 8.0,            # max nats penalty as L -> 0 for unconfirmed micro-spells
    "rec_min_len": 28,          # frames (1.12 s): start of short-recovery boost window
    "rec_max_len": 80,          # frames (3.20 s): end of short-recovery boost window
    "w_peak_rec": 3.5,          # segment bonus weight on tanh(peak_s / 2) * boundary_conf
    # Boundary entry discounts (at candidate node b_i)
    "w_bound_enter": 5.0,       # max switch discount from p_change(b_i)
    "w_snap_enter": 2.0,        # max switch discount from kinematic impulse/deflection sk_snap(b_i)
    "w_offset_enter": 2.5,      # max switch discount from continuous boundary-offset regressor agreement
    "switch_min": 5.0,          # floor on switch cost at a boundary node
    "duel_damp": 0.25,          # low-pass damping of s_team when p_change < 0.15
    "max_lookback": 50,         # max atomic intervals per segment before zero-cost continuation
}


def _offset_agreement_score(delta_t, p_change, n, radius=8):
    """Per-frame density of offset-regressed votes `u + delta_t[u] -> t` weighted by `p_change[u]`.

    Each frame `u` near a boundary predicts that the true transition sits at `u + delta_t[u]`.
    Accumulating these votes onto target frames `t` yields a sharp sub-second consensus peak at
    the exact boundary frame.
    """
    if delta_t is None:
        return np.zeros(n, dtype=np.float64)
    dt = np.asarray(delta_t, dtype=np.float64)
    pc = np.clip(np.asarray(p_change, dtype=np.float64), 0.0, 1.0)
    active = np.flatnonzero(pc >= 0.12)
    votes = np.zeros(n, dtype=np.float64)
    if not len(active):
        return votes
    targets = np.clip(np.round(active + dt[active]).astype(np.int64), 0, n - 1)
    weights = pc[active] * np.exp(-0.5 * (dt[active] / float(max(radius, 1))) ** 2)
    np.add.at(votes, targets, weights)
    # Smooth votes over a 3-frame kernel [0.5, 1.0, 0.5] and normalize to [0, 1]
    if n >= 3:
        votes = np.convolve(votes, np.array([0.4, 1.0, 0.4]), mode="same")
    return np.tanh(votes / 2.0)


def candidate_boundary_nodes(st_eff, sd, pc, pde, delta_t=None, sk_snap=None):
    """Build sorted array `0 = b_0 < b_1 < ... < b_K = n` of candidate transition frames.

    Includes all exact zero-crossings of `st_eff` and `sd` (so the Semi-Markov lattice contains
    the exact 1st-order Viterbi path as a feasible solution) PLUS local maxima of `p_change`,
    `p_dead_edge`, and offset-regressed consensus peaks.
    """
    n = len(st_eff)
    zc_team = np.flatnonzero(np.signbit(st_eff[1:]) != np.signbit(st_eff[:-1])) + 1
    zc_dead = np.flatnonzero(np.signbit(sd[1:]) != np.signbit(sd[:-1])) + 1
    pk_chg = np.flatnonzero(decode.nms_change_peaks(pc, radius=5, min_prob=0.14) >= 0.14)
    pk_de = np.flatnonzero(decode.nms_change_peaks(pde, radius=8, min_prob=0.14) >= 0.14)

    off_agree = _offset_agreement_score(delta_t, pc, n)
    pk_off = np.flatnonzero(decode.nms_change_peaks(off_agree, radius=5, min_prob=0.20) >= 0.20)

    # Also include +/- 2 frame neighbours around strong p_change peaks where zero-crossing is nearby
    raw = np.unique(np.concatenate(([0, n], zc_team, zc_dead, pk_chg, pk_de, pk_off)))
    raw = raw[(raw >= 0) & (raw <= n)]
    if len(raw) == 0 or raw[0] != 0:
        raw = np.insert(raw, 0, 0)
    if raw[-1] != n:
        raw = np.append(raw, n)

    # Insert coarse anchors every 60 frames inside long quiet stretches so `max_lookback` is never exceeded
    final_nodes = [0]
    for u, v in zip(raw[:-1], raw[1:]):
        while v - final_nodes[-1] > 65:
            final_nodes.append(final_nodes[-1] + 55)
        if v > final_nodes[-1]:
            final_nodes.append(int(v))
    return np.asarray(final_nodes, dtype=np.int64), off_agree


def semi_markov_viterbi(s_team, s_dead, p_change=None, p_dead_edge=None, delta_t=None,
                        sk_snap=None, costs=None):
    """Exact Semi-Markov Dynamic Programming over the candidate boundary lattice of one half block."""
    c = dict(DEFAULT_SEGMENT_COSTS, **(costs or {}))
    n = len(s_team)
    st = s_team.astype(np.float64)
    sd = s_dead.astype(np.float64)
    pc = np.clip(np.asarray(p_change, dtype=np.float64), 0.0, 1.0) if p_change is not None else np.zeros(n)
    pde = np.clip(np.asarray(p_dead_edge, dtype=np.float64), 0.0, 1.0) if p_dead_edge is not None else np.zeros(n)
    sk = np.asarray(sk_snap, dtype=np.float64) if sk_snap is not None else np.zeros(n)

    duel_damp = float(c.get("duel_damp", 0.25))
    if duel_damp > 0.0 and n > 9:
        st_smooth = np.convolve(st, np.ones(9) / 9.0, mode="same")
        quiet = np.clip((0.18 - pc) / 0.18, 0.0, 1.0) * duel_damp
        st_eff = (1.0 - quiet) * st + quiet * st_smooth
    else:
        st_eff = st

    log_em = decode.log_emissions(st_eff, sd)
    b_nodes, off_agree = candidate_boundary_nodes(st_eff, sd, pc, pde, delta_t=delta_t, sk_snap=sk)
    K = len(b_nodes) - 1

    cs_em = np.vstack([np.zeros((1, 3), dtype=np.float64), np.cumsum(log_em, axis=0)])
    pad_pc = np.pad(pc, 2, mode="edge")
    pc_max2 = np.maximum.reduce([pad_pc[i:i + n] for i in range(5)])

    node_pc = np.zeros(K + 1, dtype=np.float64)
    node_sk = np.zeros(K + 1, dtype=np.float64)
    node_off = np.zeros(K + 1, dtype=np.float64)
    node_dir_h2a = np.zeros(K + 1, dtype=bool)
    node_dir_a2h = np.zeros(K + 1, dtype=bool)

    cs_st = np.concatenate(([0.0], np.cumsum(st_eff)))
    for m_idx in range(1, K):
        t_m = int(b_nodes[m_idx])
        node_pc[m_idx] = pc_max2[t_m]
        node_sk[m_idx] = sk[t_m]
        node_off[m_idx] = off_agree[t_m]
        p_lo = max(0, t_m - 12)
        f_hi = min(n, t_m + 12)
        past_m = (cs_st[t_m] - cs_st[p_lo]) / max(1, t_m - p_lo)
        fut_m = (cs_st[f_hi] - cs_st[t_m]) / max(1, f_hi - t_m)
        node_dir_h2a[m_idx] = (fut_m < past_m) and (fut_m < 0.5)
        node_dir_a2h[m_idx] = (fut_m > past_m) and (fut_m > -0.5)

    atom_max_h = np.empty(K, dtype=np.float64)
    atom_max_a = np.empty(K, dtype=np.float64)
    for k_i in range(K):
        u, v = int(b_nodes[k_i]), int(b_nodes[k_i + 1])
        sl_st = st_eff[u:v]
        atom_max_h[k_i] = float(np.max(sl_st)) if v > u else 0.0
        atom_max_a[k_i] = float(np.max(-sl_st)) if v > u else 0.0

    dp = np.full((K + 1, 3), -1e18, dtype=np.float64)
    back_node = np.zeros((K + 1, 3), dtype=np.int32)
    back_state = np.full((K + 1, 3), -1, dtype=np.int8)
    dp[0, :] = 0.0

    sw_base = float(c["switch"])
    sw_min = float(c.get("switch_min", 5.0))
    to_dead = float(c["to_dead"])
    restart = float(c["restart"])
    duel_len = float(c.get("duel_len", 32))
    duel_pen = float(c.get("duel_pen", 8.0))
    rec_min = float(c.get("rec_min_len", 28))
    rec_max = float(c.get("rec_max_len", 80))
    w_be = float(c.get("w_bound_enter", 5.0))
    w_se = float(c.get("w_snap_enter", 2.0))
    w_oe = float(c.get("w_offset_enter", 2.5))
    w_pr = float(c.get("w_peak_rec", 3.5))
    max_lb = int(c.get("max_lookback", 50))

    for j in range(1, K + 1):
        t_j = int(b_nodes[j])
        em_j = cs_em[t_j]
        i_min = max(0, j - max_lb)
        peak_h = -1e9
        peak_a = -1e9

        for i in range(j - 1, i_min - 1, -1):
            if atom_max_h[i] > peak_h:
                peak_h = atom_max_h[i]
            if atom_max_a[i] > peak_a:
                peak_a = atom_max_a[i]

            t_i = int(b_nodes[i])
            L = float(t_j - t_i)
            seg_em = em_j - cs_em[t_i]

            dur_mod_h = 0.0
            dur_mod_a = 0.0
            if 0 < i and j < K:
                if L < duel_len:
                    pen = duel_pen * (1.0 - L / max(duel_len, 1.0))
                    dur_mod_h -= pen
                    dur_mod_a -= pen
                elif rec_min <= L <= rec_max:
                    b_avg = 0.5 * (node_pc[i] + node_pc[j])
                    if b_avg >= 0.18:
                        if peak_h > 0.9:
                            dur_mod_h += w_pr * b_avg * np.tanh(peak_h / 2.0)
                        if peak_a > 0.9:
                            dur_mod_a += w_pr * b_avg * np.tanh(peak_a / 2.0)

            if i == 0:
                for q in range(3):
                    sc = seg_em[q] + (dur_mod_h if q == 0 else dur_mod_a if q == 1 else 0.0)
                    if sc > dp[j, q]:
                        dp[j, q] = sc
                        back_node[j, q] = 0
                        back_state[j, q] = -1
            else:
                b_bonus_i = w_be * node_pc[i] + w_se * node_sk[i] + w_oe * node_off[i]
                sw_a2h = max(sw_min, sw_base - (b_bonus_i if node_dir_a2h[i] else 0.0))
                sw_h2a = max(sw_min, sw_base - (b_bonus_i if node_dir_h2a[i] else 0.0))

                c_from_a = dp[i, 1] - sw_a2h
                c_from_d = dp[i, 2] - restart
                if c_from_a >= c_from_d:
                    best_prev_h, p_state_h = c_from_a, 1
                else:
                    best_prev_h, p_state_h = c_from_d, 2
                if i == i_min and dp[i, 0] > best_prev_h:
                    best_prev_h, p_state_h = dp[i, 0], 0
                sc_h = best_prev_h + seg_em[0] + dur_mod_h
                if sc_h > dp[j, 0]:
                    dp[j, 0] = sc_h
                    back_node[j, 0] = i
                    back_state[j, 0] = p_state_h

                c_from_h = dp[i, 0] - sw_h2a
                if c_from_h >= c_from_d:
                    best_prev_a, p_state_a = c_from_h, 0
                else:
                    best_prev_a, p_state_a = c_from_d, 2
                if i == i_min and dp[i, 1] > best_prev_a:
                    best_prev_a, p_state_a = dp[i, 1], 1
                sc_a = best_prev_a + seg_em[1] + dur_mod_a
                if sc_a > dp[j, 1]:
                    dp[j, 1] = sc_a
                    back_node[j, 1] = i
                    back_state[j, 1] = p_state_a

                c_d_from_h = dp[i, 0] - to_dead
                c_d_from_a = dp[i, 1] - to_dead
                if c_d_from_h >= c_d_from_a:
                    best_prev_d, p_state_d = c_d_from_h, 0
                else:
                    best_prev_d, p_state_d = c_d_from_a, 1
                if i == i_min and dp[i, 2] > best_prev_d:
                    best_prev_d, p_state_d = dp[i, 2], 2
                sc_d = best_prev_d + seg_em[2]
                if sc_d > dp[j, 2]:
                    dp[j, 2] = sc_d
                    back_node[j, 2] = i
                    back_state[j, 2] = p_state_d

    path = np.empty(n, dtype=np.int8)
    cur_j = K
    cur_q = int(np.argmax(dp[K]))
    while cur_j > 0:
        prev_i = int(back_node[cur_j, cur_q])
        prev_q = int(back_state[cur_j, cur_q])
        path[b_nodes[prev_i]:b_nodes[cur_j]] = cur_q
        cur_j = prev_i
        cur_q = prev_q
    return path


def decode_segments(s_team, s_dead, blocks, n_frames, costs=None, p_change=None,
                    p_dead_edge=None, delta_t=None, sk_snap=None):
    """Decode full match using `semi_markov_viterbi` inside each graded half block."""
    labels = np.full(n_frames, "X", dtype="<U1")
    for _, first, last in blocks:
        sl = slice(first, last + 1)
        pc = None if p_change is None else p_change[sl]
        pde = None if p_dead_edge is None else p_dead_edge[sl]
        dt = None if delta_t is None else delta_t[sl]
        sk = None if sk_snap is None else sk_snap[sl]
        path = semi_markov_viterbi(
            s_team[sl], s_dead[sl], p_change=pc, p_dead_edge=pde,
            delta_t=dt, sk_snap=sk, costs=costs,
        )
        labels[sl] = decode.LETTERS[path]
    return labels
