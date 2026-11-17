"""Align broadcast tracking onto the tactical video timeline and expose a fused ball.

Algorithm (deterministic, numpy-only):
  1. Split the broadcast into continuous stretches where the homography ``valid`` flag is 1
     (live pitch view). Invalid stretches are replays, close-ups, crowd shots.
  2. For each stretch, recover a constant integer offset
     ``tactical_frame = broadcast_frame + delta`` by matching player positions (per team,
     greedy nearest / Hungarian-style assignment) against the tactical ATD.
  3. Reject a stretch when the median matched player distance exceeds ``MAX_MEDIAN_DIST_M``
     (~2.5 m) or when median player speeds disagree (slow-motion replay).
  4. Map the broadcast ball onto the tactical timeline; callers fill tactical NaNs with it.

Cache API (``cache/broadcast_aligned.npz``, never committed)::

    ball_xy          float32 (n_tactical, 2)   broadcast ball in metres on the tactical index
    accepted         uint8   (n_tactical,)     1 where a live aligned segment covers the frame
    segment_start    int32   (n_seg,)          broadcast inclusive start
    segment_end      int32   (n_seg,)          broadcast inclusive end
    segment_delta    int32   (n_seg,)          tactical = broadcast + delta (rejected -> 0)
    segment_ok       uint8   (n_seg,)          1 if the segment was accepted
    segment_med_dist float32 (n_seg,)          median player distance after align (metres)
    segment_speed_ratio float32 (n_seg,)       median_bc_speed / median_tac_speed

Aggregate helpers return counts / medians / fill rates only -- never frame rows.
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES, asset_paths, source_frames  # noqa: E402

# ---- tunables (football physics / camera sync, not match-specific) --------------------
MAX_MEDIAN_DIST_M = 2.5
SPEED_RATIO_LO = 0.55
SPEED_RATIO_HI = 1.45
MAX_BALL_SPEED_M_S = 35.0
DEFAULT_FPS = 25.0
# Coarse then fine offset search.
COARSE_STEP = 5
FINE_RADIUS = 4
# Sample every N frames inside a segment when scoring a candidate delta.
SAMPLE_STRIDE = 25
MIN_SEGMENT_FRAMES = 25  # ignore tiny valid blips (< 1 s)
MIN_MATCHED_PLAYERS = 6
SIGNATURE_STRIDE = 5

HOME, AWAY = 0, 1
PROVENANCE_TACTICAL = 0.0
PROVENANCE_BROADCAST = 1.0


# --------------------------------------------------------------------------- segments


def continuous_segments(valid, min_length=MIN_SEGMENT_FRAMES):
    """Inclusive (start, end) ranges of contiguous True values in ``valid``.

    Short blips below ``min_length`` are dropped -- they are almost never usable live action.
    """
    valid = np.asarray(valid, dtype=bool)
    if valid.size == 0:
        return []
    padded = np.concatenate(([False], valid, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1
    out = []
    for s, e in zip(starts.tolist(), ends.tolist()):
        if e - s + 1 >= min_length:
            out.append((int(s), int(e)))
    return out


# --------------------------------------------------------------------------- matching


def _pairwise_cost(a_xy, b_xy):
    """(n, m) Euclidean distances between two sets of (x, y) rows; NaN rows -> inf."""
    a = np.asarray(a_xy, dtype=np.float64)
    b = np.asarray(b_xy, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    bad_a = ~np.isfinite(a).all(axis=1)
    bad_b = ~np.isfinite(b).all(axis=1)
    d[bad_a] = np.inf
    d[:, bad_b] = np.inf
    return d


def assign_nearest(cost):
    """Deterministic greedy bipartite assignment (Hungarian surrogate for small n).

    Repeatedly takes the globally smallest finite cost among unused rows/cols. Ties break by
    (row, col) so the result is bit-stable across platforms.
    """
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.full(n, -1, dtype=np.int32), np.zeros(0, dtype=np.float64)

    flat = []
    for i in range(n):
        for j in range(m):
            c = cost[i, j]
            if np.isfinite(c):
                flat.append((c, i, j))
    flat.sort()  # cost, then row, then col

    used_r = np.zeros(n, dtype=bool)
    used_c = np.zeros(m, dtype=bool)
    pair = np.full(n, -1, dtype=np.int32)
    dists = []
    for c, i, j in flat:
        if used_r[i] or used_c[j]:
            continue
        used_r[i] = True
        used_c[j] = True
        pair[i] = j
        dists.append(c)
        if used_r.all() or used_c.all():
            break
    return pair, np.asarray(dists, dtype=np.float64)


def match_frame(tac_xy, tac_team, bc_xy, bc_team):
    """Match broadcast players to tactical by team; return distances of matched pairs."""
    dists = []
    for side in (HOME, AWAY):
        a = tac_xy[tac_team == side]
        b = bc_xy[bc_team == side]
        a = a[np.isfinite(a).all(axis=1)]
        b = b[np.isfinite(b).all(axis=1)]
        if a.size == 0 or b.size == 0:
            continue
        _, d = assign_nearest(_pairwise_cost(a, b))
        if d.size:
            dists.append(d)
    if not dists:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(dists)


def median_match_distance(tac_players_t, tac_team, bc_players_t, bc_team):
    """Median matched player distance (metres) on one pair of frames."""
    d = match_frame(tac_players_t[:, :2], tac_team, bc_players_t[:, :2], bc_team)
    if d.size < MIN_MATCHED_PLAYERS:
        return np.inf, d.size
    return float(np.median(d)), int(d.size)


def median_speed(players_t, team):
    """Median finite player speed on one frame (channel 2), both teams pooled."""
    speeds = players_t[:, 2]
    ok = np.isfinite(speeds) & np.isfinite(players_t[:, 0])
    if not ok.any():
        return np.nan
    return float(np.median(speeds[ok]))


# --------------------------------------------------------------------------- offset search


def _team_centroid_signal(players, team, side):
    """(n_frames, 2) nanmean of visible players of one team -- cheap sync signature."""
    mask = team == side
    xy = players[:, mask, :2].astype(np.float64)
    out = np.full((xy.shape[0], 2), np.nan, dtype=np.float64)
    finite = np.isfinite(xy).all(axis=2)
    counts = finite.sum(axis=1)
    # Zero-fill non-finite so the sum is well-defined; restore NaN where count==0.
    filled = np.where(finite[..., None], xy, 0.0)
    sums = filled.sum(axis=1)
    good = counts > 0
    out[good] = sums[good] / counts[good, None]
    return out


def _coarse_delta_candidates(tac_sig, bc_sig, seg_start, seg_end, n_tac, step=COARSE_STEP):
    """Shortlist of deltas from 1-D cross-correlation of centroid x (home + away)."""
    bc = bc_sig[seg_start:seg_end + 1]
    length = bc.shape[0]
    if length < MIN_SEGMENT_FRAMES:
        return []

    # Use x of home centroid; fall back to away if home is empty.
    sig_bc = bc[:, 0]
    if not np.isfinite(sig_bc).any():
        sig_bc = bc[:, 1]
    sig_tac = tac_sig[:, 0]
    if not np.isfinite(sig_tac).any():
        sig_tac = tac_sig[:, 1]

    # Subsample for speed.
    bc_idx = np.arange(0, length, SIGNATURE_STRIDE)
    bc_s = sig_bc[bc_idx]
    finite_bc = np.isfinite(bc_s)
    if finite_bc.sum() < 3:
        # Uniform grid over feasible deltas.
        delta_lo = -seg_start
        delta_hi = n_tac - 1 - seg_end
        return list(range(delta_lo, delta_hi + 1, step))

    delta_lo = int(-seg_start)
    delta_hi = int(n_tac - 1 - seg_end)
    if delta_lo > delta_hi:
        return []

    scores = {}
    for delta in range(delta_lo, delta_hi + 1, step):
        tac_idx = (seg_start + bc_idx) + delta
        if tac_idx.min() < 0 or tac_idx.max() >= n_tac:
            continue
        tac_s = sig_tac[tac_idx]
        both = finite_bc & np.isfinite(tac_s)
        if both.sum() < 3:
            continue
        err = np.median(np.abs(tac_s[both] - bc_s[both]))
        scores[delta] = float(err)

    if not scores:
        return list(range(delta_lo, delta_hi + 1, max(step * 5, 25)))

    # Keep the best few coarse peaks for fine search.
    ranked = sorted(scores, key=lambda d: (scores[d], d))
    return ranked[:5]


def score_delta(tac_players, tac_team, bc_players, bc_team, seg_start, seg_end, delta,
                sample_stride=SAMPLE_STRIDE):
    """Median-of-medians player distance and speed ratio for one (segment, delta)."""
    n_tac = tac_players.shape[0]
    dists = []
    speed_ratios = []
    for b in range(seg_start, seg_end + 1, sample_stride):
        t = b + delta
        if not 0 <= t < n_tac:
            return np.inf, np.nan, 0
        med, n_match = median_match_distance(
            tac_players[t], tac_team, bc_players[b], bc_team)
        if not np.isfinite(med):
            continue
        dists.append(med)
        vs_t = median_speed(tac_players[t], tac_team)
        vs_b = median_speed(bc_players[b], bc_team)
        if np.isfinite(vs_t) and np.isfinite(vs_b) and vs_t > 0.3:
            speed_ratios.append(vs_b / vs_t)
    if len(dists) < 2:
        return np.inf, np.nan, 0
    ratio = float(np.median(speed_ratios)) if speed_ratios else np.nan
    return float(np.median(dists)), ratio, len(dists)


def find_segment_offset(tac_players, tac_team, bc_players, bc_team, seg_start, seg_end,
                        tac_sig=None):
    """Best delta for one valid broadcast segment, plus diagnostics."""
    n_tac = tac_players.shape[0]
    delta_lo = int(-seg_start)
    delta_hi = int(n_tac - 1 - seg_end)
    if delta_lo > delta_hi:
        return {
            "delta": 0, "median_dist": float("inf"), "speed_ratio": float("nan"),
            "n_samples": 0, "accepted": False,
        }

    if tac_sig is None:
        home = _team_centroid_signal(tac_players, tac_team, HOME)
        away = _team_centroid_signal(tac_players, tac_team, AWAY)
        tac_sig = np.stack([home[:, 0], away[:, 0]], axis=1)

    bc_home = _team_centroid_signal(bc_players, bc_team, HOME)
    bc_away = _team_centroid_signal(bc_players, bc_team, AWAY)
    bc_sig = np.stack([bc_home[:, 0], bc_away[:, 0]], axis=1)

    seg_len = seg_end - seg_start + 1
    # Short segments: dense integer search is cheap. Long ones: coarse then fine.
    candidates = set()
    if seg_len <= 300 and (delta_hi - delta_lo) <= 800:
        candidates.update(range(delta_lo, delta_hi + 1))
    else:
        coarse = _coarse_delta_candidates(tac_sig, bc_sig, seg_start, seg_end, n_tac)
        for d0 in coarse:
            for d in range(d0 - FINE_RADIUS * 2, d0 + FINE_RADIUS * 2 + 1):
                if delta_lo <= d <= delta_hi:
                    candidates.add(int(d))
        # Always include a coarse grid so a weak signature cannot wipe the true peak.
        candidates.update(range(delta_lo, delta_hi + 1, COARSE_STEP))
    if not candidates:
        candidates = set(range(delta_lo, delta_hi + 1, COARSE_STEP))

    sample_stride = 10 if seg_len <= 300 else SAMPLE_STRIDE
    best = (np.inf, 0, np.nan, 0)  # med_dist, delta, speed_ratio, n_samples
    for delta in sorted(candidates):
        med, ratio, n_s = score_delta(
            tac_players, tac_team, bc_players, bc_team, seg_start, seg_end, delta,
            sample_stride=sample_stride)
        key = (med, abs(delta), delta)
        best_key = (best[0], abs(best[1]), best[1])
        if key < best_key:
            best = (med, delta, ratio, n_s)

    # Local integer polish around the winner (covers coarse-grid hits on long segments).
    polish = set()
    for d in range(best[1] - FINE_RADIUS, best[1] + FINE_RADIUS + 1):
        if delta_lo <= d <= delta_hi:
            polish.add(int(d))
    for delta in sorted(polish):
        med, ratio, n_s = score_delta(
            tac_players, tac_team, bc_players, bc_team, seg_start, seg_end, delta,
            sample_stride=max(5, sample_stride // 2))
        key = (med, abs(delta), delta)
        best_key = (best[0], abs(best[1]), best[1])
        if key < best_key:
            best = (med, delta, ratio, n_s)

    med, delta, ratio, n_s = best
    ok = (
        np.isfinite(med)
        and med <= MAX_MEDIAN_DIST_M
        and n_s >= 2
        and (not np.isfinite(ratio) or SPEED_RATIO_LO <= ratio <= SPEED_RATIO_HI)
    )
    return {
        "delta": int(delta),
        "median_dist": float(med) if np.isfinite(med) else float("inf"),
        "speed_ratio": float(ratio) if np.isfinite(ratio) else float("nan"),
        "n_samples": int(n_s),
        "accepted": bool(ok),
    }


# --------------------------------------------------------------------------- align core


def align_arrays(tac_players, tac_team, tac_ball, bc_players, bc_team, bc_ball, valid,
                 min_segment=MIN_SEGMENT_FRAMES):
    """Align broadcast arrays onto the tactical frame index.

    Parameters
    ----------
    tac_players : (n_tac, n_slots_t, 3) metres
    bc_players  : (n_bc, n_slots_b, 3) metres
    tac_ball, bc_ball : (n, 2)
    valid : (n_bc,) bool -- broadcast homography validity

    Returns
    -------
    dict with ball_xy, accepted, per-segment tables (see module docstring).
    """
    tac_players = np.asarray(tac_players, dtype=np.float32)
    bc_players = np.asarray(bc_players, dtype=np.float32)
    tac_ball = np.asarray(tac_ball, dtype=np.float32)
    bc_ball = np.asarray(bc_ball, dtype=np.float32)
    tac_team = np.asarray(tac_team, dtype=np.int8)
    bc_team = np.asarray(bc_team, dtype=np.int8)
    valid = np.asarray(valid, dtype=bool)

    n_tac = tac_players.shape[0]
    n_bc = bc_players.shape[0]
    if valid.shape[0] != n_bc:
        raise ValueError("valid length {} != broadcast frames {}".format(valid.shape[0], n_bc))

    ball_xy = np.full((n_tac, 2), np.nan, dtype=np.float32)
    accepted = np.zeros(n_tac, dtype=np.uint8)

    home = _team_centroid_signal(tac_players, tac_team, HOME)
    away = _team_centroid_signal(tac_players, tac_team, AWAY)
    tac_sig = np.stack([home[:, 0], away[:, 0]], axis=1)

    segments = continuous_segments(valid, min_length=min_segment)
    seg_start, seg_end, seg_delta, seg_ok = [], [], [], []
    seg_med, seg_ratio = [], []

    for s, e in segments:
        info = find_segment_offset(
            tac_players, tac_team, bc_players, bc_team, s, e, tac_sig=tac_sig)
        seg_start.append(s)
        seg_end.append(e)
        seg_delta.append(info["delta"])
        seg_ok.append(1 if info["accepted"] else 0)
        seg_med.append(info["median_dist"])
        seg_ratio.append(info["speed_ratio"])
        if not info["accepted"]:
            continue
        delta = info["delta"]
        for b in range(s, e + 1):
            t = b + delta
            if not 0 <= t < n_tac:
                continue
            accepted[t] = 1
            bx, by = bc_ball[b]
            if np.isfinite(bx) and np.isfinite(by):
                ball_xy[t] = (bx, by)

    return {
        "ball_xy": ball_xy,
        "accepted": accepted,
        "segment_start": np.asarray(seg_start, dtype=np.int32),
        "segment_end": np.asarray(seg_end, dtype=np.int32),
        "segment_delta": np.asarray(seg_delta, dtype=np.int32),
        "segment_ok": np.asarray(seg_ok, dtype=np.uint8),
        "segment_med_dist": np.asarray(seg_med, dtype=np.float32),
        "segment_speed_ratio": np.asarray(seg_ratio, dtype=np.float32),
        "n_tactical": np.int32(n_tac),
        "n_broadcast": np.int32(n_bc),
    }


def fuse_ball(tac_ball, bc_ball_on_tac, fps=DEFAULT_FPS, max_speed_m_s=MAX_BALL_SPEED_M_S,
              provenance=None):
    """Fill NaN tactical ball with aligned broadcast ball; kinematic gate ~35 m/s.

    Returns (ball, provenance) where provenance is float32 (n,) with
    0 = tactical, 1 = broadcast fill, NaN = still missing.
    """
    ball = np.array(tac_ball, dtype=np.float32, copy=True)
    bc = np.asarray(bc_ball_on_tac, dtype=np.float32)
    n = ball.shape[0]
    if provenance is None:
        prov = np.full(n, np.nan, dtype=np.float32)
    else:
        prov = np.array(provenance, dtype=np.float32, copy=True)

    tac_seen = np.isfinite(ball[:, 0]) & np.isfinite(ball[:, 1])
    prov[tac_seen] = PROVENANCE_TACTICAL

    missing = ~tac_seen
    bc_ok = np.isfinite(bc[:, 0]) & np.isfinite(bc[:, 1])
    fill = missing & bc_ok
    if not fill.any():
        return ball, prov

    # Kinematic sanity: reject a fill whose jump from the nearest known neighbour exceeds
    # max_speed (and reject chains that teleport). Walk forward then backward.
    max_step = max_speed_m_s / float(fps)
    candidate = ball.copy()
    candidate[fill] = bc[fill]

    good = tac_seen.copy()
    # Forward pass: accept fill only if close enough to last accepted position.
    last_xy = None
    last_t = None
    for t in range(n):
        if good[t]:
            last_xy = candidate[t].copy()
            last_t = t
            continue
        if not fill[t]:
            continue
        if last_xy is None:
            # No anchor yet -- accept tentatively; backward pass will check.
            good[t] = True
            last_xy = candidate[t].copy()
            last_t = t
            continue
        dt = t - last_t
        dist = float(np.hypot(candidate[t, 0] - last_xy[0], candidate[t, 1] - last_xy[1]))
        if dist <= max_step * dt + 1e-6:
            good[t] = True
            last_xy = candidate[t].copy()
            last_t = t

    # Backward pass from the end.
    last_xy = None
    last_t = None
    accept = np.zeros(n, dtype=bool)
    for t in range(n - 1, -1, -1):
        if tac_seen[t]:
            last_xy = candidate[t].copy()
            last_t = t
            accept[t] = True
            continue
        if not (fill[t] and good[t]):
            continue
        if last_xy is None:
            accept[t] = True
            last_xy = candidate[t].copy()
            last_t = t
            continue
        dt = last_t - t
        dist = float(np.hypot(candidate[t, 0] - last_xy[0], candidate[t, 1] - last_xy[1]))
        if dist <= max_step * dt + 1e-6:
            accept[t] = True
            last_xy = candidate[t].copy()
            last_t = t

    use = fill & accept & good
    ball[use] = bc[use]
    prov[use] = PROVENANCE_BROADCAST
    return ball, prov


def aggregate_metrics(tac_ball, aligned, fused_ball=None, graded_mask=None):
    """Aggregate-only complementarity stats (safe to print / share).

    Returns a plain dict of scalars and counts -- never per-frame rows.
    """
    tac = np.asarray(tac_ball, dtype=np.float32)
    bc = np.asarray(aligned["ball_xy"], dtype=np.float32)
    n = tac.shape[0]
    if graded_mask is None:
        graded_mask = np.ones(n, dtype=bool)
    else:
        graded_mask = np.asarray(graded_mask, dtype=bool)

    tac_seen = np.isfinite(tac[:, 0]) & np.isfinite(tac[:, 1])
    bc_seen = np.isfinite(bc[:, 0]) & np.isfinite(bc[:, 1])
    tac_miss = graded_mask & ~tac_seen
    fillable = tac_miss & bc_seen

    seg_ok = np.asarray(aligned["segment_ok"], dtype=bool)
    meds = np.asarray(aligned["segment_med_dist"], dtype=np.float64)
    out = {
        "n_frames": int(n),
        "n_graded": int(graded_mask.sum()),
        "n_segments": int(len(seg_ok)),
        "n_segments_accepted": int(seg_ok.sum()),
        "pct_segments_accepted": float(100.0 * seg_ok.mean()) if seg_ok.size else 0.0,
        "median_dist_accepted_m": float(np.median(meds[seg_ok])) if seg_ok.any() else float("nan"),
        "tac_ball_seen_pct": float(100.0 * tac_seen[graded_mask].mean()) if graded_mask.any() else 0.0,
        "bc_on_tac_seen_pct": float(100.0 * bc_seen[graded_mask].mean()) if graded_mask.any() else 0.0,
        "tac_missing_graded": int(tac_miss.sum()),
        "tac_missing_filled_by_bc": int(fillable.sum()),
        "pct_missing_filled": float(100.0 * fillable.sum() / tac_miss.sum()) if tac_miss.any() else 0.0,
    }
    if fused_ball is not None:
        fused = np.asarray(fused_ball, dtype=np.float32)
        fused_seen = np.isfinite(fused[:, 0]) & np.isfinite(fused[:, 1])
        out["fused_ball_seen_pct"] = float(
            100.0 * fused_seen[graded_mask].mean()) if graded_mask.any() else 0.0
        out["fused_gain_pp"] = out["fused_ball_seen_pct"] - out["tac_ball_seen_pct"]
    return out


# --------------------------------------------------------------------------- I/O


def default_cache_path(game=GAME, cache_dir=None):
    root = cache_dir or os.path.join(ROOT, "cache")
    return os.path.join(root, "{}_broadcast_aligned.npz".format(game))


def save_aligned(path, aligned):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    np.savez(path, **aligned)


def load_aligned(path):
    data = np.load(path)
    return {key: data[key] for key in data.files}


def _broadcast_assets_present(game, data_dir):
    paths = asset_paths(game=game, data_dir=data_dir, source="broadcast")
    needed = ("atd_meta", "atd_raw", "homography")
    return all(os.path.isfile(paths[name]) for name in needed)


def align_and_cache(game=GAME, data_dir=None, cache_path=None, tac_data=None,
                    force=False):
    """Load tactical + broadcast ATD + broadcast homography, align, write the npz cache.

    ``tac_data`` may be a preloaded tactical dict (players/ball/team) to avoid a second parse.
    Raises ``IOError`` when broadcast assets are missing -- callers treat that as "no fusion".
    """
    cache_path = cache_path or default_cache_path(game)
    if (not force) and cache_path and os.path.isfile(cache_path):
        return load_aligned(cache_path)

    if not _broadcast_assets_present(game, data_dir):
        raise IOError("broadcast assets not found under {}".format(
            data_dir or os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "data")))

    from possession import atd as atd_io
    import importlib.util

    # starter/ has no __init__.py; load the module by path (same pattern as other tooling).
    _homo_path = os.path.join(ROOT, "starter", "load_homography.py")
    _spec = importlib.util.spec_from_file_location("mtah_load_homography", _homo_path)
    load_homography = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(load_homography)

    pitch = GAMES[game]["pitch"]
    n_tac = source_frames("tactical", game)
    n_bc = source_frames("broadcast", game)

    if tac_data is None:
        t_paths = asset_paths(game=game, data_dir=data_dir, source="tactical")
        tac_cache = os.path.join(ROOT, "cache", "{}_atd.npz".format(game))
        tac_data = atd_io.load(
            t_paths["atd_meta"], t_paths["atd_raw"], n_tac, pitch,
            cache_path=tac_cache, fuse_broadcast=False,
        )

    b_paths = asset_paths(game=game, data_dir=data_dir, source="broadcast")
    bc_data = atd_io.load(
        b_paths["atd_meta"], b_paths["atd_raw"], n_bc, pitch,
        cache_path=None, fuse_broadcast=False,
    )
    _matrices, _err, valid = load_homography.load(game=game, data_dir=data_dir, source="broadcast")

    aligned = align_arrays(
        tac_data["players"], tac_data["team"], tac_data["ball"],
        bc_data["players"], bc_data["team"], bc_data["ball"],
        valid,
    )
    if cache_path:
        save_aligned(cache_path, aligned)
    return aligned
