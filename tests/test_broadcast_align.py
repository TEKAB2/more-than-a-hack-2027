"""Synthetic tests for broadcast alignment and ball fusion -- no real data/ required.

    python -m tests.test_broadcast_align

Builds two short in-memory tracks that share player geometry with a known integer offset,
plus a slow-motion "replay" segment that must be rejected. Asserts offset recovery, NaN
ball fill, kinematic gating, cache round-trip, and that atd.load stays unchanged when no
broadcast assets exist.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from broadcast.align import (  # noqa: E402
    MAX_MEDIAN_DIST_M,
    aggregate_metrics,
    align_arrays,
    continuous_segments,
    fuse_ball,
    load_aligned,
    save_aligned,
)
from possession import atd as atd_io  # noqa: E402

FPS = 25
PITCH = (105.0, 68.0)
N_TAC = 400
N_BC = 300
N_SLOTS = 22
TRUE_DELTA = 40  # tactical_frame = broadcast_frame + TRUE_DELTA
HOME, AWAY = 0, 1


def _base_positions():
    """22 fixed pitch positions (11 home, 11 away) used as a shared formation."""
    rng = np.random.default_rng(0)
    home = np.column_stack([
        rng.uniform(10, 45, 11),
        rng.uniform(5, 63, 11),
    ])
    away = np.column_stack([
        rng.uniform(55, 95, 11),
        rng.uniform(5, 63, 11),
    ])
    return np.vstack([home, away]).astype(np.float64)


def _team_vector():
    return np.array([HOME] * 11 + [AWAY] * 11, dtype=np.int8)


def build_synthetic(true_delta=TRUE_DELTA, n_tac=N_TAC, n_bc=N_BC):
    """Tactical + broadcast arrays with one live segment and one slow-mo replay.

    Live broadcast frames [50, 199] map to tactical [50+delta, 199+delta].
    Replay frames [220, 279] are half-speed copies of an earlier stretch (must be rejected).
    Homography valid only on those two stretches.
    """
    base = _base_positions()
    team = _team_vector()
    tac_players = np.full((n_tac, N_SLOTS, 3), np.nan, dtype=np.float32)
    tac_ball = np.full((n_tac, 2), np.nan, dtype=np.float32)
    bc_players = np.full((n_bc, N_SLOTS, 3), np.nan, dtype=np.float32)
    bc_ball = np.full((n_bc, 2), np.nan, dtype=np.float32)
    valid = np.zeros(n_bc, dtype=bool)

    def _formation(t):
        drift = 0.08 * t  # metres along x -- monotonic, unique in time
        phase = 0.4 * (t / FPS)
        return base + np.column_stack([
            np.sin(phase + np.arange(N_SLOTS) * 0.2) * 3.0 + drift,
            np.cos(phase + np.arange(N_SLOTS) * 0.15) * 2.0,
        ])

    # Fill the whole tactical timeline so a wrong delta still sees players (and a large error).
    for t in range(n_tac):
        pos = _formation(t)
        tac_players[t, :, 0] = pos[:, 0]
        tac_players[t, :, 1] = pos[:, 1]
        tac_players[t, :, 2] = 3.5

    # Shared live motion on broadcast, locked to tactical at TRUE_DELTA.
    live_bc = list(range(50, 200))
    for b in live_bc:
        t = b + true_delta
        pos = _formation(t)
        bc_players[b, :, 0] = pos[:, 0] + 0.05  # tiny sensor noise
        bc_players[b, :, 1] = pos[:, 1] - 0.05
        bc_players[b, :, 2] = 3.5
        bx, by = pos[0, 0] + 0.4, pos[0, 1]
        bc_ball[b] = (bx, by)
        if (t // FPS) % 2 == 0:
            tac_ball[t] = (bx, by)
        valid[b] = True

    # Slow-mo replay: reuse geometry from early live frames but at half player speed and
    # wrong temporal spacing relative to tactical -- offset search should reject it.
    replay = list(range(220, 280))
    for i, b in enumerate(replay):
        src_b = 50 + i // 2  # half-speed sampling of the live stretch
        src_t = src_b + true_delta
        bc_players[b] = tac_players[src_t]
        bc_players[b, :, 2] = tac_players[src_t, :, 2] * 0.4
        bc_ball[b] = tac_ball[src_t] if np.isfinite(tac_ball[src_t, 0]) else bc_ball[src_b]
        # Place replay content where tactical has unrelated motion so distance stays large
        # under any single delta -- shift positions by 20 m.
        bc_players[b, :, 0] = bc_players[b, :, 0] + 20.0
        valid[b] = True

    return {
        "tac_players": tac_players,
        "tac_team": team,
        "tac_ball": tac_ball,
        "bc_players": bc_players,
        "bc_team": team,
        "bc_ball": bc_ball,
        "valid": valid,
        "true_delta": true_delta,
        "live_bc": (50, 199),
    }


def test_continuous_segments():
    v = np.zeros(100, dtype=bool)
    v[10:40] = True
    v[50:55] = True  # too short if min_length=10
    v[70:95] = True
    segs = continuous_segments(v, min_length=10)
    assert segs == [(10, 39), (70, 94)], segs


def test_offset_and_replay_rejection():
    syn = build_synthetic()
    aligned = align_arrays(
        syn["tac_players"], syn["tac_team"], syn["tac_ball"],
        syn["bc_players"], syn["bc_team"], syn["bc_ball"],
        syn["valid"],
        min_segment=20,
    )
    ok = aligned["segment_ok"].astype(bool)
    assert ok.any(), "expected at least one accepted live segment"
    # The live segment must recover the true delta.
    live_hits = []
    for i, (s, e, d, accepted) in enumerate(zip(
            aligned["segment_start"], aligned["segment_end"],
            aligned["segment_delta"], aligned["segment_ok"])):
        if s == syn["live_bc"][0] and e == syn["live_bc"][1]:
            live_hits.append((int(d), int(accepted), float(aligned["segment_med_dist"][i])))
    assert live_hits, "live segment not found in segment table"
    delta, accepted, med = live_hits[0]
    assert accepted == 1, live_hits
    assert delta == syn["true_delta"], (delta, syn["true_delta"])
    assert med <= MAX_MEDIAN_DIST_M, med

    # Replay segment must be rejected.
    for s, e, accepted, med in zip(
            aligned["segment_start"], aligned["segment_end"],
            aligned["segment_ok"], aligned["segment_med_dist"]):
        if s == 220:
            assert accepted == 0, (s, e, med)
            break
    else:
        raise AssertionError("replay segment missing")


def test_fuse_and_metrics():
    syn = build_synthetic()
    aligned = align_arrays(
        syn["tac_players"], syn["tac_team"], syn["tac_ball"],
        syn["bc_players"], syn["bc_team"], syn["bc_ball"],
        syn["valid"],
        min_segment=20,
    )
    before = np.isfinite(syn["tac_ball"][:, 0]).mean()
    fused, prov = fuse_ball(syn["tac_ball"], aligned["ball_xy"], fps=FPS)
    after = np.isfinite(fused[:, 0]).mean()
    assert after > before, (before, after)
    assert (prov == 1.0).any(), "expected some broadcast provenance flags"
    assert (prov == 0.0).any(), "expected some tactical provenance flags"

    # Kinematic gate: a teleporting fill must be rejected.
    tac = syn["tac_ball"].copy()
    bc = np.full_like(tac, np.nan)
    # Known ball at t=0, insane jump at t=1.
    tac[:] = np.nan
    tac[0] = (50.0, 30.0)
    bc[1] = (90.0, 30.0)  # 40 m in 1/25 s >> 35 m/s
    fused2, prov2 = fuse_ball(tac, bc, fps=FPS, max_speed_m_s=35.0)
    assert np.isnan(fused2[1, 0]), fused2[1]
    assert not np.isfinite(prov2[1]) or prov2[1] != 1.0

    metrics = aggregate_metrics(syn["tac_ball"], aligned, fused_ball=fused)
    assert metrics["n_segments_accepted"] >= 1
    assert metrics["pct_missing_filled"] > 0
    assert metrics["fused_ball_seen_pct"] >= metrics["tac_ball_seen_pct"]
    # Aggregate dict must be scalars only.
    for key, value in metrics.items():
        assert np.ndim(value) == 0, (key, value)


def test_cache_roundtrip():
    syn = build_synthetic()
    aligned = align_arrays(
        syn["tac_players"], syn["tac_team"], syn["tac_ball"],
        syn["bc_players"], syn["bc_team"], syn["bc_ball"],
        syn["valid"],
        min_segment=20,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "broadcast_aligned.npz")
        save_aligned(path, aligned)
        loaded = load_aligned(path)
        assert loaded["segment_delta"].tolist() == aligned["segment_delta"].tolist()
        np.testing.assert_array_equal(loaded["ball_xy"], aligned["ball_xy"])


def test_atd_load_without_broadcast():
    """Bench-style call: no game/fuse flags -> identical to a plain parse."""
    with tempfile.TemporaryDirectory() as tmp:
        meta = os.path.join(tmp, "meta.xml")
        raw = os.path.join(tmp, "raw.txt")
        with open(meta, "w") as handle:
            handle.write(
                '<?xml version="1.0"?><main><Metadata><Teams>'
                '<Team id="T1"/><Team id="T2"/></Teams><Players>'
            )
            for s in range(4):
                handle.write(
                    '<Player id="P{0}" teamId="{1}"><Name>T{0}</Name></Player>'.format(
                        s + 1, "T1" if s < 2 else "T2"))
            handle.write("</Players></Metadata></main>\n")
        n = 10
        L, W = PITCH
        with open(raw, "w") as handle:
            for t in range(n):
                groups = ["{:.3f},{:.3f},1.0".format(0.2 + 0.01 * s, 0.5) for s in range(4)]
                ball = "NaN,NaN" if t % 3 == 0 else "{:.3f},{:.3f},0".format(0.4, 0.4)
                handle.write("{}:{}:{}\n".format(t + 1, ";".join(groups), ball))
        a = atd_io.load(meta, raw, n, PITCH, cache_path=None, fuse_broadcast=False)
        b = atd_io.load(meta, raw, n, PITCH, cache_path=None)  # default: no fusion
        np.testing.assert_array_equal(a["ball"], b["ball"])
        assert "ball_provenance" not in b
        # Explicit fusion with no broadcast assets / no game -> unchanged.
        c = atd_io.load(meta, raw, n, PITCH, fuse_broadcast=True, game=None)
        np.testing.assert_array_equal(a["ball"], c["ball"])


def test_apply_fusion_provenance_column():
    syn = build_synthetic()
    aligned = align_arrays(
        syn["tac_players"], syn["tac_team"], syn["tac_ball"],
        syn["bc_players"], syn["bc_team"], syn["bc_ball"],
        syn["valid"],
        min_segment=20,
    )
    data = {
        "players": syn["tac_players"],
        "ball": syn["tac_ball"].copy(),
        "ball_extra": np.full((N_TAC, 1), 0.5, dtype=np.float32),
        "team": syn["tac_team"],
    }
    out = atd_io.apply_broadcast_fusion(data, aligned, fps=FPS)
    assert out["ball_extra"].shape[1] == 2
    assert np.isfinite(out["ball"]).sum() >= np.isfinite(data["ball"]).sum()


def main():
    tests = [
        test_continuous_segments,
        test_offset_and_replay_rejection,
        test_fuse_and_metrics,
        test_cache_roundtrip,
        test_atd_load_without_broadcast,
        test_apply_fusion_provenance_column,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print("ok  {}".format(fn.__name__))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("FAIL {}: {}".format(fn.__name__, exc))
            raise
    print("all {} tests passed".format(len(tests) - failed))
    return failed


if __name__ == "__main__":
    sys.exit(main())
