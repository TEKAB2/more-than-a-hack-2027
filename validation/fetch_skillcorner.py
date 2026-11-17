"""Download SkillCorner Open Data (A-League 2024/25 broadcast tracking) into compact `.npz`.

    python -m validation.fetch_skillcorner                 # 10 matches -> validation/external_data/
    python -m validation.fetch_skillcorner --matches 1886347

Source and licence
------------------
https://github.com/SkillCorner/opendata — 10 A-League 2024/25 matches of broadcast tracking
(10 fps, computer-vision from the TV feed), Dynamic Events and Phases of Play.

**Data licence: MIT** (repository LICENSE, Copyright (c) 2020 SkillCorner). The same MIT text
covers the software *and* the data files in this release; SkillCorner asks for credit when the
data is used (soft request, not a licence restriction). Training on these OTHER games is allowed
by the challenge rules (external data only).

Conversion (matches `validation/external.py` / `validation/fetch_idsse.py`)
-------------------------------------------------------------------------
* Coordinates: metres, centre-origin → corner-origin (`+ pitch_length/2`, `+ pitch_width/2`).
* Frame rate: 10 fps → uniform 25 fps by linear interpolation of positions. A coordinate that is
  missing (or `is_detected=False`) at either endpoint stays NaN — extrapolated SkillCorner
  positions are discarded so the gaps stay realistic.
* Labels:
  - **Alive / dead**: Phases of Play are defined only while the ball is in play; frames outside
    every `[frame_start, frame_end)` interval are dead (`alive=False`). Documented SkillCorner
    rule (README: "When the ball is out of play there is no phase of play").
  - **Owner**: `possession.group` `"home team"` → 0, `"away team"` → 1. While alive with no
    group, keep the last known team (`owner=-1` only until the first known owner).
* Ball: NaN when `ball_data.x` is null or `is_detected` is not True.
* Players: NaN when `is_detected` is not True.

Output per match (`skillcorner_<id>.npz`), all arrays at 25 fps:
    players (n, n_players, 2) float32
    team    (n_players,) int8           0 home, 1 away
    ball    (n, 2) float32
    alive   (n,) bool
    owner   (n,) int8                   0 home, 1 away, -1 unknown
    period  (n,) int8
    pitch   (2,) float64                [length, width] metres
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.request

import numpy as np

SRC_FPS = 10.0
DST_FPS = 25.0
RAW_BASE = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches"
META_BASE = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches"
OUT = os.environ.get("MTAH_EXTERNAL_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "external_data"
)

# 10 broadcast-tracking matches published for A-League 2024/25 (SkillCorner Open Data #5).
MATCHES = (
    1886347,  # Auckland FC 2-0 Newcastle
    1899585,  # Auckland FC 2-1 Wellington Phoenix
    1925299,  # Brisbane Roar 0-1 Perth Glory
    1953632,  # Central Coast Mariners 1-1 Melbourne City
    1996435,  # Sydney FC 4-1 Adelaide United
    2006229,  # Melbourne City 2-0 Macarthur FC
    2011166,  # Wellington Phoenix 2-3 Melbourne Victory
    2013725,  # Western United 1-0 Sydney FC
    2015213,  # Western United 4-2 Auckland FC
    2017461,  # Melbourne Victory 0-1 Auckland FC
)


def _urlopen(url, timeout=600):
    req = urllib.request.Request(url, headers={"User-Agent": "mtah-skillcorner-fetch"})
    return urllib.request.urlopen(req, timeout=timeout)


def download(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    print("  downloading", url.split("/")[-1], "...")
    with _urlopen(url) as resp, open(path, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    return path


def _load_match_meta(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_phases(path):
    """Half-open [frame_start, frame_end) intervals at 10 fps where the ball is in play."""
    phases = []
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            phases.append((int(row["frame_start"]), int(row["frame_end"]), int(row["period"])))
    return phases


def _alive_at_src(phases, frame):
    for a, b, _ in phases:
        if a <= frame < b:
            return True
    return False


def _group_to_owner(group):
    if group == "home team":
        return 0
    if group == "away team":
        return 1
    return -1


def _parse_tracking(path, player_team):
    """Return sorted source frames: frame_id, period, ball(xy|nan), players dict, owner."""
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            frame = int(obj["frame"])
            period = obj.get("period")
            period = int(period) if period is not None else 0
            ball = obj.get("ball_data") or {}
            if ball.get("x") is not None and ball.get("is_detected") is True:
                bxy = (float(ball["x"]), float(ball["y"]))
            else:
                bxy = (np.nan, np.nan)
            pdata = {}
            for p in obj.get("player_data") or []:
                pid = p.get("player_id")
                if pid is None or pid not in player_team:
                    continue
                if p.get("is_detected") is True and p.get("x") is not None:
                    pdata[int(pid)] = (float(p["x"]), float(p["y"]))
            owner = _group_to_owner((obj.get("possession") or {}).get("group"))
            rows.append((frame, period, bxy, pdata, owner))
    rows.sort(key=lambda r: r[0])
    return rows


def _interp_series(src_t, src_y, dst_t):
    """Linear interp; NaN wherever either neighbour is NaN (no filling across gaps)."""
    out = np.full(len(dst_t), np.nan, dtype=np.float32)
    if len(src_t) == 0:
        return out
    finite = np.isfinite(src_y)
    if finite.sum() == 0:
        return out
    idx = np.flatnonzero(finite)
    breaks = np.flatnonzero(np.diff(idx) > 1) + 1
    runs = np.split(idx, breaks)
    for run in runs:
        if len(run) == 1:
            t0, y0 = src_t[run[0]], src_y[run[0]]
            j = np.flatnonzero(np.isclose(dst_t, t0, atol=1e-6))
            out[j] = y0
            continue
        t_run = src_t[run]
        y_run = src_y[run]
        lo, hi = t_run[0], t_run[-1]
        mask = (dst_t >= lo) & (dst_t <= hi)
        out[mask] = np.interp(dst_t[mask], t_run, y_run).astype(np.float32)
    return out


def _resample_match(rows, phases, player_ids, length, width):
    """Build 25 fps arrays for the whole match (periods 1 and 2 concatenated)."""
    if not rows:
        raise SystemExit("empty tracking")

    by_period = {}
    for frame, period, bxy, pdata, owner in rows:
        if period not in (1, 2):
            continue
        by_period.setdefault(period, []).append((frame, bxy, pdata, owner))

    alive_src = {frame: _alive_at_src(phases, frame) for frame, period, _, _, _ in rows}

    slots = list(player_ids)
    slot_of = {pid: i for i, pid in enumerate(slots)}
    n_players = len(slots)

    P_parts, B_parts, alive_parts, owner_parts, period_parts = [], [], [], [], []

    for period in (1, 2):
        src = by_period.get(period) or []
        if len(src) < 2:
            continue
        frames = np.array([r[0] for r in src], dtype=np.int64)
        src_t = frames.astype(np.float64) / SRC_FPS
        t0, t1 = src_t[0], src_t[-1]
        n_dst = int(np.floor((t1 - t0) * DST_FPS)) + 1
        dst_t = t0 + np.arange(n_dst, dtype=np.float64) / DST_FPS

        ball_x = _interp_series(src_t, np.array([r[1][0] for r in src], np.float64), dst_t)
        ball_y = _interp_series(src_t, np.array([r[1][1] for r in src], np.float64), dst_t)
        ball_x = ball_x + length / 2.0
        ball_y = ball_y + width / 2.0

        P = np.full((n_dst, n_players, 2), np.nan, dtype=np.float32)
        for pid in slots:
            sx = np.array([r[2].get(pid, (np.nan, np.nan))[0] for r in src], np.float64)
            sy = np.array([r[2].get(pid, (np.nan, np.nan))[1] for r in src], np.float64)
            P[:, slot_of[pid], 0] = _interp_series(src_t, sx, dst_t) + length / 2.0
            P[:, slot_of[pid], 1] = _interp_series(src_t, sy, dst_t) + width / 2.0

        owner_src = np.array([r[3] for r in src], dtype=np.int8)
        alive_s = np.array([alive_src.get(int(f), False) for f in frames], dtype=bool)
        owner_ff = owner_src.copy()
        last = -1
        for i in range(len(owner_ff)):
            if not alive_s[i]:
                last = -1
                owner_ff[i] = -1
                continue
            if owner_src[i] >= 0:
                last = int(owner_src[i])
            owner_ff[i] = last

        idx = np.searchsorted(src_t, dst_t, side="right") - 1
        idx = np.clip(idx, 0, len(src_t) - 1)
        alive = alive_s[idx]
        owner = owner_ff[idx].astype(np.int8)

        P_parts.append(P.astype(np.float32))
        B_parts.append(np.stack([ball_x, ball_y], axis=1).astype(np.float32))
        alive_parts.append(alive)
        owner_parts.append(owner)
        period_parts.append(np.full(n_dst, period, dtype=np.int8))

    players = np.concatenate(P_parts, axis=0)
    ball = np.concatenate(B_parts, axis=0)
    alive = np.concatenate(alive_parts, axis=0)
    owner = np.concatenate(owner_parts, axis=0)
    period = np.concatenate(period_parts, axis=0)
    return players, ball, alive, owner, period


def convert(match_id, raw_dir, out_dir):
    mid = str(match_id)
    match_dir = os.path.join(raw_dir, mid)
    os.makedirs(match_dir, exist_ok=True)

    meta_path = download(
        "{}/{}/{}_match.json".format(META_BASE, mid, mid),
        os.path.join(match_dir, "{}_match.json".format(mid)),
    )
    phases_path = download(
        "{}/{}/{}_phases_of_play.csv".format(META_BASE, mid, mid),
        os.path.join(match_dir, "{}_phases_of_play.csv".format(mid)),
    )
    track_path = download(
        "{}/{}/{}_tracking_extrapolated.jsonl".format(RAW_BASE, mid, mid),
        os.path.join(match_dir, "{}_tracking_extrapolated.jsonl".format(mid)),
    )

    meta = _load_match_meta(meta_path)
    length = float(meta.get("pitch_length") or 105.0)
    width = float(meta.get("pitch_width") or 68.0)
    home_id = int(meta["home_team"]["id"])
    player_team = {}
    for p in meta["players"]:
        player_team[int(p["id"])] = 0 if int(p["team_id"]) == home_id else 1

    phases = _load_phases(phases_path)
    rows = _parse_tracking(track_path, player_team)
    seen = set()
    for _, _, _, pdata, _ in rows:
        seen.update(pdata.keys())
    player_ids = sorted(seen, key=lambda pid: (player_team[pid], pid))
    if not player_ids:
        raise SystemExit("{}: no detected players".format(mid))
    team = np.array([player_team[pid] for pid in player_ids], dtype=np.int8)

    players, ball, alive, owner, period = _resample_match(
        rows, phases, player_ids, length, width
    )

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "skillcorner_{}.npz".format(mid))
    np.savez_compressed(
        path,
        players=players,
        team=team,
        ball=ball,
        alive=alive,
        owner=owner,
        period=period,
        pitch=np.array([length, width], dtype=np.float64),
    )

    ball_seen = ~np.isnan(ball[:, 0])
    in_play = alive
    dead = ~alive
    n_player_slots = players.shape[1]
    detected = (~np.isnan(players[..., 0])).sum(axis=1)
    print(
        "skillcorner_{}  {} frames@25fps  alive {:.0%}  "
        "ball seen in-play {:.0%}  ball seen dead {:.0%}  "
        "players detected/frame {:.1f}/{:.0f}  -> {}".format(
            mid,
            len(ball),
            alive.mean(),
            ball_seen[in_play].mean() if in_play.any() else float("nan"),
            ball_seen[dead].mean() if dead.any() else float("nan"),
            detected.mean(),
            n_player_slots,
            os.path.relpath(path),
        )
    )
    return path


def summarize_dir(out_dir):
    files = sorted(
        f for f in os.listdir(out_dir) if f.startswith("skillcorner_") and f.endswith(".npz")
    )
    if not files:
        print("no skillcorner_*.npz in", out_dir)
        return
    ball_alive, ball_dead, det, n_frames = [], [], [], 0
    for name in files:
        z = np.load(os.path.join(out_dir, name))
        ball, alive = z["ball"], z["alive"]
        seen = ~np.isnan(ball[:, 0])
        players = z["players"]
        det_pf = (~np.isnan(players[..., 0])).sum(axis=1)
        n = len(ball)
        n_frames += n
        if alive.any():
            ball_alive.append(seen[alive].mean())
        if (~alive).any():
            ball_dead.append(seen[~alive].mean())
        det.append(det_pf.mean())
        print(
            "  {:<28} frames {:>7}  alive {:.0%}  ball@alive {:.0%}  ball@dead {:.0%}  "
            "det/frame {:.1f}".format(
                name,
                n,
                alive.mean(),
                seen[alive].mean() if alive.any() else float("nan"),
                seen[~alive].mean() if (~alive).any() else float("nan"),
                det_pf.mean(),
            )
        )
    print(
        "AGGREGATE ({} matches, {} frames): ball seen in-play {:.1%}  "
        "ball seen dead {:.1%}  players detected/frame {:.1f}".format(
            len(files),
            n_frames,
            float(np.mean(ball_alive)) if ball_alive else float("nan"),
            float(np.mean(ball_dead)) if ball_dead else float("nan"),
            float(np.mean(det)) if det else float("nan"),
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", nargs="*", type=int, default=list(MATCHES))
    parser.add_argument("--out", default=OUT)
    parser.add_argument(
        "--raw",
        default=os.path.join(OUT, "skillcorner_raw"),
        help="cache directory for downloaded JSON/CSV/JSONL",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="only print aggregate stats over existing skillcorner_*.npz",
    )
    args = parser.parse_args()
    if args.stats_only:
        summarize_dir(args.out)
        return
    for match_id in args.matches:
        out_path = os.path.join(args.out, "skillcorner_{}.npz".format(match_id))
        if os.path.exists(out_path):
            print(match_id, "already done")
            continue
        convert(match_id, args.raw, args.out)
    summarize_dir(args.out)


if __name__ == "__main__":
    main()
