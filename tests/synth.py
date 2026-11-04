"""Synthetic match in the exact file formats of the real bundle -- for testing code, not for tuning.

    python tests/synth.py /tmp/fake_data      # writes tactical/ ATD xml+txt, smart tagging, and gt.csv

Nothing here is derived from the real match. It exists so the pipeline can be run end to end on a
machine that does not (and must not) hold the licensed data, and so a refactor that breaks the
parser or the frame alignment fails loudly.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAMES, asset_paths, half_video_blocks  # noqa: E402

FPS = 25
N_SLOTS = 46


def simulate(rng, n_frames, blocks, pitch):
    L, W = pitch
    players = np.full((n_frames, N_SLOTS, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    gt = np.full(n_frames, "X", dtype="<U1")
    pieces = []  # (start_frame, end_frame, type, side)

    home_slots = np.arange(0, 11)
    away_slots = np.arange(23, 34)
    base_home = np.column_stack([rng.uniform(5, L / 2 - 3, 11), rng.uniform(5, W - 5, 11)])
    base_away = np.column_stack([rng.uniform(L / 2 + 3, L - 5, 11), rng.uniform(5, W - 5, 11)])

    for _, first, last in blocks:
        pos = np.vstack([base_home, base_away]).astype(np.float64)
        t = first
        team = int(rng.integers(0, 2))
        pieces.append((t - 2 * FPS, t + 3 * FPS, "Kick Off (Start)", team))
        carrier = int(rng.integers(0, 11)) + 11 * team
        bpos = np.array([L / 2, W / 2])
        state = "alive"
        flight = None  # (target_idx, frames_left)
        dead_left = 0
        poss_left = int(rng.exponential(14 * FPS)) + FPS
        restart_team = team
        while t <= last:
            if state == "alive":
                speed = 4.0
                target = pos[carrier] + rng.normal(0, 0.5, 2)
                if flight is None:
                    bpos = pos[carrier] + np.array([0.4, 0.0])
                    poss_left -= 1
                    if poss_left <= 0:
                        r = rng.random()
                        if r < 0.25:   # ball goes out -> dead
                            state, dead_left = "dead", int(rng.uniform(5, 20) * FPS)
                            bpos = np.array([L + 2.0, bpos[1]])
                            restart_team = 1 - team
                        else:          # interception
                            new = int(rng.integers(0, 11)) + 11 * (1 - team)
                            flight = (new, int(rng.uniform(0.6, 1.5) * FPS), 1 - team)
                    elif rng.random() < 1 / (3 * FPS):  # pass to teammate
                        new = int(rng.integers(0, 11)) + 11 * team
                        flight = (new, int(rng.uniform(0.6, 1.5) * FPS), team)
                else:
                    target_idx, left, new_team = flight
                    bpos = bpos + (pos[target_idx] - bpos) / max(left, 1)
                    left -= 1
                    if left <= 0:
                        carrier = target_idx
                        if new_team != team:
                            team = new_team
                            poss_left = int(rng.exponential(14 * FPS)) + FPS
                        flight = None
                    else:
                        flight = (target_idx, left, new_team)
                gt[t] = "H" if team == 0 else "A"
            else:
                speed = 1.0
                dead_left -= 1
                gt[t] = "D"
                if dead_left == int(3 * FPS):
                    bpos = np.array([L - 0.2, rng.uniform(5, W - 5)])
                if dead_left <= 0:
                    state, team = "alive", restart_team
                    carrier = int(np.argmin(np.hypot(*(pos[11 * team:11 * team + 11] - bpos).T))) \
                        + 11 * team
                    poss_left = int(rng.exponential(14 * FPS)) + FPS
                    pieces.append((t - 3 * FPS, t + 2 * FPS, "Throw-in", None))
            # move players: random walk toward base shifted with the ball
            home_target = base_home + np.array([(bpos[0] - L / 2) * 0.4, 0])
            away_target = base_away + np.array([(bpos[0] - L / 2) * 0.4, 0])
            targets = np.vstack([home_target, away_target])
            if state == "alive" and flight is None:
                targets[carrier] = pos[carrier] + rng.normal(0, 1, 2)
            step = targets - pos
            norm = np.maximum(np.hypot(step[:, 0], step[:, 1]), 1e-6)
            move = step / norm[:, None] * np.minimum(norm, speed / FPS)[:, None]
            move += rng.normal(0, speed * 0.3 / FPS, move.shape)
            pos += move
            pos[:, 0] = np.clip(pos[:, 0], 1.0, L - 1.0)
            pos[:, 1] = np.clip(pos[:, 1], 1.0, W - 1.0)
            v = np.hypot(move[:, 0], move[:, 1]) * FPS
            players[t, home_slots, :2] = pos[:11]
            players[t, home_slots, 2] = v[:11]
            players[t, away_slots, :2] = pos[11:]
            players[t, away_slots, 2] = v[11:]
            ball[t] = bpos + rng.normal(0, 0.15, 2)
            t += 1

    # Ball dropouts in bursts: ~40% of frames.
    t = 0
    while t < n_frames:
        if rng.random() < 0.4:
            ball[t:t + int(rng.uniform(5, 60))] = np.nan
        t += int(rng.uniform(20, 80))
    return players, ball, gt, pieces


def write_bundle(out_dir, seed=0):
    rng = np.random.default_rng(seed)
    game = GAMES["BAR-ATH"]
    n, pitch = game["n_video_frames"], game["pitch"]
    blocks = half_video_blocks("BAR-ATH")
    players, ball, gt, pieces = simulate(rng, n, blocks, pitch)
    paths = asset_paths(data_dir=out_dir)
    os.makedirs(os.path.dirname(paths["atd_raw"]), exist_ok=True)

    with open(paths["atd_meta"], "w") as h:
        h.write('<?xml version="1.0"?>\n<main><Metadata><Teams>'
                '<Team id="T1"><Name>FC Synth</Name></Team>'
                '<Team id="T2"><Name>Athletic Fake</Name></Team></Teams><Players>')
        for s in range(N_SLOTS):
            h.write('<Player id="P{0}" teamId="{1}"><Name>Track {0}</Name></Player>'.format(
                s + 1, "T1" if s < 23 else "T2"))
        h.write("</Players></Metadata></main>\n")

    L, W = pitch
    with open(paths["atd_raw"], "w") as h:
        for t in range(n):
            groups = []
            for s in range(N_SLOTS):
                x, y, v = players[t, s]
                groups.append("NaN,NaN,NaN" if np.isnan(x) else
                              "{:.5f},{:.5f},{:.2f}".format(x / L, y / W, v))
            bx, by = ball[t]
            b = "NaN,NaN,NaN" if np.isnan(bx) else "{:.5f},{:.5f},0".format(bx / L, by / W)
            h.write("{}:{}:{}\n".format(t + 1, ";".join(groups), b))

    names = {0: "Synth", 1: "Athletic Fk", None: "N/A"}
    with open(paths["smart_tagging"], "w") as h:
        h.write("<file><ALL_INSTANCES>\n")
        for i, (s, e, kind, side) in enumerate(pieces):
            h.write("<instance><ID>{}</ID><start>{:.2f}</start><end>{:.2f}</end>"
                    "<code>SET PIECES</code><label><group>Team</group><text>{}</text></label>"
                    "<label><group>Type</group><text>{}</text></label></instance>\n".format(
                        i, s / FPS, e / FPS, names[side], kind))
        h.write("</ALL_INSTANCES></file>\n")

    gt_path = os.path.join(out_dir, "synthetic_gt.csv")
    with open(gt_path, "w") as h:
        h.write("frame,label\n")
        h.writelines("{},{}\n".format(i, l) for i, l in enumerate(gt))
    return gt_path


if __name__ == "__main__":
    print(write_bundle(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fake_data"))
