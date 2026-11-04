"""Honest validation on OTHER games: Metrica's public sample data (3 full matches).

Why this exists
---------------
The rules forbid tuning on the 5 public minutes and any human label on the provided game, but
explicitly allow labelled data from other games. Metrica publishes three full matches with
tracking (25 fps, same 0-1 coordinates as the ATD) and human event data:

    https://github.com/metrica-sports/sample-data

From the events we derive frame-level H/A/D labels (rule below), degrade the tracking so it looks
like the automatic ATD (ball missing ~41 % in bursts, noise, track gaps), run the exact same
pipeline, and score it with the official metric. Three matches give ~hundreds of possessions, so
the spellF1 noise band is ~0.04 instead of the public window's ~0.3. Every weight in the pipeline
is chosen HERE, never on BAR-ATH.

Label rule (documented in the technical note)
---------------------------------------------
* on-ball event by team T (SET PIECE, PASS, CARRY, RECOVERY, SHOT) -> T in possession, ball alive;
* BALL OUT, SHOT ending OUT or GOAL, FAULT RECEIVED -> dead from that moment;
* dead lasts until the next on-ball event (the restart);
* CHALLENGE, BALL LOST, CARD do not change anything by themselves (a loss is only a change once
  the other team records a RECOVERY / PASS / CARRY -- the "controlled possession" rule).
This approximates the corrected-tracking ground truth; it is not identical, which is fine for
choosing between versions of a method but means absolute scores here are not BAR-ATH scores.

    python -m validation.external --samples PATH/TO/sample-data            # score current params
"""

import argparse
import csv
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

PITCH = (105.0, 68.0)
FPS = 25
ON_BALL = ("SET PIECE", "PASS", "CARRY", "RECOVERY", "SHOT")


# ----------------------------------------------------------------------------------- loading
def _num(token):
    try:
        return float(token)
    except ValueError:
        return np.nan


def load_csv_game(folder, n):
    """Sample games 1 and 2: two tracking CSVs (home, away) and an event CSV."""
    stem = os.path.join(folder, "Sample_Game_{}".format(n))
    sides = []
    period = None
    for side in ("Home", "Away"):
        with open("{}_RawTrackingData_{}_Team.csv".format(stem, side)) as handle:
            rows = list(csv.reader(handle))
        header = rows[2]
        data = np.array([[_num(v) for v in r] for r in rows[3:]], dtype=np.float32)
        player_cols = [i for i, h in enumerate(header) if h.startswith("Player")]
        ball_col = header.index("Ball")
        xy = np.stack([data[:, player_cols], data[:, [c + 1 for c in player_cols]]], axis=2)
        sides.append(xy)
        ball = data[:, [ball_col, ball_col + 1]]
        period = data[:, 0].astype(int)
        frames = data[:, 1].astype(int)
    n_frames = int(frames.max())
    players = np.concatenate(sides, axis=1)
    team = np.array([0] * sides[0].shape[1] + [1] * sides[1].shape[1], dtype=np.int8)
    out_players = np.full((n_frames, players.shape[1], 3), np.nan, dtype=np.float32)
    out_ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    out_players[frames - 1, :, :2] = players
    out_ball[frames - 1] = ball

    blocks = []
    for p in sorted(set(period.tolist())):
        idx = frames[period == p] - 1
        blocks.append((p, int(idx.min()), int(idx.max())))

    events = []
    with open("{}_RawEventsData.csv".format(stem)) as handle:
        for row in csv.DictReader(handle):
            events.append({
                "team": 0 if row["Team"] == "Home" else 1,
                "type": row["Type"].upper(),
                "subtype": (row["Subtype"] or "").upper(),
                "start": int(row["Start Frame"]) - 1,
                "end": max(int(row["End Frame"] or 0), int(row["Start Frame"])) - 1,
            })
    return _scale(out_players, out_ball), team, blocks, events


def load_epts_game(folder):
    """Sample game 3: FIFA/EPTS like the ATD, with substitutions (several format specs)."""
    stem = os.path.join(folder, "Sample_Game_3")
    root = ET.parse(stem + "_metadata.xml").getroot()
    strip = lambda e: e.tag.split("}")[-1]  # noqa: E731
    team_ids = [t.get("id") for t in root.iter() if strip(t) == "Team"]
    player_team = {p.get("id"): p.get("teamId") for p in root.iter() if strip(p) == "Player"}
    channel_player = {c.get("id"): c.get("playerId")
                      for c in root.iter() if strip(c) == "PlayerChannel"}
    specs = []
    for spec in root.iter():
        if strip(spec) != "DataFormatSpecification":
            continue
        refs = [r.get("playerChannelId") for r in spec.iter() if strip(r) == "PlayerChannelRef"]
        slots = [channel_player[r] for r in refs if r.endswith("_x")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), slots))

    all_players = sorted({p for _, _, s in specs for p in s})
    slot_of = {p: i for i, p in enumerate(all_players)}
    team = np.array([0 if player_team[p] == team_ids[0] else 1 for p in all_players], np.int8)
    n_frames = max(e for _, e, _ in specs)
    players = np.full((n_frames, len(all_players), 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)

    spec_i = 0
    with open(stem + "_tracking.txt") as handle:
        for line in handle:
            fid, groups, b = line.rstrip("\n").split(":")
            fid = int(fid)
            while not specs[spec_i][0] <= fid <= specs[spec_i][1]:
                spec_i += 1
            row = fid - 1
            for pid, g in zip(specs[spec_i][2], groups.split(";")):
                v = g.split(",")
                players[row, slot_of[pid], 0] = _num(v[0])
                players[row, slot_of[pid], 1] = _num(v[1])
            v = b.split(",")
            ball[row] = (_num(v[0]), _num(v[1]))

    with open(stem + "_events.json") as handle:
        raw = json.load(handle)["data"]
    events, periods = [], {}
    for e in raw:
        sub = e.get("subtypes")
        if isinstance(sub, list):
            sub = "-".join(s["name"] for s in sub)
        elif isinstance(sub, dict):
            sub = sub["name"]
        start, end = int(e["start"]["frame"]) - 1, int(e["end"]["frame"] or e["start"]["frame"]) - 1
        events.append({"team": 0 if e["team"]["id"] == team_ids[0] else 1,
                       "type": e["type"]["name"].upper(), "subtype": (sub or "").upper(),
                       "start": start, "end": max(start, end)})
        lo, hi = periods.get(e["period"], (start, end))
        periods[e["period"]] = (min(lo, start), max(hi, end))
    blocks = [(p, lo, min(hi + 2 * FPS, n_frames - 1)) for p, (lo, hi) in sorted(periods.items())]
    return _scale(players, ball), team, blocks, events


def _scale(players, ball):
    players[..., 0] *= PITCH[0]
    players[..., 1] *= PITCH[1]
    ball = ball.copy()
    ball[:, 0] *= PITCH[0]
    ball[:, 1] *= PITCH[1]
    return players, ball


# ------------------------------------------------------------------------------------ labels
def labels_from_events(events, n_frames, blocks):
    labels = np.full(n_frames, "X", dtype="<U1")
    for _, lo, hi in blocks:
        labels[lo:hi + 1] = "D"
    marks = []  # (frame, new_state)
    for e in events:
        letter = "H" if e["team"] == 0 else "A"
        if e["type"] in ON_BALL:
            marks.append((e["start"], letter))
        if e["type"] == "BALL OUT" or (e["type"] == "SHOT" and
                                       ("OUT" in e["subtype"] or "GOAL" in e["subtype"])):
            marks.append((e["end"], "D"))
        if e["type"] == "FAULT RECEIVED":
            marks.append((e["start"], "D"))
    marks.sort(key=lambda m: (m[0], m[1] != "D"))
    for _, lo, hi in blocks:
        state, last = "D", lo
        for frame, new in marks:
            if frame < lo or frame > hi:
                continue
            labels[last:frame] = state
            state, last = new, frame
        labels[last:hi + 1] = state
    return labels


# --------------------------------------------------------------------------------- degrading
def degrade(players, ball, rng, ball_missing=0.41, track_missing=0.07, noise=0.25):
    """Make clean sample tracking look like the automatic ATD.

    Ball dropouts come in bursts and are more frequent when the ball is crowded or fast -- the
    frames the ATD loses most (DATA.md). Players get gaps too, and positional jitter.
    """
    n = len(ball)
    ball = fill_ball(ball)
    players = players.copy()
    players[..., :2] += smooth_noise(rng, players[..., :2].shape, noise)
    ball += smooth_noise(rng, ball.shape, noise)

    d = np.hypot(players[:, :, 0] - ball[:, :1], players[:, :, 1] - ball[:, 1:2])
    crowded = (np.nan_to_num(d, nan=99) < 3).sum(axis=1) >= 3
    speed = np.zeros(n)
    speed[1:] = np.hypot(*(ball[1:] - ball[:-1]).T) * FPS
    risk = 1.0 + 1.5 * crowded + 1.0 * (np.nan_to_num(speed) > 15)
    mean_burst = 30

    def dropouts(p_start, seed):
        r = np.random.default_rng(seed)
        lost = np.zeros(n, dtype=bool)
        starts = r.random(n) < p_start * risk
        lengths = (r.exponential(mean_burst, n) + 1).astype(int)
        t = 0
        for t0 in np.flatnonzero(starts):
            if t0 < t:
                continue
            lost[t0:t0 + lengths[t0]] = True
            t = t0 + lengths[t0]
        return lost

    # Bisection on the burst rate so the overall missing share matches the ATD's.
    seed = int(rng.integers(1 << 30))
    lo_p, hi_p = 1e-5, 0.2
    for _ in range(25):
        mid = (lo_p + hi_p) / 2
        if dropouts(mid, seed).mean() < ball_missing:
            lo_p = mid
        else:
            hi_p = mid
    lost = dropouts((lo_p + hi_p) / 2, seed)
    ball[lost] = np.nan

    n_slots = players.shape[1]
    for s in range(n_slots):
        t = 0
        while t < n:
            if rng.random() < track_missing / 200:
                length = int(rng.exponential(200)) + 1
                players[t:t + length, s] = np.nan
                t += length
            t += 1
    return players, ball


def smooth_noise(rng, shape, sigma, width=12):
    """Low-frequency positional error (tracker drift), not frame-to-frame jitter: the ATD's
    positions are filtered, and white noise would invent ~9 m/s of fake player speed."""
    white = rng.normal(0, 1, shape)
    cs = np.cumsum(white, axis=0)
    out = np.zeros(shape)
    out[width:] = (cs[width:] - cs[:-width]) / np.sqrt(width)
    return (out * sigma).astype(np.float32)


def fill_ball(ball):
    """The sample data removes the ball whenever it is out of play (92 % missing when dead,
    8 % when alive). The ATD keeps tracking it when it can see it, so first give the clean ball a
    plausible path through every gap (straight line between last and next sighting), then let
    degrade() remove it the ATD way. Otherwise the validator would reward "ball missing => dead",
    a cue that exists in the samples but not in the ATD."""
    out = ball.copy()
    seen = ~np.isnan(ball[:, 0])
    idx = np.arange(len(ball))
    if seen.sum() >= 2:
        for k in range(2):
            out[:, k] = np.interp(idx, idx[seen], ball[seen, k])
    return out


def speeds(players):
    out = players.copy()
    v = np.full(players.shape[:2], np.nan, dtype=np.float32)
    k = 5  # centred difference over 0.4 s, like a filtered tracker speed
    v[k:-k] = np.hypot(players[2 * k:, :, 0] - players[:-2 * k, :, 0],
                       players[2 * k:, :, 1] - players[:-2 * k, :, 1]) * FPS / (2 * k)
    out[..., 2] = np.clip(v, 0, 12)
    return out


def fake_set_piece_tags(events, rng, miss=0.15, jitter=1.0):
    """Automatic-tagging stand-in: set pieces, jittered, some dropped, throw-ins with no team."""
    pieces = []
    for e in events:
        if e["type"] != "SET PIECE" or rng.random() < miss:
            continue
        centre = e["start"] + int(rng.normal(0, jitter * FPS))
        kind = e["subtype"].lower()
        if "KICK OFF" in e["subtype"]:
            kind = "kick off (start)" if e.get("opening") else "kick off (after goal)"
        side = None if "THROW" in e["subtype"] else e["team"]
        pieces.append((centre - 3 * FPS, centre + 2 * FPS, kind, side))
    return pieces


# ----------------------------------------------------------------------------------- running
def prepare(samples_dir, seed=0):
    """[(name, atd_dict, blocks, labels, events)] for the three sample games, degraded."""
    rng = np.random.default_rng(seed)
    data_dir = os.path.join(samples_dir, "data")
    games = []
    for n in (1, 2):
        (players, ball), team, blocks, events = load_csv_game(
            os.path.join(data_dir, "Sample_Game_{}".format(n)), n)
        games.append(("sample{}".format(n), players, ball, team, blocks, events))
    (players, ball), team, blocks, events = load_epts_game(os.path.join(data_dir, "Sample_Game_3"))
    games.append(("sample3", players, ball, team, blocks, events))

    out = []
    for name, players, ball, team, blocks, events in games:
        labels = labels_from_events(events, len(ball), blocks)
        mark_openings(events, blocks)
        players, ball = degrade(players, ball, rng)
        atd = {"players": speeds(players), "ball": ball, "team": team}
        out.append((name, atd, blocks, labels, events))
    return out


def mark_openings(events, blocks):
    for _, lo, hi in blocks:
        inside = [e for e in events if lo - 10 * FPS <= e["start"] <= hi
                  and e["type"] == "SET PIECE" and "KICK OFF" in e["subtype"]]
        if inside:
            min(inside, key=lambda e: e["start"])["opening"] = True


def score_all(games, params=None, costs=None, use_tags=True, seed=0, verbose=True,
              weights=None):
    import possession_metrics as pm
    from possession import decode, emissions, features

    rng = np.random.default_rng(seed + 1)
    results = []
    for name, atd, blocks, labels, events in games:
        f = features.compute(atd, blocks, PITCH)
        pieces = fake_set_piece_tags(events, rng) if use_tags else []
        s_team, s_dead, _ = emissions.compute(f, blocks, None, pieces, [], params, weights, PITCH)
        pred = decode.decode(s_team, s_dead, blocks, len(labels), costs)
        r = pm.evaluate(pm.label_codes(labels), pm.label_codes(pred), bootstrap_n=0)
        results.append((name, r))
        if verbose:
            print("{:<8} spellF1 {:.4f}  macroF1 {:.4f}  poss gt {:>4} pred {:>4}".format(
                name, r["spell_f1_mean"], r["macro_f1"], r["n_gt_spells"], r["n_pred_spells"]))
    mean = float(np.mean([r["spell_f1_mean"] for _, r in results]))
    if verbose:
        print("mean spellF1 {:.4f}".format(mean))
    return mean, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, help="clone of metrica-sports/sample-data")
    parser.add_argument("--no-tags", action="store_true")
    args = parser.parse_args()
    score_all(prepare(args.samples), use_tags=not args.no_tags)


if __name__ == "__main__":
    main()
