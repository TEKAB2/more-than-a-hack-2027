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


MERGE_S = 1.6


def merge_brief(labels, seconds):
    """Hand an opponent spell shorter than `seconds`, sandwiched between two spells of the same
    team (no dead ball in between), back to that team.

    Why: DFL's BallPossession switches on any touch, so its possessions are much shorter (mean
    7.3 s, 43 % under 3 s) than Metrica's notion of *controlled* possession. 1.6 s is the value at
    which the DFL games reproduce the possession statistics of Metrica's own public games (about
    350 per match, 30 % under 3 s) -- chosen on those other games, never on BAR-ATH.
    """
    out = labels.copy()
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], edges, [len(labels)]))
    runs = [(labels[bounds[k]], bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]
    for k in range(1, len(runs) - 1):
        (a, _, _), (b, lo, hi), (c, _, _) = runs[k - 1], runs[k], runs[k + 1]
        if b in "HA" and a == c and a in "HA" and a != b and hi - lo < seconds * FPS:
            out[lo:hi] = a
    return out


def load_idsse(path, merge_s=None):
    """One open Bundesliga match saved by validation.fetch_idsse. Labels straight from the DFL
    corrected tracking (BallStatus, BallPossession) -- the same kind of source as the challenge's
    ground truth. Returns (players, ball), team, blocks, labels, pseudo-events."""
    z = np.load(path)
    players = np.concatenate([z["players"], np.full(z["players"].shape[:2] + (1,), np.nan,
                                                    dtype=np.float32)], axis=2)
    ball, alive, owner, period = z["ball"], z["alive"], z["owner"], z["period"]
    n = len(ball)
    blocks = []
    for p in sorted(set(period.tolist()) - {0}):
        idx = np.flatnonzero(period == p)
        blocks.append((int(p), int(idx[0]), int(idx[-1])))
    labels = np.full(n, "X", dtype="<U1")
    for _, lo, hi in blocks:
        last = "D"
        for t in range(lo, hi + 1):
            if not alive[t]:
                labels[t] = "D"
                continue
            if owner[t] >= 0:
                last = "H" if owner[t] == 0 else "A"
            elif last == "D":
                last = "H"  # unknown owner right after a restart: rare, filled below
            labels[t] = last
    for _, lo, hi in blocks:
        labels[lo:hi + 1] = merge_brief(labels[lo:hi + 1], MERGE_S if merge_s is None else merge_s)
    # pseudo-events: one restart per dead->alive transition, for fake_set_piece_tags()
    events = []
    for _, lo, hi in blocks:
        seg = labels[lo:hi + 1]
        for i in np.flatnonzero((seg[1:] != "D") & (seg[:-1] == "D")) + 1:
            events.append({"team": 0 if seg[i] == "H" else 1, "type": "SET PIECE",
                           "subtype": "RESTART", "start": lo + int(i), "end": lo + int(i)})
    return (players, ball.astype(np.float32)), z["team"].astype(np.int8), blocks, labels, events


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
# Ball visibility of the ATD, split by state. Measured on BAR-ATH with `possession.diagnose`,
# grouping frames by OUR OWN predicted labels (algorithmic pseudo-labels, no human label, no
# ground truth): ball seen on ~87 % of in-play frames and ~27 % of dead-ball frames. Consistent
# with the published overall figure (59.3 % of graded frames, DATA.md). The first version of this
# simulator dropped the ball uniformly (41 % everywhere), which hid the strongest dead-ball cue.
MISSING_ALIVE = 0.13
MISSING_DEAD = 0.73
BALL_NOISE = 0.08   # ball drift (m); 0.08 brings the touch rate in play near the ATD's (~65 vs ~43 per min)


# Gap structure of the ATD (measured on BAR-ATH from aggregates only, 168 gaps): gaps are LONG
# (median 9.6 s, p10 0.9 s) and short losses are almost absent -- the ATD bridges them itself.
# So in play the ball vanishes rarely but for seconds at a time, not in frequent short bursts.
ALIVE_GAP_MIN = 1 * FPS
ALIVE_GAP_MEAN = 5 * FPS


def _dropouts(mask, target, mean_burst, risk, seed, min_burst=1):
    """Bursts of missing ball inside `mask` only, at a rate giving `target` missing share there."""
    n = len(mask)
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    run_end = np.zeros(n, dtype=np.int64)
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        run_end[a:b] = b

    def draw(p_start):
        r = np.random.default_rng(seed)
        lost = np.zeros(n, dtype=bool)
        starts = mask & (r.random(n) < p_start * risk)
        lengths = (r.exponential(mean_burst, n) + min_burst).astype(np.int64)
        t = 0
        for t0 in np.flatnonzero(starts):
            if t0 < t:
                continue
            t = min(t0 + lengths[t0], run_end[t0])
            lost[t0:t] = True
        return lost

    if not mask.any() or target <= 0:
        return np.zeros(n, dtype=bool)
    lo_p, hi_p = 1e-6, 1.0
    for _ in range(25):
        mid = (lo_p + hi_p) / 2
        if draw(mid)[mask].mean() < target:
            lo_p = mid
        else:
            hi_p = mid
    return draw((lo_p + hi_p) / 2)


def degrade(players, ball, rng, labels=None, track_missing=0.07, noise=0.25,
            ball_noise=None):
    """Make clean tracking look like the automatic ATD.

    The ball goes missing in bursts, at the ATD's own rates for in-play and dead-ball frames
    (MISSING_ALIVE / MISSING_DEAD): short bursts in play, more often when the ball is crowded or
    fast; long bursts during stoppages, when the ball is off camera. Players get gaps and drift.
    """
    n = len(ball)
    ball = fill_ball(ball)
    players = players.copy()
    players[..., :2] += smooth_noise(rng, players[..., :2].shape, noise)
    ball += smooth_noise(rng, ball.shape, BALL_NOISE if ball_noise is None else ball_noise)

    d = np.hypot(players[:, :, 0] - ball[:, :1], players[:, :, 1] - ball[:, 1:2])
    crowded = (np.nan_to_num(d, nan=99) < 3).sum(axis=1) >= 3
    speed = np.zeros(n)
    speed[1:] = np.hypot(*(ball[1:] - ball[:-1]).T) * FPS
    risk = 1.0 + 1.5 * crowded + 1.0 * (np.nan_to_num(speed) > 15)

    if labels is None:
        alive = np.ones(n, dtype=bool)
        dead = np.zeros(n, dtype=bool)
    else:
        alive = np.isin(labels, ["H", "A"])
        dead = labels == "D"
    seed = int(rng.integers(1 << 30))
    lost = _dropouts(alive, MISSING_ALIVE, ALIVE_GAP_MEAN, risk, seed, ALIVE_GAP_MIN)
    lost |= _dropouts(dead, MISSING_DEAD, 5 * FPS, np.ones(n), seed + 1)
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
    """See below. Pseudo-restarts of the Bundesliga games are dropped more often (55 %): the
    Smart Tagging of BAR-ATH tags 59 set pieces for well over 100 stoppages."""
    if events and events[0].get("subtype") == "RESTART":
        miss = 0.55
    return _fake_tags(events, rng, miss, jitter)


def _fake_tags(events, rng, miss, jitter):
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
IDSSE_DIR = os.environ.get("MTAH_EXTERNAL_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "external_data")


def prepare(samples_dir, seed=0, idsse_dir=IDSSE_DIR):
    """[(name, atd_dict, blocks, labels, events)] for the sample games (+ the open Bundesliga
    matches when validation/external_data holds them), degraded like the ATD."""
    rng = np.random.default_rng(seed)
    data_dir = os.path.join(samples_dir, "data")
    games = []
    extra = []
    if idsse_dir and os.path.isdir(idsse_dir):
        for name in sorted(os.listdir(idsse_dir)):
            if name.startswith("idsse_") and name.endswith(".npz"):
                (players, ball), team, blocks, labels, events = load_idsse(
                    os.path.join(idsse_dir, name))
                extra.append((name[:-4], players, ball, team, blocks, events, labels))
    for n in (1, 2):
        (players, ball), team, blocks, events = load_csv_game(
            os.path.join(data_dir, "Sample_Game_{}".format(n)), n)
        games.append(("sample{}".format(n), players, ball, team, blocks, events))
    (players, ball), team, blocks, events = load_epts_game(os.path.join(data_dir, "Sample_Game_3"))
    games.append(("sample3", players, ball, team, blocks, events))

    games = [g + (None,) for g in games] + extra
    out = []
    for name, players, ball, team, blocks, events, labels in games:
        if labels is None:
            labels = labels_from_events(events, len(ball), blocks)
        mark_openings(events, blocks)
        players, ball = degrade(players, ball, rng, labels)
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
