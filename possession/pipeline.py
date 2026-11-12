"""End-to-end: raw data -> submission CSV. Deterministic, so it regenerates the same file.

    python -m possession.pipeline --team TEAM_NAME
    python -m possession.pipeline --team TEAM_NAME --check     # + official validator + public sanity score

Writes submissions/<TEAM_NAME>.csv (the name the form wants).
"""

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from possession import atd as atd_io, decode, emissions, features, model, tagging  # noqa: E402


def team_names(meta_path):
    names = [team.findtext("Name") or "" for team in ET.parse(meta_path).getroot().iter("Team")]
    return names[0], names[1]


def summarise(labels, blocks):
    spells, dead = [], 0
    for _, first, last in blocks:
        seg = labels[first:last + 1]
        dead += int((seg == "D").sum())
        start = 0
        for t in range(1, len(seg) + 1):
            if t == len(seg) or seg[t] != seg[start]:
                if seg[start] in ("H", "A"):
                    spells.append(t - start)
                start = t
    graded = sum(last - first + 1 for _, first, last in blocks)
    print("possessions {}   mean {:.1f} s   median {:.1f} s   dead-ball share {:.1%}".format(
        len(spells), np.mean(spells) / 25 if spells else 0,
        np.median(spells) / 25 if spells else 0, dead / graded))


def infer(data_dir=None, game=GAME, params=None, costs=None, use_tagging=True,
          weights_path=None, verbose=True):
    """Everything up to the labels, nothing written. Returns a dict (also used by tools/)."""
    meta, raw, tags_path = require(["atd_meta", "atd_raw", "smart_tagging"], game, data_dir)
    n = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]
    blocks = half_video_blocks(game)          # [(half, first, last)] published in DATA.md
    home, away = team_names(meta)

    cache = os.path.join(ROOT, "cache", "{}_atd.npz".format(game))
    data = atd_io.load(meta, raw, n, pitch, cache_path=cache)
    feats = features.compute(data, blocks, pitch)

    tag_prior, pieces, goals = None, [], []
    if use_tagging:
        events = tagging.load(tags_path)
        tag_prior = tagging.team_prior(events, n, home, away)
        pieces, goals = tagging.set_pieces(events, home, away)

    weights = None
    if weights_path:
        weights = model.load_weights(weights_path)
        costs = dict(weights.get("costs", {}), **(costs or {}))
        if verbose:
            print("emissions: learned ({}), held-out spellF1 on sample games {:.3f}".format(
                os.path.relpath(weights_path, ROOT),
                weights.get("held_out_spell_f1", float("nan"))))
    elif verbose:
        print("emissions: hand-set v1 parameters")
    s_team, s_dead, restarts = emissions.compute(feats, blocks, tag_prior, pieces, goals, params,
                                                 weights, pitch)
    p_change = weights.get("_last_p_change") if isinstance(weights, dict) else None
    p_dead_edge = weights.get("_last_p_dead_edge") if isinstance(weights, dict) else None
    delta_t = weights.get("_last_delta_t") if isinstance(weights, dict) else None
    labels = decode.decode(
        s_team, s_dead, blocks, n, costs,
        p_change=p_change, p_dead_edge=p_dead_edge, delta_t=delta_t,
    )
    return {"labels": labels, "s_team": s_team, "s_dead": s_dead, "restarts": restarts,
            "atd": data, "features": feats, "blocks": blocks, "pitch": pitch,
            "home": home, "away": away, "n": n}


def run(team, data_dir=None, game=GAME, params=None, costs=None, use_tagging=True,
        weights_path=None):
    out = infer(data_dir, game, params, costs, use_tagging, weights_path)
    labels = out["labels"]

    out_dir = os.path.join(ROOT, "submissions")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "{}.csv".format(team))
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))

    print("H = {}, A = {}   restarts detected: {}".format(out["home"], out["away"],
                                                          len(out["restarts"])))
    summarise(labels, out["blocks"])
    print("->", os.path.relpath(path, ROOT))
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", required=True, help="exact team name from the registration form")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--no-tagging", action="store_true")
    parser.add_argument("--v1", action="store_true",
                        help="hand-set v1 emissions instead of possession/weights.json")
    parser.add_argument("--check", action="store_true",
                        help="run the official validator, then score on the public window "
                             "(SANITY ONLY - never use it to choose between versions)")
    args = parser.parse_args()

    weights_path = None if args.v1 or not os.path.exists(model.WEIGHTS) else model.WEIGHTS
    path = run(args.team, args.data_dir, use_tagging=not args.no_tagging,
               weights_path=weights_path)
    if args.check:
        evaluation = os.path.join(ROOT, "evaluation")
        subprocess.call([sys.executable, os.path.join(evaluation, "validate_submission.py"), path])
        subprocess.call([sys.executable, os.path.join(evaluation, "run_evaluation.py"),
                         "--results", os.path.dirname(path),
                         "--out", os.path.join(ROOT, "cache", "public_check")])


if __name__ == "__main__":
    main()
