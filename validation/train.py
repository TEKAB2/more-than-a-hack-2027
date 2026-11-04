"""Fit the emission models and choose the decoder costs on Metrica's PUBLIC sample games only.

    git clone https://github.com/metrica-sports/sample-data ../sample-data
    python -m validation.train --samples ../sample-data          # writes possession/weights.json

Protocol (all on the three sample games, never on BAR-ATH):
  * each game is degraded to look like the ATD, with two different random seeds;
  * leave-one-game-out: fit on two games, decode the third with every candidate cost setting;
  * keep the cost setting with the best mean held-out spellF1;
  * refit the models on all three games and save them with those costs.
Fixed seeds, so the file is reproducible.
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from possession import decode, emissions, features, model  # noqa: E402
from validation import external as E  # noqa: E402

GRID = {
    "switch": (6.0, 10.0, 15.0, 22.0, 32.0),
    "to_dead": (4.0, 8.0, 14.0),
    "restart": (4.0, 8.0, 14.0),
    "t_team": (0.5, 1.0),
    "t_dead": (0.5, 1.0),
}
SUBSAMPLE = 4  # every 4th frame for fitting: neighbouring frames are near-duplicates


def build(samples, seeds):
    rows = []
    for seed in seeds:
        for name, atd, blocks, labels, events in E.prepare(samples, seed=seed):
            f = features.compute(atd, blocks, E.PITCH)
            ev = emissions.ball_team_evidence(f, emissions.PARAMS) / emissions.PARAMS["w_ball"]
            rows.append({
                "pieces": E.fake_set_piece_tags(events, np.random.default_rng(seed + 100)),
                "name": name, "seed": seed, "blocks": blocks, "labels": labels, "events": events,
                "f": f, "Xd": model.dead_features(f, blocks, E.PITCH),
                "Xt": model.team_features(f, blocks, E.PITCH, ev),
            })
    return rows


def fit(rows):
    Xd, yd, Xt, yt = [], [], [], []
    for r in rows:
        lab = r["labels"]
        graded = np.flatnonzero(lab != "X")[::SUBSAMPLE]
        Xd.append(r["Xd"][graded])
        yd.append((lab[graded] == "D").astype(float))
        alive = graded[np.isin(lab[graded], ["H", "A"])]
        Xt.append(r["Xt"][alive])
        yt.append((lab[alive] == "H").astype(float))
    return {"dead": model.fit_logistic(np.vstack(Xd), np.concatenate(yd)),
            "team": model.fit_logistic(np.vstack(Xt), np.concatenate(yt))}


def raw_logits(rows, weights):
    """Model logits per row, before temperatures and restarts (computed once per fold)."""
    return [(model.logit(weights["team"], r["Xt"]), model.logit(weights["dead"], r["Xd"]))
            for r in rows]


def evaluate(rows, weights, costs, logits=None):
    logits = logits or raw_logits(rows, weights)
    p = dict(emissions.PARAMS, pitch=E.PITCH)
    scores = []
    for r, (lt, ld) in zip(rows, logits):
        s_team = (lt * weights["t_team"]).astype(np.float32)
        s_dead = (ld * weights["t_dead"]).astype(np.float32)
        s_team, s_dead, _ = emissions.add_restarts(r["f"], r["blocks"], s_team, s_dead,
                                                   r["pieces"], [], p)
        pred = decode.decode(s_team, s_dead, r["blocks"], len(r["labels"]), costs)
        res = pm.evaluate(pm.label_codes(r["labels"]), pm.label_codes(pred), bootstrap_n=0)
        scores.append((res["spell_f1_mean"], res["macro_f1"], res["n_gt_spells"],
                        res["n_pred_spells"]))
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", default=model.WEIGHTS)
    args = parser.parse_args()

    rows = build(args.samples, seeds=(0, 1))
    names = sorted({r["name"] for r in rows})
    combos = list(itertools.product(*GRID.values()))
    held = {c: [] for c in combos}

    for name in names:
        train = [r for r in rows if r["name"] != name]
        test = [r for r in rows if r["name"] == name]
        base = fit(train)
        logits = raw_logits(test, base)
        for c in combos:
            setting = dict(zip(GRID, c))
            w = dict(base, t_team=setting["t_team"], t_dead=setting["t_dead"])
            costs = {k: setting[k] for k in ("switch", "to_dead", "restart")}
            held[c] += [s[0] for s in evaluate(test, w, costs, logits)]
        print("fold {} done".format(name))

    ranked = sorted(combos, key=lambda c: -np.mean(held[c]))
    for c in ranked[:5]:
        print("held-out spellF1 {:.4f}  {}".format(np.mean(held[c]), dict(zip(GRID, c))))
    best = dict(zip(GRID, ranked[0]))

    weights = fit(rows)
    weights.update(t_team=best["t_team"], t_dead=best["t_dead"],
                   costs={k: best[k] for k in ("switch", "to_dead", "restart")},
                   held_out_spell_f1=float(np.mean(held[ranked[0]])),
                   trained_on="metrica-sports/sample-data games 1-3, degraded, seeds 0-1")
    with open(args.out, "w") as handle:
        json.dump(weights, handle, indent=1)
    print("final, in-sample:")
    for (name, seed), s in zip([(r["name"], r["seed"]) for r in rows],
                               evaluate(rows, weights, weights["costs"])):
        print("  {} seed{}  spellF1 {:.4f} macroF1 {:.4f}  poss gt {} pred {}".format(
            name, seed, *s))
    print("->", os.path.relpath(args.out, ROOT))


if __name__ == "__main__":
    main()
