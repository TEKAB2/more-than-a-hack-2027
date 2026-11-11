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
    "switch": (10.0, 15.0, 22.0, 30.0),   # the possession-weighted team model needs 22-30
    "to_dead": (28.0, 40.0, 55.0, 75.0),
    "restart": (28.0, 40.0, 55.0, 75.0),
    "t_team": (0.7, 1.0, 1.4),
    "t_dead": (0.35, 0.5, 0.7),
}
TIED = True  # to_dead == restart: halves nothing that matters, quarters the search
SUBSAMPLE = 4  # every 4th frame for fitting: neighbouring frames are near-duplicates


def build(samples, seeds, team_variant, dead_variant):
    rows = []
    for seed in seeds:
        for name, atd, blocks, labels, events in E.prepare(samples, seed=seed):
            f = features.compute(atd, blocks, E.PITCH)
            ev = emissions.ball_team_evidence(f, emissions.PARAMS) / emissions.PARAMS["w_ball"]
            rows.append({
                "pieces": E.fake_set_piece_tags(events, np.random.default_rng(seed + 100)),
                "name": name, "seed": seed, "blocks": blocks, "labels": labels, "events": events,
                "f": f, "Xd": model.dead_features(f, blocks, E.PITCH, dead_variant),
                "Xt": model.team_features(f, blocks, E.PITCH, ev, team_variant),
            })
    return rows


LEARNER = "logistic"

# Team model only: spellF1 counts every possession once, but a per-frame loss counts a 20 s
# possession 20 times more than a 1 s one. Weight each frame by ref / (length of its possession),
# capped. Both constants are fixed a priori, not tuned on the bench.
SPELL_WEIGHT = True
SPELL_REF_S = 5.0
SPELL_CAP = 5.0


def spell_weights(labels):
    """Per-frame weight min(SPELL_CAP, SPELL_REF_S / duration of the frame's label run)."""
    n = len(labels)
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [n]))
    dur = np.repeat((ends - starts) / float(model.FPS), ends - starts)
    return np.minimum(SPELL_CAP, SPELL_REF_S / dur)


def fit(rows):
    Xd, yd, Xt, yt, wt = [], [], [], [], []
    for r in rows:
        lab = r["labels"]
        graded = np.flatnonzero(lab != "X")[::SUBSAMPLE]
        Xd.append(r["Xd"][graded])
        yd.append((lab[graded] == "D").astype(float))
        alive = graded[np.isin(lab[graded], ["H", "A"])]
        Xt.append(r["Xt"][alive])
        yt.append((lab[alive] == "H").astype(float))
        wt.append(spell_weights(lab)[alive])
    learn = model.fit_gbdt if LEARNER == "gbdt" else model.fit_logistic
    w = None
    if SPELL_WEIGHT:
        w = np.concatenate(wt)
        w = w / w.mean()          # mean 1: the L2 term keeps the same relative strength
    return {"dead": learn(np.vstack(Xd), np.concatenate(yd)),
            "team": learn(np.vstack(Xt), np.concatenate(yt), sample_weight=w)}


def raw_logits(rows, weights):
    """Model logits per row, before temperatures and restarts (computed once per fold)."""
    return [(model.logit_any(weights["team"], r["Xt"]), model.logit_any(weights["dead"], r["Xd"]))
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
                        res["n_pred_spells"]) + timing(r["labels"], pred)
                      + (res["spell_f1"][pm.SPELL_IOU_HEADLINE]["n_spurious"],)
                      + team_change_hits(r["labels"], pred))
    return scores


CHANGE_TOLS = (12, 25)   # frames: +/- 0.5 s and +/- 1 s, the tolerances of the official transF1


def team_change_hits(gt, pred):
    """(true team changes, predicted team changes, matched within 0.5 s, matched within 1 s).

    Only H <-> A changes, matched one to one on the destination team with the official matcher.
    Replaces the median delay, which is the same one frame whatever the model: these counts drop
    when changes are fired too early or too late.
    """
    gt_codes, pred_codes = pm.label_codes(gt), pm.label_codes(pred)
    runs = pm.valid_runs(gt_codes)
    filled = pm.fill_abstentions(pred_codes, runs)
    alive = (pm.H, pm.A)

    def team_changes(codes):
        return [c for c in pm.transitions(codes, runs) if c[1] in alive and c[2] in alive]

    g, p = team_changes(gt_codes), team_changes(filled)
    return (len(g), len(p)) + tuple(len(pm.match_transitions(g, p, tol)[0])
                                    for tol in CHANGE_TOLS)


def _changes(labels):
    out = []
    for i in np.flatnonzero(labels[1:] != labels[:-1]) + 1:
        a, b = labels[i - 1], labels[i]
        if "X" in (a, b):
            continue
        out.append((i, a, b))
    return out


def timing(gt, pred, window=5 * 25):
    """(median delay of team changes s, median delay of restarts s, short-possession recall).

    Delay = predicted frame - true frame for the nearest predicted change of the same kind within
    5 s; negative = early. Short possessions (< 3 s) count as found when a predicted spell of the
    same team overlaps them with IoU >= 0.5.
    """
    P = _changes(pred)
    pf = np.array([p[0] for p in P])
    team_d, restart_d = [], []
    for f, a, b in _changes(gt):
        kind = "team" if {a, b} == {"H", "A"} else "restart" if a == "D" else None
        if kind is None or not len(pf):
            continue
        cands = [p for p in P if abs(p[0] - f) <= window and p[2] == b
                 and ((kind == "team" and p[1] in "HA") or (kind == "restart" and p[1] == "D"))]
        if cands:
            d = min(cands, key=lambda p: abs(p[0] - f))[0] - f
            (team_d if kind == "team" else restart_d).append(d / 25.0)
    runs, short_hit, short_n = [], 0, 0
    i = 0
    while i < len(gt):
        j = i
        while j < len(gt) and gt[j] == gt[i]:
            j += 1
        if gt[i] in "HA" and j - i < 3 * 25:
            short_n += 1
            lo, hi = max(0, i - 75), min(len(pred), j + 75)
            seg = pred[lo:hi] == gt[i]
            k = 0
            while k < len(seg):
                if seg[k]:
                    m = k
                    while m < len(seg) and seg[m]:
                        m += 1
                    a, b = lo + k, lo + m
                    inter = max(0, min(b, j) - max(a, i))
                    if inter / float(max(b, j) - min(a, i)) >= 0.5:
                        short_hit += 1
                        break
                    k = m
                else:
                    k += 1
        i = j
    med = lambda x: float(np.median(x)) if x else float("nan")  # noqa: E731
    return med(team_d), med(restart_d), short_hit / max(short_n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=os.environ.get("MTAH_SAMPLES"),
                        help="clone of metrica-sports/sample-data (default: $MTAH_SAMPLES)")
    parser.add_argument("--out", default=model.WEIGHTS)
    parser.add_argument("--team-variant", default=model.TEAM_VARIANT)
    parser.add_argument("--dead-variant", default="stat")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--learner", choices=("logistic", "gbdt"), default="logistic")
    parser.add_argument("--quick", action="store_true",
                        help="no grid search: reuse the costs and temperatures of the current "
                             "possession/weights.json (fast comparison of a feature change; "
                             "does not write any weights)")
    parser.add_argument("--mini-grid", action="store_true",
                        help="like --quick, but also searches --grid-switch x --grid-t-team "
                             "(dead-ball costs unchanged); prints every setting, writes "
                             "nothing")
    parser.add_argument("--grid-switch", default="10,15,22",
                        help="switch costs searched by --mini-grid (comma-separated)")
    parser.add_argument("--grid-t-team", default="1.0,1.4,2.0",
                        help="team temperatures searched by --mini-grid (comma-separated)")
    parser.add_argument("--no-spell-weight", action="store_true",
                        help="fit the team model without the per-possession frame weights")
    args = parser.parse_args()
    if not args.samples:
        parser.error("--samples is required (or set the MTAH_SAMPLES environment variable)")
    if args.quick or args.mini_grid:
        current = model.load_weights()
        GRID.clear()
        GRID.update({"switch": (current["costs"]["switch"],),
                     "to_dead": (current["costs"]["to_dead"],),
                     "restart": (current["costs"]["restart"],),
                     "t_team": (current["t_team"],), "t_dead": (current["t_dead"],)})
    if args.mini_grid:
        GRID.update({"switch": tuple(float(v) for v in args.grid_switch.split(",")),
                     "t_team": tuple(float(v) for v in args.grid_t_team.split(","))})
    global LEARNER, SPELL_WEIGHT
    LEARNER = args.learner
    SPELL_WEIGHT = not args.no_spell_weight

    rows = build(args.samples, tuple(range(args.seeds)), args.team_variant, args.dead_variant)
    names = sorted({r["name"] for r in rows})
    combos = [c for c in itertools.product(*GRID.values()) if not TIED or c[1] == c[2]]
    held = {c: [] for c in combos}
    timings = {}
    counts = {c: [] for c in combos}
    by_source = {c: {"metrica": [], "dfl": []} for c in combos}
    changes = {c: [] for c in combos}

    for name in names:
        train = [r for r in rows if r["name"] != name]
        test = [r for r in rows if r["name"] == name]
        base = fit(train)
        logits = raw_logits(test, base)
        for c in combos:
            setting = dict(zip(GRID, c))
            w = dict(base, t_team=setting["t_team"], t_dead=setting["t_dead"])
            costs = {k: setting[k] for k in ("switch", "to_dead", "restart")}
            res = evaluate(test, w, costs, logits)
            held[c] += [s[0] for s in res]
            timings.setdefault(c, []).extend(s[4:7] for s in res)
            counts[c] += [(s[2], s[3], s[7]) for s in res]
            changes[c] += [s[8:12] for s in res]
            # the two label conventions differ (Metrica: controlled possession, DFL: per touch)
            source = "metrica" if name.startswith("sample") else "dfl"
            by_source[c][source] += [s[0] for s in res]
        print("fold {} done".format(name))

    ranked = sorted(combos, key=lambda c: -np.mean(held[c]))
    for c in (ranked if args.mini_grid else ranked[:5]):
        gt_n, pred_n, spurious = np.sum(counts[c], axis=0)
        tc = np.nanmedian(np.array(timings[c], dtype=float), axis=0)
        n_true, n_fired, hit_half, hit_one = np.sum(changes[c], axis=0)
        print("held-out spellF1 {:.4f}  {}  possessions gt {} pred {} spurious@{} {}  "
              "short found {:.0%}".format(
                  np.mean(held[c]), dict(zip(GRID, c)), int(gt_n), int(pred_n),
                  pm.SPELL_IOU_HEADLINE, int(spurious), tc[2]))
        print("    team changes gt {} pred {}: found within 0.5 s {:.1%}, within 1 s {:.1%} "
              "(of predicted: {:.1%}, {:.1%})".format(
                  int(n_true), int(n_fired), hit_half / max(n_true, 1), hit_one / max(n_true, 1),
                  hit_half / max(n_fired, 1), hit_one / max(n_fired, 1)))
        print("    by label source: " + "  ".join(
            "{} {:.4f} ({} games)".format(k, np.mean(v), len(v))
            for k, v in by_source[c].items() if v))
    best = dict(zip(GRID, ranked[0]))
    t = np.array(timings[ranked[0]], dtype=float)
    print("held-out timing (median over games): team change {:+.2f}s  restart {:+.2f}s  "
          "short possessions found {:.0%}".format(*np.nanmedian(t, axis=0)))
    if args.mini_grid:
        print("MINI-GRID done (nothing written)")
        return
    if args.quick:
        print("QUICK held-out spellF1 {:.4f} (costs of the current weights.json; nothing written)"
              .format(np.mean(held[ranked[0]])))
        return

    weights = fit(rows)
    weights.update(t_team=best["t_team"], t_dead=best["t_dead"],
                   team_variant=args.team_variant, dead_variant=args.dead_variant,
                   costs={k: best[k] for k in ("switch", "to_dead", "restart")},
                   held_out_spell_f1=float(np.mean(held[ranked[0]])),
                   trained_on="metrica-sports/sample-data games 1-3 + IDSSE open Bundesliga "
                              "matches (validation/external_data), degraded")
    if LEARNER == "gbdt":
        model_file = os.path.splitext(os.path.basename(args.out))[0] + "_trees.json"
        with open(os.path.join(os.path.dirname(os.path.abspath(args.out)), model_file), "w") as h:
            json.dump({"dead": model.export_trees(weights.pop("dead")),
                       "team": model.export_trees(weights.pop("team"))}, h)
        weights.update(kind="gbdt", model_file=model_file)
    with open(args.out, "w") as handle:
        json.dump(weights, handle, indent=1)
    if LEARNER == "gbdt":
        weights = model.load_weights(args.out)
    print("final, in-sample:")
    for (name, seed), s in zip([(r["name"], r["seed"]) for r in rows],
                               evaluate(rows, weights, weights["costs"])):
        print("  {} seed{}  spellF1 {:.4f} macroF1 {:.4f}  poss gt {} pred {}  "
              "delay team {:+.2f}s restart {:+.2f}s  short-poss recall {:.2f}".format(
                  name, seed, *s))
    print("->", os.path.relpath(args.out, ROOT))


if __name__ == "__main__":
    main()
