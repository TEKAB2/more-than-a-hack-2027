"""Leave-one-game-out evaluation and hyperparameter search for GBDT + MS-TCN ensembling and
dynamic `p_change` decoding.

Usage:
    # 1. Verify tree-only baseline + kinematic/learned change head on the 10-game bench:
    python -m validation.ensemble_eval --samples $MTAH_SAMPLES --seeds 2

    # 2. Combine v7 GBDT out-of-fold logits with MS-TCN out-of-fold predictions (.npz):
    python -m validation.ensemble_eval --samples $MTAH_SAMPLES --seq-oof path/to/seq_oof.npz
"""

import argparse
import itertools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from possession import decode, emissions, ensemble, model  # noqa: E402
from validation import external as E  # noqa: E402
from validation import train as T  # noqa: E402


def load_seq_oof(path, rows):
    """Load out-of-fold MS-TCN predictions from an `.npz` archive keyed by `<name>_seed<seed>`."""
    z = np.load(path)
    out = []
    for r in rows:
        prefix = "{}_seed{}".format(r["name"], r["seed"])
        if prefix + "_s_team" in z:
            st = z[prefix + "_s_team"]
            sd = z[prefix + "_s_dead"]
            pc = z[prefix + "_p_change"] if (prefix + "_p_change") in z else None
        else:
            st = z[r["name"] + "_s_team"]
            sd = z[r["name"] + "_s_dead"]
            pc = z[r["name"] + "_p_change"] if (r["name"] + "_p_change") in z else None
        out.append({"s_team": st, "s_dead": sd, "p_change": pc})
    return out


def score_rows(rows, tree_logits, seq_outputs, ens_cfg, cost_cfg):
    p = dict(emissions.PARAMS, pitch=E.PITCH)
    f1s, f1_m, f1_d = [], [], []
    short_hits, short_totals, spurious_total = 0, 0, 0
    gt_chg, pred_chg, hit05, hit10 = 0, 0, 0, 0

    for i, r in enumerate(rows):
        lt, ld = tree_logits[i]
        s_out = seq_outputs[i] if seq_outputs is not None else None
        res_dec = ensemble.ensemble_and_decode(
            r["f"], r["blocks"], len(r["labels"]), E.PITCH,
            tree_out=(lt, ld), seq_out=s_out,
            set_pieces=r["pieces"], params=p,
            ensemble_cfg=ens_cfg, costs=cost_cfg,
        )
        pred = res_dec["labels"]
        gt = r["labels"]
        gt_c, pred_c = pm.label_codes(gt), pm.label_codes(pred)
        ev = pm.evaluate(gt_c, pred_c, bootstrap_n=0)
        f1 = ev["spell_f1_mean"]
        f1s.append(f1)
        if r["name"].startswith("sample"):
            f1_m.append(f1)
        else:
            f1_d.append(f1)
        spurious_total += ev["spell_f1"][pm.SPELL_IOU_HEADLINE]["n_spurious"]

        runs = pm.valid_runs(gt_c)
        filled = pm.fill_abstentions(pred_c, runs)
        gt_spells = pm.possession_spells(gt_c, runs)
        pred_spells = pm.possession_spells(filled, runs)
        overlaps = pm.spell_overlaps(gt_spells, pred_spells)
        flags05 = pm.matched_gt_flags(overlaps, len(gt_spells), 0.5)
        for idx_s, (a, b, _) in enumerate(gt_spells):
            if (b - a) < 3 * 25:
                short_totals += 1
                short_hits += int(flags05[idx_s])

        nt, np_c, h05, h10 = T.team_change_hits(gt, pred)
        gt_chg += nt
        pred_chg += np_c
        hit05 += h05
        hit10 += h10

    return {
        "spell_f1": float(np.mean(f1s)),
        "metrica_f1": float(np.mean(f1_m)) if f1_m else float("nan"),
        "dfl_f1": float(np.mean(f1_d)) if f1_d else float("nan"),
        "short_recall": short_hits / max(short_totals, 1),
        "spurious": int(spurious_total),
        "chg_05_recall": hit05 / max(gt_chg, 1),
        "chg_05_prec": hit05 / max(pred_chg, 1),
        "chg_10_recall": hit10 / max(gt_chg, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=os.environ.get("MTAH_SAMPLES"))
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--team-variant", default="duel_past")
    parser.add_argument("--dead-variant", default="stat")
    parser.add_argument("--seq-oof", default=None,
                        help="optional .npz with out-of-fold MS-TCN predictions")
    parser.add_argument("--grid", action="store_true",
                        help="search ensemble weights, temperatures, and change-head costs")
    args = parser.parse_args()
    if not args.samples:
        parser.error("--samples is required (or set MTAH_SAMPLES)")

    T.LEARNER = "gbdt"
    T.SPELL_WEIGHT = True
    current = model.load_weights()
    base_costs = dict(current["costs"])

    rows = T.build(args.samples, tuple(range(args.seeds)), args.team_variant, args.dead_variant)
    names = sorted({r["name"] for r in rows})
    tree_logits = [None] * len(rows)

    for name in names:
        train_rows = [r for r in rows if r["name"] != name]
        test_idx = [i for i, r in enumerate(rows) if r["name"] == name]
        base = T.fit(train_rows)
        for i in test_idx:
            r = rows[i]
            lt = model.logit_any(base["team"], r["Xt"])
            ld = model.logit_any(base["dead"], r["Xd"])
            tree_logits[i] = (lt, ld)
        print("fold {} tree OOF done".format(name))

    seq_outputs = load_seq_oof(args.seq_oof, rows) if args.seq_oof else None

    base_ens = dict(
        ensemble.DEFAULT_ENSEMBLE,
        w_tree_team=1.0 if seq_outputs is None else 0.30,
        w_seq_team=0.0 if seq_outputs is None else 0.70,
        w_tree_dead=1.0 if seq_outputs is None else 0.35,
        w_seq_dead=0.0 if seq_outputs is None else 0.65,
        scale_tree_team=current.get("t_team", 1.0),
        scale_tree_dead=current.get("t_dead", 0.7),
    )

    res0 = score_rows(rows, tree_logits, seq_outputs, base_ens, base_costs)
    print(
        "BASELINE  spellF1 {spell_f1:.4f} (metrica {metrica_f1:.4f}, dfl {dfl_f1:.4f})  "
        "short {short_recall:.1%}  spurious {spurious}  "
        "chg@0.5s {chg_05_recall:.1%} (prec {chg_05_prec:.1%})  chg@1.0s {chg_10_recall:.1%}"
        .format(**res0)
    )

    if args.grid and seq_outputs is not None:
        w_trees = (0.15, 0.25, 0.35, 0.50)
        t_teams = (0.85, 1.0, 1.15)
        alphas = (0.0, 16.0, 20.0, 24.0)
        guards = (0.0, 6.0, 8.0)
        threshs = (0.35, 0.40)
        best_score, best_cfg = -1.0, None
        for wt, tt, al, gd, th in itertools.product(w_trees, t_teams, alphas, guards, threshs):
            if al == 0.0 and (gd != 0.0 or th != threshs[0]):
                continue
            ecfg = dict(base_ens, w_tree_team=wt, w_seq_team=1.0 - wt, t_team=tt)
            ccfg = dict(base_costs, change_alpha=al, change_guard=gd, change_thresh=th,
                        switch_min=4.0, change_nms=18, directional_gate=True)
            s = score_rows(rows, tree_logits, seq_outputs, ecfg, ccfg)
            if s["spell_f1"] > best_score:
                best_score, best_cfg = s["spell_f1"], (wt, tt, al, gd, th, s)
        wt, tt, al, gd, th, s = best_cfg
        print(
            "BEST GRID (w_tree={:.2f}, t_team={:.2f}, alpha={:.1f}, guard={:.1f}, thr={:.2f}) -> "
            "spellF1 {spell_f1:.4f}  short {short_recall:.1%}  spurious {spurious}  "
            "chg@0.5s {chg_05_recall:.1%} (prec {chg_05_prec:.1%})".format(
                wt, tt, al, gd, th, **s
            )
        )


if __name__ == "__main__":
    main()
