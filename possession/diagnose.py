"""Aggregate statistics of the tracking, split by label -- safe to share (no frame-level values).

    python -m possession.diagnose

Why: the models are trained on OTHER games whose tracking we degrade to look like the ATD. If the
ATD behaves differently during dead balls (for instance, loses the ball far more often when it
leaves the pitch), the dead-ball model is miscalibrated on BAR-ATH. These averages show it.

Two tables:
  1. whole match, split by OUR predicted label (no ground truth involved);
  2. the public 5 minutes, split by the public ground truth -- a pipeline check, not a tuning set.
Only averages over thousands of frames are printed.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import model, pipeline  # noqa: E402

FPS = 25


def table(title, mask_by_label, f, p_dead):
    act = f["activity"].astype(float)
    med = np.nanmedian(act[np.any(list(mask_by_label.values()), axis=0)])
    print("\n" + title)
    print("  {:<6} {:>8} {:>10} {:>11} {:>9} {:>10} {:>11} {:>9}".format(
        "label", "frames", "ball seen", "stationary", "ball out", "activity", "touch/min", "P(dead)"))
    for lab, m in mask_by_label.items():
        if not m.any():
            continue
        print("  {:<6} {:>8} {:>9.0%} {:>11.1%} {:>9.1%} {:>10.2f} {:>11.1f} {:>9.2f}".format(
            lab, int(m.sum()), f["ball_seen"][m].mean(), f["stationary"][m].mean(),
            f["ball_out"][m].mean(), np.nanmedian(act[m]) / med,
            (f["touch"][m] != 0).sum() / (m.sum() / FPS / 60.0), p_dead[m].mean()))


def stretches(labels, letter, blocks):
    out = []
    for _, lo, hi in blocks:
        seg = labels[lo:hi + 1] == letter
        edges = np.diff(np.concatenate(([0], seg.astype(int), [0])))
        out += list((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)) / FPS)
    return np.array(out)


def main():
    weights = model.WEIGHTS if os.path.exists(model.WEIGHTS) else None
    r = pipeline.infer(weights_path=weights, verbose=False)
    f, labels, blocks = r["features"], r["labels"], r["blocks"]
    p_dead = 1.0 / (1.0 + np.exp(-np.clip(r["s_dead"].astype(float), -30, 30)))
    graded = labels != "X"
    table("1. whole match, by predicted label", {k: graded & (labels == k) for k in "HAD"},
          f, p_dead)
    d = stretches(labels, "D", blocks)
    print("  predicted stoppages: {}  median {:.1f} s  (< 8 s: {:.0%})".format(
        len(d), np.median(d), (d < 8).mean()))

    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    with open(gt_path) as handle:
        gt = np.array([line.rstrip().split(",")[1] for line in handle][1:])
    table("2. public 5 minutes, by ground truth (pipeline check only)",
          {k: gt == k for k in "HAD"}, f, p_dead)


if __name__ == "__main__":
    main()
