# `possession/` — frame-level ball possession, v4

State-model pipeline for More than a Hack 2027, Phase 1. Drop this folder (and `tests/`) into a
fork of the official repository; it imports `games.py` from the repo root.

## Run

```bash
pip install -r requirements.txt                       # numpy + pandas, nothing else
python -m possession.inspect_atd                      # structure of the ATD (no values printed)
python -m possession.pipeline --team TEAM_NAME        # -> submissions/TEAM_NAME.csv
python -m possession.pipeline --team TEAM_NAME --check   # + official validator + public sanity score
```

Deterministic: same data in, byte-identical CSV out (no randomness, no training on the match).
Runtime ≈ 1 min the first time (parses the 199,319-line ATD into `cache/`), seconds afterwards.

## Method

1. **Parse** the raw ATD (tactical source) into arrays in metres. `video_frame = frame_id - 1`.
2. **Features** per frame, computed inside each graded half only: ball interpolated over gaps
   ≤ 0.5 s, ball speed, nearest home/away player distance, players near the ball, ball outside
   the pitch, player activity relative to the half's median, team block advancement.
3. **Evidence** as two log-odds per frame: `s_dead` (dead vs alive) and `s_team` (home vs away).
   Control evidence only counts when the ball is observed, slow, and within 3 m of a player.
   Smart Tagging adds restarts (set pieces → dead before the detected strike, alive after, team
   of the taker) and a weak phase-team prior.
4. **Decode** with Viterbi over {H, A, D} and fixed switch costs: a change of possession needs
   sustained evidence (controlled reception), a deflection or a lost duel does not pay for it.

v2: the frame evidence of step 3 comes from two logistic regressions (`model.py`, weights in
`weights.json`) trained on Metrica's public sample games, and the decoder costs are chosen there
by leave-one-game-out spellF1 — see `validation/` and `docs/TECHNICAL.md`. `--v1` restores the
hand-set evidence. Nothing is tuned on the public 5-minute window, and no human label of the
provided game is used.

| held out, 10 other games (3 Metrica + 7 Bundesliga) | spellF1 |
|---|---|
| official baseline, ±2 s window | 0.239 |
| v3 (logistic, control features) | 0.533 |
| v4 (boosted trees, + touch detection) | 0.564 |

Regenerate the weights: `git clone https://github.com/metrica-sports/sample-data ../sample-data`
then (Python 3.11+, `pip install -r validation/requirements.txt`) `python -m validation.fetch_idsse` and
`python -m validation.train --samples ../sample-data --team-variant touch --learner gbdt` (~40 min).

## Tests

`python tests/synth.py <dir>` writes a synthetic match in the exact bundle formats, so the whole
pipeline can be run and scored (`evaluation/run_evaluation.py --gt <dir>/synthetic_gt.csv`)
without the licensed data. It is a smoke test of the code, not a measure of real performance.

## Data hygiene

`data/`, `cache/` and `submissions/` are gitignored. Never commit match data or frame-level
exports of it.
