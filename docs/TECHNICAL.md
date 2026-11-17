# TercerHombre — More than a Hack 2027, Phase 1: technical note

*Entry on branch `clean-trees`: GBDT (`duel_past`) + MS-TCN 102ch ensemble, simple Viterbi.
No PFF. External data only.*

## 1. Approach

Possession is an interval, and the ranking metric (spellF1) scores whole possessions, so the
method is built around one idea: **decide possession as a hidden state, not frame by frame.**

1. **Frame evidence (trees).** From the tracking we compute two log-odds per frame: *dead vs
   alive* and *home vs away given alive*. They come from two **gradient-boosted trees**
   (scikit-learn `HistGradientBoostingClassifier`, fixed seed; team variant `duel_past`, dead
   variant `stat`) trained on **other games** (§3), over windowed features of the ball and of the
   22 players. The team model is fit with **per-possession frame weights**. Two features carry
   the possession rule directly:
   * **control**: the nearest player has the ball at his feet (< 1.5 m) *and* it travels with
     him (relative speed < 2.5 m/s). "Last team in control" (carried 2 s and 6 s) encodes
     "a contested or loose ball stays with the last team".
   * **stationary free ball**: seen, not moving for ~1 s, nobody on it: a dead ball the Smart
     Tagging often does not tag.
2. **Frame evidence (sequence).** A 2-stage MS-TCN (102 input channels: dead/team features +
   kinematic duel/deflection + carrier / ball-imputation channels) reads the same windows over a
   long temporal context and produces sequence log-odds for team and dead-ball. Weights are
   exported to `possession/weights_seq.npz` and evaluated with **numpy only** (checked against
   torch, max abs diff < 1e-5).
3. **Ensemble.** Tree and sequence emissions are combined (`possession/ensemble.py`: ~25/75 team,
   ~30/70 dead, boundary gate on trees). **Simple Viterbi** over {H, A, D} (switch / to_dead /
   restart; `change_alpha = 0` — no directional change head at decode time).
4. **Restarts.** Smart Tagging set pieces: dead before the detected strike, alive after, with the
   tagged team when named. Strike from positions (rest then move), including soft kick-offs.

## 2. Assets used

| asset | use |
|---|---|
| tactical ATD | all features (player and ball positions, player speeds) |
| Smart Tagging | set pieces → restarts; phase team as a weak prior |
| homography, videos | not used by the pipeline |
| broadcast feed | not used (no alignment to the tactical timeline) |
| ball speed channel of the ATD | not used: empty on ~89 % of frames; speed from positions |

## 3. Training data

No label of the provided game is used, in any amount.

* **IDSSE open Bundesliga data** — 7 full matches (Bassek et al., Scientific Data 12, 195, 2025;
  CC BY 4.0; `validation/fetch_idsse.py`). DFL BallStatus / BallPossession; brief opponent spells
  (< 1.6 s) merged (`merge_brief`).
* **Metrica Sports public sample data** — 3 full matches; frame labels from events.
* Tracking **degraded** with the external-only simulator (`validation/external.py`): ball
  visibility and typed stoppages from external games (Compliance), not from the provided match.
* **PFF FC World Cup data is not used.**

Held-out spellF1 (10 games, leave-one-out, 2 degradation seeds), clean-trees simulator:

| model | spellF1 10 games | Metrica alone |
|---|---|---|
| GBDT trees alone (`duel_past`) | 0.687 | 0.696 |
| **GBDT + MS-TCN 102ch, simple Viterbi (this entry)** | **0.782** | **0.814** |

## Compliance with the rules clarification of 16 November 2026 (branch `clean-trees`)

* **Training and tuning use external data only**: Metrica public sample games (3) and the IDSSE
  open Bundesliga matches (7, CC BY 4.0). The PFF FC World Cup data is **not** used here.
* **No constant is measured on the provided game.** The tracking simulator
  (`validation/external.py`) takes its ball-visibility rates from Metrica's own public sample
  games (ball lost on 7-13 % of in-play frames, gaps of mean 4.5 s; 92-100 % lost when the ball is
  out of play). Stoppages are typed from the external game itself: *ball out* (off camera) or
  *on pitch* (foul / free kick, ball still in view). The figure published in DATA.md (59.3 % of
  graded frames with a ball) is used as a sanity check only: the simulator gives 68 %.
* Effect of the typed stoppages, measured on the external bench (10 games, leave-one-game-out):
  stoppages where the ball stays in view found 94 % of the time, against 33 % for the previous
  model, which had learnt "ball missing = dead ball"; trees-alone spellF1 0.674 on the earlier
  freeze of this simulator; with 2 seeds on the same protocol as the MS-TCN LOO, trees alone
  0.687.
* Nothing is trained, fitted or calibrated on the ground-truth possession data or on the provided
  video. The public 5-minute ground truth is read by `possession/diagnose.py` and by the official
  scorer only, as an automatic check.
* Before the clarification, the team used a local viewer (`tools/viewer.html`) to understand what
  counts as possession. No label, correction or frame-specific constant was derived from it, and
  it has not been used to check outputs since.
* Inference for the submission is **numpy-only** (exported trees JSON + exported MS-TCN `.npz`);
  no pickle, no network call, no torch at submission time.

## 4. Possession rules implemented

* Loose or contested ball: possession stays with the last team (no evidence → no switch).
* Deflection, single touch, aerial duel without follow-up control: absorbed by the switch cost.
* Restarts: dead until the strike, then the restarting team.
* Second half: dead until the kick-off is detected at the centre spot.
* `X` is never written on graded frames.

## 5. Limitations

* Event-derived Metrica labels approximate corrected-tracking truth; absolute scores are not
  BAR-ATH scores.
* Dead-ball periods in sample data often lack ball tracking before degrade.
* Short possessions remain hard; the sequence model recovers more of them than trees alone
  (~59 % short recall held-out vs ~41 % for trees) but still misses many.
* The held-out bench still overestimates the public 5-minute window.

## 6. Reproduction

### Weight provenance (submission)

1. **possession/weights_seq.npz** was produced by a **final training on all 10 bench games**
   (3 Metrica + 7 IDSSE), with 2 degradation seeds and no PFF — via --write-weights after the
   leave-one-out measurement. It is **not** one of the per-fold LOO checkpoints under
   cache/seq_features_cleantrees_102/.
2. **Tree/sequence mix coefficients** (w_tree_team=0.25, w_seq_team=0.75,
   w_tree_dead=0.30, w_seq_dead=0.70, scales 1.0/0.7) were **fixed in advance** in
   alidation/train_seq.py (and ensemble.DEFAULT_ENSEMBLE); they were **not** selected by
   comparing alternatives on this held-out bench. (Only the decode choice change_alpha=0
   vs DirGated was taken from the held-out comparison on this run.)

Exact training / write command used (torch needed only for training):

`ash
python -m validation.train_seq --samples ../sample-data --carrier \
  --train-seeds 2 --eval-seeds 2 --model-seeds 1 --steps 3000 --batch-size 4 \
  --cache-dir cache/seq_features_cleantrees_102 --reuse-checkpoints --write-weights
`

(No --use-pff. LOO measurement used the same flags without --write-weights /
--reuse-checkpoints.)

`ash
pip install -r requirements.txt -r validation/requirements.txt
git clone https://github.com/metrica-sports/sample-data ../sample-data
python -m validation.fetch_idsse
# optional: retrain LOO / write weights (command above)
# submission (numpy only)
python -m possession.pipeline --team TercerHombre
`

Deterministic: the same inputs produce a byte-identical CSV.

## Explored, not used in this entry

* **PFF FC FIFA World Cup 2022** — available via `validation/fetch_pff.py`; **not** used in this
  entry (licence clarification pending; Compliance). An experimental measurement branch /
  protocol is documented in `docs/PFF_VARIANT.md` (`validation/measure_pff_gap.sh`); it must not
  replace the default submission until organisers authorise PFF.
* **SkillCorner Open Data** (10 A-League broadcast matches, MIT) — measured as extra training
  without degrade; **hurt** Metrica held-out (−0.033); not adopted.
* **DirGated + KinSnap decode** — measured beside simple Viterbi; slightly worse held-out for the
  ensemble on this run (0.777 vs 0.782); not used in the submission (`change_alpha = 0`).
* **Multi-seed MS-TCN pools / 5-model ensembles** — earlier `main` experiments; this entry ships
  **one** 102ch model + trees (one model in memory during LOO training).
