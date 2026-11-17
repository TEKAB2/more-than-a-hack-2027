# TercerHombre — More than a Hack 2027, Phase 1: technical note

*Draft. Numbers marked TODO are filled in at freeze.*

## 1. Approach

Possession is an interval, and the ranking metric (spellF1) scores whole possessions, so the
method is built around one idea: **decide possession as a hidden state, not frame by frame.**

1. **Frame evidence.** From the tracking we compute two log-odds per frame: *dead vs alive* and
   *home vs away given alive*. They come from two logistic regressions trained on **other
   games** (§3), over windowed features of the ball and of the 22 players. Two features carry
   the possession rule directly:
   * **control**: the nearest player has the ball at his feet (< 1.5 m) *and* it travels with
     him (relative speed < 2.5 m/s). A ball in flight towards a receiver, or bouncing off a
     defender, is near a player without being controlled, so it no longer pulls the decision
     towards that player's team before the reception. "Last team in control" (carried 2 s and
     6 s) encodes "a contested or loose ball stays with the last team".
   * **stationary free ball**: seen, not moving for ~1 s, nobody on it: the ball placed for a free
     kick, a corner or a goal kick, i.e. a dead ball the Smart Tagging often does not tag.
2. **Restarts.** Smart Tagging set pieces are treated as restarts: the ball is dead before the
   detected strike and alive after it, with the tagged team in possession when the tag names one.
   The strike is found from positions, not speed (the ball rested within 0.3 m for 8 frames and is
   0.6 m away 3 frames later; at the centre spot for a kick-off), so a soft kick-off pass is
   caught on the frame it is played.
3. **Decoding.** A Viterbi pass over {H, A, D} with three switch costs (team change, alive→dead,
   dead→alive). A change of possession must be paid for by sustained evidence — a controlled
   reception — while a deflection, a touch or a lost duel does not pay for it. This is the
   challenge's own definition of a change of possession, expressed as an optimisation.

## 2. Assets used

| asset | use |
|---|---|
| tactical ATD | all features (player and ball positions, player speeds) |
| Smart Tagging | set pieces → restarts; phase team as a weak prior |
| homography, videos | not used by the pipeline; the video was watched to understand the data |
| broadcast feed | not used (no alignment to the tactical timeline) |
| ball speed channel of the ATD | not used: empty on ~89 % of frames; speed is derived from positions |

## 3. Training data

No label of the provided game is used, in any amount.

* **IDSSE open Bundesliga data** — 7 full matches, 2022/23 Bundesliga 1 and 2 (Bassek et al.,
  *An integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data 12,
  195, 2025; CC BY 4.0; downloaded with kloppy by `validation/fetch_idsse.py`). Their DFL
  position data carries, on every frame, the league's corrected **BallStatus** and
  **BallPossession** — the same kind of fields the challenge's ground truth is derived from.
  DFL switches possession on any touch, so an opponent spell shorter than 1.6 s between two
  spells of the same team is handed back to that team (`merge_brief`); 1.6 s is the value at which
  these games reproduce the possession statistics of Metrica's own public games (≈ 350 per
  match, ≈ 30 % under 3 s).
* **Metrica Sports public sample data** — 3 full matches with tracking and human event data.
  Frame labels are derived from the events: an on-ball event (set piece, pass, carry, recovery,
  shot) gives possession to its team; a ball out, a shot out or in goal, or a foul starts a dead
  ball that lasts until the next on-ball event.
* All tracking is **degraded to resemble the ATD**. Ball visibility depends on the state of play,
  as in the ATD: missing on ~13 % of in-play frames (short bursts, more often when the ball is
  crowded or fast) and ~73 % of dead-ball frames (long bursts, ball off camera). In play the
  ball is lost rarely but for seconds at a time (≥ 1 s, mean ≈ 5 s): the ATD's 168 gaps have a
  median of 9.6 s and short losses are almost absent, as the ATD bridges them itself. These two rates
  were measured on BAR-ATH by `possession/diagnose.py`, grouping frames by our own predicted
  labels (algorithmic pseudo-labels, no ground truth), and agree with the published overall
  figure (59.3 % of graded frames seen). Plus low-frequency positional error and track gaps.
  Restarts are given to the model as noisy pseudo-tags (jittered, often missing, throw-ins
  without a team), like the Smart Tagging.
* Emission models: gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`, fixed
  seed), exported to plain JSON (`possession/weights_trees.json`) and evaluated with numpy, so
  producing the submission needs only numpy and pandas on any Python ≥ 3.8; scikit-learn
  (Python ≥ 3.11) is needed only to retrain. Decoder costs and temperatures are chosen by **leave-one-game-out** cross-validation on
  spellF1 over the 10 games (`validation/train.py`).

| held out, 10 games | spellF1 | delay of a change of team | short possessions found |
|---|---|---|---|
| official baseline (±2 s window) | 0.239 | — | — |
| logistic, proximity + control features | 0.533 | +0.09 s | 9 % |
| logistic, + touch detection | 0.543 | +0.08 s | 14 % |
| boosted trees, + touch detection | 0.564 | +0.04 s | 19 % |

With the state-dependent ball visibility (v5 bench, more faithful to the ATD):

| held out, 10 games | spellF1 | delay of a change of team | short possessions found |
|---|---|---|---|
| official baseline (±2 s window) | 0.283 | — | — |
| v4 model (trained on the previous bench) | 0.618 * | — | — |
| **v5: same model retrained on this bench** | **0.641** | −0.04 s | 28 % |

\* not held out: v4 was fitted on these games.

With long in-play gaps as in the ATD (v6 bench; the baseline scores 0.257 here against 0.2433 on
BAR-ATH, the closest calibration so far):

| held out on the v6 bench, 10 games | spellF1 |
|---|---|
| official baseline (±2 s window) | 0.257 |
| model fitted on the v5 simulator (short in-play gaps) | 0.591 |
| v6: GBDT fitted on the v6 simulator | 0.614 (better on all 10 games) |
| v7: GBDT (`duel_past` + `stat`, 2 seeds) | 0.681 |
| MS-TCN 58ch (2-stage dilated TCN) | 0.740 |
| MS-TCN 78ch (kinematic deflection + boundary channels) | 0.764 |
| **MS-TCN 102ch + GBDT ensemble (physics ball imputation + 22-player carrier & TTI)** | **0.775** (Metrica 0.805, DFL 0.763, macroF1 0.929, short <3s recall 61 %) |

 The baseline scores 0.283 on this bench against
0.2433 on BAR-ATH, so this bench is somewhat easier than the real task; it ranks versions, it does
not predict the final score.

**Touch detection** (`features.detect_touches`, after Link & Hoernig, *PLoS ONE* 2017, and
ELASTIC, Kim et al., 2025): a local peak of the ball's acceleration (≥ 10 m/s²) or a sharp change
of direction (≥ 25°) while a player is within 2 m. "Two touches in a row by one team" is the
observable proxy used for *controlled* possession.

## Compliance with the rules clarification of 16 November 2026 (branch `clean-trees`)

This branch is the conservative entry: gradient-boosted trees + Viterbi only.

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
  model, which had learnt "ball missing = dead ball"; spellF1 0.674 (previous model 0.670).
* Nothing is trained, fitted or calibrated on the ground-truth possession data or on the provided
  video. The public 5-minute ground truth is read by `possession/diagnose.py` and by the official
  scorer only, as an automatic check.
* Before the clarification, the team used a local viewer (`tools/viewer.html`) to understand what
  counts as possession. No label, correction or frame-specific constant was derived from it, and
  it has not been used to check outputs since.

## 4. Possession rules implemented

* Loose or contested ball: possession stays with the last team (no evidence → no switch).
* Deflection, single touch, aerial duel without follow-up control: absorbed by the switch cost.
* Restarts: dead until the strike, then the restarting team.
* Second half: dead until the kick-off is detected at the centre spot.
* `X` is never written on graded frames.

## 5. Limitations

* The event-derived labels of the sample games approximate, but are not, the corrected-tracking
  ground truth; absolute scores there are not BAR-ATH scores.
* Dead-ball periods in the sample data have no ball tracking; we interpolate a ball path through
  them before degrading, so ball-based dead cues may be miscalibrated.
* **Short possessions** (< 3 s, about 30 % of all possessions) are mostly missed: about 1 in 10
  is recovered. Re-inserting them from control episodes after decoding was tried and lowered
  spellF1 (more false possessions than recovered ones), so it is not used.
* Dead balls after a foul with no tag rely on the stationary-ball and player-activity cues only.
* TODO after freeze.

## 6. Reproduction

```bash
pip install -r requirements.txt
# optional: regenerate the learned weights (already committed in possession/weights.json)
git clone https://github.com/metrica-sports/sample-data ../sample-data
python -m validation.train --samples ../sample-data
# produce the submission (data bundle in ./data)
python -m possession.pipeline --team TercerHombre
```

Deterministic: the same inputs produce a byte-identical CSV.
