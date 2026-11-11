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
  crowded or fast) and ~73 % of dead-ball frames (long bursts, ball off camera). These two rates
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

\* not held out: v4 was fitted on these games. The baseline scores 0.283 on this bench against
0.2433 on BAR-ATH, so this bench is somewhat easier than the real task; it ranks versions, it does
not predict the final score.

**Touch detection** (`features.detect_touches`, after Link & Hoernig, *PLoS ONE* 2017, and
ELASTIC, Kim et al., 2025): a local peak of the ball's acceleration (≥ 10 m/s²) or a sharp change
of direction (≥ 25°) while a player is within 2 m. "Two touches in a row by one team" is the
observable proxy used for *controlled* possession.

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
