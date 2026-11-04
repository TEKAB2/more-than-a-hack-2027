# TercerHombre — More than a Hack 2027, Phase 1: technical note

*Draft. Numbers marked TODO are filled in at freeze.*

## 1. Approach

Possession is an interval, and the ranking metric (spellF1) scores whole possessions, so the
method is built around one idea: **decide possession as a hidden state, not frame by frame.**

1. **Frame evidence.** From the tracking we compute two log-odds per frame: *dead vs alive* and
   *home vs away given alive*. They come from two logistic regressions trained on **other
   games** (§3), over windowed features of the ball and of the 22 players.
2. **Restarts.** Smart Tagging set pieces are treated as restarts: the ball is dead before the
   detected strike (a resting ball that starts moving; at the centre spot for a kick-off) and
   alive after it, with the tagged team in possession when the tag names one.
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

* **Metrica Sports public sample data** (github.com/metrica-sports/sample-data), three full
  matches with tracking and human event data, labelled by Metrica.
* Frame labels are derived from the events by a fixed rule (`validation/external.py`): an on-ball
  event (set piece, pass, carry, recovery, shot) gives possession to its team; a ball out, a shot
  out or in goal, or a foul starts a dead ball that lasts until the next on-ball event. Challenges
  and ball losses change nothing by themselves.
* The clean tracking is **degraded to resemble the ATD**: ball missing on 41 % of frames in bursts,
  more often when the ball is crowded or fast; low-frequency positional error; player track gaps.
* Models are fitted and the decoder costs chosen by **leave-one-game-out** cross-validation on
  spellF1 (`validation/train.py`). Held-out spellF1: 0.527. The official nearest-player baseline
  (±2 s window) scores ≈ 0.25 on the same protocol (0.2433 on BAR-ATH), which suggests the protocol is a fair
  proxy for difficulty.

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
