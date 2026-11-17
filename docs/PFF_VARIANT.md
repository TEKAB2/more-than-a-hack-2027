# Experimental PFF variant (out of submission)

**Status: EXPERIMENTAL / NOT THE DEFAULT ENTRY.**  
Branch purpose: measure the held-out spellF1 gap between `clean-ensemble` (no PFF) and the
same GBDT + MS-TCN 102ch protocol trained with **PFF FC FIFA World Cup 2022** as extra
sequence-model data.

* Default submission on `clean-ensemble` stays **without PFF** until organisers explicitly
  authorise use of that dataset.
* Do **not** replace `submissions/TercerHombre.csv`, `possession/weights.json`, or
  `possession/weights_seq.npz` with PFF artefacts.
* **PFF licence: pending organiser clarification.** Cite and submit a PFF-trained model only
  after a positive answer.

## What changes vs clean-ensemble

| piece | clean-ensemble (submission) | this variant |
|---|---|---|
| Trees (GBDT `duel_past`) | 3 Metrica + 7 IDSSE | unchanged (PFF rows are **not** fed to trees in `train_seq`) |
| MS-TCN 102ch | 3 Metrica + 7 IDSSE | same + `pff_*.npz` via `--use-pff` |
| Decode | simple Viterbi (`change_alpha=0`) | same |
| Simulator | clean-trees external-only typed stoppages | same |
| Written weights | `weights.json` + `weights_seq.npz` | only if requested: `weights_pff.json` + `weights_seq_pff.npz` |

## Protocol (must match clean-ensemble LOO)

```bash
# Baseline (no PFF) — same flags as docs/TECHNICAL.md LOO measurement
python -m validation.train_seq --samples "$MTAH_SAMPLES" --carrier \
  --train-seeds 2 --eval-seeds 2 --model-seeds 1 --steps 3000 --batch-size 4 \
  --cache-dir cache/seq_features_cleantrees_102 --reuse-checkpoints

# PFF variant (MS-TCN also sees degraded PFF WC2022 matches)
python -m validation.train_seq --samples "$MTAH_SAMPLES" --carrier \
  --train-seeds 2 --eval-seeds 2 --model-seeds 1 --steps 3000 --batch-size 4 \
  --cache-dir cache/seq_features_cleantrees_102_pff \
  --use-pff --pff-seeds 1 --pff-ratio 0.30

# Optional artefacts (never overwrites submission defaults)
python -m validation.train_seq ... --use-pff --write-weights --weights-tag pff
```

Or one wrapper (checks env + data, runs both legs, tees logs under `cache/pff_gap_logs/`):

```bash
bash validation/measure_pff_gap.sh
bash validation/measure_pff_gap.sh --write-pff-weights
PFF_MAX_GAMES=8 bash validation/measure_pff_gap.sh   # smoke with fewer WC matches
```

Report from the **GLOBAL HELD-OUT SUMMARY** block: mean spellF1 (10 games), Metrica-only,
DFL-only, short recall. Adopt nothing into the default pipeline without organiser yes **and**
a meaningful held-out gain.

## Local data setup (required — not present in this cloud agent)

```bash
export MTAH_SAMPLES=/path/to/metrica-sports/sample-data
export MTAH_EXTERNAL_DIR=/path/to/external_npz   # or default: validation/external_data

git clone https://github.com/metrica-sports/sample-data "$MTAH_SAMPLES"
pip install -r requirements.txt -r validation/requirements.txt
pip install torch   # training only

python -m validation.fetch_idsse --out "$MTAH_EXTERNAL_DIR"
python -m validation.fetch_pff --download --out "$MTAH_EXTERNAL_DIR"
# expect: 7× idsse_*.npz and up to 64× pff_*.npz
```

Env vars used by the bench:

| variable | role |
|---|---|
| `MTAH_SAMPLES` | Metrica sample-data clone (required) |
| `MTAH_EXTERNAL_DIR` | folder with `idsse_*.npz` and `pff_*.npz` |
| `MTAH_PFF_RAW_DIR` | optional local raw PFF tree instead of `--download` |
| `MTAH_DATA_DIR` | provided match (pipeline only; **not** used for this measurement) |

## Guardrails in code

* `--write-weights` with `--use-pff` / `--pretrain-steps` **requires** `--weights-tag`.
* Tagged writes go to `possession/weights_<tag>.json` and `possession/weights_seq_<tag>.npz`.
* Default `python -m possession.pipeline --team TercerHombre` still loads `weights.json` (no PFF).

## What Joy Boy should ask the organisers

1. Explicit yes/no: may a Phase-1 entry train on **PFF FC open FIFA World Cup 2022** tracking/events
   (other matches only), citing them in the technical note?
2. If yes: any citation / attribution wording they require beyond the public Drive / kloppy refs.
3. If no or silence: keep submitting `clean-ensemble` without PFF (this branch stays measurement-only).

## Cloud agent note (this PR)

This environment had **no** `MTAH_SAMPLES`, **no** `MTAH_EXTERNAL_DIR`, and no `pff_*.npz` /
`idsse_*.npz` on disk. Numbers were **not** invented: re-run `measure_pff_gap.sh` locally
(or on a machine that already holds the external packs) to fill the before/after table.
