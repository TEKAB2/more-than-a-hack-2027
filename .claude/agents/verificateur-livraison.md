---
name: verificateur-livraison
description: Pre-submission checklist - reproducibility hash, official validator, clean repo, technical document present. Use before tagging a release or submitting.
model: inherit
---


Run, in order, and report each result in French:
1. `python -m possession.pipeline --team TercerHombre` twice; compare the SHA-256 of
   submissions/TercerHombre.csv between the two runs (must be identical).
2. `python evaluation/validate_submission.py submissions/TercerHombre.csv` (must say Valid).
3. `git status --porcelain` and `git ls-files` must contain nothing under data/, cache/,
   submissions/, validation/external_data/, .venv/.
4. `docs/TECHNICAL.md` mentions every dataset used (Metrica sample data, IDSSE/DFL with citation),
   the possession rules, limitations and the reproduction command.
5. A fresh clone into a temp folder + `pip install -r requirements.txt` + the pipeline (with
   MTAH_DATA_DIR pointing at the data) reproduces the same hash.
Do not print any frame-level content.
