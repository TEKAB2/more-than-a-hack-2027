# Handoff — state on Tuesday 17 November 2026 (morning, Québec)

**Deadline: Tuesday 17 November 2026, 17:59 America/New_York (= 17:59 Québec).**  
**Research freeze: 14:00 (same timezone).** The submission form is editable; the last response counts.

## 1. Rules clarification (organisers’ email — overrides older notes)

Forbidden:
- checking our outputs against the video (**do not use `tools/viewer.html` for verification**);
- human labels or human-produced data on the provided match;
- sending provided data to an AI service (**no agent reads `data/` or `cache/`**);
- training, tuning or calibrating on the provided ground truth or video.

Required: all training and parameter choices use **external data only**.  
Allowed: automatic checks on the provided data; any dataset we have rights to (verify the licence).

## 2. Version to submit: branch `clean-ensemble`

- **Model:** GBDT (`duel_past`, possession-weighted) + MS-TCN 102ch ensemble, simple Viterbi. **No PFF.**
- **Training:** Metrica (3 games) + IDSSE Bundesliga (7 games, CC BY 4.0). External-only simulator (typed stoppages).
- **Files to declare:** `submissions/TercerHombre.csv`, `TercerHombre.md`, branch URL.
- **Repository URL:** `https://github.com/TEKAB2/more-than-a-hack-2027/tree/clean-ensemble`.

Do **not** declare `clean-trees` — that freeze is superseded by this ensemble.

## 3. Numbers — never confuse the two columns

| version | held-out bench (10 other games) | public 5 min on the provided match (17 spells, very noisy) |
|---|---|---|
| official baseline | ~0.26–0.28 | 0.243 on the full match |
| clean-trees (trees alone, older freeze) | ~0.674 | ~0.594 |
| **clean-ensemble (GBDT + MS-TCN 102ch, this entry)** | **~0.782** | **~0.640** |

The public 5-minute score is a **pipeline sanity check only** — never use it to choose between versions or tune a parameter.

## 4. Closed tracks — do not reopen

| track | decision |
|---|---|
| SkillCorner Open Data | measured as extra training; **hurt** Metrica held-out (−0.033); rejected |
| PFF | licence unclear; **out of submission** until organisers say yes |
| Fiche E (broadcast ball) | measured gain too small |
| Fiche C (train ball detector on BAR–ATH) | **forbidden**; PR #9 is off the submission path |
| Viewer to verify outputs | **forbidden** since the clarification |
| Audio whistle (local only) | controls **PASS** (83 peals, hit 66.3 % ≥ 60 %); aggregate effect too small (+2 stoppages, dead 43.7 %→44.0 %, possessions 291→292). **Not adopted by default** — optional `--whistle` only if ever committed. Code still local (Joy Boy); **do not commit** without agreement. Recommendation: ship `clean-ensemble` **without** whistle. |

## 5. Checklist before submitting the form

1. Branch URL points at **`clean-ensemble`** (not `clean-trees`).
2. Run the pipeline twice → identical SHA-256 of `submissions/TercerHombre.csv`.
3. `python evaluation/validate_submission.py submissions/TercerHombre.csv` → Valid.
4. `git status`: nothing under `data/`, `cache/`, `submissions/`, `validation/external_data/`, `.venv/`.
5. `TercerHombre.md` matches the submitted model and cites every external dataset used.
6. Inference is numpy-only (no network, no pickle); no constant fitted on the provided match.
7. After 14:00 freeze: confirm whistle stays non-default unless a local surprise changes that call.

## 6. What remains for Joy Boy

- Fill / update the organisers’ **submission form** (CSV + technical note + branch URL).
- Resubmit if the form link arrives late — last response wins.
- Optional local whistle stays off the default path; no commit of `possession/whistle.py` without an explicit go-ahead.
