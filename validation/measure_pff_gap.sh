#!/usr/bin/env bash
# Experimental / out-of-submission: measure clean-ensemble LOO spellF1 with vs without PFF.
#
# Does NOT touch possession/weights.json or possession/weights_seq.npz (submission defaults).
# Optional --write-pff-weights writes possession/weights_pff.json + weights_seq_pff.npz only.
#
# Prerequisites (local machine with training stack):
#   export MTAH_SAMPLES=/path/to/metrica-sports/sample-data
#   export MTAH_EXTERNAL_DIR=/path/to/external_npz   # idsse_*.npz + pff_*.npz
#   pip install -r requirements.txt -r validation/requirements.txt
#   pip install torch   # training only; inference stays numpy
#   python -m validation.fetch_idsse --out "$MTAH_EXTERNAL_DIR"
#   python -m validation.fetch_pff --download --out "$MTAH_EXTERNAL_DIR"
#
# Usage:
#   bash validation/measure_pff_gap.sh
#   bash validation/measure_pff_gap.sh --write-pff-weights
#   bash validation/measure_pff_gap.sh --skip-baseline   # only the --use-pff leg
#   PFF_MAX_GAMES=8 bash validation/measure_pff_gap.sh  # smoke with fewer WC matches
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WRITE_PFF_WEIGHTS=0
SKIP_BASELINE=0
SKIP_PFF=0
for arg in "$@"; do
  case "$arg" in
    --write-pff-weights) WRITE_PFF_WEIGHTS=1 ;;
    --skip-baseline) SKIP_BASELINE=1 ;;
    --skip-pff) SKIP_PFF=1 ;;
    -h|--help)
      sed -n '1,25p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

: "${MTAH_SAMPLES:?Set MTAH_SAMPLES to a clone of metrica-sports/sample-data}"
EXT="${MTAH_EXTERNAL_DIR:-$ROOT/validation/external_data}"
export MTAH_EXTERNAL_DIR="$EXT"

if [[ ! -d "$MTAH_SAMPLES" ]]; then
  echo "MTAH_SAMPLES is not a directory: $MTAH_SAMPLES" >&2
  exit 1
fi
if [[ ! -d "$EXT" ]]; then
  echo "MTAH_EXTERNAL_DIR missing: $EXT" >&2
  echo "Fetch IDSSE then PFF into that folder first (see docs/PFF_VARIANT.md)." >&2
  exit 1
fi

n_idsse="$(find "$EXT" -maxdepth 1 -name 'idsse_*.npz' | wc -l | tr -d ' ')"
n_pff="$(find "$EXT" -maxdepth 1 -name 'pff_*.npz' | wc -l | tr -d ' ')"
echo "External dir: $EXT  (idsse=$n_idsse  pff=$n_pff)"
if [[ "$n_idsse" -lt 7 ]]; then
  echo "Need 7 idsse_*.npz under $EXT (found $n_idsse). Run: python -m validation.fetch_idsse --out \"$EXT\"" >&2
  exit 1
fi
if [[ "$SKIP_PFF" -eq 0 && "$n_pff" -lt 1 ]]; then
  echo "Need at least one pff_*.npz under $EXT for the PFF leg. Run: python -m validation.fetch_pff --download" >&2
  exit 1
fi

LOG_DIR="${PFF_GAP_LOG_DIR:-$ROOT/cache/pff_gap_logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CACHE_BASE="${PFF_GAP_CACHE:-$ROOT/cache/seq_features_cleantrees_102}"
PFF_MAX_ARGS=()
if [[ -n "${PFF_MAX_GAMES:-}" ]]; then
  PFF_MAX_ARGS=(--pff-max-games "$PFF_MAX_GAMES")
fi

# Same protocol as docs/TECHNICAL.md on clean-ensemble (LOO measurement; no --write-weights on baseline).
COMMON=(
  --samples "$MTAH_SAMPLES"
  --carrier
  --train-seeds 2
  --eval-seeds 2
  --model-seeds 1
  --steps 3000
  --batch-size 4
  --cache-dir "$CACHE_BASE"
)

run_leg() {
  local label="$1"
  shift
  local log="$LOG_DIR/${STAMP}_${label}.log"
  echo ""
  echo "========== $label =========="
  echo "log -> $log"
  # shellcheck disable=SC2068
  python -m validation.train_seq ${COMMON[@]} "$@" 2>&1 | tee "$log"
  echo "Finished $label"
}

summarize_log() {
  local log="$1"
  echo "---- summary lines from $(basename "$log") ----"
  grep -E 'GLOBAL HELD-OUT SUMMARY|Ensemble GBDT v7 \+ MS-TCN|spellF1' "$log" | tail -n 40 || true
}

echo "EXPERIMENTAL PFF gap measurement — default submission remains without PFF."
echo "Licence status: pending organiser clarification (do not submit a PFF-trained entry until authorised)."

if [[ "$SKIP_BASELINE" -eq 0 ]]; then
  run_leg "baseline_no_pff" --reuse-checkpoints
  summarize_log "$LOG_DIR/${STAMP}_baseline_no_pff.log"
fi

if [[ "$SKIP_PFF" -eq 0 ]]; then
  PFF_EXTRA=(--use-pff --pff-seeds 1 --pff-ratio 0.30)
  if [[ ${#PFF_MAX_ARGS[@]} -gt 0 ]]; then
    PFF_EXTRA+=("${PFF_MAX_ARGS[@]}")
  fi
  if [[ "$WRITE_PFF_WEIGHTS" -eq 1 ]]; then
    PFF_EXTRA+=(--write-weights --weights-tag pff)
  fi
  # Separate cache so PFF fold checkpoints never collide with no-PFF ones.
  PFF_CACHE="${CACHE_BASE}_pff"
  log="$LOG_DIR/${STAMP}_with_pff.log"
  echo ""
  echo "========== with_pff =========="
  echo "log -> $log"
  python -m validation.train_seq \
    --samples "$MTAH_SAMPLES" \
    --carrier \
    --train-seeds 2 \
    --eval-seeds 2 \
    --model-seeds 1 \
    --steps 3000 \
    --batch-size 4 \
    --cache-dir "$PFF_CACHE" \
    "${PFF_EXTRA[@]}" \
    2>&1 | tee "$log"
  summarize_log "$log"
fi

echo ""
echo "Done. Compare GLOBAL HELD-OUT SUMMARY lines: spellF1 (10 games), Metrica, short recall."
echo "Do not adopt PFF into the default pipeline / TercerHombre.csv without organiser yes."
echo "Logs: $LOG_DIR"
