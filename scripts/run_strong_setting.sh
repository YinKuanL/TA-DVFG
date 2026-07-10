#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON_FALLBACK:-python}"
fi

RUNS_ROOT="$ROOT/results/strong_setting_runs"
OUTPUT_ROOT="$ROOT/results/strong_setting"

"$PYTHON" "$ROOT/experiments/run_experiments.py" strong_setting \
  --results-root "$RUNS_ROOT" \
  --device "${DEVICE:-cuda}" \
  --seeds "${SEEDS:-42,43,44,45,46}" \
  --epochs "${EPOCHS:-300}" \
  --continue-on-error \
  --no-plots \
  ${STRICT_CACHE:+--strict-cache} \
  ${FORCE:+--force}

"$PYTHON" "$ROOT/experiments/strong_setting_analysis.py" \
  --input-root "$RUNS_ROOT/strong_setting" \
  --output-dir "$OUTPUT_ROOT"
