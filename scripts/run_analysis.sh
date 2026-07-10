#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON_FALLBACK:-python}"
fi

"$PYTHON" "$ROOT/experiments/aggregate_results.py" --results-root "$RESULTS_ROOT"
"$PYTHON" "$ROOT/experiments/statistical_tests.py" \
  --input "$RESULTS_ROOT/combined/raw_results.csv" \
  --output "$RESULTS_ROOT/combined/paired_significance.csv"
"$PYTHON" "$ROOT/experiments/make_ablation_plots.py" \
  --input "$RESULTS_ROOT/combined/summary_results.csv" \
  --output-dir "$RESULTS_ROOT/combined/figures"
