#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-42,43,44,45,46}"
EPOCHS="${EPOCHS:-300}"
STRICT_CACHE="${STRICT_CACHE:-1}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/Scripts/python.exe}"

ARGS=(
  "$REPO_ROOT/experiments/run_experiments.py"
  deployment_objective
  --results-root "$REPO_ROOT/results/deployment_objective_runs"
  --device "$DEVICE"
  --epochs "$EPOCHS"
  --seeds "$SEEDS"
  --no-plots
)

if [[ "$STRICT_CACHE" == "1" ]]; then ARGS+=(--strict-cache); fi
if [[ "$FORCE" == "1" ]]; then ARGS+=(--force); fi
if [[ "$DRY_RUN" == "1" ]]; then ARGS+=(--dry-run); fi

"$PYTHON" "${ARGS[@]}"

if [[ "$DRY_RUN" != "1" ]]; then
  "$PYTHON" "$REPO_ROOT/experiments/deployment_objective_analysis.py" \
    --input-root "$REPO_ROOT/results/deployment_objective_runs/deployment_objective" \
    --output-dir "$REPO_ROOT/results/deployment_objective"
fi
