#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON_FALLBACK:-python}"
fi

"$PYTHON" "$ROOT/experiments/run_experiments.py" \
  split_validation decentralized_readout active_party_identity \
  validation_label_budget topology_frequency \
  --device "${DEVICE:-cuda}" \
  --seeds "${SEEDS:-42,43,44,45,46}" \
  --epochs "${EPOCHS:-300}" \
  --strict-cache --continue-on-error --no-plots

"$PYTHON" "$ROOT/experiments/run_experiments.py" \
  strict_label_training party_scalability \
  --device "${DEVICE:-cuda}" \
  --seeds "${SEEDS:-42,43,44,45,46}" \
  --epochs "${EPOCHS:-300}" \
  --continue-on-error --no-plots
