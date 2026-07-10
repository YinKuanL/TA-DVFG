#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON_FALLBACK:-python}"
fi

"$PYTHON" "$ROOT/experiments/run_experiments.py" active_party_protocol \
  --device "${DEVICE:-cuda}" \
  --seeds "${SEEDS:-42,43,44,45,46}" \
  --epochs "${EPOCHS:-300}" \
  --strict-cache \
  --continue-on-error \
  --no-plots \
  "$@"
