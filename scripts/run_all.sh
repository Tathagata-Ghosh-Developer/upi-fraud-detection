#!/usr/bin/env bash
# Run the whole pipeline end to end. Usage (from the repository root):
#   FRAUD_WORK=/path/to/fast/disk bash scripts/run_all.sh [--engine pandas|cudf|auto] [extra tune args]
# Expects the PaySim CSV at $FRAUD_WORK/data/raw/ (see README "Get the data").
set -euo pipefail

ENGINE="pandas"
if [[ "${1:-}" == "--engine" ]]; then
  ENGINE="$2"
  shift 2
fi

export PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."

python -m pytest -q -p no:cacheprovider tests
python -m src.ingest --engine "$ENGINE"
python -m src.features --engine "$ENGINE"
python -m src.split
python -m src.baselines
python -m src.train
python -m src.tune --feature-set primary "$@"
python -m src.tune --feature-set strict "$@"
python -m src.evaluate
python -m src.export_runs
