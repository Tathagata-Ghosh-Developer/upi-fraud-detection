#!/usr/bin/env bash
# Download the PaySim CSV from a public Hugging Face mirror (no account needed)
# and verify that it is byte-identical to the original Kaggle release.
set -euo pipefail

FRAUD_WORK="${FRAUD_WORK:-$HOME/fraud_work}"
DEST="$FRAUD_WORK/data/raw"
FILE="PS_20174392719_1491204439457_log.csv"
URL="https://huggingface.co/datasets/vitaliy-sharandin/synthetic-fraud-detection/resolve/main/$FILE"
SHA256="16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b"

mkdir -p "$DEST"
if [[ ! -f "$DEST/$FILE" ]]; then
  curl -L --fail -o "$DEST/$FILE" "$URL"
fi
echo "$SHA256  $DEST/$FILE" | sha256sum --check
