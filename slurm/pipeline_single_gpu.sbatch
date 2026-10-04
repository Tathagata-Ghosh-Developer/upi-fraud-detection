#!/bin/bash
# Full pipeline on one GPU node. Submit from the repository root:
#   sbatch slurm/pipeline_single_gpu.sbatch
# Replace the <placeholders> with your site's values first.
#SBATCH --job-name=upi-fraud-pipeline
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# Large files go to scratch, not to the home or project directory.
export FRAUD_WORK="${FRAUD_WORK:-/scratch/$USER/fraud_work}"
# module load cuda   # only if your site requires it; the xgboost wheel bundles CUDA libraries

source "$FRAUD_WORK/.venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"

nvidia-smi
# Compute nodes often have no internet access: run scripts/download_data.sh
# on a login node beforehand (it is idempotent and verifies the checksum).
bash scripts/download_data.sh || echo "download skipped; expecting the CSV in $FRAUD_WORK/data/raw"
# --engine auto uses cuDF when it is installed in this environment, pandas otherwise.
# Arguments after the engine go to the tuning stage.
bash scripts/run_all.sh --engine auto --gpus-per-trial 0.5 --cpus-per-trial 4
