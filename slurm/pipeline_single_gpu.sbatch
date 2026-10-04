#!/bin/bash
# Full pipeline on one GPU node. Submit from the repository root on a login node:
#   export FRAUD_WORK=<fast shared storage>/fraud_work   # holds .venv and data/raw
#   sbatch slurm/pipeline_single_gpu.sbatch
# sbatch passes the submitting shell's environment to the job.
# Set FRAUD_RESULTS as well to keep the committed results/ untouched.
# Replace the <placeholders> with your site's values first.
#SBATCH --job-name=upi_fraud_pipeline
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# Large files go to scratch, not to the home or project directory.
: "${FRAUD_WORK:?export FRAUD_WORK (fast shared storage) before sbatch}"
# module load cuda   # only if your site requires it; the xgboost wheel bundles CUDA libraries

source "$FRAUD_WORK/.venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"

nvidia-smi
# Compute nodes often have no internet access: run scripts/download_data.sh on
# a login node beforehand. Here it finds the file and only verifies the checksum,
# so a missing or corrupt CSV stops the job instead of being silently used.
CSV="$FRAUD_WORK/data/raw/PS_20174392719_1491204439457_log.csv"
if [[ ! -f "$CSV" ]]; then
  echo "missing $CSV: run scripts/download_data.sh on a login node first" >&2
  exit 1
fi
bash scripts/download_data.sh
# --engine auto uses cuDF when it is installed in this environment, pandas otherwise.
# Arguments after the engine go to the tuning stage.
bash scripts/run_all.sh --engine auto --gpus-per-trial 0.5 --cpus-per-trial 4
