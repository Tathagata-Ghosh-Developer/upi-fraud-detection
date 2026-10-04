#!/bin/bash
# Larger Ray Tune search across several GPU nodes (one trial per GPU).
# Assumes the data stages have already run (slurm/pipeline_single_gpu.sbatch)
# and that the repository and $FRAUD_WORK are on a filesystem every node can see.
#   sbatch slurm/tune_multi_gpu.sbatch
# Replace the <placeholders> with your site's values first.
#SBATCH --job-name=upi-fraud-tune
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

export FRAUD_WORK="${FRAUD_WORK:-/scratch/$USER/fraud_work}"
source "$FRAUD_WORK/.venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"

GPUS_PER_NODE=4
PORT=6379
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD="${NODES[0]}"
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD" hostname --ip-address | awk '{print $1}')

# Ray head on the first node, one Ray worker on each remaining node.
# Recent Ray versions use token authentication; a shared $HOME makes the
# token in ~/.ray visible to every node (see the Ray security docs otherwise).
srun --nodes=1 --ntasks=1 --gpus-per-node=$GPUS_PER_NODE -w "$HEAD" \
  ray start --head --node-ip-address="$HEAD_IP" --port=$PORT \
  --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus=$GPUS_PER_NODE --block &
sleep 30

for node in "${NODES[@]:1}"; do
  srun --nodes=1 --ntasks=1 --gpus-per-node=$GPUS_PER_NODE -w "$node" \
    ray start --address="$HEAD_IP:$PORT" \
    --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus=$GPUS_PER_NODE --block &
done
sleep 30

# 8 GPUs -> 8 concurrent trials. ASHA keeps the cost of a large search close
# to that of a few full-length trials.
python -m src.tune --ray-address "$HEAD_IP:$PORT" \
  --num-samples 256 --gpus-per-trial 1 --cpus-per-trial 8 --time-budget-s 7200
python -m src.evaluate
python -m src.export_runs

ray stop || true
