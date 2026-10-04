#!/bin/bash
# Larger Ray Tune search across several GPU nodes (one trial per GPU).
# Assumes the data stages have already run (slurm/pipeline_single_gpu.sbatch)
# and that the repository and $FRAUD_WORK are on a filesystem every node can see.
# From the repository root on a login node:
#   export FRAUD_WORK=<fast shared storage>/fraud_work
#   sbatch slurm/tune_multi_gpu.sbatch
#   sbatch --nodes=4 slurm/tune_multi_gpu.sbatch --num-samples 128 --time-budget-s 1200
# Arguments after the script name go to src.tune and override the defaults below.
# Ray (2.59) enables token authentication by default and keeps the token in
# ~/.ray/auth_token; to keep it out of $HOME, export RAY_AUTH_MODE=token and
# RAY_AUTH_TOKEN_PATH=<token file on storage every node can read> before sbatch.
# Replace the <placeholders> with your site's values first.
#SBATCH --job-name=upi_fraud_tune
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

: "${FRAUD_WORK:?export FRAUD_WORK (fast shared storage) before sbatch}"
source "$FRAUD_WORK/.venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"

# Take the GPU count from SLURM: a hard-coded count larger than the real one
# makes Ray hand trials GPU ids that do not exist on the node.
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE##*:}}"
CPUS_PER_NODE="$SLURM_CPUS_PER_TASK"
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
TOTAL_GPUS=$(( GPUS_PER_NODE * ${#NODES[@]} ))
HEAD="${NODES[0]}"
# The batch script runs on the first node; use the address the other nodes resolve.
HEAD_IP=$(getent ahostsv4 "$HEAD" | awk 'NR == 1 {print $1}')
# A job-specific port: another Ray cluster on a shared node may already hold 6379.
PORT=$(( 20000 + SLURM_JOB_ID % 10000 ))
ADDR="$HEAD_IP:$PORT"
echo "Ray head $HEAD ($ADDR), ${#NODES[@]} node(s), $TOTAL_GPUS GPU(s)"

# Recent SLURM versions do not pass --cpus-per-task from sbatch on to srun, so
# set it explicitly; otherwise every Ray node is pinned to a single core.
RAY_STEP=(srun --nodes=1 --ntasks=1 --cpus-per-task="$CPUS_PER_NODE" --gres="gpu:$GPUS_PER_NODE")
PIDS=()

stop_ray() {
  # Stop only this job's Ray steps. `ray stop` would kill every Ray process of
  # the same user on the node, including those of other jobs.
  ((${#PIDS[@]})) || return 0
  kill "${PIDS[@]}" 2>/dev/null || true
  for _ in $(seq 30); do
    kill -0 "${PIDS[@]}" 2>/dev/null || break
    sleep 1
  done
  kill -9 "${PIDS[@]}" 2>/dev/null || true
  PIDS=()
}
trap stop_ray EXIT

wait_for_gpus() {
  # Poll the cluster until it reports the expected GPU count (fixed sleeps race).
  python - "$ADDR" "$1" <<'EOF'
import sys, time
import ray

addr, want = sys.argv[1], float(sys.argv[2])
deadline = time.time() + 300
while time.time() < deadline:
    try:
        ray.init(address=addr, log_to_driver=False)
        have = ray.cluster_resources().get("GPU", 0)
        ray.shutdown()
        if have >= want:
            print(f"Ray cluster ready: {have:g} GPUs")
            sys.exit(0)
    except Exception:
        pass
    time.sleep(5)
sys.exit(f"Ray cluster at {addr} did not reach {want:g} GPUs in 300 s")
EOF
}

# Ray head on the first node, one Ray worker on each remaining node.
"${RAY_STEP[@]}" -w "$HEAD" ray start --head --node-ip-address="$HEAD_IP" --port="$PORT" \
  --num-cpus="$CPUS_PER_NODE" --num-gpus="$GPUS_PER_NODE" --include-dashboard=false --block &
PIDS+=($!)
sleep 10
wait_for_gpus "$GPUS_PER_NODE"

for node in "${NODES[@]:1}"; do
  "${RAY_STEP[@]}" -w "$node" ray start --address="$ADDR" \
    --num-cpus="$CPUS_PER_NODE" --num-gpus="$GPUS_PER_NODE" --block &
  PIDS+=($!)
done
wait_for_gpus "$TOTAL_GPUS"

# One trial per GPU. ASHA keeps the cost of a large search close to that of a
# few full-length trials.
python -m src.tune --ray-address "$ADDR" --feature-set primary \
  --num-samples 256 --gpus-per-trial 1 --cpus-per-trial 8 --time-budget-s 3000 "$@"
stop_ray

python -m src.evaluate
python -m src.export_runs
