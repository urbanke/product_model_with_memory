#!/bin/bash
# Real eight-anchor enwik9 replay on node 14.  This validates the compute-node
# Python environment, production estimator, reduction, complete unequal DAG,
# cross-anchor utilization, and memory profile before the 64-anchor campaign.
#SBATCH --job-name=pmm-e9-replay
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=220G
#SBATCH --time=04:00:00
#SBATCH --partition=slurm-cluster
#SBATCH --nodelist=node14.clusterbc.epfl.ch
#SBATCH --output=output/anchored-enwik9-replay-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
ulimit -n 4096

PY="${PMM_PYTHON:-$HOME/anaconda3/bin/python3}"
SOURCE="${PMM_SOURCE:-output/streams/ours_enwik9}"
ROOT="${PMM_ROOT:-output/anchored_enwik9_replay_node14_v65536_m32768_20260812}"
REDUCED="${PMM_REDUCED:-output/replay_enwik9_v65536_stream}"

echo "host=$(hostname) commit=$(git rev-parse HEAD) python=$PY"
echo "cpus=$SLURM_CPUS_PER_TASK memory=220G"

# Validate imports and scheduler invariants in the environment that will run
# the campaign, rather than on the login node.
srun --ntasks=1 --cpus-per-task=1 "$PY" -m pytest -q \
  tests/test_checkpoint_scheduler.py \
  tests/test_anchored_potential_schedule.py \
  tests/test_unequal_anchored_schedule.py

mkdir -p "$ROOT"
[ -f "$REDUCED/stream.npy" ] || {
  echo "ERROR: transferred reduced stream is absent: $REDUCED/stream.npy"
  exit 1
}
[ -f "$REDUCED/manifest.json" ] || {
  echo "ERROR: transferred reduced manifest is absent: $REDUCED/manifest.json"
  exit 1
}

if [ ! -f "$ROOT/anchors.json" ]; then
  "$PY" scripts/plan_anchored_potential_anchors.py \
    --stream "$REDUCED/stream.npy" --minimum-prefix 2050 \
    --windows 64,256 --early-strata 2 --late-strata 2 \
    --samples-per-stratum 2 --seed 44117 --out "$ROOT/anchors.json"
fi

if [ ! -f "$ROOT/schedule.json" ]; then
  "$PY" scripts/make_unequal_anchored_schedule.py \
    --plan "$ROOT/anchors.json" --root "$ROOT/run" \
    --m1 32768 --m2 32768 --resource-profile cpu64 \
    --fitting-steps 100 --exact-interval 25 --out "$ROOT/schedule.json"
fi

srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" "$PY" -u \
  scripts/run_fixed_checkpoint_schedule.py "$ROOT/schedule.json" \
  --dependency-driven --working-directory . \
  --event-log "$ROOT/events.jsonl"

echo "replay complete: $ROOT/run/potential.json"
