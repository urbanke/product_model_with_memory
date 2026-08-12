#!/bin/bash
# Eight-anchor enwik9 utilization and portability replay on SCITAS Jed.
# Production campaigns must use the same dependency-driven DAG after this
# replay validates CPU occupancy, RSS, external artifacts, and Python runtime.
#SBATCH --job-name=pmm-e9-jed-replay
#SBATCH --account=lthc
#SBATCH --partition=standard
#SBATCH --qos=parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=440G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/urbanke/product_model_with_memory/logs/anchored-enwik9-jed-replay-%j.out

set -euo pipefail

REPOSITORY="/home/urbanke/Projects/product_model_with_memory"
SCRATCH="/scratch/urbanke/product_model_with_memory"
PY="$HOME/venvs/product-model-memory/bin/python"
REDUCED="$SCRATCH/replay_enwik9_v65536_stream"
ROOT="$SCRATCH/anchored_enwik9_replay_jed_v65536_m32768_20260812"
EXPECTED_COMMIT="${PMM_EXPECTED_COMMIT:?submit with PMM_EXPECTED_COMMIT set to the audited Git commit}"
EXPECTED_STREAM="c87fd073b923052dbe9a379e69aea2ee90d63968091f91b78dcc4fdbf317ce9b"
EXPECTED_ANCHORS="5c409244487deeba8d597ad66fa13479a8d15476d55d4509c33b2ec797b07c68"
EXPECTED_TABLE_MANIFEST="cf0449a4b704174890f0a1fb95f20954839dd7266ae307e799d1dc6d0234ffb5"

cd "$REPOSITORY"
mkdir -p "$SCRATCH/logs" "$ROOT"

export PYTHONPATH="$REPOSITORY/src${PYTHONPATH:+:$PYTHONPATH}"
export PMM_UNIVERSAL_TABLES="$REPOSITORY/tables/anchors_prod"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
ulimit -n 4096

actual_commit=$(git rev-parse HEAD)
[[ "$actual_commit" == "$EXPECTED_COMMIT" ]] || {
  echo "ERROR: expected commit $EXPECTED_COMMIT, found $actual_commit"; exit 1;
}
echo "$EXPECTED_STREAM  $REDUCED/stream.npy" | sha256sum -c -
echo "$EXPECTED_ANCHORS  $PMM_UNIVERSAL_TABLES/anchors.json" | sha256sum -c -
echo "$EXPECTED_TABLE_MANIFEST  $PMM_UNIVERSAL_TABLES/manifest.json" | sha256sum -c -

echo "host=$(hostname) commit=$actual_commit python=$($PY --version 2>&1)"
echo "cpus=$SLURM_CPUS_PER_TASK memory=440G root=$ROOT"

srun --ntasks=1 --cpus-per-task=1 "$PY" -m pytest -q \
  tests/test_production_coding.py \
  tests/test_checkpoint_scheduler.py \
  tests/test_anchored_potential_schedule.py \
  tests/test_unequal_anchored_schedule.py

if [[ ! -f "$ROOT/anchors.json" ]]; then
  "$PY" scripts/plan_anchored_potential_anchors.py \
    --stream "$REDUCED/stream.npy" --minimum-prefix 2050 \
    --windows 64,256 --early-strata 2 --late-strata 2 \
    --samples-per-stratum 2 --seed 44117 --out "$ROOT/anchors.json"
fi

if [[ ! -f "$ROOT/schedule.json" ]]; then
  "$PY" scripts/make_unequal_anchored_schedule.py \
    --plan "$ROOT/anchors.json" --root "$ROOT/run" \
    --m1 32768 --m2 32768 --resource-profile scitas \
    --python "$PY" --fitting-steps 100 --exact-interval 25 \
    --out "$ROOT/schedule.json"
fi

srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" "$PY" -u \
  scripts/run_fixed_checkpoint_schedule.py "$ROOT/schedule.json" \
  --dependency-driven --working-directory "$REPOSITORY" \
  --event-log "$ROOT/events.jsonl"

test -f "$ROOT/run/potential.json"
echo "replay complete: $ROOT/run/potential.json"
