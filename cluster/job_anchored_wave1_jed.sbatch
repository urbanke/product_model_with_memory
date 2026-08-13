#!/bin/bash
# Five independent 64-anchor production campaigns on SCITAS Jed.
# Submit as --array=0-4; every array element occupies one 72-core node.
#SBATCH --job-name=pmm-anchor-w1
#SBATCH --account=lthc
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=440G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/urbanke/product_model_with_memory/logs/anchored-wave1-%A_%a.out

set -euo pipefail

REPOSITORY="/home/urbanke/Projects/product_model_with_memory"
SCRATCH="/scratch/urbanke/product_model_with_memory"
PY="$HOME/venvs/product-model-memory/bin/python"
EXPECTED_COMMIT="${PMM_EXPECTED_COMMIT:?submit with the audited commit}"
EXPECTED_ANCHORS="5c409244487deeba8d597ad66fa13479a8d15476d55d4509c33b2ec797b07c68"
EXPECTED_TABLE_MANIFEST="cf0449a4b704174890f0a1fb95f20954839dd7266ae307e799d1dc6d0234ffb5"

case "${SLURM_ARRAY_TASK_ID:?submit as an array}" in
  0) CORPUS=enwik8; V=32768; M1=16384; M2=16384;
     STREAM="$SCRATCH/production_enwik8_v32768_stream";
     STREAM_HASH=2625ffc58442b10eef825536d72d3830be2350f466b9c6d3296f7077e69c3131 ;;
  1) CORPUS=enwik8; V=32768; M1=32768; M2=16384;
     STREAM="$SCRATCH/production_enwik8_v32768_stream";
     STREAM_HASH=2625ffc58442b10eef825536d72d3830be2350f466b9c6d3296f7077e69c3131 ;;
  2) CORPUS=enwik9; V=65536; M1=32768; M2=32768;
     STREAM="$SCRATCH/replay_enwik9_v65536_stream";
     STREAM_HASH=c87fd073b923052dbe9a379e69aea2ee90d63968091f91b78dcc4fdbf317ce9b ;;
  3) CORPUS=enwik9; V=65536; M1=65536; M2=32768;
     STREAM="$SCRATCH/replay_enwik9_v65536_stream";
     STREAM_HASH=c87fd073b923052dbe9a379e69aea2ee90d63968091f91b78dcc4fdbf317ce9b ;;
  4) CORPUS=enwik9; V=65536; M1=65536; M2=65536;
     STREAM="$SCRATCH/replay_enwik9_v65536_stream";
     STREAM_HASH=c87fd073b923052dbe9a379e69aea2ee90d63968091f91b78dcc4fdbf317ce9b ;;
  *) echo "invalid array index $SLURM_ARRAY_TASK_ID"; exit 2 ;;
esac

TAG="${CORPUS}_v${V}_m1_${M1}_m2_${M2}_gate64_w4096_seed44117"
ROOT="$SCRATCH/anchored_wave1/$TAG"
PLAN="$ROOT/anchors.json"
SCHEDULE="$ROOT/schedule.json"

cd "$REPOSITORY"
mkdir -p "$SCRATCH/logs" "$ROOT"
export PYTHONPATH="$REPOSITORY/src${PYTHONPATH:+:$PYTHONPATH}"
export PMM_UNIVERSAL_TABLES="$REPOSITORY/tables/anchors_prod"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ulimit -n 4096

actual_commit=$(git rev-parse HEAD)
[[ "$actual_commit" == "$EXPECTED_COMMIT" ]] || {
  echo "ERROR: expected commit $EXPECTED_COMMIT, found $actual_commit"; exit 1;
}
echo "$STREAM_HASH  $STREAM/stream.npy" | sha256sum -c -
echo "$EXPECTED_ANCHORS  $PMM_UNIVERSAL_TABLES/anchors.json" | sha256sum -c -
echo "$EXPECTED_TABLE_MANIFEST  $PMM_UNIVERSAL_TABLES/manifest.json" | sha256sum -c -
[[ $("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["vocabulary_size"])' "$STREAM/manifest.json") == "$V" ]]

echo "host=$(hostname) commit=$actual_commit task=$SLURM_ARRAY_TASK_ID tag=$TAG"
echo "python=$($PY --version 2>&1) cpus=$SLURM_CPUS_PER_TASK memory=440G"

if [[ ! -f "$PLAN" ]]; then
  "$PY" scripts/plan_anchored_potential_anchors.py \
    --stream "$STREAM/stream.npy" --minimum-prefix 2050 \
    --windows 8,16,32,64,128,256,512,1024,2048,4096 \
    --early-strata 8 --late-strata 24 --samples-per-stratum 2 \
    --seed 44117 --out "$PLAN"
fi

# Always regenerate the declarative schedule on submission.  Completion
# manifests, not a stale schedule JSON, are the restart authority.  This keeps
# measured resource contracts current while preserving every completed job.
if [[ "$M1" == "$V" && "$M2" == "$V" ]]; then
    "$PY" scripts/make_anchored_potential_schedule.py \
      --plan "$PLAN" --root "$ROOT/run" --python "$PY" \
      --resource-profile scitas --fitting-steps 500 --batch-size 64 \
      --exact-interval 25 --out "$SCHEDULE"
else
    "$PY" scripts/make_unequal_anchored_schedule.py \
      --plan "$PLAN" --root "$ROOT/run" --python "$PY" \
      --m1 "$M1" --m2 "$M2" --resource-profile scitas \
      --fitting-steps 500 --batch-size 64 --exact-interval 25 \
      --out "$SCHEDULE"
fi

srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" "$PY" -u \
  scripts/run_fixed_checkpoint_schedule.py "$SCHEDULE" \
  --dependency-driven --working-directory "$REPOSITORY" \
  --event-log "$ROOT/events.jsonl"

test -f "$ROOT/run/potential.json"
echo "production complete: $ROOT/run/potential.json"
