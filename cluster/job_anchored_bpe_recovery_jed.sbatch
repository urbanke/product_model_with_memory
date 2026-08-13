#!/bin/bash
# Eight independent Section-7 recovery campaigns on complete cl100k_base BPE.
# Submit as --array=0-7. Every element occupies one 72-core Jed node.
# The production Python gates independently reject any non-BPE input or plan.
#SBATCH --job-name=pmm-bpe-recovery
#SBATCH --account=lthc
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=440G
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/urbanke/product_model_with_memory/logs/anchored-bpe-recovery-%A_%a.out

set -euo pipefail

REPOSITORY="/home/urbanke/Projects/product_model_with_memory"
SCRATCH="/scratch/urbanke/product_model_with_memory"
PY="$HOME/venvs/product-model-memory/bin/python"
EXPECTED_COMMIT="${PMM_EXPECTED_COMMIT:?submit with the audited commit}"
EXPECTED_ANCHORS="5c409244487deeba8d597ad66fa13479a8d15476d55d4509c33b2ec797b07c68"
EXPECTED_TABLE_MANIFEST="cf0449a4b704174890f0a1fb95f20954839dd7266ae307e799d1dc6d0234ffb5"

case "${SLURM_ARRAY_TASK_ID:?submit as --array=0-7}" in
  0) CORPUS=text8;  V=16384; M1=16384; M2=16384;
     STREAM="$SCRATCH/production_bpe_text8_v16384_stream";
     STREAM_HASH=0ae5d896371d652ff05cfe37a13a972c00702dc3af34fc52f6c470f09ab6cade ;;
  1) CORPUS=text8;  V=16384; M1=16384; M2=8192;
     STREAM="$SCRATCH/production_bpe_text8_v16384_stream";
     STREAM_HASH=0ae5d896371d652ff05cfe37a13a972c00702dc3af34fc52f6c470f09ab6cade ;;
  2) CORPUS=text8;  V=32768; M1=16384; M2=16384;
     STREAM="$SCRATCH/production_bpe_text8_v32768_stream";
     STREAM_HASH=b8fa7bd6e88b3e2c215780a6456acf5ba8583e9f93880db770fa7df493e433d9 ;;
  3) CORPUS=enwik8; V=32768; M1=16384; M2=16384;
     STREAM="$SCRATCH/production_bpe_enwik8_v32768_stream";
     STREAM_HASH=7317d05ed3e8016ae3e28001899e08e0c1916c76da34b7bfd7aa0ebc6706ff43 ;;
  4) CORPUS=enwik8; V=32768; M1=32768; M2=16384;
     STREAM="$SCRATCH/production_bpe_enwik8_v32768_stream";
     STREAM_HASH=7317d05ed3e8016ae3e28001899e08e0c1916c76da34b7bfd7aa0ebc6706ff43 ;;
  5) CORPUS=enwik9; V=65536; M1=32768; M2=32768;
     STREAM="$SCRATCH/production_bpe_enwik9_v65536_stream";
     STREAM_HASH=3d97af5bb80994c9de4858c9a2adbdbde9a9c6e8cdcd8035d0bcfc1d01e748dc ;;
  6) CORPUS=enwik9; V=65536; M1=65536; M2=32768;
     STREAM="$SCRATCH/production_bpe_enwik9_v65536_stream";
     STREAM_HASH=3d97af5bb80994c9de4858c9a2adbdbde9a9c6e8cdcd8035d0bcfc1d01e748dc ;;
  7) CORPUS=enwik9; V=65536; M1=65536; M2=65536;
     STREAM="$SCRATCH/production_bpe_enwik9_v65536_stream";
     STREAM_HASH=3d97af5bb80994c9de4858c9a2adbdbde9a9c6e8cdcd8035d0bcfc1d01e748dc ;;
  *) echo "invalid array index $SLURM_ARRAY_TASK_ID"; exit 2 ;;
esac

TAG="bpe_${CORPUS}_v${V}_m1_${M1}_m2_${M2}_gate64_w4096_seed44117"
ROOT="$SCRATCH/anchored_bpe_recovery/$TAG"
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
"$PY" -c 'from product_model_with_memory.production_coding import require_production_reduced_stream as r; import sys; r(sys.argv[1])' "$STREAM/stream.npy"
"$PY" -c 'import json,sys; p=json.load(open(sys.argv[1])); actual=int(p["vocabulary_size"]); expected=int(sys.argv[2]); assert actual == expected, f"stream V={actual}, campaign V={expected}"' "$STREAM/manifest.json" "$V"

echo "host=$(hostname) commit=$actual_commit task=$SLURM_ARRAY_TASK_ID tag=$TAG"
echo "python=$($PY --version 2>&1) cpus=$SLURM_CPUS_PER_TASK memory=440G"

if [[ ! -f "$PLAN" ]]; then
  "$PY" scripts/plan_anchored_potential_anchors.py \
    --stream "$STREAM/stream.npy" --minimum-prefix 2050 \
    --windows 8,16,32,64,128,256,512,1024,2048,4096 \
    --early-strata 8 --late-strata 24 --samples-per-stratum 2 \
    --seed 44117 --out "$PLAN"
fi

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
