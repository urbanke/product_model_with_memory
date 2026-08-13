#!/bin/bash
# Cheap end-to-end production gate before the eight-job BPE recovery wave.
# Eight text8 anchors, two short windows, and 100 fit steps exercise the exact
# production DAG, tokenizer provenance, scheduler, manifests, and aggregate.
#SBATCH --job-name=pmm-bpe-t8-smoke
#SBATCH --account=lthc
#SBATCH --partition=standard
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=440G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/urbanke/product_model_with_memory/logs/anchored-bpe-text8-smoke-%j.out

set -euo pipefail

REPOSITORY="/home/urbanke/Projects/product_model_with_memory"
SCRATCH="/scratch/urbanke/product_model_with_memory"
PY="$HOME/venvs/product-model-memory/bin/python"
STREAM="$SCRATCH/production_bpe_text8_v32768_stream"
ROOT="$SCRATCH/anchored_bpe_text8_smoke_v16384_seed44117"
EXPECTED_COMMIT="${PMM_EXPECTED_COMMIT:?submit with the audited commit}"
EXPECTED_STREAM="b8fa7bd6e88b3e2c215780a6456acf5ba8583e9f93880db770fa7df493e433d9"
EXPECTED_ANCHORS="5c409244487deeba8d597ad66fa13479a8d15476d55d4509c33b2ec797b07c68"
EXPECTED_TABLE_MANIFEST="cf0449a4b704174890f0a1fb95f20954839dd7266ae307e799d1dc6d0234ffb5"

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
echo "$EXPECTED_STREAM  $STREAM/stream.npy" | sha256sum -c -
echo "$EXPECTED_ANCHORS  $PMM_UNIVERSAL_TABLES/anchors.json" | sha256sum -c -
echo "$EXPECTED_TABLE_MANIFEST  $PMM_UNIVERSAL_TABLES/manifest.json" | sha256sum -c -
"$PY" -c 'from product_model_with_memory.production_coding import require_production_reduced_stream as r; import sys; p=r(sys.argv[1]); assert p["source_n_tokens"] == 19429294; print(p["representation"], p["encoding"], "complete", p["complete_source"])' "$STREAM/stream.npy"

echo "host=$(hostname) commit=$actual_commit python=$($PY --version 2>&1)"
echo "cpus=$SLURM_CPUS_PER_TASK memory=440G"

if [[ ! -f "$ROOT/anchors.json" ]]; then
  "$PY" scripts/plan_anchored_potential_anchors.py \
    --stream "$STREAM/stream.npy" --minimum-prefix 2050 \
    --windows 64,256 --early-strata 2 --late-strata 2 \
    --samples-per-stratum 2 --seed 44117 --out "$ROOT/anchors.json"
fi

"$PY" scripts/make_unequal_anchored_schedule.py \
  --plan "$ROOT/anchors.json" --root "$ROOT/run" --python "$PY" \
  --m1 16384 --m2 8192 --resource-profile scitas \
  --fitting-steps 100 --batch-size 64 --exact-interval 25 \
  --out "$ROOT/schedule.json"

srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" "$PY" -u \
  scripts/run_fixed_checkpoint_schedule.py "$ROOT/schedule.json" \
  --dependency-driven --working-directory "$REPOSITORY" \
  --event-log "$ROOT/events.jsonl"

test -f "$ROOT/run/potential.json"
"$PY" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["production_eligible"] is True; assert p["tokenizer_provenance"]["encoding"] == "cl100k_base"; assert len(p["anchors"]) == 8; print("SMOKE_OK", p["delta_bits_per_token"], p["analytic_ci95"])' "$ROOT/run/potential.json"
