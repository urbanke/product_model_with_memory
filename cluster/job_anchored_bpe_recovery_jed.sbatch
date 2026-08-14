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
TASK="${SLURM_ARRAY_TASK_ID:?submit as --array=0-7}"
IFS=$'\t' read -r CORPUS N V M1 M2 MEMORY_BUDGET_GIB STREAM_NAME STREAM_HASH < <(
  "$PY" -c 'import json,sys; rows=json.load(open(sys.argv[1]))["campaigns"]; task=int(sys.argv[2]); matches=[r for r in rows if r["task"] == task]; assert len(matches) == 1, f"invalid/duplicate task {task}"; r=matches[0]; print(*(r[k] for k in ("corpus","n","v","m1","m2","memory_budget_gib","stream","stream_sha256")),sep="\t")' \
    "$REPOSITORY/cluster/anchored_bpe_recovery_campaigns.json" "$TASK"
)
STREAM="$SCRATCH/$STREAM_NAME"

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
"$PY" -c 'import json,sys; p=json.load(open(sys.argv[1])); actual=(int(p["n"]),int(p["vocabulary_size"])); expected=(int(sys.argv[2]),int(sys.argv[3])); assert actual == expected, f"stream (n,V)={actual}, campaign (n,V)={expected}"' "$STREAM/manifest.json" "$N" "$V"
(( M1 <= V && M2 <= V )) || { echo "ERROR: lag alphabet exceeds V"; exit 1; }

echo "host=$(hostname) commit=$actual_commit task=$SLURM_ARRAY_TASK_ID tag=$TAG"
echo "python=$($PY --version 2>&1) cpus=$SLURM_CPUS_PER_TASK memory=440G schedule_budget=${MEMORY_BUDGET_GIB}GiB"

if [[ ! -f "$PLAN" ]]; then
  "$PY" scripts/plan_anchored_potential_anchors.py \
    --stream "$STREAM/stream.npy" --minimum-prefix 2050 \
    --windows 8,16,32,64,128,256,512,1024,2048,4096 \
    --early-strata 8 --late-strata 24 --samples-per-stratum 2 \
    --seed 44117 --out "$PLAN"
fi
"$PY" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["production_eligible"] is True; assert p["stream_sha256"] == sys.argv[2]; assert int(p["n"]) == int(sys.argv[3]); t=p["tokenizer_provenance"]; assert (t["representation"],t["encoding"],t["complete_source"]) == ("bpe","cl100k_base",True)' "$PLAN" "$STREAM_HASH" "$N"

if [[ "$M1" == "$V" && "$M2" == "$V" ]]; then
  "$PY" scripts/make_anchored_potential_schedule.py \
    --plan "$PLAN" --root "$ROOT/run" --python "$PY" \
    --resource-profile scitas --maximum-private-memory-gib "$MEMORY_BUDGET_GIB" \
    --fitting-steps 500 --batch-size 64 \
    --exact-interval 25 --out "$SCHEDULE"
else
  "$PY" scripts/make_unequal_anchored_schedule.py \
    --plan "$PLAN" --root "$ROOT/run" --python "$PY" \
    --m1 "$M1" --m2 "$M2" --resource-profile scitas \
    --maximum-private-memory-gib "$MEMORY_BUDGET_GIB" \
    --fitting-steps 500 --batch-size 64 --exact-interval 25 \
    --out "$SCHEDULE"
fi

"$PY" -c 'import json,sys; p=json.load(open(sys.argv[1])); v,m1,m2=map(int,sys.argv[2:5]); budget=float(sys.argv[5]); assert p["production_eligible"] is True; assert p["tokenizer_provenance"]["encoding"] == "cl100k_base"; actual=(int(p.get("emission_vocabulary_size",v)),int(p.get("first_lag_parameter",v)),int(p.get("second_lag_parameter",v))); assert actual == (v,m1,m2), f"schedule (V,M1,M2)={actual}, requested={(v,m1,m2)}"; assert p["maximum_private_memory_bytes"] == int(budget*2**30), (p["maximum_private_memory_bytes"],budget)' "$SCHEDULE" "$V" "$M1" "$M2" "$MEMORY_BUDGET_GIB"

srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" "$PY" -u \
  scripts/run_fixed_checkpoint_schedule.py "$SCHEDULE" \
  --dependency-driven --working-directory "$REPOSITORY" \
  --event-log "$ROOT/events.jsonl"

test -f "$ROOT/run/potential.json"
"$PY" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["production_eligible"] is True; assert p["tokenizer_provenance"]["encoding"] == "cl100k_base"; assert len(p["anchors"]) == 64' "$ROOT/run/potential.json"
echo "production complete: $ROOT/run/potential.json"
