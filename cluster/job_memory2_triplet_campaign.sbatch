#!/bin/bash
# One shared-batch half of the declared enwik9 memory-two triplet campaign.
# Submit once for V=65536 and once for V=100277.
#SBATCH --job-name=gm2grid
#SBATCH --mail-user=ruediger@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --partition=slurm-cluster
#SBATCH --output=output/slurm-memory2-triplets-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY="${PMM_PYTHON:-$HOME/anaconda3/bin/python3}"
PLAN="${PMM_PLAN:-output/memory2_triplet_campaign_20260809/plan.json}"
V="${PMM_V:?set PMM_V to 65536 or 100277}"
JOBS="${PMM_JOBS:-12}"
STREAM="${PMM_STREAM:-output/streams/bpe_enwik9}"
ROOT="${PMM_ROOT:-output/memory2_triplet_campaign_20260809}"

case "$V" in
  65536|100277) ;;
  *) echo "ERROR: declared campaign has only V=65536 and V=100277"; exit 2 ;;
esac
[ -f "$PLAN" ] || { echo "ERROR: missing $PLAN"; exit 2; }
[ -f "$STREAM/ids.npy" ] || { echo "ERROR: missing $STREAM/ids.npy"; exit 2; }

GRID="$($PY -c '
import json, sys
plan = json.load(open(sys.argv[1]))
print(",".join(plan["grids_by_vocabulary"][sys.argv[2]]))
' "$PLAN" "$V")"
OUT="$ROOT/v${V}"
CACHE="$ROOT/cache_v${V}"
mkdir -p "$OUT" "$CACHE"

echo "V=$V members=$(awk -F, '{print NF}' <<< "$GRID") jobs=$JOBS"
srun -c "${SLURM_CPUS_PER_TASK:?submit with --cpus-per-task}" \
  "$PY" -u scripts/product_family_experiment.py \
  --ids "$STREAM" \
  --top-k "$((V - 1))" \
  --grid "$GRID" \
  --state-order frequency \
  --jobs "$JOBS" \
  --cache-dir "$CACHE" \
  --out "$OUT"
