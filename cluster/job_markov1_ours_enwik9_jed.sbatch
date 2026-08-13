#!/bin/bash
# Exact checkpoint-free first-order baseline matching the full ours_enwik9
# anchored campaigns at V=M=65,536.
#SBATCH --job-name=pmm-e9-m1
#SBATCH --account=lthc
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=440G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/urbanke/product_model_with_memory/logs/markov1-ours-enwik9-%j.out

set -euo pipefail

REPOSITORY="/home/urbanke/Projects/product_model_with_memory"
SCRATCH="/scratch/urbanke/product_model_with_memory"
PY="$HOME/venvs/product-model-memory/bin/python"
EXPECTED_COMMIT="${PMM_EXPECTED_COMMIT:?submit with the audited commit}"
STREAM="$SCRATCH/streams/ours_enwik9"
OUT="$SCRATCH/markov1_ours_enwik9_full434000723_v65536"
EXPECTED_IDS="dd8d971f8682e96b18922f49f4e14d129963cb46302df2df07710858cb88ef88"
EXPECTED_META="4b86f7503d941073dbd38c3979ad17c6762a444b1f62c7d0afc4c043f964197d"
EXPECTED_ANCHORS="5c409244487deeba8d597ad66fa13479a8d15476d55d4509c33b2ec797b07c68"
EXPECTED_TABLE_MANIFEST="cf0449a4b704174890f0a1fb95f20954839dd7266ae307e799d1dc6d0234ffb5"

cd "$REPOSITORY"
mkdir -p "$SCRATCH/logs" "$OUT"
export PYTHONPATH="$REPOSITORY/src${PYTHONPATH:+:$PYTHONPATH}"
export PMM_UNIVERSAL_TABLES="$REPOSITORY/tables/anchors_prod"
export PMM_PHI_LADDER_EVERY=1 PMM_PHI_LADDER_DEGREE=11 PMM_PHI_SADDLE_MIN_L=54
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

actual_commit=$(git rev-parse HEAD)
[[ "$actual_commit" == "$EXPECTED_COMMIT" ]] || {
  echo "ERROR: expected commit $EXPECTED_COMMIT, found $actual_commit"; exit 1;
}
echo "$EXPECTED_IDS  $STREAM/ids.npy" | sha256sum -c -
echo "$EXPECTED_META  $STREAM/stream.json" | sha256sum -c -
echo "$EXPECTED_ANCHORS  $PMM_UNIVERSAL_TABLES/anchors.json" | sha256sum -c -
echo "$EXPECTED_TABLE_MANIFEST  $PMM_UNIVERSAL_TABLES/manifest.json" | sha256sum -c -

echo "host=$(hostname) commit=$actual_commit python=$($PY --version 2>&1)"
echo "cpus=$SLURM_CPUS_PER_TASK memory=440G stream=ours_enwik9 full_tokens=434000723"

if [[ ! -f "$OUT/results.json" ]]; then
  srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    "$PY" -u scripts/state_family_experiment.py \
      --ids "$STREAM" --top-k 65535 --m-grid 65536 --state-order id \
      --jobs "$SLURM_CPUS_PER_TASK" --alphabet-grid-size 8 --out "$OUT"
fi

test -f "$OUT/results.json"
"$PY" - <<'PY'
import json
from pathlib import Path
p = json.loads(Path("/scratch/urbanke/product_model_with_memory/markov1_ours_enwik9_full434000723_v65536/results.json").read_text())
a = p["honest_accounting"]
assert a["sequence_estimator"] == "layered_depth_averaged_product_simplex_v1"
assert int(p["n_tokens"]) == 434000723
print("honest_bpc", a["honest_member_bits_per_character"]["65536"])
PY
