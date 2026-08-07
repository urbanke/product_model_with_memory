#!/bin/bash
#SBATCH --job-name=gme8-v4096-shared
#SBATCH --mail-user=ruediger@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --time=4-00:00:00
#SBATCH --mem=220G
#SBATCH --nodes=1
#SBATCH -c 64
#SBATCH --partition=slurm-cluster
#SBATCH -w node14.clusterbc.epfl.ch
#SBATCH --output=output/calibration_enwik8_v4096_server/slurm-shared-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ROOT=output/calibration_enwik8_v4096_server
STREAM=output/streams/bpe_enwik8
PROBLEMS="$ROOT/problems_c32_jobs64"
STORE="$ROOT/shared_graph_c32"
FITTED="$ROOT/fitted_shared_c32_w12"
mkdir -p "$ROOT"

PY=""
for cand in "$(command -v python3 2>/dev/null || true)" \
            "$HOME/anaconda3/envs/pmm/bin/python" \
            "$HOME/miniforge3/envs/pmm/bin/python" \
            "$HOME/anaconda3/bin/python" \
            /usr/bin/python3; do
  if [ -x "$cand" ] && "$cand" -c \
      'import sys; assert sys.version_info >= (3,11); import numpy, scipy, setuptools' \
      2>/dev/null; then
    PY="$cand"
    break
  fi
done
[ -n "$PY" ] || {
  echo "ERROR: no existing Python >=3.11 with numpy/scipy/setuptools was found."
  exit 1
}
echo "using python: $PY"
"$PY" -c 'import sys, numpy, scipy; print(sys.version); print(numpy.__version__, scipy.__version__)'

"$PY" setup.py build_ext --inplace
"$PY" -c 'from product_model_with_memory import _graphical_margin_c; print("native extension: OK")'

[ -f "$PROBLEMS/results.json" ] || {
  echo "ERROR: completed construction is absent: $PROBLEMS/results.json"
  exit 1
}
[ -f "$PROBLEMS/states/checkpoint_031.npz" ] || {
  echo "ERROR: final checkpoint is absent"
  exit 1
}

echo "=== build shared YA- and AB-major graphs ==="
if [ ! -f "$STORE/manifest.json" ] || \
   [ ! -f "$STORE/ab_graph/manifest.json" ]; then
  srun -c 64 "$PY" -u scripts/build_layered_intersection_store.py \
    --problems "$PROBLEMS" --out "$STORE" --ab-major
else
  echo "reusing $STORE"
fi

echo "=== fit all checkpoints with direct sampled graph traversal ==="
if [ ! -f "$FITTED/summary.json" ]; then
  srun -c 64 "$PY" -u scripts/fit_shared_graph_checkpoints.py \
    --store "$STORE" --problems "$PROBLEMS" --out "$FITTED" \
    --workers 12 --steps 4000 --tolerance 1e-2 \
    --exact-interval 50 --blocks 128 --cache 16 --stop 32
else
  echo "reusing $FITTED/summary.json"
fi

echo "=== exact honest scoring with 31 workers ==="
if [ ! -f "$FITTED/scoring.json" ]; then
  srun -c 64 "$PY" -u scripts/calibration_score_states.py \
    --run "$FITTED" --ids "$STREAM" --top-k 4095 --n 25793085 \
    --workers 31 --out "$FITTED/scoring.json"
else
  echo "reusing $FITTED/scoring.json"
fi

echo "completed: $FITTED"
