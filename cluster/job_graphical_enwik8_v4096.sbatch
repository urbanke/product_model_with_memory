#!/bin/bash
#SBATCH --job-name=gme8-v4096
#SBATCH --mail-user=ruediger@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --time=4-00:00:00
#SBATCH --mem=220G
#SBATCH --nodes=1
#SBATCH -c 64
#SBATCH --partition=slurm-cluster
#SBATCH -w node14.clusterbc.epfl.ch
#SBATCH --output=output/calibration_enwik8_v4096_server/slurm-%j.out

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
FITTED="$ROOT/fitted_c32_w12"
mkdir -p "$ROOT"

PY=""
for cand in "$(command -v python3 2>/dev/null || true)" \
            "$HOME/anaconda3/envs/pmm/bin/python" \
            "$HOME/miniforge3/envs/pmm/bin/python" \
            /usr/bin/python3; do
  if [ -x "$cand" ] && "$cand" -c \
      'import sys; assert sys.version_info >= (3,11); import numpy, scipy' \
      2>/dev/null; then
    PY="$cand"
    break
  fi
done
[ -n "$PY" ] || {
  echo "ERROR: no existing Python >=3.11 with numpy/scipy was found."
  exit 1
}
echo "using python: $PY"
"$PY" -c 'import sys, numpy, scipy; print(sys.version); print(numpy.__version__, scipy.__version__)'

# Build only the native extension; do not install or change packages.
"$PY" setup.py build_ext --inplace
"$PY" -c 'from product_model_with_memory import _graphical_margin_c; print("native extension: OK")'

if [ ! -f "$STREAM/ids.npy" ] || [ ! -f "$STREAM/stream.json" ]; then
  echo "ERROR: $STREAM is absent. Copy the prepared stream from the Mac."
  exit 1
fi

echo "=== construct V4096/C32 with 64 workers ==="
if [ ! -f "$PROBLEMS/results.json" ]; then
  srun -c 64 "$PY" scripts/calibration_checkpoint_probe.py \
    --ids "$STREAM" --top-k 4095 --n 25793085 \
    --checkpoints 32 --first-checkpoint 65536 --interleave 1 \
    --margin-workers 1 --solver lbfgs --evaluator union \
    --initialization unigram --checkpoint-transfer copy \
    --jobs 64 --iterations 2000 --tolerance 0.0001 \
    --sparse-upstream --stream-checkpoints --uncompressed-states \
    --construct-only --out "$PROBLEMS"
else
  echo "reusing $PROBLEMS"
fi

echo "=== calibrate with matched 12-replica batch and 12 workers ==="
if [ ! -f "$FITTED/results.json" ]; then
  srun -c 64 "$PY" scripts/calibration_fit_precomputed.py \
    --problems "$PROBLEMS" --out "$FITTED" \
    --steps 4000 --workers 12 --replicas 12 --edge-blocks 128 \
    --learning-rate 0.03 --minimum-learning-rate 0.003 \
    --exact-interval 50 --trust-radius 8.0 --tolerance 0.01
else
  echo "reusing $FITTED"
fi

echo "=== exact honest scoring with 31 workers ==="
srun -c 64 "$PY" scripts/calibration_score_states.py \
  --run "$FITTED" --ids "$STREAM" --top-k 4095 --n 25793085 \
  --workers 31 --out "$FITTED/scoring.json"

echo "completed: $ROOT"
