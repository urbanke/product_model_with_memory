#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY="${PMM_PYTHON:-.venv/bin/python3}"
ROOT=output/calibration_text8_v16384_c32_m4
PROBLEMS="$ROOT/problems_jobs8"
STORE="$ROOT/shared_graph_c32"
FITTED="$ROOT/fitted_shared_w12"
STREAM=output/streams/bpe_text8

[ -x "$PY" ] || {
  echo "ERROR: $PY is absent"
  exit 1
}
[ -f "$PROBLEMS/results.json" ] || {
  echo "ERROR: completed V16384 construction is absent"
  exit 1
}
[ -f "$PROBLEMS/states/checkpoint_031.npz" ] || {
  echo "ERROR: final V16384 checkpoint is absent"
  exit 1
}

"$PY" setup.py build_ext --inplace
"$PY" -c \
  'from product_model_with_memory import _graphical_margin_c; print("native extension: OK")'

echo "=== build/reuse shared YA- and AB-major graphs ==="
if [ ! -f "$STORE/manifest.json" ] || \
   [ ! -f "$STORE/ab_graph/manifest.json" ]; then
  "$PY" -u scripts/build_layered_intersection_store.py \
    --problems "$PROBLEMS" --out "$STORE" --ab-major
else
  echo "reusing $STORE"
fi

echo "=== fit V16384/C32 with direct sampled graph traversal ==="
if [ ! -f "$FITTED/summary.json" ]; then
  "$PY" -u scripts/fit_shared_graph_checkpoints.py \
    --store "$STORE" --problems "$PROBLEMS" --out "$FITTED" \
    --workers 12 --max-stochastic-steps 20000 --tolerance 1e-2 \
    --exact-interval 50 --blocks 128 --cache 16 --stop 32
else
  echo "reusing $FITTED/summary.json"
fi

echo "=== exact honest scoring with 15 workers ==="
if [ ! -f "$FITTED/scoring.json" ]; then
  "$PY" -u scripts/calibration_score_states.py \
    --run "$FITTED" --ids "$STREAM" --top-k 16383 --n 19429294 \
    --workers 15 --out "$FITTED/scoring.json"
else
  echo "reusing $FITTED/scoring.json"
fi

echo "completed: $FITTED"
