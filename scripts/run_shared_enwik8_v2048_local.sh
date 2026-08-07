#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

PY="${PMM_PYTHON:-.venv/bin/python3}"
ROOT=output/calibration_enwik8_v2048_c32_local
STREAM=output/streams/bpe_enwik8
PROBLEMS="$ROOT/problems_jobs4"
STORE="$ROOT/shared_graph_c32"
FITTED="$ROOT/fitted_shared_w6"
mkdir -p "$ROOT"

"$PY" setup.py build_ext --inplace
"$PY" -c 'from product_model_with_memory import _graphical_margin_c; print("native extension: OK")'

echo "=== construct enwik8 V2048/C32 with 4 workers ==="
if [ ! -f "$PROBLEMS/results.json" ]; then
  "$PY" -u scripts/calibration_checkpoint_probe.py \
    --ids "$STREAM" --top-k 2047 --n 25793085 \
    --checkpoints 32 --first-checkpoint 32768 --interleave 1 \
    --margin-workers 1 --solver lbfgs --evaluator union \
    --initialization unigram --checkpoint-transfer copy \
    --jobs 4 --iterations 2000 --tolerance 0.0001 \
    --projection-tolerance 1e-10 --projection-iterations 100000 \
    --sparse-upstream --stream-checkpoints --uncompressed-states \
    --construct-only --resume-streamed --out "$PROBLEMS"
else
  echo "reusing $PROBLEMS/results.json"
fi

echo "=== build/reuse shared YA- and AB-major graphs ==="
if [ ! -f "$STORE/manifest.json" ] || [ ! -f "$STORE/ab_graph/manifest.json" ]; then
  "$PY" -u scripts/build_layered_intersection_store.py \
    --problems "$PROBLEMS" --out "$STORE" --ab-major
else
  echo "reusing $STORE"
fi

echo "=== relaxed stochastic fit with 6 workers / 12 replicas ==="
if [ ! -f "$FITTED/summary.json" ]; then
  "$PY" -u scripts/fit_shared_graph_checkpoints.py \
    --store "$STORE" --problems "$PROBLEMS" --out "$FITTED" \
    --workers 6 --replicas 12 --max-stochastic-steps 20000 \
    --relaxed --slack-precision 1 --stationarity-tolerance 1e-4 \
    --tolerance 1e-2 --exact-interval 5 --blocks 128 --cache 16 \
    --persistent-reference-positions \
    --progress-interval 100 --stop 32
else
  echo "reusing $FITTED/summary.json"
fi

echo "=== honest scoring ==="
if [ ! -f "$FITTED/scoring.json" ]; then
  "$PY" -u scripts/calibration_score_states.py \
    --run "$FITTED" --ids "$STREAM" --top-k 2047 --n 25793085 \
    --workers 8 --out "$FITTED/scoring.json"
else
  echo "reusing $FITTED/scoring.json"
fi

echo "completed: $FITTED"
