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
STREAM="output/streams/bpe_text8"
ROOT="output/calibration_text8_v16384_c32_m4"
PROBLEMS="$ROOT/problems_jobs8"
FITTED="$ROOT/fitted_w12"

[ -x "$PY" ] || {
  echo "ERROR: $PY is absent. Create the project environment first."
  exit 1
}
"$PY" -c 'import numpy, scipy, setuptools, tiktoken'

if [ ! -f "$STREAM/ids.npy" ] || [ ! -f "$STREAM/stream.json" ]; then
  [ -f data/text8 ] || {
    echo "ERROR: data/text8 is absent."
    exit 1
  }
  echo "=== build deterministic cl100k_base text8 stream ==="
  mkdir -p vocab_cache output/streams
  "$PY" scripts/make_stream.py \
    --representation bpe --file data/text8 --encoding cl100k_base \
    --vocab-dir vocab_cache --charge-vocabulary 776019 \
    --out "$STREAM"
fi

if [ ! -f tables/anchors_prod/anchors.json ]; then
  echo "=== build production anchor store with 12 workers ==="
  "$PY" scripts/build_anchor_store.py \
    --out tables/anchors_prod --factor 0.083071 --levels 2-53 \
    --r-max 200000000 --pad-anchors 8 --dense-below 256 \
    --targets-per-level 40 --jobs 12 --go
fi

mkdir -p "$ROOT"
"$PY" setup.py build_ext --inplace
"$PY" -c \
  'from product_model_with_memory import _graphical_margin_c; print("native extension: OK")'

echo "=== construct text8 V16384/C32 with 8 workers ==="
if [ ! -f "$PROBLEMS/results.json" ]; then
  "$PY" scripts/calibration_checkpoint_probe.py \
    --ids "$STREAM" --top-k 16383 --n 19429294 \
    --checkpoints 32 --first-checkpoint 262144 --interleave 1 \
    --margin-workers 1 --solver lbfgs --evaluator union \
    --initialization unigram --checkpoint-transfer copy \
    --jobs 8 --iterations 2000 --tolerance 0.0001 \
    --projection-tolerance 1e-10 --projection-iterations 100000 \
    --sparse-upstream --stream-checkpoints --uncompressed-states \
    --construct-only --resume-streamed --out "$PROBLEMS"
else
  echo "reusing $PROBLEMS"
fi

echo "=== calibrate with fixed 12-replica batch and 12 workers ==="
if [ ! -f "$FITTED/results.json" ]; then
  "$PY" scripts/calibration_fit_precomputed.py \
    --problems "$PROBLEMS" --out "$FITTED" \
    --steps 4000 --workers 12 --replicas 12 --edge-blocks 128 \
    --learning-rate 0.03 --minimum-learning-rate 0.003 \
    --exact-interval 50 --trust-radius 8.0 --tolerance 0.01
else
  echo "reusing $FITTED"
fi

echo "=== exact honest scoring with 15 workers ==="
if [ ! -f "$FITTED/scoring.json" ]; then
  "$PY" scripts/calibration_score_states.py \
    --run "$FITTED" --ids "$STREAM" --top-k 16383 --n 19429294 \
    --workers 15 --out "$FITTED/scoring.json"
else
  echo "reusing $FITTED/scoring.json"
fi

echo "completed: $ROOT"
