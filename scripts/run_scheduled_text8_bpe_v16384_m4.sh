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
ROOT="${PMM_ROOT:-output/scheduler_text8_bpe_v16384_full_c32_cold_1000}"
MAXIMUM_WORKERS="${PMM_MAXIMUM_WORKERS:-12}"
CONSTRUCTION_WORKERS="${PMM_CONSTRUCTION_WORKERS:-4}"
FITTING_WORKERS="${PMM_FITTING_WORKERS:-4}"
STEPS="${PMM_STEPS:-1000}"

"$PY" setup.py build_ext --inplace
"$PY" -c \
  'from product_model_with_memory import _graphical_margin_c; print("native extension: OK")'

mkdir -p "$ROOT"

if [ ! -f "$ROOT/jobs.json" ]; then
  "$PY" scripts/make_fixed_checkpoint_schedule.py \
    --root "$ROOT" \
    --ids output/streams/bpe_text8 \
    --top-k 16383 \
    --n 19429294 \
    --checkpoints 32 \
    --first-checkpoint 65536 \
    --policy pipeline \
    --python "$PY" \
    --maximum-workers "$MAXIMUM_WORKERS" \
    --construction-workers "$CONSTRUCTION_WORKERS" \
    --fitting-workers "$FITTING_WORKERS" \
    --fitting-replicas 12 \
    --evaluation-workers 1 \
    --fitting-steps "$STEPS" \
    --out "$ROOT/jobs.json"
fi

if [ ! -f "$ROOT/plan.json" ]; then
  "$PY" scripts/plan_analytic_checkpoint_schedule.py \
    --ids output/streams/bpe_text8 \
    --top-k 16383 \
    --n 19429294 \
    --checkpoints 32 \
    --first-checkpoint 65536 \
    --maximum-workers "$MAXIMUM_WORKERS" \
    --construction-maximum-workers "$CONSTRUCTION_WORKERS" \
    --fitting-maximum-workers "$FITTING_WORKERS" \
    --stochastic-steps "$STEPS" \
    --replicas 12 \
    --blocks 16 \
    --exact-interval 5 \
    --out "$ROOT/plan.json"
fi

exec caffeinate -i "$PY" -u scripts/run_analytic_checkpoint_schedule.py \
  --jobs "$ROOT/jobs.json" \
  --plan "$ROOT/plan.json" \
  --working-directory . \
  --maximum-workers "$MAXIMUM_WORKERS" \
  --event-log "$ROOT/scheduler_events.jsonl"
