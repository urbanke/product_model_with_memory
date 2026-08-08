#!/bin/bash
# Generic scheduled full-file BPE experiment; configure with PMM_* exports.
#SBATCH --job-name=gmbpe
#SBATCH --mail-user=ruediger@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --partition=slurm-cluster
#SBATCH --output=output/slurm-bpe-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Late checkpoints map all graph deltas.  Make the required descriptor budget
# explicit so behavior does not depend on the login shell or Slurm default.
ulimit -n 4096
echo "open-file limit: $(ulimit -n)"

PY="${PMM_PYTHON:-$HOME/anaconda3/bin/python3}"
CORPUS="${PMM_CORPUS:?set PMM_CORPUS, for example enwik9}"
STREAM="${PMM_STREAM:-output/streams/bpe_${CORPUS}}"
TOP_K="${PMM_TOP_K:?set PMM_TOP_K to V-1}"
VOCABULARY_SIZE=$((TOP_K + 1))
STEPS="${PMM_STEPS:-1000}"
CPUS="${SLURM_CPUS_PER_TASK:?submit with --cpus-per-task}"
MAXIMUM_WORKERS="${PMM_MAXIMUM_WORKERS:-$CPUS}"
CONSTRUCTION_WORKERS="${PMM_CONSTRUCTION_WORKERS:-4}"
FITTING_WORKERS="${PMM_FITTING_WORKERS:-4}"
ROOT="${PMM_ROOT:-output/scheduler_${CORPUS}_bpe_v${VOCABULARY_SIZE}_full_c32_cold_${STEPS}}"

[ -f "$STREAM/ids.npy" ] || {
  echo "ERROR: $STREAM/ids.npy is absent on the server"
  exit 1
}
N="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["n_tokens"])' "$STREAM/stream.json")"

"$PY" -c 'import sys, numpy, scipy; print(sys.version); print(numpy.__version__, scipy.__version__)'
"$PY" -c \
  'from product_model_with_memory import _graphical_margin_c; print("native extension: OK")'

mkdir -p "$ROOT"

# Prepare this once before analytic planning.  Both construction and every
# evaluation task mmap the compact result; none rereads/reduces enwik9.
if [ ! -f "$ROOT/reduced_stream/manifest.json" ]; then
  "$PY" -u scripts/prepare_reduced_checkpoint_stream.py \
    --ids "$STREAM" \
    --top-k "$TOP_K" \
    --n "$N" \
    --checkpoints 32 \
    --first-checkpoint 65536 \
    --out "$ROOT/reduced_stream"
fi

if [ ! -f "$ROOT/jobs.json" ]; then
  "$PY" scripts/make_fixed_checkpoint_schedule.py \
    --root "$ROOT" \
    --ids "$STREAM" \
    --top-k "$TOP_K" \
    --n "$N" \
    --checkpoints 32 \
    --first-checkpoint 65536 \
    --policy pipeline \
    --python "$PY" \
    --maximum-workers "$MAXIMUM_WORKERS" \
    --construction-workers "$CONSTRUCTION_WORKERS" \
    --fitting-workers "$FITTING_WORKERS" \
    --fitting-replicas 12 \
    --fitting-blocks 128 \
    --fitting-block-cache 128 \
    --fitting-exact-interval 50 \
    --evaluation-workers 1 \
    --fitting-steps "$STEPS" \
    --out "$ROOT/jobs.json"
fi

if [ ! -f "$ROOT/plan.json" ]; then
  "$PY" scripts/plan_analytic_checkpoint_schedule.py \
    --ids "$STREAM" \
    --reduced-stream "$ROOT/reduced_stream" \
    --top-k "$TOP_K" \
    --n "$N" \
    --checkpoints 32 \
    --first-checkpoint 65536 \
    --maximum-workers "$MAXIMUM_WORKERS" \
    --construction-maximum-workers "$CONSTRUCTION_WORKERS" \
    --fitting-maximum-workers "$FITTING_WORKERS" \
    --stochastic-steps "$STEPS" \
    --replicas 12 \
    --blocks 128 \
    --exact-interval 50 \
    --out "$ROOT/plan.json"
fi

srun -c "$CPUS" "$PY" -u scripts/run_analytic_checkpoint_schedule.py \
  --jobs "$ROOT/jobs.json" \
  --plan "$ROOT/plan.json" \
  --working-directory . \
  --maximum-workers "$MAXIMUM_WORKERS" \
  --event-log "$ROOT/scheduler_events.jsonl"

srun -c 1 "$PY" -u scripts/aggregate_scheduled_scores.py \
  --root "$ROOT" \
  --ids "$STREAM" \
  --top-k "$TOP_K" \
  --n "$N" \
  --out "$ROOT/accounting.json"
