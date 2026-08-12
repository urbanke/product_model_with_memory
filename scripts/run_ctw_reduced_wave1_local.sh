#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY=".venv/bin/python3"
RUNNER="scripts/production_context_tree_experiment.py"
ROOT="output/ctw_reduced_wave1_20260810"
GRID_SIZE=16
JOBS="${CTW_JOBS:-10}"

run_one() {
  local corpus="$1"
  local v="$2"
  local depth="$3"
  local first_order="$4"
  local out="$ROOT/${corpus}_v${v}_d${depth}"
  local result="$out/results.json"

  if [[ -f "$result" ]]; then
    echo "SKIP completed $corpus V=$v D=$depth"
    return
  fi

  echo "START $corpus V=$v D=$depth jobs=$JOBS"
  "$PY" "$RUNNER" \
    --ids "output/streams/bpe_${corpus}" \
    --V "$v" \
    --depth "$depth" \
    --candidate-grid-size "$GRID_SIZE" \
    --first-order-results "$first_order" \
    --jobs "$JOBS" \
    --out "$out"
  echo "DONE  $corpus V=$v D=$depth"
}

# Most likely alphabets first within each corpus.  Candidates remain a single
# predeclared 16-member family regardless of execution order.
for depth in 2 3; do
  run_one text8 16384 "$depth" \
    output/markov1_state_sweep_20260809/text8/v16384/results.json
done
for v in 8192 32768; do
  for depth in 2 3; do
    run_one text8 "$v" "$depth" \
      "output/markov1_state_sweep_20260809/text8/v${v}/results.json"
  done
done

for depth in 2 3; do
  run_one enwik8 32768 "$depth" \
    output/markov1_alphabet_sweep_20260809/enwik8/v32768/results.json
done
for v in 16384 65536; do
  for depth in 2 3; do
    run_one enwik8 "$v" "$depth" \
      "output/markov1_alphabet_sweep_20260809/enwik8/v${v}/results.json"
  done
done

for v in 65536 32768; do
  for depth in 2 3; do
    run_one enwik9 "$v" "$depth" \
      "output/markov1_alphabet_sweep_20260809/enwik9/v${v}/results.json"
  done
done

echo "CTW reduced-alphabet Wave 1 complete"
