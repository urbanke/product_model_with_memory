#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY=".venv/bin/python3"
RUNNER="scripts/production_context_tree_experiment.py"
ROOT="output/ctw_common_m_wave1_20260811"

# The honest selected model is chosen from the union of the 16 completed
# (V,D) candidates and these 12 new (V,M,D) candidates.
GLOBAL_GRID_SIZE=28
JOBS="${CTW_JOBS:-10}"

run_one() {
  local corpus="$1"
  local v="$2"
  local m="$3"
  local depth="$4"
  local out="$ROOT/${corpus}_v${v}_m${m}_d${depth}"

  if [[ -f "$out/results.json" ]]; then
    echo "SKIP completed $corpus V=$v M=$m D=$depth"
    return
  fi

  echo "START $corpus V=$v M=$m D=$depth jobs=$JOBS"
  "$PY" "$RUNNER" \
    --ids "output/streams/bpe_${corpus}" \
    --V "$v" \
    --M "$m" \
    --depth "$depth" \
    --candidate-grid-size "$GLOBAL_GRID_SIZE" \
    --jobs "$JOBS" \
    --out "$out"
  echo "DONE  $corpus V=$v M=$m D=$depth"
}

# Hold V at the completed Wave-1 winner for each corpus and test a common
# context resolution at V/2 and V/4.  M=V is already measured at both depths.
for spec in "text8 16384" "enwik8 32768" "enwik9 65536"; do
  read -r corpus v <<< "$spec"
  for m in "$((v / 2))" "$((v / 4))"; do
    for depth in 2 3; do
      run_one "$corpus" "$v" "$m" "$depth"
    done
  done
done

echo "CTW common-M Wave 1 complete"
