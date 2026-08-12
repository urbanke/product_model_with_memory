#!/usr/bin/env bash
set -euo pipefail

# Exact checkpoint-free Markov-1 controls for the two declared unequal
# anchored experiments. Run sequentially and only when no memory-heavy
# anchored campaign is active. These use the same corpus-frequency retained
# state subsets as anchored_state_maps.py and charge them enumeratively.

project_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$project_root"
python_bin=${PMM_PYTHON:-.venv/bin/python3}
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PMM_UNIVERSAL_TABLES=${PMM_UNIVERSAL_TABLES:-tables/anchors_prod}
export PMM_PHI_LADDER_EVERY=1
export PMM_PHI_LADDER_DEGREE=11
export PMM_PHI_SADDLE_MIN_L=54

run_one() {
    local v=$1 m=$2 out=$3
    if [[ -f "$out/results.json" ]]; then
        echo "reuse exact control V=$v M=$m: $out/results.json"
        return
    fi
    mkdir -p "$out"
    "$python_bin" -u scripts/state_family_experiment.py \
        --ids output/streams/ours_text8 \
        --top-k "$((v-1))" \
        --m-grid "$m" \
        --state-order frequency \
        --jobs 13 \
        --alphabet-grid-size 1 \
        --out "$out"
}

run_one 16384 16384 output/markov1_ours_text8_v16384_m16384_exact_for_anchored
run_one 32768 16384 output/markov1_ours_text8_v32768_m16384_exact_for_anchored
