#!/usr/bin/env bash
set -euo pipefail

# Test only the two coordinate neighbors of a diagonal Markov-1 point.
# Usage: bash scripts/run_markov1_state_neighbors.sh CORPUS V V_PLUS
# Example: ... enwik9 65536 100277

if (( $# != 3 )); then
    echo "usage: $0 CORPUS V V_PLUS" >&2
    exit 2
fi
corpus=$1
v=$2
v_plus=$3
if (( v <= 1 || v_plus <= v )); then
    echo "require 1 < V < V_PLUS" >&2
    exit 2
fi
m_lower=$((v / 2))

project_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$project_root"
python_bin=${PMM_SWEEP_PYTHON:-.venv/bin/python3}
jobs=${PMM_SWEEP_JOBS:-8}
output_root=${PMM_SWEEP_ROOT:-output/markov1_state_neighbors_20260809}
cache_root=${PMM_SWEEP_CACHE_ROOT:-output/markov1_alphabet_sweep_20260809}
stream="output/streams/bpe_${corpus}"

export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PMM_UNIVERSAL_TABLES=${PMM_UNIVERSAL_TABLES:-tables/anchors_prod}
export PMM_PHI_LADDER_EVERY=${PMM_PHI_LADDER_EVERY:-1}
export PMM_PHI_LADDER_DEGREE=${PMM_PHI_LADDER_DEGREE:-11}
export PMM_PHI_SADDLE_MIN_L=${PMM_PHI_SADDLE_MIN_L:-54}

run_point() {
    local output_v=$1
    local m_grid=$2
    local point="$output_root/$corpus/v${output_v}"
    mkdir -p "$point"
    if [[ -f "$point/results.json" ]]; then
        echo "skip complete $point/results.json"
        return
    fi
    "$python_bin" -u scripts/state_family_experiment.py \
        --ids "$stream" \
        --top-k $((output_v - 1)) \
        --m-grid "$m_grid" \
        --state-order frequency \
        --jobs "$jobs" \
        --alphabet-grid-size 8 \
        --cache-dir "$cache_root/$corpus/v${output_v}/cache" \
        --out "$point" \
        2>&1 | tee "$point/run.log"
}

# (M,V) = (V/2,V), compared with the current diagonal point (V,V).
run_point "$v" "$m_lower,$v"
# (M,V) = (V,V_PLUS), compared with the next diagonal point.
run_point "$v_plus" "$v,$v_plus"
