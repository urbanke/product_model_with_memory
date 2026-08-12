#!/usr/bin/env bash
set -euo pipefail

# Honest checkpoint-free Markov-1 sweep over both output alphabet V and
# previous-token state resolution M.  The retained output alphabet and the
# frequency-selected M-state subset are both explicitly described.
#
# Environment overrides mirror run_markov1_alphabet_sweep.sh.  In addition:
#   PMM_STATE_GRID=0,16,64,256,1024,4096,16384,65536
#   PMM_SWEEP_ROOT=output/markov1_state_sweep_20260809
#   PMM_SWEEP_CACHE_ROOT=output/markov1_alphabet_sweep_20260809

project_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$project_root"

python_bin=${PMM_SWEEP_PYTHON:-.venv/bin/python3}
output_root=${PMM_SWEEP_ROOT:-output/markov1_state_sweep_20260809}
corpora_csv=${PMM_SWEEP_CORPORA:-text8,enwik8,enwik9}
grid_csv=${PMM_SWEEP_GRID:-1024,2048,4096,8192,16384,32768,65536,100277}
state_grid_csv=${PMM_STATE_GRID:-0,16,64,256,1024,4096,16384,65536}
jobs=${PMM_SWEEP_JOBS:-8}
cache_root=${PMM_SWEEP_CACHE_ROOT:-$output_root}

IFS=',' read -r -a corpora <<< "$corpora_csv"
IFS=',' read -r -a grid <<< "$grid_csv"
IFS=',' read -r -a state_grid <<< "$state_grid_csv"
grid_size=${PMM_SWEEP_DECLARED_GRID_SIZE:-${#grid[@]}}
if (( grid_size < ${#grid[@]} )); then
    echo "declared V-grid size cannot be smaller than requested sweep" >&2
    exit 1
fi

mkdir -p "$output_root"

export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PMM_UNIVERSAL_TABLES=${PMM_UNIVERSAL_TABLES:-tables/anchors_prod}
export PMM_PHI_LADDER_EVERY=${PMM_PHI_LADDER_EVERY:-1}
export PMM_PHI_LADDER_DEGREE=${PMM_PHI_LADDER_DEGREE:-11}
export PMM_PHI_SADDLE_MIN_L=${PMM_PHI_SADDLE_MIN_L:-54}

for corpus in "${corpora[@]}"; do
    stream="output/streams/bpe_${corpus}"
    if [[ ! -f "$stream/ids.npy" || ! -f "$stream/stream.json" ]]; then
        echo "missing stream: $stream" >&2
        exit 1
    fi

    for vocabulary_size in "${grid[@]}"; do
        top_k=$((vocabulary_size - 1))
        m_values=()
        for m in "${state_grid[@]}"; do
            if (( m < vocabulary_size )); then
                m_values+=("$m")
            fi
        done
        m_values+=("$vocabulary_size")
        m_csv=$(IFS=,; echo "${m_values[*]}")

        point="$output_root/$corpus/v${vocabulary_size}"
        result="$point/results.json"
        log="$point/run.log"
        mkdir -p "$point"

        if [[ -f "$result" ]]; then
            echo "skip complete $corpus V=$vocabulary_size: $result"
            continue
        fi

        echo "start $corpus V=$vocabulary_size M=$m_csv"
        "$python_bin" -u scripts/state_family_experiment.py \
            --ids "$stream" \
            --top-k "$top_k" \
            --m-grid "$m_csv" \
            --state-order frequency \
            --jobs "$jobs" \
            --alphabet-grid-size "$grid_size" \
            --cache-dir "$cache_root/$corpus/v${vocabulary_size}/cache" \
            --out "$point" \
            2>&1 | tee "$log"
        echo "finish $corpus V=$vocabulary_size"
    done
done

echo "sweep complete: $output_root"
