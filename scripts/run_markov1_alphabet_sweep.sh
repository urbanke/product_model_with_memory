#!/usr/bin/env bash
set -euo pipefail

# Exact, checkpoint-free Markov-1 sweep used by paper Table 5.
#
# Environment overrides:
#   PMM_SWEEP_CORPORA=text8,enwik8,enwik9
#   PMM_SWEEP_GRID=1024,2048,4096,8192,16384,32768,65536,100277
#   PMM_SWEEP_DECLARED_GRID_SIZE=8
#   PMM_SWEEP_JOBS=8
#   PMM_SWEEP_ROOT=output/markov1_alphabet_sweep_20260809
#   PMM_SWEEP_PYTHON=.venv/bin/python3
#
# The result directories are restartable. A point is skipped only when its
# results.json exists; use a new PMM_SWEEP_ROOT for a genuinely fresh sweep.

project_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$project_root"

python_bin=${PMM_SWEEP_PYTHON:-.venv/bin/python3}
output_root=${PMM_SWEEP_ROOT:-output/markov1_alphabet_sweep_20260809}
corpora_csv=${PMM_SWEEP_CORPORA:-text8,enwik8,enwik9}
grid_csv=${PMM_SWEEP_GRID:-1024,2048,4096,8192,16384,32768,65536,100277}
jobs=${PMM_SWEEP_JOBS:-8}

IFS=',' read -r -a corpora <<< "$corpora_csv"
IFS=',' read -r -a grid <<< "$grid_csv"
grid_size=${PMM_SWEEP_DECLARED_GRID_SIZE:-${#grid[@]}}
if (( grid_size < ${#grid[@]} )); then
    echo "declared grid size cannot be smaller than requested sweep" >&2
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
        point="$output_root/$corpus/v${vocabulary_size}"
        result="$point/results.json"
        log="$point/run.log"
        mkdir -p "$point"

        if [[ -f "$result" ]]; then
            echo "skip complete $corpus V=$vocabulary_size: $result"
            continue
        fi

        echo "start $corpus V=$vocabulary_size"
        "$python_bin" -u scripts/state_family_experiment.py \
            --ids "$stream" \
            --top-k "$top_k" \
            --m-grid "$vocabulary_size" \
            --state-order id \
            --jobs "$jobs" \
            --alphabet-grid-size "$grid_size" \
            --out "$point" \
            2>&1 | tee "$log"
        echo "finish $corpus V=$vocabulary_size"
    done
done

echo "sweep complete: $output_root"
