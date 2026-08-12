#!/usr/bin/env python3
"""Recover predictive scoring from persisted sparse calibration states."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import tempfile
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from product_model_with_memory.graphical_calibration import (
    SparseGroupedProblem,
    SparseGroupedResult,
    SparseProjectedPair,
    sparse_gated_log_probabilities,
    sparse_pair_log_probabilities,
    sparse_star_log_probabilities,
)
from product_model_with_memory.production_coding import (
    PRODUCTION_SEQUENCE_ESTIMATOR,
    layered_sequence_code,
    require_production_sequence_estimator,
)
from product_model_with_memory.streams import load_stream, reduce_ids


def load_state(path: Path):
    data = np.load(path)
    estimator = (
        str(data["sequence_estimator"])
        if "sequence_estimator" in data.files
        else PRODUCTION_SEQUENCE_ESTIMATOR
    )
    require_production_sequence_estimator(estimator, source=str(path))
    problem = SparseGroupedProblem(
        vocabulary_size=len(data["target_y"]),
        edge_a=data["edge_a"], edge_b=data["edge_b"],
        edge_probability=data["edge_probability"],
        target_y=data["target_y"],
        active_ya_y=data["active_ya_y"],
        active_ya_a=data["active_ya_a"], target_ya=data["target_ya"],
        active_yb_y=data["active_yb_y"],
        active_yb_b=data["active_yb_b"], target_yb=data["target_yb"],
    )
    result = SparseGroupedResult(
        data["log_base_y"], data["correction_ya"], data["correction_yb"],
        0, np.nan, np.nan, np.nan, True,
    )

    def pair(name):
        return SparseProjectedPair(
            problem.vocabulary_size,
            data[f"fallback_{name}_left"],
            data[f"fallback_{name}_right"],
            data[f"fallback_{name}_background"],
            data[f"fallback_{name}_active_y"],
            data[f"fallback_{name}_active_context"],
            data[f"fallback_{name}_delta"],
        )

    return problem, result, pair("ya"), pair("yb"), int(data["prefix"])


def score_interval(task):
    """Score one checkpoint interval using the exact reference calculation."""

    interval_index, state_path, next_state_path, reduced_path = task
    problem, result, p_ya, p_yb, prefix = load_state(Path(state_path))
    with np.load(next_state_path) as next_data:
        next_prefix = int(next_data["prefix"])
    x = np.load(reduced_path, mmap_mode="r")
    target = x[prefix:next_prefix]
    lag1 = x[prefix - 1:next_prefix - 1]
    lag2 = x[prefix - 2:next_prefix - 2]
    candidate = sparse_gated_log_probabilities(
        problem, result, target, lag1, lag2, p_ya, p_yb
    )
    star = sparse_star_log_probabilities(p_ya, p_yb, target, lag1, lag2)
    pair1 = sparse_pair_log_probabilities(p_ya, target, lag1)
    support = np.sort(
        problem.edge_a * problem.vocabulary_size + problem.edge_b
    )
    keys = lag1 * problem.vocabulary_size + lag2
    positions = np.searchsorted(support, keys)
    covered = (positions < len(support)) & (
        support[np.minimum(positions, len(support) - 1)] == keys
    )
    candidate_sum = float(candidate.sum())
    star_sum = float(star.sum())
    pair1_sum = float(pair1.sum())
    scale = -1.0 / (len(target) * np.log(2.0))
    return {
        "interval_index": interval_index,
        "fit_prefix": prefix,
        "scored_records": len(target),
        "supported_fraction": float(covered.mean()),
        "candidate_bpc": candidate_sum * scale,
        "star_bpc": star_sum * scale,
        "pair1_bpc": pair1_sum * scale,
        "calibrated_gain_over_star_bpc": (
            (star_sum - candidate_sum) * scale
        ),
        "candidate_bits": -candidate_sum / np.log(2.0),
        "star_bits": -star_sum / np.log(2.0),
        "pair1_bits": -pair1_sum / np.log(2.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--ids", default="output/streams/bpe_text8")
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    paths = sorted((Path(args.run) / "states").glob("checkpoint_*.npz"))
    if len(paths) < 2:
        raise ValueError("at least two checkpoint states are required")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    first_state = load_state(paths[0])
    ids, metadata = load_stream(args.ids)
    original = ids[:args.n].astype(np.int64)
    x, _, _, keep = reduce_ids(
        original, args.top_k, return_keep=True
    )
    x = x.astype(np.int64)
    with np.load(paths[-1]) as last_data:
        final_prefix = int(last_data["prefix"])
    if final_prefix > len(x):
        raise ValueError(
            f"stream has {len(x)} records but final checkpoint is "
            f"{final_prefix}"
        )
    if int(x.max(initial=-1)) >= first_state[0].vocabulary_size:
        raise ValueError(
            "reduced stream alphabet does not match calibration states; "
            "check --top-k"
        )
    started = time.time()
    prefixes = []
    for path in paths:
        with np.load(path) as data:
            prefixes.append(int(data["prefix"]))
    tasks = [
        (index, str(a), str(b), prefixes[index + 1] - prefixes[index])
        for index, (a, b) in enumerate(pairwise(paths))
    ]
    with tempfile.TemporaryDirectory(prefix="calibration_score_") as temp_dir:
        reduced_path = Path(temp_dir) / "reduced.npy"
        np.save(reduced_path, x)
        worker_tasks = [
            (index, a, b, str(reduced_path))
            for index, a, b, _ in tasks
        ]
        if args.workers == 1:
            scored = list(map(score_interval, worker_tasks))
        else:
            # Longest-processing-time first avoids leaving the largest final
            # checkpoint interval as a single-process tail.  Restore causal
            # order below before accumulating floating-point totals.
            worker_tasks.sort(
                key=lambda task: tasks[task[0]][3], reverse=True
            )
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers
            ) as executor:
                scored = list(executor.map(score_interval, worker_tasks))
            scored.sort(key=lambda row: row["interval_index"])
    rows = []
    total_candidate = total_star = total_pair1 = 0.0
    total_records = 0
    for row in scored:
        row.pop("interval_index")
        total_candidate += row.pop("candidate_bits")
        total_star += row.pop("star_bits")
        total_pair1 += row.pop("pair1_bits")
        total_records += row["scored_records"]
        rows.append(row)
    first_prefix = first_state[4]
    initial_code = layered_sequence_code(
        np.bincount(x[:first_prefix], minlength=args.top_k + 1),
        args.top_k + 1,
    )
    initial_reduced_bits = initial_code.bits
    reduced_full_bits = initial_reduced_bits + total_candidate
    tokenizer_alphabet = int(metadata.get(
        "alphabet", int(original.max()) + 1
    ))
    subset_bits = float(
        (
            gammaln(tokenizer_alphabet + 1)
            - gammaln(len(keep) + 1)
            - gammaln(tokenizer_alphabet - len(keep) + 1)
        ) / np.log(2.0)
    )
    tokenizer_vocabulary_bits = float(metadata.get("fixed_bits", 0.0))
    vocabulary_bits = tokenizer_vocabulary_bits + subset_bits
    retained = np.zeros(tokenizer_alphabet, dtype=bool)
    retained[keep] = True
    original_counts = np.bincount(original, minlength=tokenizer_alphabet)
    escaped_counts = original_counts[~retained]
    escaped_total = int(escaped_counts.sum())
    escape_code = layered_sequence_code(
        escaped_counts, tokenizer_alphabet - len(keep)
    )
    escape_bits = escape_code.bits
    positive_escape = escaped_counts[escaped_counts > 0]
    escape_probability = positive_escape / escaped_total
    oracle_escape_bits = -float(np.sum(
        positive_escape * np.log2(escape_probability)
    ))
    honest_bits = reduced_full_bits + vocabulary_bits + escape_bits
    accounting = {
        "sequence_estimator": PRODUCTION_SEQUENCE_ESTIMATOR,
        "unit": "bits_per_bpe_token",
        "initial_prefix_tokens": first_prefix,
        "initial_prefix_reduced_bits": initial_reduced_bits,
        "initial_prefix_code": {
            "estimator": initial_code.estimator,
            "alphabet_size": initial_code.alphabet_size,
            "l_max": initial_code.l_max,
        },
        "reduced_alphabet_predictive_bpc": reduced_full_bits / len(original),
        "tokenizer_vocabulary_bits": tokenizer_vocabulary_bits,
        "selected_subset_description_bits": subset_bits,
        "vocabulary_description_bpc": vocabulary_bits / len(original),
        "escaped_tokens": escaped_total,
        "escape_fraction": escaped_total / len(original),
        "escaped_token_payload_bpc": escape_bits / len(original),
        "escaped_token_payload_code": {
            "estimator": escape_code.estimator,
            "alphabet_size": escape_code.alphabet_size,
            "l_max": escape_code.l_max,
        },
        "oracle_escape_payload_bpc_noncode": (
            oracle_escape_bits / len(original)
        ),
        "honest_full_stream_bpc": honest_bits / len(original),
    }
    if len(original) == int(metadata.get("n_tokens", -1)):
        accounting["honest_full_stream_bits_per_input_byte"] = (
            honest_bits / float(metadata["n_bytes"])
        )
    payload = {
        "source": args.run,
        "V": first_state[0].vocabulary_size,
        "n": args.n,
        "scored_records": total_records,
        "candidate_bpc": total_candidate / total_records,
        "star_bpc": total_star / total_records,
        "pair1_bpc": total_pair1 / total_records,
        "calibrated_gain_over_star_bpc": (
            (total_star - total_candidate) / total_records
        ),
        "accounting": accounting,
        "scoring_seconds": time.time() - started,
        "workers": args.workers,
        "rows": rows,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
