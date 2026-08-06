#!/usr/bin/env python3
"""Benchmark the compact layered graph against the explicit plan."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intersection_topology_audit import keys, load_problem, mark_births
from product_model_with_memory.graphical_calibration import (
    build_sparse_intersection_plan,
    layered_intersection_graph_from_plan,
    sparse_factorized_margins,
    sparse_factorized_margins_layered,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--factors", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--explicit-workers", type=int, default=1)
    args = parser.parse_args()

    paths = sorted((Path(args.problems) / "states").glob("checkpoint_*.npz"))
    factor_paths = sorted((Path(args.factors) / "states").glob("checkpoint_*.npz"))
    if not paths or len(paths) != len(factor_paths):
        parser.error("problem and factor runs must have equal checkpoints")
    final = load_problem(paths[-1])
    final_keys = keys(final)
    births = [np.full(len(value), 255, dtype=np.uint8) for value in final_keys]
    for depth, path in enumerate(paths):
        for final_key, checkpoint_key, birth in zip(
            final_keys, keys(load_problem(path)), births
        ):
            mark_births(final_key, checkpoint_key, depth, birth)

    started = time.perf_counter()
    plan = build_sparse_intersection_plan(final)
    plan_build_seconds = time.perf_counter() - started
    triangle_birth = np.maximum.reduce([
        births[0][plan.correction_ya],
        births[1][plan.correction_yb],
        births[2][plan.edge],
    ])
    started = time.perf_counter()
    graph = layered_intersection_graph_from_plan(
        final, plan, triangle_birth, layers=len(paths)
    )
    graph_build_seconds = time.perf_counter() - started
    with np.load(factor_paths[-1], allow_pickle=False) as data:
        log_base = data["log_base_y"]
        c1 = data["correction_ya"]
        c2 = data["correction_yb"]

    boundaries = np.linspace(
        0, len(plan.edge), args.explicit_workers + 1, dtype=np.int64
    )
    shards = [
        (int(lo), int(hi)) for lo, hi in zip(boundaries[:-1], boundaries[1:])
        if hi > lo
    ]
    explicit = sparse_factorized_margins(
        final, plan, log_base, c1, c2, intersection_shards=shards
    )
    layered = sparse_factorized_margins_layered(
        final, graph, len(paths) - 1, log_base, c1, c2
    )

    def measure(function):
        times = []
        for _ in range(args.repetitions):
            started = time.perf_counter()
            function()
            times.append(time.perf_counter() - started)
        return times

    explicit_times = measure(lambda: sparse_factorized_margins(
        final, plan, log_base, c1, c2, intersection_shards=shards
    ))
    layered_times = measure(lambda: sparse_factorized_margins_layered(
        final, graph, len(paths) - 1, log_base, c1, c2
    ))
    worst_edge = int(np.argmax(np.abs(
        explicit.log_normalizer - layered.log_normalizer
    )))
    a = int(final.edge_a[worst_edge])
    b = int(final.edge_b[worst_edge])
    score = log_base - logsumexp(log_base)
    score = score.copy()
    selected1 = np.flatnonzero(final.active_ya_a == a)
    selected2 = np.flatnonzero(final.active_yb_b == b)
    score[final.active_ya_y[selected1]] += c1[selected1]
    score[final.active_yb_y[selected2]] += c2[selected2]
    direct_log_z = float(logsumexp(score))
    largest = np.argsort(np.abs(
        explicit.log_normalizer - layered.log_normalizer
    ))[-100:]
    explicit_direct_error = []
    layered_direct_error = []
    for edge in largest:
        aa = int(final.edge_a[edge])
        bb = int(final.edge_b[edge])
        direct_score = log_base - logsumexp(log_base)
        direct_score = direct_score.copy()
        first_position = np.flatnonzero(final.active_ya_a == aa)
        second_position = np.flatnonzero(final.active_yb_b == bb)
        direct_score[final.active_ya_y[first_position]] += c1[first_position]
        direct_score[final.active_yb_y[second_position]] += c2[second_position]
        truth = float(logsumexp(direct_score))
        explicit_direct_error.append(abs(
            float(explicit.log_normalizer[edge]) - truth
        ))
        layered_direct_error.append(abs(
            float(layered.log_normalizer[edge]) - truth
        ))
    print(json.dumps({
        "triangles": len(plan.edge),
        "plan_build_seconds": plan_build_seconds,
        "graph_conversion_seconds": graph_build_seconds,
        "explicit_bytes": sum(array.nbytes for array in (
            plan.edge, plan.target_y, plan.correction_ya, plan.correction_yb
        )),
        "layered_bytes": graph.nbytes,
        "explicit_workers": args.explicit_workers,
        "explicit_seconds": explicit_times,
        "layered_sequential_seconds": layered_times,
        "maximum_difference": max(
            float(np.max(np.abs(left - right)))
            for left, right in zip(
                (explicit.target_y, explicit.active_ya,
                 explicit.active_yb, explicit.log_normalizer),
                (layered.target_y, layered.active_ya,
                 layered.active_yb, layered.log_normalizer),
            )
        ),
        "differences": {
            name: {
                "maximum": float(np.max(np.abs(left - right))),
                "l1": float(np.sum(np.abs(left - right))),
                "nonfinite": int(np.count_nonzero(~np.isfinite(right))),
            }
            for name, left, right in zip(
                ("target_y", "active_ya", "active_yb", "log_normalizer"),
                (explicit.target_y, explicit.active_ya,
                 explicit.active_yb, explicit.log_normalizer),
                (layered.target_y, layered.active_ya,
                 layered.active_yb, layered.log_normalizer),
            )
        },
        "worst_normalizer": {
            "edge": worst_edge,
            "a": a,
            "b": b,
            "explicit": float(explicit.log_normalizer[worst_edge]),
            "layered": float(layered.log_normalizer[worst_edge]),
            "direct_logsumexp": direct_log_z,
        },
        "largest_100_difference_direct_check": {
            "explicit_maximum_error": max(explicit_direct_error),
            "layered_maximum_error": max(layered_direct_error),
            "explicit_l1_error": sum(explicit_direct_error),
            "layered_l1_error": sum(layered_direct_error),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
