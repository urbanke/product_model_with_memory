#!/usr/bin/env python3
"""Benchmark the native layered Hessian product without running an optimizer."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_relaxed_newton import load_problem
from validate_layered_checkpoint_store import reorder_values

from product_model_with_memory.graphical_calibration import (
    BirthMajorSparseSupport,
    SparseGroupedProblem,
    checkpoint_in_birth_major_support,
    load_layered_intersection_graph,
    sparse_factorized_dual_hessian_product_layered,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--workers", default="1,4,12")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()

    original, source = load_problem(Path(args.problem))
    with np.load(Path(args.candidate), allow_pickle=False) as saved:
        candidate = {name: np.array(saved[name], copy=True) for name in saved.files}
    store = Path(args.store)
    manifest = json.loads((store / "manifest.json").read_text())
    support_dir = store / "support"

    def load(name: str) -> np.ndarray:
        return np.load(support_dir / f"{name}.npy", mmap_mode="r")

    final = SparseGroupedProblem(
        int(manifest["vocabulary_size"]), load("edge_a"), load("edge_b"),
        np.zeros(len(load("edge_a"))), load("target_y"),
        load("active_ya_y"), load("active_ya_a"),
        np.zeros(len(load("active_ya_y"))), load("active_yb_y"),
        load("active_yb_b"), np.zeros(len(load("active_yb_y"))),
    )
    support = BirthMajorSparseSupport(
        final, load("birth_ya"), load("birth_yb"), load("birth_ab")
    )
    problem = checkpoint_in_birth_major_support(
        original, support, args.checkpoint
    )
    graph = load_layered_intersection_graph(store / "graph")
    if "active_ya_y" in candidate:
        c1 = reorder_values(
            candidate["active_ya_y"], candidate["active_ya_a"],
            candidate["correction_ya"], problem.active_ya_y,
            problem.active_ya_a, problem.vocabulary_size,
        )
        c2 = reorder_values(
            candidate["active_yb_y"], candidate["active_yb_b"],
            candidate["correction_yb"], problem.active_yb_y,
            problem.active_yb_b, problem.vocabulary_size,
        )
    else:
        c1 = candidate["correction_ya"]
        c2 = candidate["correction_yb"]
        if len(c1) != len(problem.target_ya) or len(c2) != len(problem.target_yb):
            raise ValueError(
                "candidate without support keys is not aligned to this checkpoint"
            )
    rng = np.random.default_rng(args.seed)
    direction = rng.normal(
        size=problem.vocabulary_size + len(c1) + len(c2)
    )
    rows = []
    reference = None
    for workers in [int(value) for value in args.workers.split(",")]:
        # Warm mapped pages and native worker state before timing.
        product = sparse_factorized_dual_hessian_product_layered(
            problem, graph, args.checkpoint, candidate["log_base_y"],
            c1, c2, direction, workers=workers,
        )
        started = time.perf_counter()
        for _ in range(args.repeats):
            product = sparse_factorized_dual_hessian_product_layered(
                problem, graph, args.checkpoint, candidate["log_base_y"],
                c1, c2, direction, workers=workers,
            )
        seconds = time.perf_counter() - started
        if reference is None:
            reference = product.copy()
            maximum_difference = 0.0
        else:
            maximum_difference = float(np.max(np.abs(product - reference)))
        rows.append({
            "workers": workers,
            "repeats": args.repeats,
            "seconds": seconds,
            "seconds_per_product": seconds / args.repeats,
            "products_per_second": args.repeats / seconds,
            "checksum": float(np.sum(product)),
            "maximum_difference_from_first": maximum_difference,
        })
    baseline = rows[0]["seconds_per_product"]
    for row in rows:
        row["wall_speedup_from_first"] = (
            baseline / row["seconds_per_product"]
        )
    print(json.dumps({
        "checkpoint": args.checkpoint,
        "prefix": int(source["prefix"]),
        "parameters": int(len(direction)),
        "triangles": int(sum(len(x) for x in graph.edge_ab)),
        "rows": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
