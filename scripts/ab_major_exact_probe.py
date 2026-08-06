#!/usr/bin/env python3
"""Compare exact YA-major and AB-major evaluations without an old plan."""

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

from intersection_topology_audit import load_problem
from validate_layered_checkpoint_store import reorder_values
from product_model_with_memory.graphical_calibration import (
    BirthMajorSparseSupport, SparseGroupedProblem,
    checkpoint_in_birth_major_support,
    load_ab_major_intersection_graph, load_layered_intersection_graph,
    sparse_factorized_margins_ab_major, sparse_factorized_margins_layered,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--factors", required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    store = Path(args.store)
    manifest = json.loads((store / "manifest.json").read_text())
    support_dir = store / "support"
    load = lambda name: np.load(  # noqa: E731
        support_dir / f"{name}.npy", mmap_mode="r"
    )
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
    name = f"checkpoint_{args.checkpoint:03d}.npz"
    original = load_problem(Path(args.problems) / "states" / name)
    problem = checkpoint_in_birth_major_support(
        original, support, args.checkpoint
    )
    with np.load(Path(args.factors) / "states" / name) as state:
        lb = state["log_base_y"]
        c1 = reorder_values(
            original.active_ya_y, original.active_ya_a,
            state["correction_ya"], problem.active_ya_y,
            problem.active_ya_a, problem.vocabulary_size,
        )
        c2 = reorder_values(
            original.active_yb_y, original.active_yb_b,
            state["correction_yb"], problem.active_yb_y,
            problem.active_yb_b, problem.vocabulary_size,
        )
    ya_graph = load_layered_intersection_graph(store / "graph")
    ab_graph = load_ab_major_intersection_graph(store / "ab_graph")
    started = time.perf_counter()
    ya = sparse_factorized_margins_layered(
        problem, ya_graph, args.checkpoint, lb, c1, c2,
        workers=args.workers,
    )
    ya_seconds = time.perf_counter() - started
    started = time.perf_counter()
    ab = sparse_factorized_margins_ab_major(
        problem, ab_graph, args.checkpoint, 0, lb, c1, c2,
        workers=args.workers,
    )
    ab_seconds = time.perf_counter() - started
    difference = np.abs(ab.log_normalizer - ya.log_normalizer)
    worst = int(np.argmax(difference))
    score = np.asarray(lb, dtype=np.float64) - logsumexp(lb)
    score = score.copy()
    chosen1 = np.flatnonzero(problem.active_ya_a == problem.edge_a[worst])
    chosen2 = np.flatnonzero(problem.active_yb_b == problem.edge_b[worst])
    score[problem.active_ya_y[chosen1]] += c1[chosen1]
    score[problem.active_yb_y[chosen2]] += c2[chosen2]
    direct = float(logsumexp(score))
    print(json.dumps({
        "checkpoint": args.checkpoint,
        "ya_major_seconds": ya_seconds,
        "ab_major_seconds": ab_seconds,
        "difference_target_y": float(np.max(np.abs(ab.target_y-ya.target_y))),
        "difference_ya": float(np.max(np.abs(ab.active_ya-ya.active_ya))),
        "difference_yb": float(np.max(np.abs(ab.active_yb-ya.active_yb))),
        "difference_log_z": float(difference[worst]),
        "ya_error_at_worst": float(abs(ya.log_normalizer[worst]-direct)),
        "ab_error_at_worst": float(abs(ab.log_normalizer[worst]-direct)),
    }, indent=2))


if __name__ == "__main__":
    main()
