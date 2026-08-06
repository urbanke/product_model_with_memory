#!/usr/bin/env python3
"""Validate a shared layered store against real checkpoint-specific plans."""

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
from product_model_with_memory.graphical_calibration import (
    BirthMajorSparseSupport,
    SparseGroupedProblem,
    build_sparse_intersection_plan,
    checkpoint_in_birth_major_support,
    intersection_plan_from_layered_graph,
    load_layered_intersection_graph,
    sparse_factorized_margins,
    sparse_factorized_margins_layered,
)


def reorder_values(
    left: np.ndarray,
    right: np.ndarray,
    values: np.ndarray,
    expected_left: np.ndarray,
    expected_right: np.ndarray,
    vocabulary_size: int,
) -> np.ndarray:
    key = left.astype(np.int64) * vocabulary_size + right
    expected = (
        expected_left.astype(np.int64) * vocabulary_size + expected_right
    )
    order = np.argsort(key, kind="stable")
    position = np.searchsorted(key[order], expected)
    if not np.array_equal(key[order[position]], expected):
        raise RuntimeError("checkpoint values do not match global prefix")
    return np.asarray(values)[order[position]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--factors", required=True)
    parser.add_argument("--checkpoints", default="0,7,15,23,31")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    store = Path(args.store)
    manifest = json.loads((store / "manifest.json").read_text())
    support_dir = store / "support"
    arrays = {
        name: np.load(support_dir / f"{name}.npy", mmap_mode="r")
        for name in (
            "edge_a", "edge_b", "target_y", "active_ya_y", "active_ya_a",
            "active_yb_y", "active_yb_b", "birth_ya", "birth_yb", "birth_ab",
        )
    }
    final = SparseGroupedProblem(
        vocabulary_size=int(manifest["vocabulary_size"]),
        edge_a=arrays["edge_a"], edge_b=arrays["edge_b"],
        edge_probability=np.zeros(len(arrays["edge_a"])),
        target_y=arrays["target_y"],
        active_ya_y=arrays["active_ya_y"],
        active_ya_a=arrays["active_ya_a"],
        target_ya=np.zeros(len(arrays["active_ya_y"])),
        active_yb_y=arrays["active_yb_y"],
        active_yb_b=arrays["active_yb_b"],
        target_yb=np.zeros(len(arrays["active_yb_y"])),
    )
    support = BirthMajorSparseSupport(
        final, arrays["birth_ya"], arrays["birth_yb"], arrays["birth_ab"]
    )
    graph = load_layered_intersection_graph(store / "graph")
    rows = []
    for checkpoint in map(int, args.checkpoints.split(",")):
        problem_path = Path(args.problems) / "states" / f"checkpoint_{checkpoint:03d}.npz"
        factor_path = Path(args.factors) / "states" / f"checkpoint_{checkpoint:03d}.npz"
        original = load_problem(problem_path)
        aligned = checkpoint_in_birth_major_support(original, support, checkpoint)
        with np.load(factor_path, allow_pickle=False) as state:
            log_base = state["log_base_y"]
            old_c1 = state["correction_ya"]
            old_c2 = state["correction_yb"]
        c1 = reorder_values(
            original.active_ya_y, original.active_ya_a, old_c1,
            aligned.active_ya_y, aligned.active_ya_a,
            aligned.vocabulary_size,
        )
        c2 = reorder_values(
            original.active_yb_y, original.active_yb_b, old_c2,
            aligned.active_yb_y, aligned.active_yb_b,
            aligned.vocabulary_size,
        )
        old_plan = build_sparse_intersection_plan(original)
        aligned_plan = build_sparse_intersection_plan(aligned)
        layered_plan = intersection_plan_from_layered_graph(
            aligned, graph, checkpoint
        )
        topology_equal = all(np.array_equal(left, right) for left, right in (
            (aligned_plan.edge, layered_plan.edge),
            (aligned_plan.target_y, layered_plan.target_y),
            (aligned_plan.correction_ya, layered_plan.correction_ya),
            (aligned_plan.correction_yb, layered_plan.correction_yb),
        ))
        old = sparse_factorized_margins(
            original, old_plan, log_base, old_c1, old_c2
        )
        started = time.perf_counter()
        new = sparse_factorized_margins_layered(
            aligned, graph, checkpoint, log_base, c1, c2,
            workers=args.workers,
        )
        seconds = time.perf_counter() - started
        old_ya = reorder_values(
            original.active_ya_y, original.active_ya_a, old.active_ya,
            aligned.active_ya_y, aligned.active_ya_a,
            aligned.vocabulary_size,
        )
        old_yb = reorder_values(
            original.active_yb_y, original.active_yb_b, old.active_yb,
            aligned.active_yb_y, aligned.active_yb_b,
            aligned.vocabulary_size,
        )
        old_z = reorder_values(
            original.edge_a, original.edge_b, old.log_normalizer,
            aligned.edge_a, aligned.edge_b, aligned.vocabulary_size,
        )
        worst_edge = int(np.argmax(np.abs(new.log_normalizer - old_z)))
        score = np.asarray(log_base, dtype=np.float64).copy()
        score -= logsumexp(score)
        selected1 = np.flatnonzero(
            aligned.active_ya_a == aligned.edge_a[worst_edge]
        )
        selected2 = np.flatnonzero(
            aligned.active_yb_b == aligned.edge_b[worst_edge]
        )
        score[aligned.active_ya_y[selected1]] += c1[selected1]
        score[aligned.active_yb_y[selected2]] += c2[selected2]
        direct_z = float(logsumexp(score))
        rows.append({
            "checkpoint": checkpoint,
            "seconds": seconds,
            "triangles": int(sum(len(graph.edge_ab[d]) for d in range(checkpoint + 1))),
            "explicit_triangles": len(aligned_plan.edge),
            "topology_equal": topology_equal,
            "difference_target_y": float(np.max(np.abs(new.target_y - old.target_y))),
            "difference_ya": float(np.max(np.abs(new.active_ya - old_ya))),
            "difference_yb": float(np.max(np.abs(new.active_yb - old_yb))),
            "difference_log_z": float(np.max(np.abs(new.log_normalizer - old_z))),
            "worst_edge": worst_edge,
            "old_log_z_error_at_worst": float(
                abs(old_z[worst_edge] - direct_z)
            ),
            "layered_log_z_error_at_worst": float(
                abs(new.log_normalizer[worst_edge] - direct_z)
            ),
        })
        print(json.dumps(rows[-1]), flush=True)
    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
