#!/usr/bin/env python3
"""Seek a scalable dual-recession witness of margin infeasibility."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intersection_topology_audit import load_problem
from validate_layered_checkpoint_store import reorder_values
from product_model_with_memory.graphical_calibration import (
    BirthMajorSparseSupport,
    SparseGroupedProblem,
    checkpoint_in_birth_major_support,
    load_layered_intersection_graph,
    sparse_factorized_dual_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--start-state", required=True)
    parser.add_argument("--end-state", required=True)
    parser.add_argument("--checkpoint", type=int, default=0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--scales", default="1,2,4,8,16,32,64,128",
        help="comma-separated multiples of the unit-Linf recession direction",
    )
    args = parser.parse_args()

    store = Path(args.store)
    manifest = json.loads((store / "manifest.json").read_text())
    support_dir = store / "support"

    def load(name):
        return np.load(
            support_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False
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
    problem_path = (
        Path(args.problems) / "states"
        / f"checkpoint_{args.checkpoint:03d}.npz"
    )
    original = load_problem(problem_path)
    problem = checkpoint_in_birth_major_support(
        original, support, args.checkpoint
    )

    def state_vector(path: str) -> np.ndarray:
        with np.load(path, allow_pickle=False) as state:
            source_ya_y = (
                np.asarray(state["active_ya_y"])
                if "active_ya_y" in state else original.active_ya_y
            )
            source_ya_a = (
                np.asarray(state["active_ya_a"])
                if "active_ya_a" in state else original.active_ya_a
            )
            source_yb_y = (
                np.asarray(state["active_yb_y"])
                if "active_yb_y" in state else original.active_yb_y
            )
            source_yb_b = (
                np.asarray(state["active_yb_b"])
                if "active_yb_b" in state else original.active_yb_b
            )
            return np.concatenate([
                np.asarray(state["log_base_y"]),
                reorder_values(
                    source_ya_y, source_ya_a, state["correction_ya"],
                    problem.active_ya_y, problem.active_ya_a,
                    problem.vocabulary_size,
                ),
                reorder_values(
                    source_yb_y, source_yb_b, state["correction_yb"],
                    problem.active_yb_y, problem.active_yb_b,
                    problem.vocabulary_size,
                ),
            ])

    start = state_vector(args.start_state)
    end = state_vector(args.end_state)
    direction = end - start
    # Remove the harmless global baseline gauge before normalization.
    direction[:problem.vocabulary_size] -= np.mean(
        direction[:problem.vocabulary_size]
    )
    raw_linf = float(np.max(np.abs(direction)))
    if not np.isfinite(raw_linf) or raw_linf == 0.0:
        raise ValueError("the two states define no finite nonzero direction")
    direction /= raw_linf
    first = problem.vocabulary_size
    second = first + len(problem.target_ya)
    graph = load_layered_intersection_graph(store / "graph")

    def evaluate(vector):
        return sparse_factorized_dual_evaluation(
            problem, vector[:first], vector[first:second], vector[second:],
            compute_certificate=True,
            layered_graph=graph,
            layered_checkpoint=args.checkpoint,
            margin_workers=args.workers,
        )

    rows = []
    for scale in [float(item) for item in args.scales.split(",")]:
        evaluation = evaluate(scale * direction)
        gradient = evaluation.gradient()
        rows.append({
            "scale": scale,
            "objective": float(evaluation.objective),
            "objective_per_scale": float(evaluation.objective / scale),
            "directional_derivative": float(gradient @ direction),
            "certificate": float(evaluation.certificate),
        })

    variables = problem.vocabulary_size * len(problem.edge_probability)
    print(json.dumps({
        "checkpoint": args.checkpoint,
        "vocabulary_size": problem.vocabulary_size,
        "retained_ab_edges": len(problem.edge_probability),
        "dense_feasibility_lp_variables": variables,
        "direction_raw_linf": raw_linf,
        "rows": rows,
        "interpretation": (
            "a negative limiting objective_per_scale is a rigorous dual "
            "recession witness of primal infeasibility; a nonnegative value "
            "is inconclusive"
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
