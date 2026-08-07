#!/usr/bin/env python3
"""Run the relaxed Newton--CG finisher from a persisted checkpoint state."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_layered_checkpoint_store import reorder_values

from product_model_with_memory.graphical_calibration import (
    BirthMajorSparseSupport,
    SparseGroupedProblem,
    checkpoint_in_birth_major_support,
    empirical_pair_slack_variances,
    load_layered_intersection_graph,
    sparse_grouped_newton_cg,
)


def load_problem(path: Path) -> tuple[SparseGroupedProblem, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as saved:
        state = {name: np.array(saved[name], copy=True) for name in saved.files}
    problem = SparseGroupedProblem(
        len(state["target_y"]),
        state["edge_a"], state["edge_b"], state["edge_probability"],
        state["target_y"],
        state["active_ya_y"], state["active_ya_a"], state["target_ya"],
        state["active_yb_y"], state["active_yb_b"], state["target_yb"],
    )
    return problem, state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--store")
    parser.add_argument("--checkpoint", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--slack-precision", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--hessian-products", type=int, default=200)
    parser.add_argument("--max-precondition-scale", type=float, default=1e6)
    args = parser.parse_args()

    original, source = load_problem(Path(args.problem))
    problem = original
    candidate_problem, candidate = load_problem(Path(args.candidate))
    prefix = int(source["prefix"])
    graph = (
        load_layered_intersection_graph(Path(args.store) / "graph")
        if args.store else None
    )
    if args.store:
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
        problem = checkpoint_in_birth_major_support(
            original, support, args.checkpoint
        )
    variance_ya, variance_yb = empirical_pair_slack_variances(problem, prefix)
    c1 = reorder_values(
        candidate_problem.active_ya_y, candidate_problem.active_ya_a,
        candidate["correction_ya"], problem.active_ya_y,
        problem.active_ya_a, problem.vocabulary_size,
    )
    c2 = reorder_values(
        candidate_problem.active_yb_y, candidate_problem.active_yb_b,
        candidate["correction_yb"], problem.active_yb_y,
        problem.active_yb_b, problem.vocabulary_size,
    )
    started = time.perf_counter()
    result = sparse_grouped_newton_cg(
        problem,
        log_base_y=candidate["log_base_y"],
        correction_ya=c1,
        correction_yb=c2,
        max_iterations=args.iterations,
        tolerance=args.tolerance,
        max_hessian_products=args.hessian_products,
        precondition_max_scale=args.max_precondition_scale,
        pair_slack_precision=args.slack_precision,
        pair_slack_variance_ya=variance_ya,
        pair_slack_variance_yb=variance_yb,
        layered_graph=graph,
        layered_checkpoint=args.checkpoint if graph is not None else None,
        margin_workers=args.workers,
    )
    print(json.dumps({
        "prefix": prefix,
        "seconds": time.perf_counter() - started,
        "converged": result.converged,
        "stationarity": result.stationarity,
        "objective": result.objective,
        "final_stationarity": result.final_stationarity,
        "final_objective": result.final_objective,
        "raw_margin_certificate": max(
            result.residual_y_l1,
            result.grouped_residual_ya_l1,
            result.grouped_residual_yb_l1,
        ),
        "iterations": result.iterations,
        "margin_evaluations": result.margin_evaluations,
        "hessian_products": result.hessian_products,
    }, indent=2))


if __name__ == "__main__":
    main()
