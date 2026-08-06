#!/usr/bin/env python3
"""Benchmark matrix-free Newton-CG on one persisted checkpoint warm start."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from product_model_with_memory.graphical_calibration import (
    SparseGroupedProblem,
    build_sparse_intersection_plan,
    sparse_factorized_dual_evaluation,
    sparse_grouped_newton_cg,
)


def load_problem(path: Path) -> tuple[SparseGroupedProblem, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as saved:
        state = {key: saved[key] for key in saved.files}
    problem = SparseGroupedProblem(
        len(state["target_y"]),
        state["edge_a"], state["edge_b"], state["edge_probability"],
        state["target_y"],
        state["active_ya_y"], state["active_ya_a"], state["target_ya"],
        state["active_yb_y"], state["active_yb_b"], state["target_yb"],
    )
    return problem, state


def transfer(
    old_y: np.ndarray,
    old_context: np.ndarray,
    old_value: np.ndarray,
    new_y: np.ndarray,
    new_context: np.ndarray,
    vocabulary_size: int,
) -> np.ndarray:
    old_key = old_y * vocabulary_size + old_context
    new_key = new_y * vocabulary_size + new_context
    order = np.argsort(old_key, kind="stable")
    key = old_key[order]
    value = old_value[order]
    position = np.searchsorted(key, new_key)
    valid = position < len(key)
    valid[valid] &= key[position[valid]] == new_key[valid]
    result = np.zeros(len(new_key))
    result[valid] = value[position[valid]]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--warm", required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-2)
    parser.add_argument("--hessian-products", type=int, default=40)
    parser.add_argument("--max-scale", type=float, default=10.0)
    parser.add_argument("--no-precondition", action="store_true")
    args = parser.parse_args()

    problem_path = Path(args.problems) / "states" / f"checkpoint_{args.checkpoint:03d}.npz"
    warm_path = Path(args.warm) / "states" / f"checkpoint_{args.checkpoint - 1:03d}.npz"
    problem, _ = load_problem(problem_path)
    old_problem, old = load_problem(warm_path)
    v = problem.vocabulary_size
    c1 = transfer(
        old_problem.active_ya_y, old_problem.active_ya_a, old["correction_ya"],
        problem.active_ya_y, problem.active_ya_a, v,
    )
    c2 = transfer(
        old_problem.active_yb_y, old_problem.active_yb_b, old["correction_yb"],
        problem.active_yb_y, problem.active_yb_b, v,
    )
    log_base = old["log_base_y"]
    plan = build_sparse_intersection_plan(problem)
    initial = sparse_factorized_dual_evaluation(
        problem, log_base, c1, c2,
        intersection_plan=plan, compute_certificate=True,
    )
    started = time.perf_counter()
    result = sparse_grouped_newton_cg(
        problem,
        log_base_y=log_base,
        correction_ya=c1,
        correction_yb=c2,
        tolerance=args.tolerance,
        jacobi_precondition=not args.no_precondition,
        precondition_max_scale=args.max_scale,
        max_hessian_products=args.hessian_products,
    )
    print(json.dumps({
        "checkpoint": args.checkpoint,
        "initial_certificate": float(initial.certificate),
        "final_certificate": max(
            result.residual_y_l1,
            result.grouped_residual_ya_l1,
            result.grouped_residual_yb_l1,
        ),
        "converged": result.converged,
        "seconds": time.perf_counter() - started,
        "iterations": result.iterations,
        "evaluations": result.margin_evaluations,
        "hessian_products": result.hessian_products,
        "preconditioned": not args.no_precondition,
    }, indent=2))


if __name__ == "__main__":
    main()
