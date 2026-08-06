#!/usr/bin/env python3
"""Compare Newton-CG and stochastic continuation from one near-fit state."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from newton_checkpoint_probe import load_problem, transfer
from product_model_with_memory.graphical_calibration import (
    sparse_grouped_newton_cg,
    stochastic_sparse_dual_approach,
)


def stochastic_fit(
    problem,
    initial,
    *,
    tolerance: float,
    seed: int,
    workers: int,
    steps: int,
    exact_interval: int,
):
    return stochastic_sparse_dual_approach(
        problem, *initial,
        steps=steps,
        batch_size=1,
        learning_rate=0.03,
        minimum_learning_rate=0.003,
        exact_interval=exact_interval,
        seed=seed,
        trust_radius=8.0,
        sampling="blocks",
        replicas=12,
        stochastic_workers=workers,
        edge_blocks=128,
        variance_reduction=True,
        certificate_tolerance=tolerance,
        exact_margin_workers=workers,
        optimizer="adam_plateau",
    )


def certificate(result) -> float:
    return float(result.best_exact_certificate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--warm", required=True)
    parser.add_argument("--checkpoint", type=int, default=23)
    parser.add_argument("--switch", type=float, default=0.02)
    parser.add_argument("--target", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--approach-steps", type=int, default=500)
    parser.add_argument("--finish-steps", type=int, default=500)
    parser.add_argument("--hessian-products", type=int, default=200)
    parser.add_argument("--approach-exact-interval", type=int, default=50)
    parser.add_argument("--finish-exact-interval", type=int, default=50)
    args = parser.parse_args()

    current_path = (
        Path(args.problems) / "states"
        / f"checkpoint_{args.checkpoint:03d}.npz"
    )
    previous_path = (
        Path(args.warm) / "states"
        / f"checkpoint_{args.checkpoint - 1:03d}.npz"
    )
    problem, _ = load_problem(current_path)
    old_problem, old = load_problem(previous_path)
    v = problem.vocabulary_size
    initial = (
        old["log_base_y"],
        transfer(
            old_problem.active_ya_y, old_problem.active_ya_a,
            old["correction_ya"], problem.active_ya_y,
            problem.active_ya_a, v,
        ),
        transfer(
            old_problem.active_yb_y, old_problem.active_yb_b,
            old["correction_yb"], problem.active_yb_y,
            problem.active_yb_b, v,
        ),
    )

    approach_started = time.perf_counter()
    approach = stochastic_fit(
        problem, initial,
        tolerance=args.switch,
        seed=1000 + args.checkpoint,
        workers=args.workers,
        steps=args.approach_steps,
        exact_interval=args.approach_exact_interval,
    )
    approach_seconds = time.perf_counter() - approach_started
    switch_state = (
        approach.log_base_y,
        approach.correction_ya,
        approach.correction_yb,
    )

    stochastic_started = time.perf_counter()
    stochastic = stochastic_fit(
        problem, switch_state,
        tolerance=args.target,
        seed=2000 + args.checkpoint,
        workers=args.workers,
        steps=args.finish_steps,
        exact_interval=args.finish_exact_interval,
    )
    stochastic_seconds = time.perf_counter() - stochastic_started

    newton_started = time.perf_counter()
    newton = sparse_grouped_newton_cg(
        problem,
        log_base_y=switch_state[0],
        correction_ya=switch_state[1],
        correction_yb=switch_state[2],
        tolerance=args.target,
        max_hessian_products=args.hessian_products,
    )
    newton_seconds = time.perf_counter() - newton_started
    newton_certificate = max(
        newton.residual_y_l1,
        newton.grouped_residual_ya_l1,
        newton.grouped_residual_yb_l1,
    )
    print(json.dumps({
        "checkpoint": args.checkpoint,
        "switch_target": args.switch,
        "target": args.target,
        "approach": {
            "seconds": approach_seconds,
            "steps": approach.steps,
            "certificate": certificate(approach),
            "exact_evaluations": approach.exact_evaluations,
        },
        "stochastic_finish": {
            "seconds": stochastic_seconds,
            "steps": stochastic.steps,
            "certificate": certificate(stochastic),
            "converged": certificate(stochastic) <= args.target,
            "exact_evaluations": stochastic.exact_evaluations,
        },
        "newton_finish": {
            "seconds": newton_seconds,
            "iterations": newton.iterations,
            "hessian_products": newton.hessian_products,
            "certificate": newton_certificate,
            "converged": newton.converged,
            "margin_evaluations": newton.margin_evaluations,
        },
    }, indent=2))


if __name__ == "__main__":
    main()
