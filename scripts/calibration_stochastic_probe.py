#!/usr/bin/env python3
"""Probe block-SVRG warm starts on persisted calibration checkpoints."""

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
    SparseGroupedResult,
    sparse_grouped_ipf,
    stochastic_sparse_dual_approach,
    transfer_sparse_warm_start,
)


def load_state(path: Path) -> tuple[SparseGroupedProblem, SparseGroupedResult]:
    with np.load(path) as data:
        problem = SparseGroupedProblem(
            vocabulary_size=len(data["target_y"]),
            edge_a=data["edge_a"],
            edge_b=data["edge_b"],
            edge_probability=data["edge_probability"],
            target_y=data["target_y"],
            active_ya_y=data["active_ya_y"],
            active_ya_a=data["active_ya_a"],
            target_ya=data["target_ya"],
            active_yb_y=data["active_yb_y"],
            active_yb_b=data["active_yb_b"],
            target_yb=data["target_yb"],
        )
        result = SparseGroupedResult(
            log_base_y=data["log_base_y"],
            correction_ya=data["correction_ya"],
            correction_yb=data["correction_yb"],
            iterations=0,
            grouped_residual_ya_l1=np.nan,
            grouped_residual_yb_l1=np.nan,
            residual_y_l1=np.nan,
            converged=False,
        )
    return problem, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--warm-index", type=int, required=True)
    parser.add_argument(
        "--initial-factors",
        help="optional factors.npz to continue instead of transferred warm start",
    )
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--edge-blocks", type=int, default=256)
    parser.add_argument(
        "--replicas", type=int, default=4,
        help="fixed number of sampled block gradients averaged per update",
    )
    parser.add_argument(
        "--stochastic-workers", type=int,
        help="threads evaluating the fixed replica batch (default: replicas)",
    )
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument(
        "--optimizer",
        choices=("adam", "adam_cosine", "adam_plateau", "svrg_bb"),
        default="adam"
    )
    parser.add_argument("--minimum-learning-rate", type=float, default=0.003)
    parser.add_argument("--exact-interval", type=int, default=25)
    parser.add_argument("--trust-radius", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", required=True)
    parser.add_argument("--finish", action="store_true")
    parser.add_argument("--finish-iterations", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--margin-workers", type=int, default=5)
    parser.add_argument("--certificate-tolerance", type=float, default=0.01)
    parser.add_argument("--exact-margin-workers", type=int, default=5)
    args = parser.parse_args()

    states = sorted((Path(args.run) / "states").glob("checkpoint_*.npz"))
    if not states:
        parser.error("run contains no persisted checkpoint states")
    for name, index in (("index", args.index), ("warm-index", args.warm_index)):
        if index < 0 or index >= len(states):
            parser.error(f"{name} lies outside the persisted checkpoints")
    if args.warm_index >= args.index:
        parser.error("warm-index must precede index")

    old_problem, old_result = load_state(states[args.warm_index])
    problem, _ = load_state(states[args.index])
    warm = transfer_sparse_warm_start(old_problem, old_result, problem)
    if args.initial_factors:
        with np.load(args.initial_factors) as initial:
            supplied = SparseGroupedResult(
                initial["log_base_y"],
                initial["correction_ya"],
                initial["correction_yb"],
                0, np.nan, np.nan, np.nan, False,
            )
        warm = transfer_sparse_warm_start(old_problem, supplied, problem)
    started = time.perf_counter()
    result = stochastic_sparse_dual_approach(
        problem,
        *warm,
        steps=args.steps,
        batch_size=1,
        learning_rate=args.learning_rate,
        exact_interval=args.exact_interval,
        seed=args.seed,
        trust_radius=args.trust_radius,
        sampling="blocks",
        replicas=args.replicas,
        stochastic_workers=args.stochastic_workers,
        edge_blocks=args.edge_blocks,
        variance_reduction=True,
        certificate_tolerance=args.certificate_tolerance,
        exact_margin_workers=args.exact_margin_workers,
        optimizer=args.optimizer,
        minimum_learning_rate=args.minimum_learning_rate,
    )
    elapsed = time.perf_counter() - started

    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination / "factors.npz",
        log_base_y=result.log_base_y,
        correction_ya=result.correction_ya,
        correction_yb=result.correction_yb,
    )
    payload = {
        "run": args.run,
        "index": args.index,
        "warm_index": args.warm_index,
        "initial_factors": args.initial_factors,
        "steps": args.steps,
        "edge_blocks": args.edge_blocks,
        "replicas": args.replicas,
        "stochastic_workers": args.stochastic_workers,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "minimum_learning_rate": args.minimum_learning_rate,
        "exact_interval": args.exact_interval,
        "trust_radius": args.trust_radius,
        "seed": args.seed,
        "certificate_tolerance": args.certificate_tolerance,
        "exact_margin_workers": args.exact_margin_workers,
        "elapsed_seconds": elapsed,
        "sampled_edge_evaluations": result.sampled_edges,
        "exact_evaluations": result.exact_evaluations,
        "best_exact_objective": result.best_exact_objective,
        "best_exact_certificate": result.best_exact_certificate,
        "exact_seconds": result.exact_seconds,
        "sampled_gradient_seconds": result.sampled_gradient_seconds,
        "optimizer_seconds": result.optimizer_seconds,
        "reference_cache_seconds": result.reference_cache_seconds,
        "trace": list(result.trace),
    }
    if args.finish:
        finishing = {}
        for name, factors in (
            ("original", warm),
            ("stochastic", (
                result.log_base_y,
                result.correction_ya,
                result.correction_yb,
            )),
        ):
            fit_started = time.perf_counter()
            fitted = sparse_grouped_ipf(
                problem,
                max_iterations=args.finish_iterations,
                tolerance=args.tolerance,
                log_base_y=factors[0],
                correction_ya=factors[1],
                correction_yb=factors[2],
                solver="lbfgs",
                margin_workers=args.margin_workers,
                evaluator="factorized",
            )
            finishing[name] = {
                "seconds": time.perf_counter() - fit_started,
                "iterations": fitted.iterations,
                "margin_evaluations": fitted.margin_evaluations,
                "converged": fitted.converged,
                "residual_y_l1": fitted.residual_y_l1,
                "residual_ya_l1": fitted.grouped_residual_ya_l1,
                "residual_yb_l1": fitted.grouped_residual_yb_l1,
            }
        payload["finishing"] = finishing
    (destination / "results.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
