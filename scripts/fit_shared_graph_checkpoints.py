#!/usr/bin/env python3
"""Fit saved checkpoints using shared exact and sampled graph layouts."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intersection_topology_audit import load_problem
from validate_layered_checkpoint_store import reorder_values
from product_model_with_memory.graphical_calibration import (
    BirthMajorSparseSupport, SparseGroupedProblem, SparseGroupedResult,
    first_pair_warm_start,
    checkpoint_in_birth_major_support, load_ab_major_intersection_graph,
    load_layered_intersection_graph, pair_midpoint_warm_start,
    pair_product_warm_start, second_pair_warm_start,
    sparse_factorized_dual_evaluation, sparse_grouped_ipf,
    stochastic_sparse_dual_approach, transfer_sparse_warm_start,
)


def select_warm_start(
    problem: SparseGroupedProblem,
    exact_graph,
    checkpoint: int,
    workers: int,
    transferred: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], list[dict], str]:
    """Choose a generic initializer by its exact convex dual objective."""

    unigram = (
        np.log(np.maximum(problem.target_y, np.finfo(float).tiny)),
        np.zeros(len(problem.target_ya)),
        np.zeros(len(problem.target_yb)),
    )
    candidates = {
        "unigram": unigram,
        "first_pair": first_pair_warm_start(problem),
        "second_pair": second_pair_warm_start(problem),
        "pair_midpoint": pair_midpoint_warm_start(problem),
        "pair_product": pair_product_warm_start(problem),
    }
    if transferred is not None:
        candidates["transferred"] = transferred

    rows = []
    valid = []
    for name, factors in candidates.items():
        started = time.perf_counter()
        try:
            evaluation = sparse_factorized_dual_evaluation(
                problem, *factors, compute_certificate=True,
                layered_graph=exact_graph,
                layered_checkpoint=checkpoint, margin_workers=workers,
            )
            objective = float(evaluation.objective)
            certificate = float(evaluation.certificate)
        except FloatingPointError:
            objective = certificate = float("inf")
        factor_abs_max = max(
            float(np.max(np.abs(component), initial=0.0))
            for component in factors
        )
        row = {
            "name": name,
            "objective": objective,
            "certificate": certificate,
            "factor_abs_max": factor_abs_max,
            "evaluation_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        if np.isfinite(objective) and np.isfinite(certificate):
            valid.append((objective, certificate, factor_abs_max, name))
    if not valid:
        raise FloatingPointError("all initialization candidates are nonfinite")
    selected = min(valid)[-1]
    return candidates[selected], rows, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--max-stochastic-steps", "--steps", dest="steps", type=int,
        default=20_000,
        help=(
            "safety limit; normal termination is controlled adaptively by "
            "the exact certificate and plateau scheduler"
        ),
    )
    parser.add_argument("--tolerance", type=float, default=1e-2)
    parser.add_argument("--exact-interval", type=int, default=50)
    parser.add_argument("--blocks", type=int, default=128)
    parser.add_argument("--cache", type=int, default=16)
    parser.add_argument(
        "--save-fallback-candidates", action="store_true",
        help="persist the post-stochastic state before each exact fallback",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
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
    exact_graph = load_layered_intersection_graph(store / "graph")
    sampled_graph = load_ab_major_intersection_graph(store / "ab_graph")
    paths = sorted((Path(args.problems) / "states").glob("checkpoint_*.npz"))
    stop = len(paths) if args.stop is None else min(args.stop, len(paths))
    if not 0 <= args.start < stop:
        parser.error("invalid checkpoint range")
    out = Path(args.out)
    (out / "states").mkdir(parents=True, exist_ok=True)
    previous_problem = None
    previous_result = None
    rows = []
    run_started = time.perf_counter()
    for checkpoint in range(args.start, stop):
        original = load_problem(paths[checkpoint])
        problem = checkpoint_in_birth_major_support(
            original, support, checkpoint
        )
        transferred = None
        if previous_result is not None:
            transferred = transfer_sparse_warm_start(
                previous_problem, previous_result, problem
            )
        (lb, c1, c2), initialization_rows, initialization = (
            select_warm_start(
                problem, exact_graph, checkpoint, args.workers, transferred
            )
        )
        print(json.dumps({
            "checkpoint": checkpoint,
            "initialization": initialization,
            "initialization_candidates": initialization_rows,
        }), flush=True)
        started = time.perf_counter()
        stochastic = stochastic_sparse_dual_approach(
            problem, lb, c1, c2,
            steps=args.steps, batch_size=1,
            sampling="blocks", edge_blocks=args.blocks,
            replicas=args.workers, stochastic_workers=args.workers,
            variance_reduction=True, exact_interval=args.exact_interval,
            exact_margin_workers=args.workers,
            certificate_tolerance=args.tolerance,
            optimizer="adam_plateau",
            exact_layered_graph=exact_graph,
            exact_layered_checkpoint=checkpoint,
            sampled_ab_major_graph=sampled_graph,
            lazy_block_cache=args.cache,
        )
        stochastic_seconds = time.perf_counter() - started
        fallback = stochastic.best_exact_certificate > args.tolerance
        fallback_seconds = 0.0
        if fallback:
            if args.save_fallback_candidates:
                candidate_dir = out / "fallback_candidates"
                candidate_dir.mkdir(parents=True, exist_ok=True)
                np.savez(
                    candidate_dir / f"checkpoint_{checkpoint:03d}.npz",
                    vocabulary_size=problem.vocabulary_size,
                    edge_a=problem.edge_a,
                    edge_b=problem.edge_b,
                    edge_probability=problem.edge_probability,
                    target_y=problem.target_y,
                    active_ya_y=problem.active_ya_y,
                    active_ya_a=problem.active_ya_a,
                    target_ya=problem.target_ya,
                    active_yb_y=problem.active_yb_y,
                    active_yb_b=problem.active_yb_b,
                    target_yb=problem.target_yb,
                    log_base_y=stochastic.log_base_y,
                    correction_ya=stochastic.correction_ya,
                    correction_yb=stochastic.correction_yb,
                    stochastic_certificate=stochastic.best_exact_certificate,
                )
            fallback_started = time.perf_counter()
            result = sparse_grouped_ipf(
                problem, solver="lbfgs", evaluator="layered",
                tolerance=args.tolerance, max_iterations=5_000,
                log_base_y=stochastic.log_base_y,
                correction_ya=stochastic.correction_ya,
                correction_yb=stochastic.correction_yb,
                margin_workers=args.workers,
                _layered_graph=exact_graph,
                _layered_checkpoint=checkpoint,
            )
            fallback_seconds = time.perf_counter() - fallback_started
        else:
            record = min(stochastic.trace, key=lambda row: row["exact_certificate"])
            result = SparseGroupedResult(
                stochastic.log_base_y, stochastic.correction_ya,
                stochastic.correction_yb, stochastic.steps,
                float(record["residual_ya_l1"]),
                float(record["residual_yb_l1"]),
                float(record["residual_y_l1"]), True,
                stochastic.exact_evaluations,
            )
        original_c1 = reorder_values(
            problem.active_ya_y, problem.active_ya_a, result.correction_ya,
            original.active_ya_y, original.active_ya_a,
            problem.vocabulary_size,
        )
        original_c2 = reorder_values(
            problem.active_yb_y, problem.active_yb_b, result.correction_yb,
            original.active_yb_y, original.active_yb_b,
            problem.vocabulary_size,
        )
        with np.load(paths[checkpoint], allow_pickle=False) as source:
            payload = {name: source[name] for name in source.files}
        payload.update({
            "log_base_y": result.log_base_y,
            "correction_ya": original_c1,
            "correction_yb": original_c2,
        })
        np.savez(
            out / "states" / f"checkpoint_{checkpoint:03d}.npz", **payload
        )
        row = {
            "checkpoint": checkpoint,
            "initialization": initialization,
            "initialization_candidates": initialization_rows,
            "stochastic_seconds": stochastic_seconds,
            "fallback": fallback,
            "fallback_seconds": fallback_seconds,
            "certificate": max(
                result.residual_y_l1, result.grouped_residual_ya_l1,
                result.grouped_residual_yb_l1,
            ),
            "steps": stochastic.steps,
            "stochastic_stop_reason": stochastic.stop_reason,
            "sampled_topology_cache_bytes": stochastic.intersection_plan_bytes,
            "reference_cache_bytes": stochastic.reference_cache_bytes,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        previous_problem, previous_result = problem, result
    (out / "summary.json").write_text(json.dumps({
        "elapsed_seconds": time.perf_counter() - run_started,
        "peak_resident_bytes": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
        "rows": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
