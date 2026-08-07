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
    AppendOnlySparseSupportState, BirthMajorSparseSupport,
    SparseGroupedProblem, SparseGroupedResult, append_checkpoint_support,
    first_pair_warm_start,
    checkpoint_in_birth_major_support, load_ab_major_intersection_graph,
    load_layered_intersection_graph, load_sparse_intersection_delta,
    pair_midpoint_warm_start,
    pair_product_warm_start, second_pair_warm_start,
    empirical_pair_slack_variances, exact_sparse_dual_wolfe,
    sparse_factorized_dual_evaluation, sparse_grouped_ipf,
    sparse_deltas_as_layered_graph, stochastic_sparse_dual_approach,
    transfer_sparse_warm_start,
)


def select_warm_start(
    problem: SparseGroupedProblem,
    exact_graph,
    checkpoint: int,
    workers: int,
    transferred: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    restart: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    policy: str,
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
    if restart is not None:
        candidates["restart"] = restart

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
    selected = (
        "restart" if restart is not None
        else "transferred" if policy == "legacy" and transferred is not None
        else "pair_product" if policy == "legacy"
        else min(valid)[-1]
    )
    return candidates[selected], rows, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    topology = parser.add_mutually_exclusive_group(required=True)
    topology.add_argument("--store", help="legacy cumulative graph store")
    topology.add_argument(
        "--delta-store", help="append-only graph-delta store"
    )
    parser.add_argument("--problems", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--process-shards", type=int, default=1,
        help=(
            "experimental persistent process shards for one stochastic "
            "gradient; each shard locally threads and reduces its replicas"
        ),
    )
    parser.add_argument(
        "--replicas", type=int, default=12,
        help="fixed total stochastic batch, independent of --workers",
    )
    parser.add_argument(
        "--seed", type=int, default=71,
        help="base stochastic seed; checkpoint index is added",
    )
    parser.add_argument(
        "--initialization-policy", choices=("legacy", "portfolio"),
        default="legacy",
        help=(
            "legacy uses pair-product once then certified transfers; "
            "portfolio is an experimental objective-selected diagnostic"
        ),
    )
    parser.add_argument(
        "--max-stochastic-steps", "--steps", dest="steps", type=int,
        default=20_000,
        help=(
            "safety limit; normal termination is controlled adaptively by "
            "the exact certificate and plateau scheduler"
        ),
    )
    parser.add_argument("--tolerance", type=float, default=1e-2)
    parser.add_argument(
        "--stationarity-tolerance", type=float, default=1e-5,
        help="regularized-gradient infinity-norm tolerance",
    )
    parser.add_argument(
        "--relaxed", action="store_true",
        help="allow uncertainty-weighted YA/YB margin slack",
    )
    parser.add_argument(
        "--slack-precision", type=float, default=1.0,
        help="dimensionless confidence multiplier for relaxed pair margins",
    )
    parser.add_argument(
        "--exact-interval", type=int, default=5,
        help=(
            "SVRG reference-refresh and exact-certificate interval; shared-"
            "graph timing audits favor 5 over the historical value 50"
        ),
    )
    parser.add_argument("--blocks", type=int, default=128)
    parser.add_argument(
        "--block-replica-schedule",
        choices=("independent", "systematic"),
        default="independent",
        help=(
            "assign independently sampled blocks to replicas, or use one "
            "randomized systematic draw that spreads replicas over block mass"
        ),
    )
    parser.add_argument("--cache", type=int, default=16)
    parser.add_argument(
        "--persistent-reference-positions", action="store_true",
        help=(
            "retain block YA/YB support indices across numerical SVRG "
            "reference refreshes; diagnostic until its memory scaling is known"
        ),
    )
    parser.add_argument(
        "--lazy-sampled-intersections", action="store_true",
        help=(
            "construct and cache only sampled block intersections even when "
            "a legacy AB-major graph is available"
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument(
        "--minimum-learning-rate", type=float, default=3e-3,
    )
    parser.add_argument(
        "--plateau-relative-threshold", type=float, default=1e-4,
        help="relative exact-dual improvement counted by the plateau scheduler",
    )
    parser.add_argument(
        "--record-trace", action="store_true",
        help="include the exact-check scheduler trajectory in each result row",
    )
    parser.add_argument(
        "--progress-interval", type=int, default=0,
        help=(
            "print a live certificate record every this many stochastic "
            "steps; zero disables live progress"
        ),
    )
    parser.add_argument(
        "--save-fallback-candidates", action="store_true",
        help="persist the post-stochastic state before each exact fallback",
    )
    parser.add_argument(
        "--snapshot-certificates",
        help="comma-separated exact-certificate thresholds to persist",
    )
    parser.add_argument(
        "--skip-fallback", action="store_true",
        help="probe only: retain the best stochastic state without exact finish",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument(
        "--restart-state",
        help=(
            "scoring-compatible state for the first selected checkpoint; "
            "reuse its factors while resetting optimizer and scheduler state"
        ),
    )
    args = parser.parse_args()
    if args.progress_interval < 0:
        parser.error("--progress-interval must be nonnegative")
    snapshot_thresholds = (
        [] if not args.snapshot_certificates else sorted(
            {float(value) for value in args.snapshot_certificates.split(",")},
            reverse=True,
        )
    )
    if any(value <= 0.0 for value in snapshot_thresholds):
        parser.error("snapshot certificate thresholds must be positive")

    paths = sorted((Path(args.problems) / "states").glob("checkpoint_*.npz"))
    stop = len(paths) if args.stop is None else min(args.stop, len(paths))
    if not 0 <= args.start < stop:
        parser.error("invalid checkpoint range")
    if args.delta_store:
        delta_store = Path(args.delta_store)
        support_path = (
            delta_store / "support" / f"checkpoint_{stop - 1:03d}.npz"
        )
        with np.load(support_path, allow_pickle=False) as saved:
            state = AppendOnlySparseSupportState(
                int(saved["vocabulary_size"]), saved["keys_ya"],
                saved["keys_yb"], saved["keys_ab"], saved["birth_ya"],
                saved["birth_yb"], saved["birth_ab"],
            )
        state, support = append_checkpoint_support(
            state, load_problem(paths[stop - 1]), stop - 1
        )
        deltas = tuple(
            load_sparse_intersection_delta(
                delta_store / "deltas" / f"checkpoint_{k:03d}"
            )
            for k in range(stop)
        )
        # Only the O(number of YA edges x layers) row directories are
        # expanded.  Triangle payloads remain mmap-backed in their immutable
        # deltas and are never copied into a cumulative graph.
        exact_graph = sparse_deltas_as_layered_graph(
            deltas, len(support.problem.target_ya)
        )
        sampled_graph = None
    else:
        store = Path(args.store)
        manifest = json.loads((store / "manifest.json").read_text())
        support_dir = store / "support"
        load = lambda name: np.load(  # noqa: E731
            support_dir / f"{name}.npy", mmap_mode="r"
        )
        final = SparseGroupedProblem(
            int(manifest["vocabulary_size"]), load("edge_a"),
            load("edge_b"), np.zeros(len(load("edge_a"))),
            load("target_y"), load("active_ya_y"), load("active_ya_a"),
            np.zeros(len(load("active_ya_y"))), load("active_yb_y"),
            load("active_yb_b"), np.zeros(len(load("active_yb_y"))),
        )
        support = BirthMajorSparseSupport(
            final, load("birth_ya"), load("birth_yb"), load("birth_ab")
        )
        exact_graph = load_layered_intersection_graph(store / "graph")
        sampled_graph = (
            None if args.lazy_sampled_intersections
            else load_ab_major_intersection_graph(store / "ab_graph")
        )
    out = Path(args.out)
    (out / "states").mkdir(parents=True, exist_ok=True)
    previous_problem = None
    previous_result = None
    if args.start:
        previous_checkpoint = args.start - 1
        previous_state_path = (
            out / "states" / f"checkpoint_{previous_checkpoint:03d}.npz"
        )
        if not previous_state_path.exists():
            raise FileNotFoundError(
                f"cannot resume at checkpoint {args.start}: missing "
                f"{previous_state_path}"
            )
        previous_original = load_problem(paths[previous_checkpoint])
        previous_problem = checkpoint_in_birth_major_support(
            previous_original, support, previous_checkpoint
        )
        with np.load(previous_state_path, allow_pickle=False) as state:
            previous_result = SparseGroupedResult(
                log_base_y=state["log_base_y"],
                correction_ya=reorder_values(
                    previous_original.active_ya_y,
                    previous_original.active_ya_a,
                    state["correction_ya"],
                    previous_problem.active_ya_y,
                    previous_problem.active_ya_a,
                    previous_problem.vocabulary_size,
                ),
                correction_yb=reorder_values(
                    previous_original.active_yb_y,
                    previous_original.active_yb_b,
                    state["correction_yb"],
                    previous_problem.active_yb_y,
                    previous_problem.active_yb_b,
                    previous_problem.vocabulary_size,
                ),
                iterations=0,
                grouped_residual_ya_l1=0.0,
                grouped_residual_yb_l1=0.0,
                residual_y_l1=0.0,
                converged=True,
            )
    rows = []
    run_started = time.perf_counter()
    for checkpoint in range(args.start, stop):
        original = load_problem(paths[checkpoint])
        problem = checkpoint_in_birth_major_support(
            original, support, checkpoint
        )
        with np.load(paths[checkpoint], allow_pickle=False) as source:
            prefix = int(source["prefix"])
        pair_variances = (
            empirical_pair_slack_variances(problem, prefix)
            if args.relaxed else (None, None)
        )
        transferred = None
        if previous_result is not None:
            transferred = transfer_sparse_warm_start(
                previous_problem, previous_result, problem
            )
        restart = None
        if checkpoint == args.start and args.restart_state:
            restart_path = Path(args.restart_state)
            with np.load(restart_path, allow_pickle=False) as state:
                with np.load(paths[checkpoint], allow_pickle=False) as source:
                    expected_prefix = int(source["prefix"])
                if (
                    "prefix" in state
                    and int(state["prefix"]) != expected_prefix
                ):
                    raise ValueError(
                        "restart state belongs to a different checkpoint prefix"
                    )
                restart = (
                    np.asarray(state["log_base_y"]),
                    reorder_values(
                        np.asarray(state["active_ya_y"])
                        if "active_ya_y" in state else original.active_ya_y,
                        np.asarray(state["active_ya_a"])
                        if "active_ya_a" in state else original.active_ya_a,
                        state["correction_ya"], problem.active_ya_y,
                        problem.active_ya_a, problem.vocabulary_size,
                    ),
                    reorder_values(
                        np.asarray(state["active_yb_y"])
                        if "active_yb_y" in state else original.active_yb_y,
                        np.asarray(state["active_yb_b"])
                        if "active_yb_b" in state else original.active_yb_b,
                        state["correction_yb"], problem.active_yb_y,
                        problem.active_yb_b, problem.vocabulary_size,
                    ),
                )
        (lb, c1, c2), initialization_rows, initialization = (
            select_warm_start(
                problem, exact_graph, checkpoint, args.workers, transferred,
                restart, args.initialization_policy,
            )
        )
        print(json.dumps({
            "checkpoint": checkpoint,
            "initialization": initialization,
            "initialization_policy": args.initialization_policy,
            "initialization_candidates": initialization_rows,
        }), flush=True)
        saved_thresholds = set()

        def save_candidate(path, factors, certificate_value):
            np.savez(
                path,
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
                log_base_y=factors[0],
                correction_ya=factors[1],
                correction_yb=factors[2],
                stochastic_certificate=certificate_value,
            )

        def snapshot_callback(step, certificate_value, sb, s1, s2):
            if (
                args.progress_interval > 0
                and step % args.progress_interval == 0
            ):
                print(json.dumps({
                    "checkpoint": checkpoint,
                    "stochastic_step": step,
                    "certificate": certificate_value,
                    "elapsed_seconds": time.perf_counter() - started,
                }), flush=True)
            for threshold in snapshot_thresholds:
                if threshold in saved_thresholds or certificate_value > threshold:
                    continue
                snapshot_dir = (
                    out / "stochastic_snapshots"
                    / f"checkpoint_{checkpoint:03d}"
                )
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                save_candidate(
                    snapshot_dir / f"threshold_{threshold:.6g}.npz",
                    (sb, s1, s2), certificate_value,
                )
                saved_thresholds.add(threshold)
                print(json.dumps({
                    "checkpoint": checkpoint,
                    "snapshot_threshold": threshold,
                    "snapshot_step": step,
                    "snapshot_certificate": certificate_value,
                }), flush=True)
        started = time.perf_counter()
        stochastic = stochastic_sparse_dual_approach(
            problem, lb, c1, c2,
            steps=args.steps, batch_size=1,
            sampling="blocks", edge_blocks=args.blocks,
            replicas=args.replicas, stochastic_workers=args.workers,
            variance_reduction=True, exact_interval=args.exact_interval,
            exact_margin_workers=args.workers,
            certificate_tolerance=(
                args.stationarity_tolerance if args.relaxed
                else args.tolerance
            ),
            optimizer="adam_plateau",
            learning_rate=args.learning_rate,
            minimum_learning_rate=args.minimum_learning_rate,
            plateau_relative_threshold=args.plateau_relative_threshold,
            seed=args.seed + checkpoint,
            exact_layered_graph=exact_graph,
            exact_layered_checkpoint=checkpoint,
            sampled_ab_major_graph=sampled_graph,
            block_replica_schedule=args.block_replica_schedule,
            persistent_reference_positions=args.persistent_reference_positions,
            process_shards=args.process_shards,
            lazy_block_cache=args.cache,
            pair_slack_precision=(
                args.slack_precision if args.relaxed else float("inf")
            ),
            pair_slack_variance_ya=pair_variances[0],
            pair_slack_variance_yb=pair_variances[1],
            exact_record_callback=(
                snapshot_callback
                if snapshot_thresholds or args.progress_interval > 0
                else None
            ),
        )
        stochastic_seconds = time.perf_counter() - started
        stopping_value = (
            stochastic.best_exact_stationarity if args.relaxed
            else stochastic.best_exact_certificate
        )
        target_tolerance = (
            args.stationarity_tolerance if args.relaxed else args.tolerance
        )
        fallback = stopping_value > target_tolerance
        fallback_seconds = 0.0
        final_stationarity = (
            stochastic.best_exact_stationarity if args.relaxed else None
        )
        fallback_converged = None
        if fallback and not args.skip_fallback:
            if args.save_fallback_candidates:
                candidate_dir = out / "fallback_candidates"
                candidate_dir.mkdir(parents=True, exist_ok=True)
                save_candidate(
                    candidate_dir / f"checkpoint_{checkpoint:03d}.npz",
                    (
                        stochastic.log_base_y, stochastic.correction_ya,
                        stochastic.correction_yb,
                    ),
                    stochastic.best_exact_certificate,
                )
            fallback_started = time.perf_counter()
            if args.relaxed:
                finished = exact_sparse_dual_wolfe(
                    problem, stochastic.log_base_y,
                    stochastic.correction_ya, stochastic.correction_yb,
                    tolerance=args.stationarity_tolerance,
                    max_iterations=5_000,
                    margin_workers=args.workers,
                    layered_graph=exact_graph,
                    layered_checkpoint=checkpoint,
                    pair_slack_precision=args.slack_precision,
                    pair_slack_variance_ya=pair_variances[0],
                    pair_slack_variance_yb=pair_variances[1],
                )
                diagnostic = sparse_factorized_dual_evaluation(
                    problem, finished.log_base_y,
                    finished.correction_ya, finished.correction_yb,
                    compute_certificate=True, layered_graph=exact_graph,
                    layered_checkpoint=checkpoint,
                    margin_workers=args.workers,
                )
                result = SparseGroupedResult(
                    finished.log_base_y, finished.correction_ya,
                    finished.correction_yb, finished.iterations,
                    float(diagnostic.residual_ya_l1),
                    float(diagnostic.residual_yb_l1),
                    float(diagnostic.residual_y_l1),
                    finished.converged, finished.evaluations,
                )
                final_stationarity = finished.stationarity
                fallback_converged = finished.converged
            else:
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
            record = min(
                stochastic.trace,
                key=lambda row: row[
                    "regularized_stationarity"
                    if args.relaxed else "exact_certificate"
                ],
            )
            result = SparseGroupedResult(
                stochastic.log_base_y, stochastic.correction_ya,
                stochastic.correction_yb, stochastic.steps,
                float(record["residual_ya_l1"]),
                float(record["residual_yb_l1"]),
                float(record["residual_y_l1"]), not fallback,
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
            payload = {
                name: source[name] for name in source.files
                if not name.startswith("construction_")
            }
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
            "initialization_policy": args.initialization_policy,
            "initialization_candidates": initialization_rows,
            "stochastic_seconds": stochastic_seconds,
            "stochastic_seed": args.seed + checkpoint,
            "fallback": fallback,
            "fallback_skipped": bool(fallback and args.skip_fallback),
            "fallback_seconds": fallback_seconds,
            "certificate": max(
                result.residual_y_l1, result.grouped_residual_ya_l1,
                result.grouped_residual_yb_l1,
            ),
            "regularized_stationarity": (
                final_stationarity
            ),
            "fallback_converged": fallback_converged,
            "relaxed": args.relaxed,
            "slack_precision": args.slack_precision,
            "steps": stochastic.steps,
            "stochastic_stop_reason": stochastic.stop_reason,
            "sampled_topology_cache_bytes": stochastic.intersection_plan_bytes,
            "reference_cache_bytes": stochastic.reference_cache_bytes,
            "reference_cache_seconds": stochastic.reference_cache_seconds,
            "sampled_gradient_seconds": stochastic.sampled_gradient_seconds,
            "optimizer_seconds": stochastic.optimizer_seconds,
            "exact_seconds": stochastic.exact_seconds,
            "sampled_phase_timing": stochastic.sampled_phase_timing,
            "stochastic_trace": (
                stochastic.trace if args.record_trace else None
            ),
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
