#!/usr/bin/env python3
"""Time sparse three-pair calibration over causal geometric checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from calibration_structure_probe import project_margins

from product_model_with_memory.codelength import default_l_max
from product_model_with_memory.graphical_calibration import (
    SparseGroupedCheckpoint,
    SparseGroupedResult,
    build_sparse_intersection_plan,
    fit_sparse_grouped_checkpoints,
    project_sparse_layered_pair,
    sparse_layered_pair,
    sparse_relaxed_problem_from_layered_pairs,
    restrict_margins_to_observed_contexts,
    restrict_sparse_margins_to_observed_contexts,
    sparse_factorized_dual_evaluation,
    sparse_gated_log_probabilities,
    sparse_problem_from_dense,
    sparse_problem_from_projected,
    sparse_star_log_probabilities,
    star_log_probabilities,
)
from product_model_with_memory.pooled_lags import (
    SparseCountRows,
    _layered_log_sparse_tables,
    _layered_log_tables,
    _LayeredPredictiveBuilder,
)
from product_model_with_memory.production_coding import (
    PRODUCTION_SEQUENCE_ESTIMATOR,
)
from product_model_with_memory.streams import load_stream, reduce_ids


def geometric_edges(
    start: int, stop: int, count: int, first_prefix: int = 2_050
) -> np.ndarray:
    """Match the paper experiments' 2048-token geometric schedule."""

    available = stop - start
    first = min(
        max(1, first_prefix - start), max(1, available // count)
    )
    lo, hi = 1.0, 4.0
    for _ in range(200):
        ratio = (lo + hi) / 2.0
        total = first * (
            count if abs(ratio - 1.0) < 1e-12
            else (ratio**count - 1.0) / (ratio - 1.0)
        )
        if total < available:
            lo = ratio
        else:
            hi = ratio
    ratio = (lo + hi) / 2.0
    edges = [start]
    accumulated = 0.0
    for k in range(count):
        accumulated += first * ratio**k
        edges.append(min(stop, start + round(accumulated)))
    edges[-1] = stop
    return np.unique(np.asarray(edges, dtype=np.int64))


def merge_sparse_counts(
    old_keys: np.ndarray, old_counts: np.ndarray, new_keys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Merge a block of transition keys into sorted cumulative counts."""

    block_keys, block_counts = np.unique(new_keys, return_counts=True)
    if not len(old_keys) and not len(block_keys):
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    keys = np.concatenate([old_keys, block_keys])
    counts = np.concatenate([old_counts, block_counts])
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    counts = counts[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(keys))]
    return keys[starts], np.add.reduceat(counts, starts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", default="output/streams/bpe_text8")
    parser.add_argument("--top-k", type=int, default=31)
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--checkpoints", type=int, default=8)
    parser.add_argument("--first-checkpoint", type=int, default=2_050)
    parser.add_argument("--interleave", type=int, default=2)
    parser.add_argument("--margin-workers", type=int, default=1)
    parser.add_argument(
        "--solver",
        choices=("lbfgs", "ipf", "stochastic", "exact-first-stochastic"),
        default="lbfgs",
    )
    parser.add_argument(
        "--evaluator", choices=("union", "factorized", "auto"),
        default="union"
    )
    parser.add_argument("--lbfgs-trust-radius", type=float, default=16.0)
    parser.add_argument(
        "--initialization",
        choices=(
            "unigram", "first_pair", "second_pair", "pair_midpoint",
            "pair_product",
        ),
        default="unigram",
    )
    parser.add_argument(
        "--checkpoint-transfer",
        choices=("copy", "tree_delta"),
        default="copy",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--stochastic-replicas", type=int, default=12)
    parser.add_argument("--stochastic-edge-blocks", type=int, default=128)
    parser.add_argument("--stochastic-learning-rate", type=float, default=3e-2)
    parser.add_argument("--stochastic-exact-interval", type=int, default=50)
    parser.add_argument("--stochastic-trust-radius", type=float, default=8.0)
    parser.add_argument(
        "--stochastic-start-certificate", type=float, default=0.1,
        help="use exact fallback immediately when the warm start is farther away",
    )
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--projection-tolerance", type=float, default=1e-12,
        help="L1 tolerance for sparse pair-margin Sinkhorn projections",
    )
    parser.add_argument(
        "--projection-iterations", type=int, default=10_000,
        help="maximum Sinkhorn iterations for sparse pair projections",
    )
    parser.add_argument("--reference-tolerance", type=float)
    parser.add_argument("--reference-iterations", type=int, default=5_000)
    parser.add_argument("--sparse-upstream", action="store_true")
    parser.add_argument(
        "--stream-checkpoints", action="store_true",
        help="fit, persist and release each interleaved checkpoint batch",
    )
    parser.add_argument(
        "--uncompressed-states", action="store_true",
        help="write checkpoints without slow single-threaded ZIP compression",
    )
    parser.add_argument(
        "--construct-only", action="store_true",
        help=(
            "construct and persist reusable checkpoint problems without "
            "running calibration"
        ),
    )
    parser.add_argument(
        "--resume-streamed", action="store_true",
        help=(
            "reuse structurally valid completed checkpoint files in --out; "
            "resume cumulative token counts from the newest completed state"
        ),
    )
    parser.add_argument(
        "--stop-after-checkpoint", type=int,
        help=(
            "stop after publishing this construction checkpoint; requires "
            "--construct-only --stream-checkpoints --interleave 1"
        ),
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.stream_checkpoints and args.reference_tolerance is not None:
        parser.error("streamed checkpoints do not support reference refits")
    if args.stream_checkpoints and args.checkpoint_transfer != "copy":
        parser.error("streamed checkpoints currently require copy transfer")
    if args.construct_only and not args.stream_checkpoints:
        parser.error("construction-only mode requires --stream-checkpoints")
    if args.resume_streamed and not (
        args.stream_checkpoints and args.construct_only
    ):
        parser.error(
            "resume-streamed currently requires --stream-checkpoints "
            "--construct-only"
        )
    if args.resume_streamed and not args.sparse_upstream:
        parser.error("resume-streamed requires sparse upstream counts")
    if args.stop_after_checkpoint is not None and not (
        args.construct_only and args.stream_checkpoints and args.interleave == 1
    ):
        parser.error(
            "--stop-after-checkpoint requires construction-only streamed "
            "checkpoints with --interleave 1"
        )
    if args.solver == "exact-first-stochastic" and (
        not args.stream_checkpoints or args.interleave != 1
    ):
        parser.error(
            "exact-first-stochastic currently requires streamed checkpoints "
            "and one consecutive chain"
        )

    for key, value in (
        ("PMM_UNIVERSAL_TABLES", "tables/anchors_prod"),
        ("PMM_PHI_LADDER_EVERY", "1"),
        ("PMM_PHI_LADDER_DEGREE", "11"),
        ("PMM_PHI_SADDLE_MIN_L", "54"),
    ):
        os.environ.setdefault(key, value)

    ids, _ = load_stream(args.ids)
    reduced, vocabulary_size, _ = reduce_ids(ids[: args.n], args.top_k)
    x = reduced.astype(np.int64)
    edges = geometric_edges(
        2, len(x), args.checkpoints, args.first_checkpoint
    )[1:]
    builder = _LayeredPredictiveBuilder(
        vocabulary_size,
        default_l_max(vocabulary_size),
        None,
        args.jobs,
        None,
    )

    points = []
    fallback_margins = []
    construction = []
    streamed_rows = []
    streamed_fit_seconds = 0.0
    streamed_persistence_seconds = 0.0
    compact_previous = [None] * args.interleave
    out = Path(args.out)
    if args.stream_checkpoints:
        out.mkdir(parents=True, exist_ok=True)
        (out / "states").mkdir(exist_ok=True)
    resume_fingerprint = {
        "sequence_estimator": PRODUCTION_SEQUENCE_ESTIMATOR,
        "ids": str(Path(args.ids).resolve()),
        "n": len(x),
        "V": vocabulary_size,
        "top_k": args.top_k,
        "edges": [int(edge) for edge in edges],
        "sparse_upstream": args.sparse_upstream,
        "margin_preprocessing": (
            "raw_relaxed" if args.sparse_upstream else "projected_exact"
        ),
        "universal_tables": os.environ["PMM_UNIVERSAL_TABLES"],
        "phi_ladder_every": os.environ["PMM_PHI_LADDER_EVERY"],
        "phi_ladder_degree": os.environ["PMM_PHI_LADDER_DEGREE"],
        "phi_saddle_min_l": os.environ["PMM_PHI_SADDLE_MIN_L"],
    }
    fingerprint_path = out / "construction_fingerprint.json"
    if args.resume_streamed and fingerprint_path.exists():
        recorded = json.loads(fingerprint_path.read_text())
        if recorded != resume_fingerprint:
            raise RuntimeError(
                "refusing to resume: construction fingerprint differs"
            )
    if args.stream_checkpoints:
        temporary = fingerprint_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(resume_fingerprint, indent=2))
        temporary.replace(fingerprint_path)
    v = vocabulary_size
    unigram = np.zeros(v, dtype=np.float64)
    if args.sparse_upstream:
        empty_i = np.empty(0, dtype=np.int64)
        keys1 = keys2 = keys12 = empty_i
        counts1 = counts2 = counts12 = empty_i
    else:
        n1 = np.zeros((v, v), dtype=np.float64)
        n2 = np.zeros((v, v), dtype=np.float64)
        n12 = np.zeros((v, v), dtype=np.float64)
    previous_edge = 0
    first_checkpoint_to_build = 0

    def compact_result(point, result):
        problem = point.problem
        return (
            result.log_base_y.copy(),
            problem.active_ya_y * v + problem.active_ya_a,
            result.correction_ya.copy(),
            problem.active_yb_y * v + problem.active_yb_b,
            result.correction_yb.copy(),
        )

    def expand_compact(compact, problem):
        log_base, old_ya, old_c1, old_yb, old_c2 = compact

        def transfer(old_keys, old_values, new_keys):
            order = np.argsort(old_keys, kind="stable")
            keys = old_keys[order]
            values = old_values[order]
            positions = np.searchsorted(keys, new_keys)
            valid = positions < len(keys)
            valid[valid] &= keys[positions[valid]] == new_keys[valid]
            answer = np.zeros(len(new_keys))
            answer[valid] = values[positions[valid]]
            return answer

        new_ya = problem.active_ya_y * v + problem.active_ya_a
        new_yb = problem.active_yb_y * v + problem.active_yb_b
        return SparseGroupedResult(
            log_base.copy(),
            transfer(old_ya, old_c1, new_ya),
            transfer(old_yb, old_c2, new_yb),
            0, np.nan, np.nan, np.nan, False,
        )

    def persist_state(index, point, result, fallback):
        problem = point.problem
        state = {
            "prefix": np.asarray(edges[index]),
            "sequence_estimator": np.asarray(
                PRODUCTION_SEQUENCE_ESTIMATOR
            ),
            "margin_preprocessing": np.asarray(
                "raw_relaxed" if args.sparse_upstream else "projected_exact"
            ),
            "edge_a": problem.edge_a,
            "edge_b": problem.edge_b,
            "edge_probability": problem.edge_probability,
            "target_y": problem.target_y,
            "active_ya_y": problem.active_ya_y,
            "active_ya_a": problem.active_ya_a,
            "target_ya": problem.target_ya,
            "active_yb_y": problem.active_yb_y,
            "active_yb_b": problem.active_yb_b,
            "target_yb": problem.target_yb,
            "log_base_y": result.log_base_y,
            "correction_ya": result.correction_ya,
            "correction_yb": result.correction_yb,
        }
        if args.sparse_upstream:
            state.update({
                "construction_unigram": unigram,
                "construction_keys1": keys1,
                "construction_counts1": counts1,
                "construction_keys2": keys2,
                "construction_counts2": counts2,
                "construction_keys12": keys12,
                "construction_counts12": counts12,
            })
            for name, pair in zip(("ya", "yb"), fallback):
                state.update({
                    f"fallback_{name}_left": pair.left,
                    f"fallback_{name}_right": pair.right,
                    f"fallback_{name}_background": pair.background,
                    f"fallback_{name}_active_y": pair.active_y,
                    f"fallback_{name}_active_context": pair.active_context,
                    f"fallback_{name}_delta": pair.delta,
                })
        else:
            state["fallback_ya"] = fallback[0]
            state["fallback_yb"] = fallback[1]
        started = time.perf_counter()
        writer = np.savez if args.uncompressed_states else np.savez_compressed
        writer(out / "states" / f"checkpoint_{index:03d}.npz", **state)
        return time.perf_counter() - started

    # Resume from the newest contiguous checkpoint carrying the cumulative
    # sufficient statistics.  Older checkpoint files predate this contract;
    # refusing them avoids silently replaying the stream and distorting C_k
    # costs seen by the scheduler.
    if args.resume_streamed:
        required_counts = (
            "construction_unigram", "construction_keys1",
            "construction_counts1", "construction_keys2",
            "construction_counts2", "construction_keys12",
            "construction_counts12",
        )
        for index, edge in enumerate(edges):
            state_path = out / "states" / f"checkpoint_{index:03d}.npz"
            if not state_path.exists():
                break
            with np.load(state_path, allow_pickle=False) as saved:
                if int(saved["prefix"]) != int(edge):
                    raise RuntimeError(
                        f"checkpoint {index} has the wrong prefix"
                    )
                if not all(name in saved for name in required_counts):
                    raise RuntimeError(
                        f"checkpoint {index} lacks persisted construction "
                        "counts; rebuild this construction chain once"
                    )
                unigram = np.asarray(saved["construction_unigram"]).copy()
                keys1 = np.asarray(saved["construction_keys1"]).copy()
                counts1 = np.asarray(saved["construction_counts1"]).copy()
                keys2 = np.asarray(saved["construction_keys2"]).copy()
                counts2 = np.asarray(saved["construction_counts2"]).copy()
                keys12 = np.asarray(saved["construction_keys12"]).copy()
                counts12 = np.asarray(saved["construction_counts12"]).copy()
            previous_edge = int(edge)
            first_checkpoint_to_build = index + 1
            row = {
                "prefix": int(edge), "retained_ab_mass": None,
                "context_edges": 0, "ya_corrections": 0,
                "yb_corrections": 0, "construction_seconds": 0.0,
                "persistence_seconds": 0.0, "constructed_only": True,
                "resumed": True,
            }
            construction.append(row)
            streamed_rows.append(row)

    if (
        args.stop_after_checkpoint is not None
        and first_checkpoint_to_build > args.stop_after_checkpoint
    ):
        print(json.dumps({
            "checkpoint": args.stop_after_checkpoint,
            "prefix": int(edges[args.stop_after_checkpoint]),
            "resumed": True,
            "peak_resident_bytes": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
        }), flush=True)

    construction_started = time.time()
    for checkpoint_index in range(first_checkpoint_to_build, len(edges)):
        edge = edges[checkpoint_index]
        started = time.time()
        unigram += np.bincount(
            x[previous_edge:edge], minlength=v
        )
        reveal_start = max(2, previous_edge)
        target = x[reveal_start:edge]
        lag1 = x[reveal_start - 1:edge - 1]
        lag2 = x[reveal_start - 2:edge - 2]
        if args.sparse_upstream:
            keys1, counts1 = merge_sparse_counts(
                keys1, counts1, lag1 * v + target
            )
            keys2, counts2 = merge_sparse_counts(
                keys2, counts2, lag2 * v + target
            )
            keys12, counts12 = merge_sparse_counts(
                keys12, counts12, lag1 * v + lag2
            )
        else:
            n1 += np.bincount(
                lag1 * v + target, minlength=v * v
            ).reshape(v, v)
            n2 += np.bincount(
                lag2 * v + target, minlength=v * v
            ).reshape(v, v)
            n12 += np.bincount(
                lag1 * v + lag2, minlength=v * v
            ).reshape(v, v)
        previous_edge = int(edge)
        if args.sparse_upstream:
            sparse_counts1 = SparseCountRows.from_sorted_keys(
                v, keys1, counts1
            )
            sparse_counts2 = SparseCountRows.from_sorted_keys(
                v, keys2, counts2
            )
            log_m, sparse_tables = _layered_log_sparse_tables(
                builder, unigram, [sparse_counts1, sparse_counts2]
            )
            marginal = np.exp2(log_m)

            def layered_pair(table, marginal=marginal):
                contexts = np.repeat(np.arange(v), np.diff(table["ptr"]))
                return sparse_layered_pair(
                    marginal,
                    np.exp2(table["unseen"]),
                    table["idx"],
                    contexts,
                    np.exp2(table["val"]),
                )

            p_ya = layered_pair(sparse_tables[0])
            p_yb = layered_pair(sparse_tables[1])
            observed_a = keys12 // v
            observed_b = keys12 % v
            problem, retained_ab_mass = sparse_relaxed_problem_from_layered_pairs(
                p_ya, p_yb, observed_a, observed_b, marginal,
            )
        else:
            log_m, (log_p1, log_p2) = _layered_log_tables(
                builder, unigram, [n1, n2]
            )
            marginal = np.exp2(log_m)
            p_ya = project_margins(
                (marginal[:, None] * np.exp2(log_p1)).T, marginal
            )
            p_yb = project_margins(
                (marginal[:, None] * np.exp2(log_p2)).T, marginal
            )
            restricted = restrict_margins_to_observed_contexts(
                p_ya, p_yb, p_ya, n12 > 0
            )
            problem = sparse_problem_from_dense(
                restricted.p_ya, restricted.p_yb, restricted.p_ab,
                n1.T > 0, n2.T > 0,
            )
        points.append(SparseGroupedCheckpoint(
            problem,
            np.log(marginal),
            p_ya if args.sparse_upstream else None,
            p_yb if args.sparse_upstream else None,
        ))
        fallback_margins.append((p_ya, p_yb))
        construction.append({
            "prefix": int(edge),
            "retained_ab_mass": (
                retained_ab_mass if args.sparse_upstream
                else restricted.retained_ab_mass
            ),
            "context_edges": len(problem.edge_probability),
            "ya_corrections": len(problem.target_ya),
            "yb_corrections": len(problem.target_yb),
            "construction_seconds": time.time() - started,
        })
        batch_ready = (
            len(points) == args.interleave
            or checkpoint_index == len(edges) - 1
        )
        if args.stream_checkpoints and batch_ready:
            batch_start = checkpoint_index + 1 - len(points)
            if args.construct_only:
                for local_index, (point, fallback) in enumerate(zip(
                    points, fallback_margins
                )):
                    index = batch_start + local_index
                    problem = point.problem
                    raw = SparseGroupedResult(
                        point.log_base_y.copy(),
                        np.zeros(len(problem.target_ya)),
                        np.zeros(len(problem.target_yb)),
                        0, np.nan, np.nan, np.nan, False,
                    )
                    persistence_seconds = persist_state(
                        index, point, raw, fallback
                    )
                    streamed_persistence_seconds += persistence_seconds
                    streamed_rows.append(construction[index] | {
                        "persistence_seconds": persistence_seconds,
                        "constructed_only": True,
                    })
                    print(json.dumps({
                        "checkpoint": index,
                        "prefix": int(edges[index]),
                        "construction_seconds": construction[index][
                            "construction_seconds"
                        ],
                        "persistence_seconds_total": (
                            streamed_persistence_seconds
                        ),
                        "peak_resident_bytes": resource.getrusage(
                            resource.RUSAGE_SELF
                        ).ru_maxrss,
                    }), flush=True)
                points.clear()
                fallback_margins.clear()
                if (
                    args.stop_after_checkpoint is not None
                    and checkpoint_index >= args.stop_after_checkpoint
                ):
                    break
                continue
            starts = []
            for local_index, point in enumerate(points):
                chain = (batch_start + local_index) % args.interleave
                previous = compact_previous[chain]
                starts.append(
                    None if previous is None
                    else expand_compact(previous, point.problem)
                )
            initial_results = None if all(x is None for x in starts) else starts
            if initial_results is not None and any(x is None for x in starts):
                raise RuntimeError("incomplete streamed warm-start batch")
            fit_started = time.time()
            batch_solver = args.solver
            preemptive_fallback = False
            if args.solver == "exact-first-stochastic":
                batch_solver = "lbfgs" if batch_start == 0 else "stochastic"
            if batch_solver == "stochastic" and initial_results is not None:
                point = points[0]
                initial = initial_results[0]
                initial_evaluation = sparse_factorized_dual_evaluation(
                    point.problem,
                    initial.log_base_y,
                    initial.correction_ya,
                    initial.correction_yb,
                    intersection_plan=build_sparse_intersection_plan(
                        point.problem
                    ),
                    compute_certificate=True,
                )
                if (
                    float(initial_evaluation.certificate)
                    > args.stochastic_start_certificate
                ):
                    batch_solver = "lbfgs"
                    preemptive_fallback = True
            batch_results = fit_sparse_grouped_checkpoints(
                points,
                initial_results=initial_results,
                interleave=len(points),
                max_iterations=args.iterations,
                tolerance=args.tolerance,
                solver=batch_solver,
                margin_workers=args.margin_workers,
                evaluator=args.evaluator,
                lbfgs_trust_radius=args.lbfgs_trust_radius,
                initialization=args.initialization,
                checkpoint_transfer=args.checkpoint_transfer,
                stochastic_replicas=args.stochastic_replicas,
                stochastic_edge_blocks=args.stochastic_edge_blocks,
                stochastic_learning_rate=args.stochastic_learning_rate,
                stochastic_exact_interval=args.stochastic_exact_interval,
                stochastic_trust_radius=args.stochastic_trust_radius,
            )
            streamed_fit_seconds += time.time() - fit_started
            exact_fallback = [preemptive_fallback] * len(batch_results)
            if batch_solver == "stochastic":
                failed = [
                    local
                    for local, result in enumerate(batch_results)
                    if not result.converged
                ]
                if failed:
                    # Hybrid mode never propagates an uncertified iterate.
                    # With its required one-checkpoint batches, polish the
                    # best stochastic candidate using the reliable solver.
                    fallback_started = time.time()
                    batch_results = fit_sparse_grouped_checkpoints(
                        points,
                        initial_results=initial_results,
                        interleave=1,
                        max_iterations=args.iterations,
                        tolerance=args.tolerance,
                        solver="lbfgs",
                        margin_workers=args.margin_workers,
                        evaluator=args.evaluator,
                        lbfgs_trust_radius=args.lbfgs_trust_radius,
                        initialization=args.initialization,
                        checkpoint_transfer=args.checkpoint_transfer,
                    )
                    streamed_fit_seconds += time.time() - fallback_started
                    exact_fallback = [True] * len(batch_results)
            failed_fallback = [
                batch_start + local
                for local, result in enumerate(batch_results)
                if not result.converged
            ]
            if failed_fallback:
                raise RuntimeError(
                    "certified fallback failed at checkpoint(s) "
                    f"{failed_fallback}; refusing to propagate"
                )
            for local_index, (point, result, fallback) in enumerate(zip(
                points, batch_results, fallback_margins
            )):
                index = batch_start + local_index
                chain = index % args.interleave
                compact_previous[chain] = compact_result(point, result)
                persistence_seconds = persist_state(
                    index, point, result, fallback
                )
                streamed_persistence_seconds += persistence_seconds
                streamed_rows.append(construction[index] | {
                    "persistence_seconds": persistence_seconds,
                    "exact_fallback": exact_fallback[local_index],
                    "iterations": result.iterations,
                    "margin_evaluations": result.margin_evaluations,
                    "converged": result.converged,
                    "residual_ya_l1": result.grouped_residual_ya_l1,
                    "residual_yb_l1": result.grouped_residual_yb_l1,
                    "residual_y_l1": result.residual_y_l1,
                })
                print(json.dumps({
                    "checkpoint": index,
                    "prefix": int(edges[index]),
                    "fit_seconds_total": streamed_fit_seconds,
                    "persistence_seconds_total": streamed_persistence_seconds,
                    "peak_resident_bytes": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss,
                    "certificate": max(
                        result.grouped_residual_ya_l1,
                        result.grouped_residual_yb_l1,
                        result.residual_y_l1,
                    ),
                    "exact_fallback": exact_fallback[local_index],
                }), flush=True)
            points.clear()
            fallback_margins.clear()

    construction_seconds = time.time() - construction_started
    if args.stream_checkpoints:
        payload = {
            "sequence_estimator": PRODUCTION_SEQUENCE_ESTIMATOR,
            "ids": args.ids,
            "V": vocabulary_size,
            "n": len(x),
            "checkpoints": len(edges),
            "first_checkpoint": args.first_checkpoint,
            "interleave": args.interleave,
            "margin_workers": args.margin_workers,
            "solver": args.solver,
            "evaluator": args.evaluator,
            "lbfgs_trust_radius": args.lbfgs_trust_radius,
            "initialization": args.initialization,
            "checkpoint_transfer": args.checkpoint_transfer,
            "sparse_upstream": args.sparse_upstream,
            "margin_preprocessing": (
                "raw_relaxed" if args.sparse_upstream else "projected_exact"
            ),
            "stream_checkpoints": True,
            "construct_only": args.construct_only,
            "construction_jobs": args.jobs,
            "tolerance": args.tolerance,
            "max_iterations_per_phase": args.iterations,
            "construction_seconds": sum(
                row["construction_seconds"] for row in construction
            ),
            "fit_seconds": streamed_fit_seconds,
            "persistence_seconds": streamed_persistence_seconds,
            "state_compression": (
                "none" if args.uncompressed_states else "zip"
            ),
            "elapsed_seconds": construction_seconds,
            "peak_resident_bytes": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "rows": streamed_rows,
            "predictive_comparison": None,
        }
        completed_all = len(streamed_rows) == len(edges)
        payload["complete"] = completed_all
        path = out / (
            "results.json" if completed_all else "partial_results.json"
        )
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2))
        temporary.replace(path)
        print(json.dumps(payload, indent=2), flush=True)
        print(f"written: {path}", flush=True)
        return
    started = time.time()
    results = fit_sparse_grouped_checkpoints(
        points,
        interleave=args.interleave,
        max_iterations=args.iterations,
        tolerance=args.tolerance,
        solver=args.solver,
        margin_workers=args.margin_workers,
        evaluator=args.evaluator,
        lbfgs_trust_radius=args.lbfgs_trust_radius,
        initialization=args.initialization,
        checkpoint_transfer=args.checkpoint_transfer,
        stochastic_replicas=args.stochastic_replicas,
        stochastic_edge_blocks=args.stochastic_edge_blocks,
        stochastic_learning_rate=args.stochastic_learning_rate,
        stochastic_exact_interval=args.stochastic_exact_interval,
        stochastic_trust_radius=args.stochastic_trust_radius,
    )
    fit_seconds = time.time() - started
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state_dir = out / "states"
    state_dir.mkdir(exist_ok=True)
    for index, (point, result, fallback) in enumerate(zip(
        points, results, fallback_margins
    )):
        problem = point.problem
        state = {
            "prefix": np.asarray(edges[index]),
            "sequence_estimator": np.asarray(
                PRODUCTION_SEQUENCE_ESTIMATOR
            ),
            "margin_preprocessing": np.asarray(
                "raw_relaxed" if args.sparse_upstream else "projected_exact"
            ),
            "edge_a": problem.edge_a,
            "edge_b": problem.edge_b,
            "edge_probability": problem.edge_probability,
            "target_y": problem.target_y,
            "active_ya_y": problem.active_ya_y,
            "active_ya_a": problem.active_ya_a,
            "target_ya": problem.target_ya,
            "active_yb_y": problem.active_yb_y,
            "active_yb_b": problem.active_yb_b,
            "target_yb": problem.target_yb,
            "log_base_y": result.log_base_y,
            "correction_ya": result.correction_ya,
            "correction_yb": result.correction_yb,
        }
        if args.sparse_upstream:
            for name, pair in zip(("ya", "yb"), fallback):
                state.update({
                    f"fallback_{name}_left": pair.left,
                    f"fallback_{name}_right": pair.right,
                    f"fallback_{name}_background": pair.background,
                    f"fallback_{name}_active_y": pair.active_y,
                    f"fallback_{name}_active_context": pair.active_context,
                    f"fallback_{name}_delta": pair.delta,
                })
        else:
            state["fallback_ya"] = fallback[0]
            state["fallback_yb"] = fallback[1]
        np.savez_compressed(
            state_dir / f"checkpoint_{index:03d}.npz", **state
        )
    rows = []
    for built, result in zip(construction, results):
        rows.append(built | {
            "iterations": result.iterations,
            "margin_evaluations": result.margin_evaluations,
            "converged": result.converged,
            "residual_ya_l1": result.grouped_residual_ya_l1,
            "residual_yb_l1": result.grouped_residual_yb_l1,
            "residual_y_l1": result.residual_y_l1,
        })
    comparison = None
    if args.reference_tolerance is not None:
        reference_started = time.time()
        reference = fit_sparse_grouped_checkpoints(
            points,
            initial_results=results,
            # This is a controlled tolerance comparison, not a production
            # warm-start chain: polish every checkpoint from its own accepted
            # candidate so tolerance is the only changed variable.
            interleave=len(points),
            max_iterations=args.reference_iterations,
            tolerance=args.reference_tolerance,
            solver="ipf",
            margin_workers=args.margin_workers,
            evaluator=args.evaluator,
            lbfgs_trust_radius=args.lbfgs_trust_radius,
        )
        reference_fit_seconds = time.time() - reference_started
        candidate_bits = 0.0
        reference_bits = 0.0
        star_bits = 0.0
        scored = 0
        score_rows = []
        for index in range(len(points) - 1):
            lo = int(edges[index])
            hi = int(edges[index + 1])
            target = x[lo:hi]
            lag1 = x[lo - 1:hi - 1]
            lag2 = x[lo - 2:hi - 2]
            p_ya, p_yb = fallback_margins[index]
            candidate_logp = sparse_gated_log_probabilities(
                points[index].problem, results[index],
                target, lag1, lag2, p_ya, p_yb,
            )
            reference_logp = sparse_gated_log_probabilities(
                points[index].problem, reference[index],
                target, lag1, lag2, p_ya, p_yb,
            )
            if args.sparse_upstream:
                star_logp = sparse_star_log_probabilities(
                    p_ya, p_yb, target, lag1, lag2
                )
            else:
                star_logp = star_log_probabilities(
                    target, lag1, lag2, p_ya, p_yb
                )
            scale = -1.0 / np.log(2.0)
            candidate_block = float(candidate_logp.sum() * scale)
            reference_block = float(reference_logp.sum() * scale)
            star_block = float(star_logp.sum() * scale)
            candidate_bits += candidate_block
            reference_bits += reference_block
            star_bits += star_block
            scored += len(target)
            supported = {
                int(a) * vocabulary_size + int(b)
                for a, b in zip(
                    points[index].problem.edge_a,
                    points[index].problem.edge_b,
                )
            }
            keys = lag1 * vocabulary_size + lag2
            supported_fraction = float(np.mean([
                int(key) in supported for key in keys
            ]))
            score_rows.append({
                "fit_prefix": lo,
                "scored_records": len(target),
                "supported_fraction": supported_fraction,
                "candidate_bpc": candidate_block / len(target),
                "reference_bpc": reference_block / len(target),
                "star_bpc": star_block / len(target),
                "calibrated_gain_over_star_bpc": (
                    star_block - candidate_block
                ) / len(target),
                "candidate_minus_reference_bpc": (
                    candidate_block - reference_block
                ) / len(target),
                "reference_max_residual": max(
                    reference[index].grouped_residual_ya_l1,
                    reference[index].grouped_residual_yb_l1,
                    reference[index].residual_y_l1,
                ),
            })
        comparison = {
            "reference_tolerance": args.reference_tolerance,
            "reference_max_iterations_per_phase": args.reference_iterations,
            "reference_fit_seconds": reference_fit_seconds,
            "scored_records": scored,
            "candidate_bpc": candidate_bits / scored,
            "reference_bpc": reference_bits / scored,
            "star_bpc": star_bits / scored,
            "calibrated_gain_over_star_bpc": (
                star_bits - candidate_bits
            ) / scored,
            "candidate_minus_reference_bpc": (
                candidate_bits - reference_bits
            ) / scored,
            "rows": score_rows,
        }
    payload = {
        "sequence_estimator": PRODUCTION_SEQUENCE_ESTIMATOR,
        "ids": args.ids,
        "V": vocabulary_size,
        "n": len(x),
        "checkpoints": len(points),
        "first_checkpoint": args.first_checkpoint,
        "interleave": args.interleave,
        "margin_workers": args.margin_workers,
        "solver": args.solver,
        "evaluator": args.evaluator,
        "lbfgs_trust_radius": args.lbfgs_trust_radius,
        "initialization": args.initialization,
        "checkpoint_transfer": args.checkpoint_transfer,
        "sparse_upstream": args.sparse_upstream,
        "margin_preprocessing": (
            "raw_relaxed" if args.sparse_upstream else "projected_exact"
        ),
        "tolerance": args.tolerance,
        "max_iterations_per_phase": args.iterations,
        "construction_seconds": construction_seconds,
        "fit_seconds": fit_seconds,
        "peak_resident_bytes": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
        "rows": rows,
        "predictive_comparison": comparison,
    }
    path = out / "results.json"
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)
    print(f"written: {path}", flush=True)


if __name__ == "__main__":
    main()
