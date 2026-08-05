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
    fit_sparse_grouped_checkpoints,
    project_sparse_layered_pair,
    restrict_margins_to_observed_contexts,
    restrict_sparse_margins_to_observed_contexts,
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
from product_model_with_memory.streams import load_stream, reduce_ids


def geometric_edges(start: int, stop: int, count: int) -> np.ndarray:
    """Match the paper experiments' 2048-token geometric schedule."""

    available = stop - start
    first = min(2_048, max(1, available // count))
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
    parser.add_argument("--interleave", type=int, default=2)
    parser.add_argument("--margin-workers", type=int, default=1)
    parser.add_argument(
        "--initialization",
        choices=(
            "unigram", "first_pair", "second_pair", "pair_midpoint",
            "pair_product",
        ),
        default="unigram",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--reference-tolerance", type=float)
    parser.add_argument("--reference-iterations", type=int, default=5_000)
    parser.add_argument("--sparse-upstream", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

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
    edges = geometric_edges(2, len(x), args.checkpoints)[1:]
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
    construction_started = time.time()
    for edge in edges:
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

            def projected_pair(table, marginal=marginal):
                contexts = np.repeat(np.arange(v), np.diff(table["ptr"]))
                return project_sparse_layered_pair(
                    marginal,
                    np.exp2(table["unseen"]),
                    table["idx"],
                    contexts,
                    np.exp2(table["val"]),
                    marginal,
                )

            p_ya = projected_pair(sparse_tables[0])
            p_yb = projected_pair(sparse_tables[1])
            observed_a = keys12 // v
            observed_b = keys12 % v
            restricted = restrict_sparse_margins_to_observed_contexts(
                p_ya, p_yb, observed_a, observed_b
            )
            problem = sparse_problem_from_projected(restricted)
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
        points.append(SparseGroupedCheckpoint(problem, np.log(marginal)))
        fallback_margins.append((p_ya, p_yb))
        construction.append({
            "prefix": int(edge),
            "retained_ab_mass": restricted.retained_ab_mass,
            "context_edges": len(problem.edge_probability),
            "ya_corrections": len(problem.target_ya),
            "yb_corrections": len(problem.target_yb),
            "construction_seconds": time.time() - started,
        })

    construction_seconds = time.time() - construction_started
    started = time.time()
    results = fit_sparse_grouped_checkpoints(
        points,
        interleave=args.interleave,
        max_iterations=args.iterations,
        tolerance=args.tolerance,
        solver="lbfgs",
        margin_workers=args.margin_workers,
        initialization=args.initialization,
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
        "ids": args.ids,
        "V": vocabulary_size,
        "n": len(x),
        "checkpoints": len(points),
        "interleave": args.interleave,
        "margin_workers": args.margin_workers,
        "initialization": args.initialization,
        "sparse_upstream": args.sparse_upstream,
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
