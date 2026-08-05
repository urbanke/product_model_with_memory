#!/usr/bin/env python3
"""Probe whether three-pair calibration preserves implicit backgrounds.

This is a small-alphabet diagnostic, not a compressor.  It builds the three
pair margins with the layered estimator, fits the exact conditional-IPF
triangle, and asks whether the fitted factors' unobserved cells can still be
written as row constants plus one shared target gauge.  If so, the current
background-plus-corrections representation is closed under calibration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from product_model_with_memory.codelength import default_l_max
from product_model_with_memory.graphical_calibration import (
    conditional_ipf,
    grouped_conditional_ipf,
    restrict_margins_to_observed_contexts,
    sparse_grouped_ipf,
    sparse_problem_from_dense,
)
from product_model_with_memory.pooled_lags import (
    _layered_log_tables,
    _LayeredPredictiveBuilder,
)
from product_model_with_memory.streams import load_stream, reduce_ids


def project_margins(
    joint: np.ndarray,
    marginal: np.ndarray,
    *,
    iterations: int = 10_000,
    tolerance: float = 1e-13,
) -> np.ndarray:
    """Rescale both margins of a positive pair joint to ``marginal``."""

    out = joint.copy()
    for _ in range(iterations):
        out *= (marginal / out.sum(axis=1))[:, None]
        out *= (marginal / out.sum(axis=0))[None, :]
        if max(
            np.abs(out.sum(axis=1) - marginal).sum(),
            np.abs(out.sum(axis=0) - marginal).sum(),
        ) < tolerance:
            return out
    raise RuntimeError("pair-margin projection did not converge")


def background_fit(
    log_f_ya: np.ndarray,
    log_f_yb: np.ndarray,
    inactive_ya: np.ndarray,
    inactive_yb: np.ndarray,
) -> dict[str, float | int]:
    """Fit the gauge-invariant implicit-background equations.

    A tables-free representation exists with one background per A/B row if

        log f_ya[y,a] = h[y] + c[a]  on inactive YA cells,
        log f_yb[y,b] = -h[y] + d[b] on inactive YB cells.

    The target gauge ``h`` may move freely between the two factors without
    changing their product.  Least squares therefore tests the representation
    rather than one arbitrary gauge chosen by IPF.
    """

    ny, na = log_f_ya.shape
    _, nb = log_f_yb.shape
    iya = np.argwhere(inactive_ya)
    iyb = np.argwhere(inactive_yb)
    rows = len(iya) + len(iyb)
    design = np.zeros((rows, ny + na + nb), dtype=np.float64)
    target = np.empty(rows, dtype=np.float64)
    k = 0
    for y, a in iya:
        design[k, y] = 1.0
        design[k, ny + a] = 1.0
        target[k] = log_f_ya[y, a]
        k += 1
    for y, b in iyb:
        design[k, y] = -1.0
        design[k, ny + na + b] = 1.0
        target[k] = log_f_yb[y, b]
        k += 1
    coef, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    error = target - design @ coef
    return {
        "inactive_cells": rows,
        "rms_log_error": float(np.sqrt(np.mean(error * error))),
        "max_abs_log_error": float(np.max(np.abs(error))),
        "p99_abs_log_error": float(np.quantile(np.abs(error), 0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", default="output/streams/bpe_text8")
    parser.add_argument("--top-k", type=int, default=31)
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--out", default="output/calibration_structure_probe")
    parser.add_argument("--sparse-only", action="store_true")
    parser.add_argument("--sparse-iters", type=int, default=20_000)
    parser.add_argument("--sparse-tol", type=float, default=1e-8)
    parser.add_argument("--sparse-solver",
                        choices=("ipf", "anderson", "lbfgs"),
                        default="anderson")
    args = parser.parse_args()

    for key, value in (
        ("PMM_UNIVERSAL_TABLES", "tables/anchors_prod"),
        ("PMM_PHI_LADDER_EVERY", "1"),
        ("PMM_PHI_LADDER_DEGREE", "11"),
        ("PMM_PHI_SADDLE_MIN_L", "54"),
    ):
        os.environ.setdefault(key, value)

    ids, _ = load_stream(args.ids)
    ids = ids[: args.n]
    x, vocabulary_size, _ = reduce_ids(ids, args.top_k)
    x = x.astype(np.int64)
    t = np.arange(2, len(x))
    target = x[t]
    lag1 = x[t - 1]
    lag2 = x[t - 2]

    uni = np.bincount(x, minlength=vocabulary_size).astype(np.float64)
    n1 = np.bincount(
        lag1 * vocabulary_size + target,
        minlength=vocabulary_size**2,
    ).reshape(vocabulary_size, vocabulary_size).astype(np.float64)
    n2 = np.bincount(
        lag2 * vocabulary_size + target,
        minlength=vocabulary_size**2,
    ).reshape(vocabulary_size, vocabulary_size).astype(np.float64)
    n12 = np.bincount(
        lag1 * vocabulary_size + lag2,
        minlength=vocabulary_size**2,
    ).reshape(vocabulary_size, vocabulary_size).astype(np.float64)

    builder = _LayeredPredictiveBuilder(
        vocabulary_size,
        default_l_max(vocabulary_size),
        None,
        args.jobs,
        None,
    )
    log_m, (log_p1, log_p2) = _layered_log_tables(builder, uni, [n1, n2])
    marginal = np.exp2(log_m)
    p1 = np.exp2(log_p1)
    p2 = np.exp2(log_p2)
    p_ya = project_margins((marginal[:, None] * p1).T, marginal)
    p_yb = project_margins((marginal[:, None] * p2).T, marginal)
    # Stationarity makes the adjacent context pair the same oriented pair
    # distribution as (Y, lag 1), as in pairwise_arena.py.
    p_ab = p_ya.copy()
    active_ya = n1.T > 0
    active_yb = n2.T > 0
    observed_ab = n12 > 0
    restricted = restrict_margins_to_observed_contexts(
        p_ya, p_yb, p_ab, observed_ab
    )
    sparse_problem = sparse_problem_from_dense(
        restricted.p_ya,
        restricted.p_yb,
        restricted.p_ab,
        active_ya,
        active_yb,
    )
    if args.sparse_only:
        started = time.time()
        sparse = sparse_grouped_ipf(
            sparse_problem,
            max_iterations=args.sparse_iters,
            tolerance=args.sparse_tol,
            log_base_y=np.log(marginal),
            solver=args.sparse_solver,
        )
        payload = {
            "ids": args.ids,
            "n": len(x),
            "V": vocabulary_size,
            "retained_layered_p_ab_mass": restricted.retained_ab_mass,
            "iterations": sparse.iterations,
            "converged": sparse.converged,
            "grouped_residual_ya_l1": sparse.grouped_residual_ya_l1,
            "grouped_residual_yb_l1": sparse.grouped_residual_yb_l1,
            "residual_y_l1": sparse.residual_y_l1,
            "seconds": time.time() - started,
            "stored_context_edges": len(sparse_problem.edge_probability),
            "stored_ya_corrections": len(sparse_problem.target_ya),
            "stored_yb_corrections": len(sparse_problem.target_yb),
        }
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "results.json"
        path.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        print(f"written: {path}")
        return

    fit = conditional_ipf(
        p_ya,
        p_yb,
        p_ab,
        max_iterations=20_000,
        tolerance=1e-11,
    )
    grouped = grouped_conditional_ipf(
        p_ya,
        p_yb,
        p_ab,
        active_ya,
        active_yb,
        max_iterations=20_000,
        tolerance=1e-11,
        log_base_y=np.log(marginal),
    )
    q_full = fit.joint(p_ab)
    q_grouped = grouped.joint(p_ab)
    positive = q_grouped > 0.0
    kl_grouped_from_full_bits = float(np.sum(
        q_grouped[positive]
        * np.log2(q_grouped[positive] / q_full[positive])
    ))

    supported = grouped_conditional_ipf(
        restricted.p_ya,
        restricted.p_yb,
        restricted.p_ab,
        active_ya,
        active_yb,
        max_iterations=20_000,
        tolerance=1e-8,
        log_base_y=np.log(marginal),
    )
    sparse_started = time.time()
    sparse_supported = sparse_grouped_ipf(
        sparse_problem,
        max_iterations=args.sparse_iters,
        tolerance=args.sparse_tol,
        log_base_y=np.log(marginal),
        solver=args.sparse_solver,
    )
    sparse_seconds = time.time() - sparse_started
    cond_supported = supported.conditional()
    py = p_ya.sum(axis=1)
    pa = p_ya.sum(axis=0)
    pb = p_yb.sum(axis=0)
    star_score = (
        (p_ya / pa[None, :])[:, :, None]
        * (p_yb / pb[None, :])[:, None, :]
        / py[:, None, None]
    )
    cond_star = star_score / star_score.sum(axis=0, keepdims=True)
    cond_gated = np.where(
        observed_ab[None, :, :], cond_supported, cond_star
    )
    cond_full = fit.conditional()
    gated_positive = cond_gated > 0.0
    gated_kl = float(np.sum(
        p_ab[None, :, :]
        * np.where(
            gated_positive,
            cond_gated * np.log2(cond_gated / cond_full),
            0.0,
        )
    ))
    structure = background_fit(
        fit.log_f_ya,
        fit.log_f_yb,
        inactive_ya=(n1.T == 0),
        inactive_yb=(n2.T == 0),
    )
    result = {
        "ids": args.ids,
        "n": len(x),
        "V": int(vocabulary_size),
        "active_ya": int(np.count_nonzero(n1)),
        "active_yb": int(np.count_nonzero(n2)),
        "factor_cells_each": int(vocabulary_size**2),
        "iterations": fit.iterations,
        "converged": fit.converged,
        "residual_ya_l1": fit.residual_ya_l1,
        "residual_yb_l1": fit.residual_yb_l1,
        "residual_ab_l1": fit.residual_ab_l1,
        "implicit_background_fit": structure,
        "grouped_model": {
            "iterations": grouped.iterations,
            "converged": grouped.converged,
            "grouped_residual_ya_l1": grouped.grouped_residual_ya_l1,
            "grouped_residual_yb_l1": grouped.grouped_residual_yb_l1,
            "full_residual_ya_l1": grouped.residual_ya_l1,
            "full_residual_yb_l1": grouped.residual_yb_l1,
            "residual_ab_l1": grouped.residual_ab_l1,
            "residual_y_l1": grouped.residual_y_l1,
            "kl_from_full_triangle_bits_per_record": (
                kl_grouped_from_full_bits
            ),
        },
        "observed_context_model": {
            "observed_context_pairs": int(np.count_nonzero(observed_ab)),
            "retained_layered_p_ab_mass": restricted.retained_ab_mass,
            "iterations": supported.iterations,
            "converged": supported.converged,
            "grouped_residual_ya_l1": supported.grouped_residual_ya_l1,
            "grouped_residual_yb_l1": supported.grouped_residual_yb_l1,
            "residual_y_l1": supported.residual_y_l1,
            "gated_kl_from_full_triangle_bits_per_record": gated_kl,
            "fallback": "two-pair maximum-entropy product",
            "matrix_free": {
                "iterations": sparse_supported.iterations,
                "converged": sparse_supported.converged,
                "grouped_residual_ya_l1": (
                    sparse_supported.grouped_residual_ya_l1
                ),
                "grouped_residual_yb_l1": (
                    sparse_supported.grouped_residual_yb_l1
                ),
                "residual_y_l1": sparse_supported.residual_y_l1,
                "seconds": sparse_seconds,
                "stored_context_edges": len(sparse_problem.edge_probability),
                "stored_ya_corrections": len(sparse_problem.target_ya),
                "stored_yb_corrections": len(sparse_problem.target_yb),
            },
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"written: {path}")


if __name__ == "__main__":
    main()
