#!/usr/bin/env python3
"""Compare exact L-BFGS and bounded Newton-CG from one identical state."""

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
    load_layered_intersection_graph,
    sparse_grouped_ipf,
    sparse_grouped_newton_cg,
)


def certificate(result) -> float:
    return max(
        result.residual_y_l1,
        result.grouped_residual_ya_l1,
        result.grouped_residual_yb_l1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--tolerance", type=float, default=1e-2)
    parser.add_argument("--newton-products", type=int, default=80)
    parser.add_argument(
        "--lbfgs-caps", default="10,25,50,100,250",
        help="comma-separated L-BFGS approach budgets, each followed by IPF",
    )
    parser.add_argument("--lbfgs-precondition", action="store_true")
    args = parser.parse_args()

    with np.load(args.candidate, allow_pickle=False) as saved:
        problem = SparseGroupedProblem(
            int(saved["vocabulary_size"]),
            saved["edge_a"], saved["edge_b"], saved["edge_probability"],
            saved["target_y"], saved["active_ya_y"], saved["active_ya_a"],
            saved["target_ya"], saved["active_yb_y"],
            saved["active_yb_b"], saved["target_yb"],
        )
        warm = tuple(np.array(saved[name], copy=True) for name in (
            "log_base_y", "correction_ya", "correction_yb"
        ))
        initial_certificate = float(saved["stochastic_certificate"])

    graph = load_layered_intersection_graph(Path(args.store) / "graph")
    rows = []

    caps = [int(value) for value in args.lbfgs_caps.split(",")]
    if not caps or any(value < 1 for value in caps):
        parser.error("L-BFGS caps must be positive integers")
    for cap in caps:
        started = time.perf_counter()
        trace = []
        phase_timing = {}
        lbfgs = sparse_grouped_ipf(
            problem, solver="lbfgs", evaluator="layered",
            tolerance=args.tolerance, max_iterations=cap,
            log_base_y=warm[0], correction_ya=warm[1],
            correction_yb=warm[2], margin_workers=args.workers,
            lbfgs_precondition=args.lbfgs_precondition,
            trace=trace, trace_interval=max(1, cap // 10),
            phase_timing=phase_timing,
            _layered_graph=graph, _layered_checkpoint=args.checkpoint,
        )
        phase_counts = {}
        for record in trace:
            phase = record["phase"]
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        rows.append({
            "solver": "layered_lbfgs_then_ipf",
            "lbfgs_cap": cap,
            "preconditioned": args.lbfgs_precondition,
            "seconds": time.perf_counter() - started,
            "certificate": certificate(lbfgs),
            "converged": lbfgs.converged,
            "iterations": lbfgs.iterations,
            "margin_evaluations": lbfgs.margin_evaluations,
            "trace_phase_counts": phase_counts,
            "phase_timing": phase_timing,
        })

    started = time.perf_counter()
    newton = sparse_grouped_newton_cg(
        problem, log_base_y=warm[0], correction_ya=warm[1],
        correction_yb=warm[2], tolerance=args.tolerance,
        max_hessian_products=args.newton_products,
    )
    rows.append({
        "solver": "newton_cg",
        "seconds": time.perf_counter() - started,
        "certificate": certificate(newton),
        "converged": newton.converged,
        "iterations": newton.iterations,
        "margin_evaluations": newton.margin_evaluations,
        "hessian_products": newton.hessian_products,
    })
    print(json.dumps({
        "checkpoint": args.checkpoint,
        "initial_certificate": initial_certificate,
        "rows": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
