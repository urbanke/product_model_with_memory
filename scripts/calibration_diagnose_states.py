#!/usr/bin/env python3
"""Trace convergence and factor growth from persisted sparse checkpoints."""

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
    sparse_grouped_ipf,
)


def load_problem(data: np.lib.npyio.NpzFile) -> SparseGroupedProblem:
    """Load the problem arrays shared by dense- and sparse-upstream states."""

    return SparseGroupedProblem(
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


def summarize_trace(trace: list[dict]) -> dict:
    certificates = np.array([row["certificate"] for row in trace])
    factors = np.array([row["factor_abs_p99"] for row in trace])
    best = np.minimum.accumulate(certificates)
    tail = best[max(0, len(best) // 2):]
    relative_tail_improvement = (
        float((tail[0] - tail[-1]) / max(tail[0], np.finfo(float).tiny))
        if len(tail) > 1 else 0.0
    )
    limiting = {}
    for row in trace:
        name = row["limiting_margin"]
        limiting[name] = limiting.get(name, 0) + 1
    return {
        "trace_points": len(trace),
        "initial_certificate": float(certificates[0]),
        "final_certificate": float(certificates[-1]),
        "best_certificate": float(best[-1]),
        "relative_tail_improvement": relative_tail_improvement,
        "plateau_suspected": bool(
            len(tail) >= 3 and relative_tail_improvement < 0.01
        ),
        "initial_factor_abs_p99": float(factors[0]),
        "final_factor_abs_p99": float(factors[-1]),
        "max_factor_abs": float(max(row["factor_abs_max"] for row in trace)),
        "limiting_margin_counts": limiting,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--indices", default="all",
        help="comma-separated zero-based checkpoints, or all",
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--trace-interval", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--margin-workers", type=int, default=1)
    parser.add_argument(
        "--evaluator", choices=("union", "factorized", "auto"),
        default="union"
    )
    parser.add_argument("--lbfgs-trust-radius", type=float, default=16.0)
    parser.add_argument(
        "--solver", choices=("ipf", "lbfgs"), default="ipf"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = sorted((Path(args.run) / "states").glob("checkpoint_*.npz"))
    indices = (
        list(range(len(paths))) if args.indices == "all"
        else [int(value) for value in args.indices.split(",")]
    )
    rows = []
    for index in indices:
        with np.load(paths[index]) as data:
            problem = load_problem(data)
            trace: list[dict] = []
            phase_timing: dict[str, float] = {}
            started = time.time()
            result = sparse_grouped_ipf(
                problem,
                max_iterations=args.iterations,
                tolerance=args.tolerance,
                log_base_y=data["log_base_y"],
                correction_ya=data["correction_ya"],
                correction_yb=data["correction_yb"],
                solver=args.solver,
                trace=trace,
                trace_interval=args.trace_interval,
                margin_workers=args.margin_workers,
                evaluator=args.evaluator,
                lbfgs_trust_radius=args.lbfgs_trust_radius,
                phase_timing=phase_timing,
            )
            prefix = int(data["prefix"])
        rows.append({
            "index": index,
            "prefix": prefix,
            "seconds": time.time() - started,
            "continued_iterations": result.iterations,
            "margin_evaluations": result.margin_evaluations,
            "continued_converged": result.converged,
            "summary": summarize_trace(trace),
            "phase_timing": phase_timing,
            "trace": trace,
        })
    payload = {
        "source": args.run,
        "solver": args.solver,
        "iterations": args.iterations,
        "trace_interval": args.trace_interval,
        "tolerance": args.tolerance,
        "margin_workers": args.margin_workers,
        "evaluator": args.evaluator,
        "lbfgs_trust_radius": args.lbfgs_trust_radius,
        "rows": rows,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "source": args.run,
        "rows": [
            {"index": row["index"], **row["summary"]} for row in rows
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
