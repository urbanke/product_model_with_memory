#!/usr/bin/env python3
"""Fit a warm-started chain of precomputed three-pair checkpoints."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from product_model_with_memory.graphical_calibration import (
    SparseGroupedProblem,
    SparseGroupedResult,
    pair_product_warm_start,
    sparse_grouped_ipf,
    stochastic_sparse_dual_approach,
    transfer_sparse_warm_start,
)


def current_resident_bytes() -> int | None:
    """Return current macOS RSS, distinct from the lifetime peak."""

    try:
        library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")

        class MachTaskBasicInfo(ctypes.Structure):
            _fields_ = [
                ("virtual_size", ctypes.c_uint64),
                ("resident_size", ctypes.c_uint64),
                ("resident_size_max", ctypes.c_uint64),
                ("user_seconds", ctypes.c_int),
                ("user_microseconds", ctypes.c_int),
                ("system_seconds", ctypes.c_int),
                ("system_microseconds", ctypes.c_int),
                ("policy", ctypes.c_int),
                ("suspend_count", ctypes.c_int),
            ]

        library.mach_task_self.restype = ctypes.c_uint
        information = MachTaskBasicInfo()
        count = ctypes.c_uint(
            ctypes.sizeof(information) // ctypes.sizeof(ctypes.c_int)
        )
        status = library.task_info(
            library.mach_task_self(), 20,
            ctypes.byref(information), ctypes.byref(count),
        )
        return int(information.resident_size) if status == 0 else None
    except (AttributeError, OSError):
        return None


def load_problem(path: Path) -> tuple[SparseGroupedProblem, int]:
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
        prefix = int(data["prefix"])
    return problem, prefix


def persist_fitted_state(
    source: Path, destination: Path, result: SparseGroupedResult
) -> float:
    started = time.perf_counter()
    with np.load(source) as data:
        state = {name: data[name] for name in data.files}
    state.update({
        "log_base_y": result.log_base_y,
        "correction_ya": result.correction_ya,
        "correction_yb": result.correction_yb,
    })
    np.savez(destination, **state)
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--replicas", type=int, default=12)
    parser.add_argument("--edge-blocks", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--minimum-learning-rate", type=float, default=0.003)
    parser.add_argument("--exact-interval", type=int, default=50)
    parser.add_argument("--trust-radius", type=float, default=8.0)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--exact-iterations", type=int, default=4000)
    parser.add_argument("--lbfgs-trust-radius", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()

    source = Path(args.problems)
    paths = sorted((source / "states").glob("checkpoint_*.npz"))
    if not paths:
        parser.error("problem directory contains no checkpoint states")
    metadata = json.loads((source / "results.json").read_text())
    if not metadata.get("construct_only"):
        parser.error("input is not a construction-only checkpoint run")

    destination = Path(args.out)
    state_dir = destination / "states"
    state_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    previous_problem = None
    previous_result = None
    total_started = time.perf_counter()

    for index, path in enumerate(paths):
        rss_before_load = current_resident_bytes()
        problem, prefix = load_problem(path)
        if previous_problem is None:
            warm = pair_product_warm_start(problem)
        else:
            warm = transfer_sparse_warm_start(
                previous_problem, previous_result, problem
            )

        fit_started = time.perf_counter()
        rss_before_fit = current_resident_bytes()
        stochastic = stochastic_sparse_dual_approach(
            problem, *warm,
            steps=args.steps,
            batch_size=1,
            learning_rate=args.learning_rate,
            minimum_learning_rate=args.minimum_learning_rate,
            exact_interval=args.exact_interval,
            seed=args.seed + index,
            trust_radius=args.trust_radius,
            sampling="blocks",
            replicas=args.replicas,
            stochastic_workers=args.workers,
            edge_blocks=args.edge_blocks,
            variance_reduction=True,
            certificate_tolerance=args.tolerance,
            exact_margin_workers=args.workers,
            optimizer="adam_plateau",
        )
        record = min(
            stochastic.trace,
            key=lambda item: float(item["exact_certificate"]),
        )
        result = SparseGroupedResult(
            stochastic.log_base_y,
            stochastic.correction_ya,
            stochastic.correction_yb,
            stochastic.steps,
            float(record["residual_ya_l1"]),
            float(record["residual_yb_l1"]),
            float(record["residual_y_l1"]),
            stochastic.best_exact_certificate <= args.tolerance,
            stochastic.exact_evaluations,
        )
        stochastic_seconds = time.perf_counter() - fit_started
        rss_after_stochastic = current_resident_bytes()
        exact_fallback = False
        exact_seconds = 0.0
        if not result.converged:
            exact_fallback = True
            exact_started = time.perf_counter()
            result = sparse_grouped_ipf(
                problem,
                max_iterations=args.exact_iterations,
                tolerance=args.tolerance,
                log_base_y=result.log_base_y,
                correction_ya=result.correction_ya,
                correction_yb=result.correction_yb,
                solver="lbfgs",
                margin_workers=args.workers,
                evaluator="factorized",
                lbfgs_trust_radius=args.lbfgs_trust_radius,
            )
            exact_seconds = time.perf_counter() - exact_started
        if not result.converged:
            raise RuntimeError(
                f"exact fallback failed at checkpoint {index}; "
                "refusing to propagate an uncertified result"
            )

        persistence_seconds = persist_fitted_state(
            path, state_dir / path.name, result
        )
        certificate = max(
            result.grouped_residual_ya_l1,
            result.grouped_residual_yb_l1,
            result.residual_y_l1,
        )
        row = {
            "checkpoint": index,
            "prefix": prefix,
            "stochastic_seconds": stochastic_seconds,
            "exact_fallback": exact_fallback,
            "exact_seconds": exact_seconds,
            "persistence_seconds": persistence_seconds,
            "stochastic_steps": stochastic.steps,
            "exact_evaluations": stochastic.exact_evaluations,
            "certificate": certificate,
            "sampled_gradient_seconds": stochastic.sampled_gradient_seconds,
            "exact_evaluation_seconds": stochastic.exact_seconds,
            "reference_cache_seconds": stochastic.reference_cache_seconds,
            "optimizer_seconds": stochastic.optimizer_seconds,
            "intersection_plan_bytes": stochastic.intersection_plan_bytes,
            "reference_cache_bytes": stochastic.reference_cache_bytes,
            "rejected_nonfinite_records": sum(
                bool(item["rejected_nonfinite"])
                for item in stochastic.trace
            ),
            "rss_before_load": rss_before_load,
            "rss_before_fit": rss_before_fit,
            "rss_after_stochastic": rss_after_stochastic,
        }
        previous_problem = problem
        previous_result = result
        del warm, stochastic
        gc.collect()
        row["rss_after_release"] = current_resident_bytes()
        rows.append(row)
        print(json.dumps(row), flush=True)

    payload = {
        "problems": str(source),
        "checkpoints": len(paths),
        "workers": args.workers,
        "replicas": args.replicas,
        "edge_blocks": args.edge_blocks,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "minimum_learning_rate": args.minimum_learning_rate,
        "exact_interval": args.exact_interval,
        "trust_radius": args.trust_radius,
        "tolerance": args.tolerance,
        "elapsed_seconds": time.perf_counter() - total_started,
        "peak_resident_bytes": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
        "rows": rows,
    }
    (destination / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"written: {destination / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
