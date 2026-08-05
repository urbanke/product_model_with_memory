#!/usr/bin/env python3
"""Continue selected persisted sparse calibration checkpoints."""

from __future__ import annotations

import argparse
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
    sparse_gated_log_probabilities,
    sparse_grouped_ipf,
)
from product_model_with_memory.streams import load_stream, reduce_ids


def load_state(path: Path) -> tuple[SparseGroupedProblem, SparseGroupedResult,
                                    np.ndarray, np.ndarray, int]:
    data = np.load(path)
    problem = SparseGroupedProblem(
        vocabulary_size=len(data["target_y"]),
        edge_a=data["edge_a"], edge_b=data["edge_b"],
        edge_probability=data["edge_probability"],
        target_y=data["target_y"],
        active_ya_y=data["active_ya_y"],
        active_ya_a=data["active_ya_a"], target_ya=data["target_ya"],
        active_yb_y=data["active_yb_y"],
        active_yb_b=data["active_yb_b"], target_yb=data["target_yb"],
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
    return (
        problem, result, data["fallback_ya"], data["fallback_yb"],
        int(data["prefix"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--indices", required=True,
                        help="comma-separated zero-based checkpoints")
    parser.add_argument("--ids", default="output/streams/bpe_text8")
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--tolerance", type=float, default=5e-4)
    parser.add_argument("--solver", choices=("ipf", "lbfgs"), default="ipf")
    parser.add_argument("--dense-fallback-mass", type=float, default=0.99)
    parser.add_argument("--margin-workers", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run = Path(args.run)
    result_payload = json.loads((run / "results.json").read_text())
    state_paths = sorted((run / "states").glob("checkpoint_*.npz"))
    ids, _ = load_stream(args.ids)
    x, _, _ = reduce_ids(ids[: result_payload["n"]], args.top_k)
    x = x.astype(np.int64)
    rows = []
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    for index in [int(value) for value in args.indices.split(",")]:
        problem, old, p_ya, p_yb, prefix = load_state(state_paths[index])
        started = time.time()
        new = sparse_grouped_ipf(
            problem,
            max_iterations=args.iterations,
            tolerance=args.tolerance,
            log_base_y=old.log_base_y,
            correction_ya=old.correction_ya,
            correction_yb=old.correction_yb,
            solver=args.solver,
            dense_fallback_mass=args.dense_fallback_mass,
            margin_workers=args.margin_workers,
        )
        row = {
            "index": index,
            "prefix": prefix,
            "seconds": time.time() - started,
            "iterations": new.iterations,
            "converged": new.converged,
            "residual_ya_l1": new.grouped_residual_ya_l1,
            "residual_yb_l1": new.grouped_residual_yb_l1,
            "residual_y_l1": new.residual_y_l1,
        }
        if index + 1 < len(state_paths):
            _, _, _, _, next_prefix = load_state(state_paths[index + 1])
            target = x[prefix:next_prefix]
            lag1 = x[prefix - 1:next_prefix - 1]
            lag2 = x[prefix - 2:next_prefix - 2]
            scale = -1.0 / (len(target) * np.log(2.0))
            old_logp = sparse_gated_log_probabilities(
                problem, old, target, lag1, lag2, p_ya, p_yb
            )
            new_logp = sparse_gated_log_probabilities(
                problem, new, target, lag1, lag2, p_ya, p_yb
            )
            row["old_bpc"] = float(old_logp.sum() * scale)
            row["refit_bpc"] = float(new_logp.sum() * scale)
            row["refit_minus_old_bpc"] = row["refit_bpc"] - row["old_bpc"]
        np.savez_compressed(
            output / f"checkpoint_{index:03d}.npz",
            log_base_y=new.log_base_y,
            correction_ya=new.correction_ya,
            correction_yb=new.correction_yb,
        )
        rows.append(row)
    payload = {
        "source": str(run), "solver": args.solver,
        "tolerance": args.tolerance, "max_iterations": args.iterations,
        "dense_fallback_mass": args.dense_fallback_mass,
        "margin_workers": args.margin_workers,
        "rows": rows,
        "peak_resident_bytes": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
