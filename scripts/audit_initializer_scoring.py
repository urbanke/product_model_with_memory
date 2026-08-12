#!/usr/bin/env python3
"""Compare fitted and unfitted initializers on scheduled score intervals.

This is an audit utility.  It reads an existing scheduled run and writes one
separate JSON report; it does not modify fitted states, score files, or the
production accounting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from calibration_score_states import load_state
from product_model_with_memory.graphical_calibration import (
    SparseGroupedResult,
    first_pair_warm_start,
    pair_product_warm_start,
    sparse_gated_log_probabilities,
    sparse_pair_log_probabilities,
)


def as_result(factors) -> SparseGroupedResult:
    return SparseGroupedResult(
        *factors, 0, np.nan, np.nan, np.nan, True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoints", default="all")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    reduced = np.load(root / "reduced_stream" / "stream.npy", mmap_mode="r")
    if args.checkpoints == "all":
        checkpoints = range(31)
    else:
        checkpoints = [int(value) for value in args.checkpoints.split(",")]

    rows = []
    totals = {name: 0.0 for name in ("fitted", "pair_product", "first_pair")}
    total_records = 0
    for checkpoint in checkpoints:
        state = (
            root / "fitted" / f"checkpoint_{checkpoint:03d}" / "states"
            / f"checkpoint_{checkpoint:03d}.npz"
        )
        next_problem = (
            root / "problems" / "states"
            / f"checkpoint_{checkpoint + 1:03d}.npz"
        )
        problem, fitted, p_ya, p_yb, prefix = load_state(state)
        with np.load(next_problem, allow_pickle=False) as saved:
            next_prefix = int(saved["prefix"])
        target = np.asarray(reduced[prefix:next_prefix], dtype=np.int64)
        lag1 = np.asarray(reduced[prefix - 1:next_prefix - 1], dtype=np.int64)
        lag2 = np.asarray(reduced[prefix - 2:next_prefix - 2], dtype=np.int64)
        results = {
            "fitted": fitted,
            "pair_product": as_result(pair_product_warm_start(problem)),
            "first_pair": as_result(first_pair_warm_start(problem)),
        }
        bits = {}
        for name, result in results.items():
            log_probability = sparse_gated_log_probabilities(
                problem, result, target, lag1, lag2, p_ya, p_yb,
            )
            bits[name] = -float(log_probability.sum()) / np.log(2.0)
            totals[name] += bits[name]
        pair1_bits = -float(sparse_pair_log_probabilities(
            p_ya, target, lag1,
        ).sum()) / np.log(2.0)
        total_records += len(target)
        rows.append({
            "checkpoint": checkpoint,
            "records": len(target),
            "bits": bits,
            "pair1_bits": pair1_bits,
            "bits_per_token": {
                name: value / len(target) for name, value in bits.items()
            },
            "pair1_bits_per_token": pair1_bits / len(target),
        })
        print(json.dumps(rows[-1]), flush=True)

    payload = {
        "version": 1,
        "root": str(root),
        "records": total_records,
        "totals": totals,
        "bits_per_token": {
            name: value / total_records for name, value in totals.items()
        },
        "rows": rows,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "written": str(destination),
        "records": total_records,
        "bits_per_token": payload["bits_per_token"],
    }, indent=2))


if __name__ == "__main__":
    main()
