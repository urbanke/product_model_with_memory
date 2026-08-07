#!/usr/bin/env python3
"""Run the exact large sparse feasibility LP for one shared checkpoint."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intersection_topology_audit import load_problem
from product_model_with_memory.graphical_calibration import (
    BirthMajorSparseSupport,
    SparseGroupedProblem,
    check_grouped_feasibility_lp,
    checkpoint_in_birth_major_support,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--checkpoint", type=int, default=0)
    parser.add_argument("--max-variables", type=int, default=100_000_000)
    parser.add_argument("--time-limit", type=float, default=21_600.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    store = Path(args.store)
    manifest = json.loads((store / "manifest.json").read_text())
    support_dir = store / "support"

    def load(name):
        return np.load(
            support_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False
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
    problem_path = (
        Path(args.problems) / "states"
        / f"checkpoint_{args.checkpoint:03d}.npz"
    )
    problem = checkpoint_in_birth_major_support(
        load_problem(problem_path), support, args.checkpoint
    )
    print(json.dumps({
        "phase": "start",
        "checkpoint": args.checkpoint,
        "vocabulary_size": problem.vocabulary_size,
        "retained_ab_edges": len(problem.edge_probability),
        "variables": problem.vocabulary_size * len(problem.edge_probability),
        "time_limit": args.time_limit,
    }), flush=True)
    started = time.perf_counter()
    result = check_grouped_feasibility_lp(
        problem,
        max_variables=args.max_variables,
        time_limit=args.time_limit,
        progress=True,
    )
    row = {
        **asdict(result),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_resident_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
