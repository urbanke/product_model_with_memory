#!/usr/bin/env python3
"""Fit one saved shared-graph checkpoint by exact strong-Wolfe search."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intersection_topology_audit import load_problem
from validate_layered_checkpoint_store import reorder_values
from product_model_with_memory.graphical_calibration import (
    BirthMajorSparseSupport,
    SparseGroupedProblem,
    checkpoint_in_birth_major_support,
    exact_sparse_dual_wolfe,
    load_layered_intersection_graph,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", type=int, default=0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--tolerance", type=float, default=1e-2)
    parser.add_argument("--progress-interval", type=int, default=1)
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
    original = load_problem(problem_path)
    problem = checkpoint_in_birth_major_support(
        original, support, args.checkpoint
    )
    with np.load(args.state, allow_pickle=False) as state:
        factors = (
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

    started = time.perf_counter()

    def progress(row):
        iteration = int(row["iteration"])
        if iteration % args.progress_interval == 0:
            print(json.dumps({
                **row, "elapsed_seconds": time.perf_counter() - started,
            }), flush=True)

    result = exact_sparse_dual_wolfe(
        problem, *factors,
        max_iterations=args.iterations,
        tolerance=args.tolerance,
        margin_workers=args.workers,
        layered_graph=load_layered_intersection_graph(store / "graph"),
        layered_checkpoint=args.checkpoint,
        progress_callback=progress,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / f"checkpoint_{args.checkpoint:03d}.npz"
    np.savez(
        state_path,
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
        log_base_y=result.log_base_y,
        correction_ya=result.correction_ya,
        correction_yb=result.correction_yb,
        certificate=result.certificate,
    )
    final_state_path = out / f"checkpoint_{args.checkpoint:03d}_final.npz"
    np.savez(
        final_state_path,
        active_ya_y=problem.active_ya_y,
        active_ya_a=problem.active_ya_a,
        active_yb_y=problem.active_yb_y,
        active_yb_b=problem.active_yb_b,
        log_base_y=result.final_log_base_y,
        correction_ya=result.final_correction_ya,
        correction_yb=result.final_correction_yb,
        certificate=result.final_certificate,
    )
    print(json.dumps({
        "checkpoint": args.checkpoint,
        "iterations": result.iterations,
        "evaluations": result.evaluations,
        "objective": result.objective,
        "certificate": result.certificate,
        "final_objective": result.final_objective,
        "final_certificate": result.final_certificate,
        "converged": result.converged,
        "elapsed_seconds": time.perf_counter() - started,
        "state": str(state_path),
        "final_state": str(final_state_path),
    }), flush=True)


if __name__ == "__main__":
    main()
