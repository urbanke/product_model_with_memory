#!/usr/bin/env python3
"""Publish one append-only exact/AB-major graph delta for checkpoint k."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intersection_topology_audit import load_problem
from product_model_with_memory.graphical_calibration import (
    append_checkpoint_support,
    build_sparse_intersection_delta_incremental,
    save_sparse_intersection_delta,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    args = parser.parse_args()
    if args.checkpoint < 0:
        parser.error("checkpoint must be nonnegative")

    problem_paths = sorted(
        (Path(args.problems) / "states").glob("checkpoint_*.npz")
    )
    if args.checkpoint >= len(problem_paths):
        parser.error("requested checkpoint problem has not been published")
    destination = (
        Path(args.store) / "deltas" / f"checkpoint_{args.checkpoint:03d}"
    )
    if (destination / "manifest.json").exists():
        print(json.dumps({
            "checkpoint": args.checkpoint,
            "reused": True,
            "delta": str(destination),
        }))
        return

    state = None
    support = None
    for checkpoint, path in enumerate(problem_paths[:args.checkpoint + 1]):
        state, support = append_checkpoint_support(
            state, load_problem(path), checkpoint
        )
    delta = build_sparse_intersection_delta_incremental(
        support.problem, support.birth_ya, support.birth_yb,
        support.birth_ab, args.checkpoint,
    )

    support_dir = Path(args.store) / "support"
    support_dir.mkdir(parents=True, exist_ok=True)
    support_path = support_dir / f"checkpoint_{args.checkpoint:03d}.npz"
    temporary_support = support_path.with_suffix(".npz.tmp")
    with temporary_support.open("wb") as output:
        np.savez(
            output,
            vocabulary_size=np.asarray(support.problem.vocabulary_size),
            keys_ya=state.keys_ya,
            keys_yb=state.keys_yb,
            keys_ab=state.keys_ab,
            birth_ya=state.birth_ya,
            birth_yb=state.birth_yb,
            birth_ab=state.birth_ab,
        )
    temporary_support.replace(support_path)
    save_sparse_intersection_delta(delta, destination)
    print(json.dumps({
        "checkpoint": args.checkpoint,
        "triangles": delta.triangles,
        "bytes": delta.nbytes,
        "support": str(support_path),
        "delta": str(destination),
    }))


if __name__ == "__main__":
    main()
