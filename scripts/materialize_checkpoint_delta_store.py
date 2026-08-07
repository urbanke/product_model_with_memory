#!/usr/bin/env python3
"""Materialize graph deltas through k in the legacy fitter store format."""

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
from product_model_with_memory.graphical_calibration import (
    append_checkpoint_support,
    load_sparse_intersection_delta,
    save_ab_major_intersection_graph,
    save_layered_intersection_graph,
    sparse_deltas_as_ab_major_graph,
    sparse_deltas_as_layered_graph,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--delta-store", required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    destination = Path(args.out)
    if (destination / "manifest.json").exists():
        print(json.dumps({"checkpoint": args.checkpoint, "reused": True}))
        return

    paths = sorted((Path(args.problems) / "states").glob("checkpoint_*.npz"))
    if not 0 <= args.checkpoint < len(paths):
        parser.error("checkpoint problem is unavailable")
    started = time.perf_counter()
    state = None
    support = None
    for checkpoint, path in enumerate(paths[:args.checkpoint + 1]):
        state, support = append_checkpoint_support(
            state, load_problem(path), checkpoint
        )
    deltas = tuple(load_sparse_intersection_delta(
        Path(args.delta_store) / "deltas" / f"checkpoint_{checkpoint:03d}"
    ) for checkpoint in range(args.checkpoint + 1))
    exact = sparse_deltas_as_layered_graph(
        deltas, len(support.problem.target_ya)
    )
    ab_major = sparse_deltas_as_ab_major_graph(
        deltas, len(support.problem.edge_probability)
    )

    destination.mkdir(parents=True, exist_ok=True)
    save_layered_intersection_graph(exact, destination / "graph")
    save_ab_major_intersection_graph(ab_major, destination / "ab_graph")
    support_dir = destination / "support"
    support_dir.mkdir(exist_ok=True)
    arrays = {
        "edge_a": support.problem.edge_a,
        "edge_b": support.problem.edge_b,
        "target_y": support.problem.target_y,
        "active_ya_y": support.problem.active_ya_y,
        "active_ya_a": support.problem.active_ya_a,
        "active_yb_y": support.problem.active_yb_y,
        "active_yb_b": support.problem.active_yb_b,
        "birth_ya": support.birth_ya,
        "birth_yb": support.birth_yb,
        "birth_ab": support.birth_ab,
    }
    for name, array in arrays.items():
        np.save(support_dir / f"{name}.npy", array)
    temporary = destination / ".manifest.json.tmp"
    temporary.write_text(json.dumps({
        "version": 1,
        "problems": args.problems,
        "checkpoints": args.checkpoint + 1,
        "vocabulary_size": support.problem.vocabulary_size,
        "ya_edges": len(support.problem.target_ya),
        "yb_edges": len(support.problem.target_yb),
        "ab_edges": len(support.problem.edge_probability),
        "triangles": exact.edges,
        "graph_bytes": exact.nbytes,
        "ab_major_bytes": ab_major.nbytes,
        "materialization_seconds": time.perf_counter() - started,
        "compatibility_bridge": True,
    }, indent=2))
    temporary.replace(destination / "manifest.json")
    print((destination / "manifest.json").read_text())


if __name__ == "__main__":
    main()
