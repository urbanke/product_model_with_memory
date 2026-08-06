#!/usr/bin/env python3
"""Build one memory-mappable birth-layered graph for a checkpoint chain."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intersection_topology_audit import keys, load_problem, mark_births
from product_model_with_memory.graphical_calibration import (
    birth_major_sparse_support,
    build_ab_major_intersection_graph,
    build_layered_intersection_graph,
    save_layered_intersection_graph,
    save_ab_major_intersection_graph,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ab-major", action="store_true")
    args = parser.parse_args()

    source = Path(args.problems)
    paths = sorted((source / "states").glob("checkpoint_*.npz"))
    if not paths:
        parser.error("problem run contains no checkpoint states")
    final = load_problem(paths[-1])
    final_keys = keys(final)
    births = [np.full(len(value), 255, dtype=np.uint8) for value in final_keys]
    for depth, path in enumerate(paths):
        checkpoint_keys = keys(load_problem(path))
        for final_key, checkpoint_key, birth in zip(
            final_keys, checkpoint_keys, births
        ):
            mark_births(final_key, checkpoint_key, depth, birth)
    if any(np.any(birth == 255) for birth in births):
        raise RuntimeError("some final pair edges have no birth checkpoint")

    support = birth_major_sparse_support(final, *births)
    started = time.perf_counter()
    graph = build_layered_intersection_graph(
        support.problem,
        support.birth_ya, support.birth_yb, support.birth_ab,
        layers=len(paths),
    )
    construction_seconds = time.perf_counter() - started

    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    save_started = time.perf_counter()
    save_layered_intersection_graph(graph, destination / "graph")
    ab_graph = None
    ab_construction_seconds = 0.0
    if args.ab_major:
        ab_started = time.perf_counter()
        ab_graph = build_ab_major_intersection_graph(
            support.problem,
            support.birth_ya, support.birth_yb, support.birth_ab,
        )
        ab_construction_seconds = time.perf_counter() - ab_started
        save_ab_major_intersection_graph(ab_graph, destination / "ab_graph")
    problem = support.problem
    arrays = {
        "edge_a": problem.edge_a,
        "edge_b": problem.edge_b,
        "target_y": problem.target_y,
        "active_ya_y": problem.active_ya_y,
        "active_ya_a": problem.active_ya_a,
        "active_yb_y": problem.active_yb_y,
        "active_yb_b": problem.active_yb_b,
        "birth_ya": support.birth_ya,
        "birth_yb": support.birth_yb,
        "birth_ab": support.birth_ab,
    }
    support_dir = destination / "support"
    support_dir.mkdir(exist_ok=True)
    for name, array in arrays.items():
        np.save(support_dir / f"{name}.npy", array)
    persistence_seconds = time.perf_counter() - save_started
    payload = {
        "version": 1,
        "problems": str(source),
        "checkpoints": len(paths),
        "vocabulary_size": problem.vocabulary_size,
        "ya_edges": len(problem.target_ya),
        "yb_edges": len(problem.target_yb),
        "ab_edges": len(problem.edge_probability),
        "triangles": graph.edges,
        "graph_bytes": graph.nbytes,
        "construction_seconds": construction_seconds,
        "persistence_seconds": persistence_seconds,
        "ab_major_bytes": None if ab_graph is None else ab_graph.nbytes,
        "ab_major_construction_seconds": ab_construction_seconds,
    }
    temporary = destination / "manifest.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(destination / "manifest.json")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
