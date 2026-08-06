#!/usr/bin/env python3
"""Audit a single depth-layered intersection graph across checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from product_model_with_memory.graphical_calibration import (
    SparseGroupedProblem,
    build_sparse_intersection_plan,
)


def load_problem(path: Path) -> SparseGroupedProblem:
    with np.load(path, allow_pickle=False) as data:
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


def keys(problem: SparseGroupedProblem) -> tuple[np.ndarray, ...]:
    v = problem.vocabulary_size
    return (
        problem.active_ya_y.astype(np.int64) * v + problem.active_ya_a,
        problem.active_yb_y.astype(np.int64) * v + problem.active_yb_b,
        problem.edge_a.astype(np.int64) * v + problem.edge_b,
    )


def mark_births(
    final_keys: np.ndarray,
    checkpoint_keys: np.ndarray,
    depth: int,
    births: np.ndarray,
) -> None:
    order = np.argsort(final_keys, kind="stable")
    sorted_keys = final_keys[order]
    position = np.searchsorted(sorted_keys, checkpoint_keys)
    if np.any(position == len(sorted_keys)):
        raise RuntimeError("checkpoint support is absent from final support")
    final_position = order[position]
    if not np.array_equal(final_keys[final_position], checkpoint_keys):
        raise RuntimeError("checkpoint support is absent from final support")
    births[final_position] = np.minimum(births[final_position], depth)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    paths = sorted((Path(args.run) / "states").glob("checkpoint_*.npz"))
    if not paths:
        parser.error("run contains no checkpoint states")
    final = load_problem(paths[-1])
    final_keys = keys(final)
    births = [np.full(len(value), 255, dtype=np.uint8) for value in final_keys]
    previous_sets: tuple[set[int], ...] | None = None
    support_rows = []
    for depth, path in enumerate(paths):
        problem = load_problem(path)
        if problem.vocabulary_size != final.vocabulary_size:
            raise RuntimeError("vocabulary changes across checkpoints")
        checkpoint_keys = keys(problem)
        current_sets = tuple(set(map(int, value)) for value in checkpoint_keys)
        monotone = (
            previous_sets is None
            or all(old <= new for old, new in zip(previous_sets, current_sets))
        )
        if not monotone:
            raise RuntimeError(f"support is not monotone at checkpoint {depth}")
        for final_key, checkpoint_key, birth in zip(
            final_keys, checkpoint_keys, births
        ):
            mark_births(final_key, checkpoint_key, depth, birth)
        support_rows.append({
            "checkpoint": depth,
            "ya_edges": len(checkpoint_keys[0]),
            "yb_edges": len(checkpoint_keys[1]),
            "ab_edges": len(checkpoint_keys[2]),
        })
        previous_sets = current_sets
    if any(np.any(birth == 255) for birth in births):
        raise RuntimeError("some final-support edges have no birth checkpoint")

    plan = build_sparse_intersection_plan(final)
    triangle_birth = np.maximum.reduce([
        births[0][plan.correction_ya],
        births[1][plan.correction_yb],
        births[2][plan.edge],
    ])
    triangle_birth_counts = np.bincount(
        triangle_birth, minlength=len(paths)
    )
    triangles = len(plan.edge)
    ya_degree = np.bincount(plan.correction_ya, minlength=len(final.target_ya))
    yb_degree = np.bincount(plan.correction_yb, minlength=len(final.target_yb))
    blue_by_y = np.bincount(
        final.active_ya_y, minlength=final.vocabulary_size
    ).astype(object)
    red_by_y = np.bincount(
        final.active_yb_y, minlength=final.vocabulary_size
    ).astype(object)
    candidate_paths = int(sum(blue_by_y * red_by_y))
    current_bytes = 16 * triangles
    payload = {
        "run": args.run,
        "checkpoints": len(paths),
        "vocabulary_size": final.vocabulary_size,
        "support_monotone": True,
        "final_support": {
            "ya_edges": len(final.target_ya),
            "yb_edges": len(final.target_yb),
            "ab_edges": len(final.edge_probability),
        },
        "candidate_blue_red_paths": candidate_paths,
        "retained_triangles": triangles,
        "retained_fraction_of_candidate_paths": (
            triangles / candidate_paths if candidate_paths else 0.0
        ),
        "triangle_birth_counts": triangle_birth_counts.tolist(),
        "triangle_cumulative_counts": np.cumsum(
            triangle_birth_counts
        ).tolist(),
        "degree": {
            "ya_node_max": int(ya_degree.max(initial=0)),
            "yb_node_max": int(yb_degree.max(initial=0)),
            "ya_node_mean": float(ya_degree.mean()),
            "yb_node_mean": float(yb_degree.mean()),
        },
        "storage_bytes": {
            "current_four_int32": current_bytes,
            "three_int32": 12 * triangles,
            "node_major_two_int32_plus_uint8_depth_and_int64_rowptr": (
                9 * triangles + 8 * (len(final.target_ya) + 1)
            ),
            "layered_csr_two_int32_plus_32_int64_rowptr": (
                8 * triangles
                + 8 * len(paths) * (len(final.target_ya) + 1)
            ),
        },
        "supports": support_rows,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
