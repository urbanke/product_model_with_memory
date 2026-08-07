#!/usr/bin/env python3
"""Audit whether checkpoint supports admit append-only graph deltas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def sorted_keys(left: np.ndarray, right: np.ndarray, vocabulary_size: int):
    keys = (
        np.asarray(left, dtype=np.int64) * vocabulary_size
        + np.asarray(right, dtype=np.int64)
    )
    return np.unique(keys)


def missing_from(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Return keys present previously but absent from the current support."""

    positions = np.searchsorted(current, previous)
    present = positions < len(current)
    present[present] &= current[positions[present]] == previous[present]
    return previous[~present]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", required=True)
    parser.add_argument(
        "--store",
        help="optional monolithic layered graph whose birth arrays are counted",
    )
    parser.add_argument("--out")
    args = parser.parse_args()

    paths = sorted((Path(args.problems) / "states").glob("checkpoint_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no checkpoint states below {args.problems}")

    rows = []
    previous = None
    vocabulary_size = None
    monotone = {"ya": True, "yb": True, "ab": True}
    for checkpoint, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as state:
            current_v = len(state["target_y"])
            if vocabulary_size is None:
                vocabulary_size = current_v
            elif current_v != vocabulary_size:
                raise ValueError("vocabulary size changes between checkpoints")
            current = {
                "ya": sorted_keys(
                    state["active_ya_y"], state["active_ya_a"], current_v
                ),
                "yb": sorted_keys(
                    state["active_yb_y"], state["active_yb_b"], current_v
                ),
                "ab": sorted_keys(state["edge_a"], state["edge_b"], current_v),
            }
            prefix = int(state["prefix"])

        row = {"checkpoint": checkpoint, "prefix": prefix}
        for name, keys in current.items():
            removed = (
                np.empty(0, dtype=np.int64)
                if previous is None else missing_from(previous[name], keys)
            )
            added = (
                keys if previous is None else missing_from(keys, previous[name])
            )
            monotone[name] &= len(removed) == 0
            row[f"{name}_edges"] = len(keys)
            row[f"new_{name}_edges"] = len(added)
            row[f"removed_{name}_edges"] = len(removed)
            if len(removed):
                row[f"first_removed_{name}_keys"] = removed[:10].tolist()
        rows.append(row)
        previous = current

    graph_births = None
    if args.store:
        store = Path(args.store)
        birth_paths = {
            "ya": store / "support" / "birth_ya.npy",
            "yb": store / "support" / "birth_yb.npy",
            "ab": store / "support" / "birth_ab.npy",
        }
        graph_births = {}
        for name, path in birth_paths.items():
            if not path.exists():
                graph_births[name] = {"missing": str(path)}
                continue
            values = np.load(path, mmap_mode="r")
            counts = np.bincount(
                np.asarray(values, dtype=np.int64), minlength=len(paths)
            )
            graph_births[name] = {
                "total": len(values),
                "new_by_checkpoint": counts[:len(paths)].tolist(),
            }
        exact_layers = [
            store / "graph" / f"correction_yb_{depth:03d}.npy"
            for depth in range(len(paths))
        ]
        if all(path.exists() for path in exact_layers):
            triangle_counts = [
                len(np.load(path, mmap_mode="r")) for path in exact_layers
            ]
            graph_births["triangles"] = {
                "total": sum(triangle_counts),
                "new_by_checkpoint": triangle_counts,
                "representation": "physical exact-graph birth layers",
            }
        else:
            ab_birth = store / "ab_graph" / "birth.npy"
            if ab_birth.exists():
                values = np.load(ab_birth, mmap_mode="r")
                counts = np.bincount(
                    np.asarray(values, dtype=np.int64), minlength=len(paths)
                )
                graph_births["triangles"] = {
                    "total": len(values),
                    "new_by_checkpoint": counts[:len(paths)].tolist(),
                    "representation": "AB-major triangle birth labels",
                }

    payload = {
        "problems": args.problems,
        "store": args.store,
        "checkpoints": len(paths),
        "vocabulary_size": vocabulary_size,
        "append_only_supports": all(monotone.values()),
        "monotone": monotone,
        "rows": rows,
        "monolithic_graph_births": graph_births,
    }
    rendered = json.dumps(payload, indent=2)
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(rendered)
        temporary.replace(destination)
    print(rendered)


if __name__ == "__main__":
    main()
