#!/usr/bin/env python3
"""Assemble independently estimated pairs and AB support into one problem."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from product_model_with_memory.graphical_calibration import (
    SparseProjectedPair, sparse_relaxed_problem_from_layered_pairs,
)


def load_pair(path: Path) -> tuple[np.ndarray, SparseProjectedPair]:
    marginal = np.load(path / "marginal.npy", mmap_mode="r")
    arrays = {name: np.load(path / f"{name}.npy", mmap_mode="r") for name in
              ("left", "right", "background", "active_y",
               "active_context", "delta")}
    return marginal, SparseProjectedPair(len(marginal), **arrays)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--counts", required=True)
    p.add_argument("--ya", required=True)
    p.add_argument("--yb", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    counts = Path(a.counts)
    manifest = json.loads((counts / "manifest.json").read_text())
    marginal_a, p_ya = load_pair(Path(a.ya))
    marginal_b, p_yb = load_pair(Path(a.yb))
    if not np.array_equal(marginal_a, marginal_b):
        raise RuntimeError("independent pair jobs produced different unigrams")
    v = len(marginal_a)
    ab = np.load(counts / "keys_ab.npy", mmap_mode="r")
    problem, retained = sparse_relaxed_problem_from_layered_pairs(
        p_ya, p_yb, ab // v, ab % v, marginal_a,
    )
    state = {"prefix": np.asarray(manifest["prefix"]),
             "margin_preprocessing": np.asarray("raw_relaxed"),
             "edge_a": problem.edge_a, "edge_b": problem.edge_b,
             "edge_probability": problem.edge_probability,
             "target_y": problem.target_y,
             "active_ya_y": problem.active_ya_y,
             "active_ya_a": problem.active_ya_a,
             "target_ya": problem.target_ya,
             "active_yb_y": problem.active_yb_y,
             "active_yb_b": problem.active_yb_b,
             "target_yb": problem.target_yb,
             "log_base_y": np.log(marginal_a),
             "correction_ya": np.zeros(len(problem.target_ya)),
             "correction_yb": np.zeros(len(problem.target_yb))}
    for label, pair in (("ya", p_ya), ("yb", p_yb)):
        for name in ("left", "right", "background", "active_y",
                     "active_context", "delta"):
            state[f"fallback_{label}_{name}"] = getattr(pair, name)
    destination = Path(a.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **state)
    print(json.dumps({"checkpoint": manifest["checkpoint"],
                      "retained_ab_mass": retained,
                      "context_edges": len(problem.edge_probability)}), flush=True)


if __name__ == "__main__":
    main()
