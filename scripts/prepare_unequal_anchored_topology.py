#!/usr/bin/env python3
"""Publish a full layered topology with an explicitly estimated AB law."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from product_model_with_memory.anchored_pair_graph import (
    ANCHORED_PAIR_GRAPH_MODEL,
    FullImplicitPairProblem,
    full_implicit_pair_problem_explicit_ab,
)
from product_model_with_memory.graphical_calibration import (
    SparseGroupedProblem,
    SparseProjectedPair,
)
from product_model_with_memory.production_coding import require_production_sequence_estimator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source = Path(args.problem).resolve()
    destination = Path(args.out).resolve()
    source_hash = sha256(source)
    with np.load(source, allow_pickle=False) as data:
        require_production_sequence_estimator(str(data["sequence_estimator"]), source=str(source))
        v = len(data["target_y"])
        problem = SparseGroupedProblem(
            vocabulary_size=v, edge_a=data["edge_a"], edge_b=data["edge_b"],
            edge_probability=data["edge_probability"], target_y=data["target_y"],
            active_ya_y=data["active_ya_y"], active_ya_a=data["active_ya_a"],
            target_ya=data["target_ya"], active_yb_y=data["active_yb_y"],
            active_yb_b=data["active_yb_b"], target_yb=data["target_yb"],
        )
        def pair(label):
            return SparseProjectedPair(v, *[
                np.array(data[f"fallback_{label}_{name}"]) for name in
                ("left", "right", "background", "active_y", "active_context", "delta")
            ])
        p_ya, p_yb, p_ab = pair("ya"), pair("yb"), pair("ab")
        prefix = int(data["prefix"])
        dimensions = {name: int(data[name]) for name in
                      ("emission_vocabulary_size", "first_lag_parameter",
                       "second_lag_parameter", "first_lag_alphabet_size",
                       "second_lag_alphabet_size")}
    topology = full_implicit_pair_problem_explicit_ab(problem, p_ya, p_yb, p_ab)
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for field in fields(FullImplicitPairProblem):
        if field.name == "vocabulary_size":
            continue
        final = destination / f"{field.name}.npy"
        with (destination / f".{field.name}.npy.tmp").open("wb") as output:
            np.save(output, getattr(topology, field.name))
        (destination / f".{field.name}.npy.tmp").replace(final)
        hashes[field.name] = sha256(final)
    manifest = {
        "version": 4, "model": ANCHORED_PAIR_GRAPH_MODEL,
        "topology": "full_layered_ya_explicit_ab_unequal_v1",
        "vocabulary_size": v, "problem_sha256": source_hash,
        "prefix": prefix, "yb_factors": len(topology.target_yb),
        "ab_factors": len(topology.target_ab), "array_sha256": hashes,
        **dimensions,
    }
    (destination / ".manifest.json.tmp").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (destination / ".manifest.json.tmp").replace(destination / "manifest.json")
    print(json.dumps({"out": str(destination), **manifest}))


if __name__ == "__main__":
    main()
