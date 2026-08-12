#!/usr/bin/env python3
"""Publish an immutable intersection topology for one hard-YA checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from product_model_with_memory.graphical_calibration import (
    SparseGroupedProblem,
    SparseProjectedPair,
)
from product_model_with_memory.anchored_pair_graph import (
    ANCHORED_PAIR_GRAPH_MODEL,
    FullImplicitPairProblem,
    full_implicit_pair_problem,
)
from product_model_with_memory.production_coding import (
    require_production_sequence_estimator,
)


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
    manifest_path = destination / "manifest.json"
    source_hash = sha256(source)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("version") != 3:
            raise RuntimeError("existing topology predates full layered AB support")
        if manifest.get("problem_sha256") != source_hash:
            raise RuntimeError("topology belongs to a different checkpoint problem")
        for name, expected in manifest["array_sha256"].items():
            if sha256(destination / f"{name}.npy") != expected:
                raise RuntimeError("persisted topology array is corrupt")
        print(json.dumps({"reused": True, "out": str(destination)}))
        return
    with np.load(source, allow_pickle=False) as data:
        require_production_sequence_estimator(
            str(data["sequence_estimator"]), source=str(source),
        )
        problem = SparseGroupedProblem(
            vocabulary_size=len(data["target_y"]),
            edge_a=data["edge_a"], edge_b=data["edge_b"],
            edge_probability=data["edge_probability"],
            target_y=data["target_y"],
            active_ya_y=data["active_ya_y"], active_ya_a=data["active_ya_a"],
            target_ya=data["target_ya"],
            active_yb_y=data["active_yb_y"], active_yb_b=data["active_yb_b"],
            target_yb=data["target_yb"],
        )
        def pair(label):
            return SparseProjectedPair(problem.vocabulary_size, *[
                np.array(data[f"fallback_{label}_{name}"]) for name in (
                "left", "right", "background", "active_y",
                "active_context", "delta",
            )])
        p_ya, p_yb = pair("ya"), pair("yb")
        prefix = int(data["prefix"])
    topology = full_implicit_pair_problem(problem, p_ya, p_yb)
    destination.mkdir(parents=True, exist_ok=True)
    arrays = {
        field.name: getattr(topology, field.name)
        for field in fields(FullImplicitPairProblem)
        if field.name != "vocabulary_size"
    }
    hashes = {}
    for name, values in arrays.items():
        final = destination / f"{name}.npy"
        temporary = destination / f".{name}.npy.tmp"
        with temporary.open("wb") as output:
            np.save(output, values)
        temporary.replace(final)
        hashes[name] = sha256(final)
    manifest = {
        "version": 3,
        "model": ANCHORED_PAIR_GRAPH_MODEL,
        "topology": "full_layered_ya_ab_sparse_factors_v1",
        "vocabulary_size": topology.vocabulary_size,
        "problem_sha256": source_hash,
        "prefix": prefix,
        "yb_factors": len(topology.target_yb),
        "ab_factors": len(topology.target_ab),
        "array_sha256": hashes,
    }
    temporary = destination / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    temporary.replace(manifest_path)
    print(json.dumps({"reused": False, "out": str(destination), **manifest}))


if __name__ == "__main__":
    main()
