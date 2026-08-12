#!/usr/bin/env python3
"""Fit one hard-YA checkpoint from an immutable sparse problem artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from product_model_with_memory.anchored_pair_graph import (
    ANCHORED_PAIR_GRAPH_INITIALIZER,
    ANCHORED_PAIR_GRAPH_MODEL,
    FullImplicitPairProblem,
    cold_pair_midpoint_from_projected,
    full_implicit_pair_problem,
    full_implicit_pair_sgd,
    full_implicit_validation_objective,
)
from product_model_with_memory.graphical_calibration import (
    SparseGroupedProblem,
    SparseProjectedPair,
)
from product_model_with_memory.production_coding import (
    PRODUCTION_SEQUENCE_ESTIMATOR,
    require_production_sequence_estimator,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_problem(path: Path):
    with np.load(path, allow_pickle=False) as data:
        problem = SparseGroupedProblem(
            vocabulary_size=len(data["target_y"]),
            edge_a=np.array(data["edge_a"]),
            edge_b=np.array(data["edge_b"]),
            edge_probability=np.array(data["edge_probability"]),
            target_y=np.array(data["target_y"]),
            active_ya_y=np.array(data["active_ya_y"]),
            active_ya_a=np.array(data["active_ya_a"]),
            target_ya=np.array(data["target_ya"]),
            active_yb_y=np.array(data["active_yb_y"]),
            active_yb_b=np.array(data["active_yb_b"]),
            target_yb=np.array(data["target_yb"]),
        )
        estimator = str(data["sequence_estimator"])
        require_production_sequence_estimator(estimator, source=str(path))
        def pair(label):
            return SparseProjectedPair(
                problem.vocabulary_size,
                *[np.array(data[f"fallback_{label}_{name}"]) for name in (
                    "left", "right", "background", "active_y",
                    "active_context", "delta",
                )],
            )
        p_ya, p_yb = pair("ya"), pair("yb")
        prefix = int(data["prefix"])
    return problem, p_ya, p_yb, prefix, estimator


def empirical_variance(probability: np.ndarray, sample_size: int) -> np.ndarray:
    inverse = 1.0 / float(sample_size)
    return np.maximum(probability * (1.0 - probability) * inverse, inverse * inverse)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--topology")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--exact-interval", type=int, default=50)
    parser.add_argument("--validation-seed", type=int, default=700001)
    parser.add_argument("--validation-samples", type=int, default=4096)
    parser.add_argument("--slack-precision", type=float, default=1.0)
    args = parser.parse_args()

    source = Path(args.problem).resolve()
    destination = Path(args.out).resolve()
    state_path = destination / "state.npz"
    manifest_path = destination / "manifest.json"
    source_hash = sha256(source)
    configuration = {
        "version": 1,
        "model": ANCHORED_PAIR_GRAPH_MODEL,
        "sequence_estimator": PRODUCTION_SEQUENCE_ESTIMATOR,
        "initialization": ANCHORED_PAIR_GRAPH_INITIALIZER,
        "warm_start": False,
        "problem_sha256": source_hash,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "exact_interval": args.exact_interval,
        "validation_seed": args.validation_seed,
        "validation_samples": args.validation_samples,
        "slack_precision": args.slack_precision,
        "iterate_selection": "best_fixed_validation_objective_v1",
        "sample_schedule": "seed_step_batch_size_v1",
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        comparable = {key: old.get(key) for key in configuration}
        if comparable != configuration:
            raise RuntimeError("completed fit artifact has different configuration")
        if not state_path.exists() or sha256(state_path) != old.get("state_sha256"):
            raise RuntimeError("completed fit artifact is incomplete or corrupt")
        print(json.dumps({"reused": True, "out": str(destination)}))
        return

    problem, p_ya, p_yb, prefix, estimator = load_problem(source)
    if args.topology:
        topology = Path(args.topology).resolve()
        topology_manifest = json.loads((topology / "manifest.json").read_text())
        if topology_manifest.get("problem_sha256") != source_hash:
            raise RuntimeError("topology belongs to a different checkpoint problem")
        if topology_manifest.get("version") not in (3, 4):
            raise RuntimeError("topology does not retain full layered AB support")
        values = {}
        for name in topology_manifest["array_sha256"]:
            path = topology / f"{name}.npy"
            expected = topology_manifest["array_sha256"][name]
            if sha256(path) != expected:
                raise RuntimeError("persisted topology array is corrupt")
            values[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        anchored = FullImplicitPairProblem(
            vocabulary_size=int(topology_manifest["vocabulary_size"]),
            **values,
        )
    else:
        topology_manifest = None
        anchored = full_implicit_pair_problem(problem, p_ya, p_yb)
    initial = cold_pair_midpoint_from_projected(problem, p_ya, p_yb)
    variance_yb = empirical_variance(problem.target_yb, prefix)
    variance_ab = empirical_variance(anchored.target_ab, prefix)
    result = full_implicit_pair_sgd(
        anchored, *initial, steps=args.steps, batch_size=args.batch_size,
        training_seed=args.seed, validation_seed=args.validation_seed,
        validation_samples=args.validation_samples,
        workers=args.workers, learning_rate=args.learning_rate,
        validation_interval=args.exact_interval,
        slack_precision=args.slack_precision,
        variance_yb=variance_yb, variance_ab=variance_ab,
    )
    selected_validation = full_implicit_validation_objective(
        anchored, result.correction_yb, result.correction_ab,
        validation_seed=args.validation_seed,
        validation_samples=args.validation_samples,
        slack_precision=args.slack_precision,
        variance_yb=variance_yb, variance_ab=variance_ab,
    )
    destination.mkdir(parents=True, exist_ok=True)
    temporary_state = destination / ".state.npz.tmp"
    with temporary_state.open("wb") as output:
        np.savez(
            output,
            model=np.asarray(ANCHORED_PAIR_GRAPH_MODEL),
            sequence_estimator=np.asarray(estimator),
            initialization=np.asarray(ANCHORED_PAIR_GRAPH_INITIALIZER),
            prefix=np.asarray(prefix),
            active_yb_y=problem.active_yb_y,
            active_yb_b=problem.active_yb_b,
            edge_a=problem.edge_a,
            edge_b=problem.edge_b,
            correction_yb=result.correction_yb,
            correction_ab=result.correction_ab,
            best_validation_objective=np.asarray(result.best_validation_objective),
        )
    temporary_state.replace(state_path)
    manifest = {
        **configuration,
        "execution_workers": args.workers,
        "prefix": prefix,
        "yb_edges": len(problem.target_yb),
        "ab_edges": len(anchored.target_ab),
        "topology_problem_sha256": (
            None if topology_manifest is None
            else topology_manifest["problem_sha256"]
        ),
        "initial_validation_objective": result.initial_validation_objective,
        "best_validation_objective": result.best_validation_objective,
        "final_validation_objective": result.final_validation_objective,
        "selected_validation_objective": selected_validation,
        "state_sha256": sha256(state_path),
    }
    temporary_manifest = destination / ".manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    temporary_manifest.replace(manifest_path)
    print(json.dumps({"reused": False, "out": str(destination), **manifest}))


if __name__ == "__main__":
    main()
