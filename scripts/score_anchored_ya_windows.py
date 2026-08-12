#!/usr/bin/env python3
"""Score nested windows for one anchored-YA fit and its Markov-1 control."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from product_model_with_memory.anchored_pair_graph import (
    ANCHORED_PAIR_GRAPH_MODEL,
    FullImplicitPairProblem,
    full_implicit_log_probabilities,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--fit", required=True)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--windows", default="64,256,1024,4096")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    problem_path = Path(args.problem).resolve()
    topology_path = Path(args.topology).resolve()
    fit_path = Path(args.fit).resolve()
    stream_path = Path(args.stream).resolve()
    destination = Path(args.out).resolve()
    windows = sorted(set(int(value) for value in args.windows.split(",")))
    if not windows or windows[0] < 1:
        parser.error("windows must be positive integers")

    topology_manifest = json.loads((topology_path / "manifest.json").read_text())
    fit_manifest = json.loads((fit_path / "manifest.json").read_text())
    if topology_manifest.get("version") != 3:
        raise RuntimeError("scoring requires full layered-AB topology version 3")
    if fit_manifest.get("model") != ANCHORED_PAIR_GRAPH_MODEL:
        raise RuntimeError("fit uses a different graphical model")
    require_production_sequence_estimator(
        fit_manifest.get("sequence_estimator"), source=str(fit_path),
    )
    problem_hash = sha256(problem_path)
    if topology_manifest.get("problem_sha256") != problem_hash:
        raise RuntimeError("topology belongs to a different checkpoint")
    if fit_manifest.get("problem_sha256") != problem_hash:
        raise RuntimeError("fit belongs to a different checkpoint")
    values = {}
    for name, expected in topology_manifest["array_sha256"].items():
        path = topology_path / f"{name}.npy"
        if sha256(path) != expected:
            raise RuntimeError("topology array hash mismatch")
        values[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    topology = FullImplicitPairProblem(
        vocabulary_size=int(topology_manifest["vocabulary_size"]), **values,
    )
    state_path = fit_path / "state.npz"
    if sha256(state_path) != fit_manifest.get("state_sha256"):
        raise RuntimeError("fit state hash mismatch")
    with np.load(state_path, allow_pickle=False) as state:
        prefix = int(state["prefix"])
        u = np.array(state["correction_yb"])
        v = np.array(state["correction_ab"])
    stream = np.load(stream_path, mmap_mode="r", allow_pickle=False)
    maximum = min(windows[-1], len(stream) - prefix)
    windows = [length for length in windows if length <= maximum]
    if not windows or prefix < 2:
        raise ValueError("stream does not contain the requested score windows")
    stop = prefix + windows[-1]
    target = np.asarray(stream[prefix:stop], dtype=np.int64)
    lag1 = np.asarray(stream[prefix - 1:stop - 1], dtype=np.int64)
    lag2 = np.asarray(stream[prefix - 2:stop - 2], dtype=np.int64)
    anchored_log, markov_log = full_implicit_log_probabilities(
        topology, u, v, target, lag1, lag2,
    )
    covered = np.ones(len(target), dtype=bool)
    scale = -1.0 / np.log(2.0)
    anchored_bits = anchored_log * scale
    markov_bits = markov_log * scale
    delta_bits = anchored_bits - markov_bits

    def totals(lo: int, hi: int) -> dict:
        delta = delta_bits[lo:hi]
        return {
            "start_offset": lo, "stop_offset": hi, "tokens": hi - lo,
            "anchored_bits": float(np.sum(anchored_bits[lo:hi])),
            "markov1_bits": float(np.sum(markov_bits[lo:hi])),
            "delta_bits": float(np.sum(delta)),
            "delta_bits_squared": float(np.dot(delta, delta)),
            "covered_tokens": int(np.sum(covered[lo:hi])),
        }

    cumulative = [totals(0, length) for length in windows]
    boundaries = [0] + windows
    rings = [totals(lo, hi) for lo, hi in zip(boundaries, boundaries[1:])]
    payload = {
        "version": 1,
        "model": ANCHORED_PAIR_GRAPH_MODEL,
        "sequence_estimator": PRODUCTION_SEQUENCE_ESTIMATOR,
        "fallback": "none_full_layered_ab_v1",
        "problem_sha256": problem_hash,
        "topology_manifest_sha256": sha256(topology_path / "manifest.json"),
        "fit_manifest_sha256": sha256(fit_path / "manifest.json"),
        "stream_sha256": sha256(stream_path),
        "fit_prefix": prefix,
        "windows": windows,
        "cumulative": cumulative,
        "rings": rings,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(destination)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
