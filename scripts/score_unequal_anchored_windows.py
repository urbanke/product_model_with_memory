#!/usr/bin/env python3
"""Score unequal mapped windows against their matching Markov-1 control."""

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
from product_model_with_memory.anchored_state_maps import map_reduced_context
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
    problem_path, topology_path = Path(args.problem).resolve(), Path(args.topology).resolve()
    fit_path, stream_path = Path(args.fit).resolve(), Path(args.stream).resolve()
    windows = sorted(set(map(int, args.windows.split(","))))
    topology_manifest = json.loads((topology_path / "manifest.json").read_text())
    fit_manifest = json.loads((fit_path / "manifest.json").read_text())
    if topology_manifest.get("version") != 4:
        raise RuntimeError("unequal scoring requires explicit-AB topology version 4")
    if fit_manifest.get("model") != ANCHORED_PAIR_GRAPH_MODEL:
        raise RuntimeError("fit uses a different graphical model")
    require_production_sequence_estimator(fit_manifest.get("sequence_estimator"), source=str(fit_path))
    problem_hash = sha256(problem_path)
    if topology_manifest.get("problem_sha256") != problem_hash or fit_manifest.get("problem_sha256") != problem_hash:
        raise RuntimeError("fit/topology belongs to a different problem")
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
        correction_yb, correction_ab = np.array(state["correction_yb"]), np.array(state["correction_ab"])
    with np.load(problem_path, allow_pickle=False) as problem_data:
        v = int(problem_data["emission_vocabulary_size"])
        m1 = int(problem_data["first_lag_parameter"])
        m2 = int(problem_data["second_lag_parameter"])
        subset_bits = float(problem_data["state_subset_description_bits"])
    stream = np.load(stream_path, mmap_mode="r", allow_pickle=False)
    maximum = min(windows[-1], len(stream) - prefix)
    windows = [length for length in windows if 0 < length <= maximum]
    if not windows or prefix < 2:
        raise ValueError("stream does not contain requested score windows")
    stop = prefix + windows[-1]
    target = np.asarray(stream[prefix:stop], dtype=np.int64)
    lag1 = map_reduced_context(stream[prefix - 1:stop - 1], v, m1)
    lag2 = map_reduced_context(stream[prefix - 2:stop - 2], v, m2)
    anchored_log, markov_log = full_implicit_log_probabilities(
        topology, correction_yb, correction_ab, target, lag1, lag2,
    )
    scale = -1.0 / np.log(2.0)
    anchored_bits, markov_bits = anchored_log * scale, markov_log * scale
    delta_bits = anchored_bits - markov_bits
    def totals(lo, hi):
        delta = delta_bits[lo:hi]
        return {"start_offset": lo, "stop_offset": hi, "tokens": hi-lo,
                "anchored_bits": float(anchored_bits[lo:hi].sum()),
                "markov1_bits": float(markov_bits[lo:hi].sum()),
                "delta_bits": float(delta.sum()),
                "delta_bits_squared": float(np.dot(delta, delta)),
                "covered_tokens": hi-lo}
    boundaries = [0] + windows
    payload = {
        "version": 2, "model": ANCHORED_PAIR_GRAPH_MODEL,
        "sequence_estimator": PRODUCTION_SEQUENCE_ESTIMATOR,
        "fallback": "none_full_layered_explicit_ab_v1",
        "problem_sha256": problem_hash, "fit_prefix": prefix,
        "V": v, "M1": m1, "M2": m2,
        "state_subset_description_bits": subset_bits, "windows": windows,
        "cumulative": [totals(0, length) for length in windows],
        "rings": [totals(lo, hi) for lo, hi in zip(boundaries, boundaries[1:])],
    }
    destination = Path(args.out).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(destination)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
