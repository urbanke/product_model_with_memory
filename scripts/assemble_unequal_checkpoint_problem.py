#!/usr/bin/env python3
"""Assemble unequal YA/YB/AB layered laws into one V-embedded problem."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from product_model_with_memory.graphical_calibration import SparseProjectedPair
from product_model_with_memory.production_coding import (
    PRODUCTION_SEQUENCE_ESTIMATOR,
    require_production_sequence_estimator,
)


PAIR_ARRAYS = ("left", "right", "background", "active_y",
               "active_context", "delta")


def load_pair(path: Path, label: str) -> tuple[dict, SparseProjectedPair]:
    manifest = json.loads((path / "manifest.json").read_text())
    require_production_sequence_estimator(
        manifest.get("sequence_estimator"), source=str(path),
    )
    if manifest.get("pair") != label:
        raise RuntimeError("pair artifact has the wrong label")
    size = int(manifest["padded_pair_size"])
    arrays = [np.load(path / f"{name}.npy", mmap_mode="r") for name in PAIR_ARRAYS]
    return manifest, SparseProjectedPair(size, *arrays)


def embed_pair(pair: SparseProjectedPair, v: int) -> SparseProjectedPair:
    """Zero-pad a natural pair for graph indexing without re-estimating it."""

    if pair.vocabulary_size > v:
        raise ValueError("natural pair exceeds graph alphabet")
    if pair.vocabulary_size == v:
        return pair
    n = pair.vocabulary_size
    background = np.zeros(v, dtype=np.float64)
    background[:n] = pair.background
    return SparseProjectedPair(
        v, np.ones(v), np.ones(v), background,
        np.asarray(pair.active_y), np.asarray(pair.active_context),
        np.asarray(pair.delta),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", required=True)
    parser.add_argument("--ya", required=True)
    parser.add_argument("--yb", required=True)
    parser.add_argument("--ab", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    counts = Path(args.counts)
    source_manifest = json.loads((counts / "manifest.json").read_text())
    v = int(source_manifest["emission_vocabulary_size"])
    a_size = int(source_manifest["first_lag_alphabet_size"])
    b_size = int(source_manifest["second_lag_alphabet_size"])
    ya_manifest, p_ya = load_pair(Path(args.ya), "ya")
    yb_manifest, p_yb = load_pair(Path(args.yb), "yb")
    ab_manifest, natural_ab = load_pair(Path(args.ab), "ab")
    for manifest in (ya_manifest, yb_manifest, ab_manifest):
        for key in ("emission_vocabulary_size", "first_lag_parameter",
                    "second_lag_parameter", "prefix", "state_map"):
            if manifest.get(key) != source_manifest.get(key):
                raise RuntimeError(f"pair and count artifacts disagree on {key}")
    if p_ya.vocabulary_size != v or p_yb.vocabulary_size != v:
        raise RuntimeError("YA/YB pairs must use emission target alphabet V")
    if natural_ab.vocabulary_size != a_size:
        raise RuntimeError("AB pair must use natural A target alphabet")
    p_ab = embed_pair(natural_ab, v)

    keys = np.load(counts / "keys_ab.npy", mmap_mode="r")
    edge_a = np.asarray(keys % a_size, dtype=np.int64)
    edge_b = np.asarray(keys // a_size, dtype=np.int64)
    if len(edge_a) == 0 or np.any(edge_a >= a_size) or np.any(edge_b >= b_size):
        raise RuntimeError("AB support lies outside declared state alphabets")
    edge_probability = p_ab.values(edge_a, edge_b)
    retained = float(edge_probability.sum())
    if not 0.0 < retained <= 1.0 + 1e-10:
        raise RuntimeError("explicit AB support has invalid probability mass")
    edge_probability = edge_probability / retained
    target_y = np.load(Path(args.ya) / "target_marginal.npy", mmap_mode="r")
    if target_y.shape != (v,):
        raise RuntimeError("YA target marginal has the wrong alphabet")
    state = {
        "prefix": np.asarray(source_manifest["prefix"]),
        "sequence_estimator": np.asarray(PRODUCTION_SEQUENCE_ESTIMATOR),
        "margin_preprocessing": np.asarray("raw_relaxed_explicit_ab_v1"),
        "emission_vocabulary_size": np.asarray(v),
        "first_lag_parameter": np.asarray(source_manifest["first_lag_parameter"]),
        "second_lag_parameter": np.asarray(source_manifest["second_lag_parameter"]),
        "first_lag_alphabet_size": np.asarray(a_size),
        "second_lag_alphabet_size": np.asarray(b_size),
        "state_subset_description_bits": np.asarray(
            source_manifest["state_subset_description_bits"]
        ),
        "edge_a": edge_a, "edge_b": edge_b,
        "edge_probability": edge_probability, "target_y": target_y,
        "active_ya_y": p_ya.active_y, "active_ya_a": p_ya.active_context,
        "target_ya": p_ya.active_values(),
        "active_yb_y": p_yb.active_y, "active_yb_b": p_yb.active_context,
        "target_yb": p_yb.active_values(),
        "log_base_y": np.log(target_y),
        "correction_ya": np.zeros(len(p_ya.delta)),
        "correction_yb": np.zeros(len(p_yb.delta)),
        "retained_ab_mass": np.asarray(retained),
    }
    for label, pair in (("ya", p_ya), ("yb", p_yb), ("ab", p_ab)):
        for name in PAIR_ARRAYS:
            state[f"fallback_{label}_{name}"] = getattr(pair, name)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **state)
    print(json.dumps({"checkpoint": source_manifest["checkpoint"],
                      "retained_ab_mass": retained,
                      "context_edges": len(edge_probability),
                      "V": v, "M1_alphabet": a_size,
                      "M2_alphabet": b_size}), flush=True)


if __name__ == "__main__":
    main()
