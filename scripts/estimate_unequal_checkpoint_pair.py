#!/usr/bin/env python3
"""Estimate one natural-alphabet layered pair for an unequal checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from product_model_with_memory.codelength import default_l_max
from product_model_with_memory.graphical_calibration import sparse_layered_pair
from product_model_with_memory.pooled_lags import (
    SparseCountRows,
    _LayeredPredictiveBuilder,
    _layered_log_sparse_conditionals,
)
from product_model_with_memory.production_coding import (
    PRODUCTION_SEQUENCE_ESTIMATOR,
    configure_production_tables,
    require_production_sequence_estimator,
)


def load_marginal(path: Path, symbol: str, expected: int) -> np.ndarray:
    manifest = json.loads((path / "manifest.json").read_text())
    require_production_sequence_estimator(
        manifest.get("sequence_estimator"), source=str(path),
    )
    if manifest.get("symbol") != symbol or int(manifest.get("alphabet_size", -1)) != expected:
        raise RuntimeError("marginal artifact has the wrong symbol alphabet")
    value = np.load(path / "marginal.npy", mmap_mode="r")
    if value.shape != (expected,) or not np.isclose(float(value.sum()), 1.0):
        raise RuntimeError("marginal artifact is malformed")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", required=True)
    parser.add_argument("--target-marginal", required=True)
    parser.add_argument("--context-marginal", required=True)
    parser.add_argument("--pair", choices=("ya", "yb", "ab"), required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    configure_production_tables()
    source = Path(args.counts)
    manifest = json.loads((source / "manifest.json").read_text())
    v = int(manifest["emission_vocabulary_size"])
    a_size = int(manifest["first_lag_alphabet_size"])
    b_size = int(manifest["second_lag_alphabet_size"])
    specification = {
        "ya": ("y", v, "a", a_size),
        "yb": ("y", v, "b", b_size),
        "ab": ("a", a_size, "b", b_size),
    }[args.pair]
    target_symbol, target_size, context_symbol, context_size = specification
    if context_size > target_size:
        raise RuntimeError("padded sparse evaluator requires context <= target alphabet")
    target_marginal = load_marginal(
        Path(args.target_marginal), target_symbol, target_size,
    )
    context_marginal = load_marginal(
        Path(args.context_marginal), context_symbol, context_size,
    )
    padded_context = np.zeros(target_size, dtype=np.float64)
    padded_context[:context_size] = context_marginal
    keys = np.load(source / f"keys_{args.pair}.npy", mmap_mode="r")
    values = np.load(source / f"counts_{args.pair}.npy", mmap_mode="r")
    rows = SparseCountRows.from_sorted_keys(target_size, keys, values)
    builder = _LayeredPredictiveBuilder(
        target_size, default_l_max(target_size), None, args.jobs, None,
    )
    table = _layered_log_sparse_conditionals(builder, [rows])[0]
    contexts = np.repeat(np.arange(target_size), np.diff(table["ptr"]))
    pair = sparse_layered_pair(
        padded_context, np.exp2(table["unseen"]), table["idx"], contexts,
        np.exp2(table["val"]),
    )
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "target_marginal.npy", target_marginal)
    np.save(destination / "context_marginal.npy", context_marginal)
    for name in ("left", "right", "background", "active_y",
                 "active_context", "delta"):
        np.save(destination / f"{name}.npy", getattr(pair, name))
    payload = {
        **manifest, "version": 1, "kind": "unequal_pair", "pair": args.pair,
        "target_symbol": target_symbol, "target_alphabet_size": target_size,
        "context_symbol": context_symbol, "context_alphabet_size": context_size,
        "padded_pair_size": target_size,
        "sequence_estimator": PRODUCTION_SEQUENCE_ESTIMATOR,
        "l_max": default_l_max(target_size),
    }
    (destination / "manifest.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"pair": args.pair, "active": len(pair.delta),
                      "target_alphabet_size": target_size,
                      "context_alphabet_size": context_size}), flush=True)


if __name__ == "__main__":
    main()
