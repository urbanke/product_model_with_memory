#!/usr/bin/env python3
"""Estimate one natural-alphabet marginal for an unequal checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from product_model_with_memory.codelength import default_l_max
from product_model_with_memory.pooled_lags import (
    _LayeredPredictiveBuilder,
    _layered_log_sparse_tables,
)
from product_model_with_memory.production_coding import (
    PRODUCTION_SEQUENCE_ESTIMATOR,
    configure_production_tables,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", required=True)
    parser.add_argument("--symbol", choices=("y", "a", "b"), required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    configure_production_tables()
    source = Path(args.counts)
    manifest = json.loads((source / "manifest.json").read_text())
    sizes = {
        "y": int(manifest["emission_vocabulary_size"]),
        "a": int(manifest["first_lag_alphabet_size"]),
        "b": int(manifest["second_lag_alphabet_size"]),
    }
    alphabet = sizes[args.symbol]
    counts = np.load(source / f"unigram_{args.symbol}.npy", mmap_mode="r")
    if len(counts) != alphabet:
        raise RuntimeError("marginal counts disagree with declared alphabet")
    builder = _LayeredPredictiveBuilder(
        alphabet, default_l_max(alphabet), None, args.jobs, None,
    )
    log_marginal, tables = _layered_log_sparse_tables(builder, counts, [])
    if tables:
        raise RuntimeError("unigram-only evaluation unexpectedly made rows")
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "marginal.npy", np.exp2(log_marginal))
    payload = {
        **manifest, "version": 1, "kind": "unequal_unigram",
        "symbol": args.symbol, "alphabet_size": alphabet,
        "sequence_estimator": PRODUCTION_SEQUENCE_ESTIMATOR,
        "l_max": default_l_max(alphabet),
    }
    (destination / "manifest.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"checkpoint": manifest["checkpoint"],
                      "symbol": args.symbol, "alphabet_size": alphabet}),
          flush=True)


if __name__ == "__main__":
    main()
