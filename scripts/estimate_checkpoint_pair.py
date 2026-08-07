#!/usr/bin/env python3
"""Estimate one raw layered pair law from a persisted count snapshot."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from product_model_with_memory.codelength import default_l_max
from product_model_with_memory.graphical_calibration import sparse_layered_pair
from product_model_with_memory.pooled_lags import (
    SparseCountRows, _LayeredPredictiveBuilder,
    _layered_log_sparse_conditionals,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--counts", required=True)
    p.add_argument("--unigram", required=True)
    p.add_argument("--pair", choices=("ya", "yb"), required=True)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.environ.setdefault("PMM_UNIVERSAL_TABLES", "tables/anchors_prod")
    os.environ.setdefault("PMM_PHI_LADDER_EVERY", "1")
    # These are part of the estimator definition and must match the
    # monolithic construction, not merely be performance defaults.
    os.environ.setdefault("PMM_PHI_LADDER_DEGREE", "11")
    os.environ.setdefault("PMM_PHI_SADDLE_MIN_L", "54")
    counts = Path(a.counts)
    manifest = json.loads((counts / "manifest.json").read_text())
    v = int(manifest["vocabulary_size"])
    marginal = np.load(Path(a.unigram) / "marginal.npy", mmap_mode="r")
    if len(marginal) != v:
        raise RuntimeError("unigram and count snapshot vocabularies differ")
    keys = np.load(counts / f"keys_{a.pair}.npy", mmap_mode="r")
    values = np.load(counts / f"counts_{a.pair}.npy", mmap_mode="r")
    rows = SparseCountRows.from_sorted_keys(v, keys, values)
    builder = _LayeredPredictiveBuilder(v, default_l_max(v), None, a.jobs, None)
    tables = _layered_log_sparse_conditionals(builder, [rows])
    table = tables[0]
    contexts = np.repeat(np.arange(v), np.diff(table["ptr"]))
    pair = sparse_layered_pair(marginal, np.exp2(table["unseen"]),
                               table["idx"], contexts, np.exp2(table["val"]))
    destination = Path(a.out)
    destination.mkdir(parents=True, exist_ok=True)
    arrays = {"marginal": marginal, "left": pair.left, "right": pair.right,
              "background": pair.background, "active_y": pair.active_y,
              "active_context": pair.active_context, "delta": pair.delta}
    for name, value in arrays.items():
        np.save(destination / f"{name}.npy", value)
    payload = {"version": 1, "pair": a.pair, **manifest}
    (destination / "manifest.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"pair": a.pair, "active": len(pair.delta)}), flush=True)


if __name__ == "__main__":
    main()
