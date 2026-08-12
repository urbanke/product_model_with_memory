#!/usr/bin/env python3
"""Estimate and persist the one unigram shared by both pair jobs."""

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
    _layered_log_sparse_tables,
    _LayeredPredictiveBuilder,
)
from product_model_with_memory.production_coding import (
    PRODUCTION_SEQUENCE_ESTIMATOR,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--counts", required=True)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    for key, value in (("PMM_UNIVERSAL_TABLES", "tables/anchors_prod"),
                       ("PMM_PHI_LADDER_EVERY", "1"),
                       ("PMM_PHI_LADDER_DEGREE", "11"),
                       ("PMM_PHI_SADDLE_MIN_L", "54")):
        os.environ.setdefault(key, value)
    source = Path(a.counts)
    manifest = json.loads((source / "manifest.json").read_text())
    v = int(manifest["vocabulary_size"])
    unigram = np.load(source / "unigram.npy", mmap_mode="r")
    builder = _LayeredPredictiveBuilder(v, default_l_max(v), None, a.jobs, None)
    log_m, tables = _layered_log_sparse_tables(builder, unigram, [])
    if tables:
        raise RuntimeError("unigram-only evaluation unexpectedly made rows")
    destination = Path(a.out)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "marginal.npy", np.exp2(log_m))
    # Production invariant: this artifact must remain layered. A different
    # estimator requires a separately named scientific experiment.
    (destination / "manifest.json").write_text(json.dumps({
        **manifest,
        "version": 2,
        "kind": "unigram",
        "sequence_estimator": PRODUCTION_SEQUENCE_ESTIMATOR,
        "l_max": default_l_max(v),
    }, indent=2))
    print(json.dumps({"checkpoint": manifest["checkpoint"],
                      "vocabulary_size": v}), flush=True)


if __name__ == "__main__":
    main()
