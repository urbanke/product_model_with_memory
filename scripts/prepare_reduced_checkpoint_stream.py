#!/usr/bin/env python3
"""Prepare one mmap-friendly reduced stream and its checkpoint schedule."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from calibration_checkpoint_probe import geometric_edges
from product_model_with_memory.streams import load_stream, reduce_ids


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", required=True)
    p.add_argument("--top-k", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--checkpoints", type=int, required=True)
    p.add_argument("--first-checkpoint", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    ids, _ = load_stream(a.ids, mmap_mode="r")
    reduced, v, _ = reduce_ids(ids[:a.n], a.top_k)
    edges = geometric_edges(
        2, len(reduced), a.checkpoints, a.first_checkpoint
    )[1:]
    destination = Path(a.out)
    destination.mkdir(parents=True, exist_ok=True)
    # Choose the smallest portable unsigned representation.  All later jobs
    # mmap this file read-only rather than reloading and reducing the corpus.
    dtype = np.uint16 if v <= np.iinfo(np.uint16).max + 1 else np.uint32
    np.save(destination / "stream.npy", reduced.astype(dtype, copy=False))
    payload = {
        "version": 1, "ids": str(Path(a.ids).resolve()), "n": len(reduced),
        "top_k": a.top_k, "vocabulary_size": int(v),
        "dtype": np.dtype(dtype).str, "edges": [int(x) for x in edges],
    }
    (destination / "manifest.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
