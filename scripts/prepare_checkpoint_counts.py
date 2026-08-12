#!/usr/bin/env python3
"""Advance one cumulative count checkpoint and persist mmap-friendly arrays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from calibration_checkpoint_probe import geometric_edges, merge_sparse_counts
from product_model_with_memory.streams import load_stream, reduce_ids


NAMES = ("unigram", "keys_ya", "counts_ya", "keys_yb", "counts_yb",
         "keys_ab", "counts_ab")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", required=True)
    p.add_argument("--top-k", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--checkpoints", type=int, required=True)
    p.add_argument("--first-checkpoint", type=int, required=True)
    p.add_argument("--checkpoint", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    ids, _ = load_stream(a.ids)
    reduced, v, _ = reduce_ids(ids[:a.n], a.top_k)
    x = reduced.astype(np.int64, copy=False)
    edges = geometric_edges(2, len(x), a.checkpoints, a.first_checkpoint)[1:]
    if not 0 <= a.checkpoint < len(edges):
        raise ValueError("checkpoint outside geometric schedule")
    root = Path(a.out)
    destination = root / f"checkpoint_{a.checkpoint:03d}"
    destination.mkdir(parents=True, exist_ok=True)

    if a.checkpoint:
        previous = root / f"checkpoint_{a.checkpoint - 1:03d}"
        manifest = json.loads((previous / "manifest.json").read_text())
        prior = {name: np.load(previous / f"{name}.npy", mmap_mode="r")
                 for name in NAMES}
        unigram = np.asarray(prior["unigram"]).copy()
        keys1, counts1 = prior["keys_ya"], prior["counts_ya"]
        keys2, counts2 = prior["keys_yb"], prior["counts_yb"]
        keys12, counts12 = prior["keys_ab"], prior["counts_ab"]
        start = int(manifest["prefix"])
    else:
        unigram = np.zeros(v, dtype=np.int64)
        empty = np.empty(0, dtype=np.int64)
        keys1 = counts1 = keys2 = counts2 = keys12 = counts12 = empty
        start = 0
    edge = int(edges[a.checkpoint])
    unigram += np.bincount(x[start:edge], minlength=v)
    reveal = max(2, start)
    target = x[reveal:edge]
    lag1 = x[reveal - 1:edge - 1]
    lag2 = x[reveal - 2:edge - 2]
    keys1, counts1 = merge_sparse_counts(keys1, counts1, lag1 * v + target)
    keys2, counts2 = merge_sparse_counts(keys2, counts2, lag2 * v + target)
    keys12, counts12 = merge_sparse_counts(keys12, counts12, lag1 * v + lag2)
    arrays = dict(unigram=unigram, keys_ya=keys1, counts_ya=counts1,
                  keys_yb=keys2, counts_yb=counts2,
                  keys_ab=keys12, counts_ab=counts12)
    for name, value in arrays.items():
        np.save(destination / f"{name}.npy", np.asarray(value))
    payload = {"version": 1, "checkpoint": a.checkpoint, "prefix": edge,
               "vocabulary_size": int(v), "ids": str(Path(a.ids).resolve()),
               "n": len(x), "top_k": a.top_k,
               "edges": [int(value) for value in edges]}
    (destination / "manifest.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
