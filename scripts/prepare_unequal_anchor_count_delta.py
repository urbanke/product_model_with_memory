#!/usr/bin/env python3
"""Count one unequal-alphabet anchored checkpoint interval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from product_model_with_memory.anchored_state_maps import (
    map_reduced_context,
    state_map_manifest,
)


def sparse_counts(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, counts = np.unique(keys, return_counts=True)
    return unique.astype(np.int64, copy=False), counts.astype(np.int64, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--m1", type=int, required=True)
    parser.add_argument("--m2", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source = Path(args.stream)
    base = json.loads((source / "manifest.json").read_text())
    edges = base["edges"]
    if not 0 <= args.checkpoint < len(edges):
        raise ValueError("checkpoint outside schedule")
    stream_path = Path(base.get("stream_path", source / "stream.npy"))
    x = np.load(stream_path, mmap_mode="r", allow_pickle=False)
    v = int(base["vocabulary_size"])
    maps = state_map_manifest(v, args.m1, args.m2)
    a_size = int(maps["first_lag_alphabet_size"])
    b_size = int(maps["second_lag_alphabet_size"])
    start = 0 if args.checkpoint == 0 else int(edges[args.checkpoint - 1])
    edge = int(edges[args.checkpoint])

    segment = np.asarray(x[start:edge], dtype=np.int64)
    mapped_a = map_reduced_context(segment, v, args.m1)
    mapped_b = map_reduced_context(segment, v, args.m2)
    unigram_y = np.bincount(segment, minlength=v).astype(np.int64, copy=False)
    unigram_a = np.bincount(mapped_a, minlength=a_size).astype(np.int64, copy=False)
    unigram_b = np.bincount(mapped_b, minlength=b_size).astype(np.int64, copy=False)

    reveal = max(2, start)
    target = np.asarray(x[reveal:edge], dtype=np.int64)
    lag1 = map_reduced_context(x[reveal - 1:edge - 1], v, args.m1)
    lag2 = map_reduced_context(x[reveal - 2:edge - 2], v, args.m2)
    k_ya, c_ya = sparse_counts(lag1 * v + target)
    k_yb, c_yb = sparse_counts(lag2 * v + target)
    # AB orientation is target=A, context=B.  Its natural target alphabet is
    # A_size, not V; estimating this table directly is essential when the two
    # context maps differ.
    k_ab, c_ab = sparse_counts(lag2 * a_size + lag1)

    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    arrays = {
        "unigram_y": unigram_y, "unigram_a": unigram_a,
        "unigram_b": unigram_b, "keys_ya": k_ya, "counts_ya": c_ya,
        "keys_yb": k_yb, "counts_yb": c_yb,
        "keys_ab": k_ab, "counts_ab": c_ab,
    }
    for name, value in arrays.items():
        np.save(destination / f"{name}.npy", value)
    payload = {
        **base, **maps, "version": 1, "kind": "unequal_count_delta",
        "checkpoint": args.checkpoint, "start": start, "prefix": edge,
    }
    if "anchor_ids" in base:
        payload["anchor_id"] = int(base["anchor_ids"][args.checkpoint])
    (destination / "manifest.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"checkpoint": args.checkpoint, "prefix": edge,
                      "ya_edges": len(k_ya), "yb_edges": len(k_yb),
                      "ab_edges": len(k_ab)}), flush=True)


if __name__ == "__main__":
    main()
