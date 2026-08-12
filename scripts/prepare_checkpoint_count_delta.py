#!/usr/bin/env python3
"""Count one disjoint checkpoint interval from the shared reduced stream."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def sparse_counts(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, counts = np.unique(keys, return_counts=True)
    return unique.astype(np.int64, copy=False), counts.astype(np.int64, copy=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stream", required=True)
    p.add_argument("--checkpoint", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    source = Path(a.stream)
    manifest = json.loads((source / "manifest.json").read_text())
    edges = manifest["edges"]
    if not 0 <= a.checkpoint < len(edges):
        raise ValueError("checkpoint outside schedule")
    stream_path = Path(manifest.get("stream_path", source / "stream.npy"))
    x = np.load(stream_path, mmap_mode="r")
    v = int(manifest["vocabulary_size"])
    start = 0 if a.checkpoint == 0 else int(edges[a.checkpoint - 1])
    edge = int(edges[a.checkpoint])
    unigram = np.bincount(
        np.asarray(x[start:edge], dtype=np.int64), minlength=v
    ).astype(np.int64, copy=False)
    reveal = max(2, start)
    target = np.asarray(x[reveal:edge], dtype=np.int64)
    lag1 = np.asarray(x[reveal - 1:edge - 1], dtype=np.int64)
    lag2 = np.asarray(x[reveal - 2:edge - 2], dtype=np.int64)
    k1, c1 = sparse_counts(lag1 * v + target)
    k2, c2 = sparse_counts(lag2 * v + target)
    k12, c12 = sparse_counts(lag1 * v + lag2)
    destination = Path(a.out)
    destination.mkdir(parents=True, exist_ok=True)
    arrays = dict(unigram=unigram, keys_ya=k1, counts_ya=c1,
                  keys_yb=k2, counts_yb=c2, keys_ab=k12, counts_ab=c12)
    for name, value in arrays.items():
        np.save(destination / f"{name}.npy", value)
    payload = {**manifest, "version": 1, "kind": "count_delta",
               "checkpoint": a.checkpoint, "start": start, "prefix": edge}
    if "anchor_ids" in manifest:
        payload["anchor_id"] = int(manifest["anchor_ids"][a.checkpoint])
    (destination / "manifest.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"checkpoint": a.checkpoint, "start": start,
                      "prefix": edge, "ya_edges": len(k1),
                      "yb_edges": len(k2), "ab_edges": len(k12)}), flush=True)


if __name__ == "__main__":
    main()
