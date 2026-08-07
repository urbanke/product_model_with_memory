#!/usr/bin/env python3
"""Merge one count delta into the preceding cumulative mmap snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PAIR_NAMES = ("ya", "yb", "ab")


def merge_weighted(pk, pc, dk, dc):
    """Merge two sorted key/count arrays without expanding observations."""
    if not len(pk):
        return np.asarray(dk).copy(), np.asarray(dc).copy()
    if not len(dk):
        return np.asarray(pk).copy(), np.asarray(pc).copy()
    all_keys = np.concatenate((pk, dk))
    all_counts = np.concatenate((pc, dc))
    order = np.argsort(all_keys, kind="stable")
    all_keys, all_counts = all_keys[order], all_counts[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(all_keys))]
    return all_keys[starts], np.add.reduceat(all_counts, starts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--delta", required=True)
    p.add_argument("--previous")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    delta = Path(a.delta)
    manifest = json.loads((delta / "manifest.json").read_text())
    destination = Path(a.out)
    destination.mkdir(parents=True, exist_ok=True)
    d_uni = np.load(delta / "unigram.npy", mmap_mode="r")
    if a.previous:
        previous = Path(a.previous)
        unigram = np.asarray(
            np.load(previous / "unigram.npy", mmap_mode="r")
        ).copy()
        unigram += d_uni
    else:
        unigram = np.asarray(d_uni).copy()
    np.save(destination / "unigram.npy", unigram)
    edge_counts = {}
    for label in PAIR_NAMES:
        dk = np.load(delta / f"keys_{label}.npy", mmap_mode="r")
        dc = np.load(delta / f"counts_{label}.npy", mmap_mode="r")
        if a.previous:
            pk = np.load(Path(a.previous) / f"keys_{label}.npy", mmap_mode="r")
            pc = np.load(Path(a.previous) / f"counts_{label}.npy", mmap_mode="r")
        else:
            pk = pc = np.empty(0, dtype=np.int64)
        keys, counts = merge_weighted(pk, pc, dk, dc)
        np.save(destination / f"keys_{label}.npy", keys)
        np.save(destination / f"counts_{label}.npy", counts)
        edge_counts[label] = len(keys)
    payload = {**manifest, "kind": "cumulative_counts"}
    (destination / "manifest.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"checkpoint": manifest["checkpoint"], **edge_counts}),
          flush=True)


if __name__ == "__main__":
    main()
