#!/usr/bin/env python3
"""Merge unequal-alphabet count deltas into a cumulative snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PAIR_NAMES = ("ya", "yb", "ab")
UNIGRAM_NAMES = ("y", "a", "b")


def merge_weighted(pk, pc, dk, dc):
    if not len(pk):
        return np.asarray(dk).copy(), np.asarray(dc).copy()
    if not len(dk):
        return np.asarray(pk).copy(), np.asarray(pc).copy()
    keys = np.concatenate((pk, dk))
    counts = np.concatenate((pc, dc))
    order = np.argsort(keys, kind="stable")
    keys, counts = keys[order], counts[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(keys))]
    return keys[starts], np.add.reduceat(counts, starts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", required=True)
    parser.add_argument("--previous")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    delta = Path(args.delta)
    manifest = json.loads((delta / "manifest.json").read_text())
    previous = None if args.previous is None else Path(args.previous)
    if previous is not None:
        old = json.loads((previous / "manifest.json").read_text())
        for key in ("emission_vocabulary_size", "first_lag_parameter",
                    "second_lag_parameter", "state_map"):
            if old.get(key) != manifest.get(key):
                raise RuntimeError(f"count snapshots disagree on {key}")
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    for label in UNIGRAM_NAMES:
        value = np.asarray(np.load(delta / f"unigram_{label}.npy", mmap_mode="r")).copy()
        if previous is not None:
            value += np.load(previous / f"unigram_{label}.npy", mmap_mode="r")
        np.save(destination / f"unigram_{label}.npy", value)
    edge_counts = {}
    for label in PAIR_NAMES:
        dk = np.load(delta / f"keys_{label}.npy", mmap_mode="r")
        dc = np.load(delta / f"counts_{label}.npy", mmap_mode="r")
        if previous is None:
            pk = pc = np.empty(0, dtype=np.int64)
        else:
            pk = np.load(previous / f"keys_{label}.npy", mmap_mode="r")
            pc = np.load(previous / f"counts_{label}.npy", mmap_mode="r")
        keys, counts = merge_weighted(pk, pc, dk, dc)
        np.save(destination / f"keys_{label}.npy", keys)
        np.save(destination / f"counts_{label}.npy", counts)
        edge_counts[label] = len(keys)
    (destination / "manifest.json").write_text(json.dumps(
        {**manifest, "kind": "unequal_cumulative_counts"}, indent=2
    ))
    print(json.dumps({"checkpoint": manifest["checkpoint"], **edge_counts}),
          flush=True)


if __name__ == "__main__":
    main()
