#!/usr/bin/env python3
"""Publish sorted causal construction edges for a fixed anchor plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    stream = Path(plan["stream"]).resolve()
    x = np.load(stream, mmap_mode="r", allow_pickle=False)
    ordered = sorted(plan["anchors"], key=lambda row: int(row["prefix"]))
    edges = [int(row["prefix"]) for row in ordered]
    if len(edges) != len(set(edges)) or edges != sorted(edges):
        raise RuntimeError("anchor prefixes must be distinct")
    payload = {
        "version": 2, "kind": "anchor_construction_stream",
        "stream_path": str(stream), "stream_sha256": plan["stream_sha256"],
        "n": len(x), "vocabulary_size": int(x.max()) + 1,
        "edges": edges,
        "anchor_ids": [int(row["anchor_id"]) for row in ordered],
        "sampling_design": plan["design"],
    }
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / ".manifest.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(destination / "manifest.json")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
