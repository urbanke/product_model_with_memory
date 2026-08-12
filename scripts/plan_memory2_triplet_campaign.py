#!/usr/bin/env python3
"""Declare a finite, deduplicated memory-two triplet campaign."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from product_model_with_memory.memory2_frontier import (
    MemoryTwoPoint,
    declared_triplet_grid,
)


def _integers(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v-grid", default="65536,100277")
    parser.add_argument(
        "--m1-grid", default="2048,4096,8192,16384,32768,65536,100277"
    )
    parser.add_argument(
        "--m2-grid", default="1024,2048,4096,8192,16384,32768,65536"
    )
    parser.add_argument("--minimum-m2", type=int, default=1024)
    parser.add_argument(
        "--bridge-m2", default="128,256,512",
        help="fixed V=M1=65536 bridge points below minimum-m2",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    broad_points = declared_triplet_grid(
        vocabulary_grid=_integers(args.v_grid),
        first_lag_grid=_integers(args.m1_grid),
        second_lag_grid=_integers(args.m2_grid),
        minimum_second_lag=args.minimum_m2,
    )
    bridge = tuple(
        MemoryTwoPoint(65536, 65536, value)
        for value in _integers(args.bridge_m2)
    )
    points = tuple(sorted(set(broad_points) | set(bridge)))
    by_v: dict[str, list[str]] = {}
    # Finish the already-improving fixed-V bridge first, then enter the broad
    # combinatorial grid.  This affects run order only, never the declared set.
    run_order = bridge + tuple(point for point in points if point not in bridge)
    for point in run_order:
        by_v.setdefault(str(point.vocabulary_size), []).append(
            f"{point.first_lag_states}:{point.second_lag_states}"
        )
    payload = {
        "version": 1,
        "rule": (
            "three fixed V=M1=65536 bridge points, then M2 < M1 <= V; "
            "powers-of-two lag grids; broad grid M2 >= minimum_m2"
        ),
        "vocabulary_grid": sorted({point.vocabulary_size for point in points}),
        "minimum_m2": args.minimum_m2,
        "bridge_triplets": [point.as_tuple() for point in bridge],
        "triplets": [point.as_tuple() for point in points],
        "triplet_count": len(points),
        "triplet_selection_bits": math.log2(len(points)),
        "grids_by_vocabulary": by_v,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"written": str(out), "triplets": len(points),
                      "by_vocabulary": {k: len(v) for k, v in by_v.items()}}))


if __name__ == "__main__":
    main()
