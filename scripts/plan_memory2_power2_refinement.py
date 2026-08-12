#!/usr/bin/env python3
"""Declare the frozen power-of-two memory-two refinement campaign."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from product_model_with_memory.memory2_frontier import MemoryTwoPoint


VOCABULARY_GRID = (32768, 65536)
FIRST_LAG_GRID = (16384, 32768, 65536)
SECOND_LAG_GRID = (128, 256, 512, 1024)


def build_plan() -> dict:
    points = tuple(
        MemoryTwoPoint(vocabulary, first_lag, second_lag)
        for vocabulary in VOCABULARY_GRID
        for first_lag in FIRST_LAG_GRID
        for second_lag in SECOND_LAG_GRID
        if second_lag < first_lag <= vocabulary
    )
    if len(points) != 20 or len(set(points)) != 20:
        raise RuntimeError("power-of-two refinement must contain 20 triplets")
    for point in points:
        if any(
            value <= 0 or value & (value - 1)
            for value in point.as_tuple()
        ):
            raise RuntimeError(f"non-power-of-two point {point.as_tuple()}")

    by_v: dict[str, list[str]] = {}
    for point in points:
        by_v.setdefault(str(point.vocabulary_size), []).append(
            f"{point.first_lag_states}:{point.second_lag_states}"
        )
    return {
        "version": 1,
        "campaign": "memory2_power2_refinement_20260810",
        "rule": (
            "predeclared power-of-two refinement; V in {32768,65536}; "
            "M1 in {16384,32768,65536}; M2 in {128,256,512,1024}; "
            "M2 < M1 <= V"
        ),
        "vocabulary_grid": list(VOCABULARY_GRID),
        "first_lag_grid": list(FIRST_LAG_GRID),
        "second_lag_grid": list(SECOND_LAG_GRID),
        "triplets": [point.as_tuple() for point in points],
        "triplet_count": len(points),
        "triplet_selection_bits": math.log2(len(points)),
        "grids_by_vocabulary": by_v,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="output/memory2_power2_refinement_20260810/plan.json",
    )
    args = parser.parse_args()
    payload = build_plan()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "written": str(out),
        "triplets": payload["triplet_count"],
        "selection_bits": payload["triplet_selection_bits"],
        "by_vocabulary": {
            key: len(value)
            for key, value in payload["grids_by_vocabulary"].items()
        },
    }))


if __name__ == "__main__":
    main()
