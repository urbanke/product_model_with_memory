#!/usr/bin/env python3
"""Compare two persisted calibration-problem checkpoint collections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


STRUCTURAL = (
    "prefix", "active_ya_y", "active_ya_a", "active_yb_y", "active_yb_b",
    "edge_a", "edge_b", "fallback_ya_active_y",
    "fallback_ya_active_context", "fallback_yb_active_y",
    "fallback_yb_active_context", "margin_preprocessing",
)
NUMERIC = (
    "edge_probability", "target_y", "target_ya", "target_yb", "log_base_y",
    "correction_ya", "correction_yb", "fallback_ya_left",
    "fallback_ya_right", "fallback_ya_background", "fallback_ya_delta",
    "fallback_yb_left", "fallback_yb_right", "fallback_yb_background",
    "fallback_yb_delta",
)


def compare_checkpoint(left: Path, right: Path, atol: float) -> dict:
    rows = {}
    equivalent = True
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        for name in STRUCTURAL:
            present = name in a and name in b
            same = present and a[name].shape == b[name].shape and np.array_equal(
                a[name], b[name]
            )
            rows[name] = {"kind": "structural", "equal": bool(same)}
            equivalent &= same
        for name in NUMERIC:
            present = name in a and name in b
            same_shape = present and a[name].shape == b[name].shape
            maximum = (
                float(np.max(np.abs(a[name] - b[name]), initial=0.0))
                if same_shape else float("inf")
            )
            same = same_shape and maximum <= atol
            rows[name] = {
                "kind": "numeric", "within_tolerance": bool(same),
                "maximum_absolute_difference": maximum,
            }
            equivalent &= same
    return {"equivalent": bool(equivalent), "arrays": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--checkpoints", default="0,1")
    parser.add_argument("--atol", type=float, default=1e-15)
    parser.add_argument("--out")
    args = parser.parse_args()
    selected = sorted({int(value) for value in args.checkpoints.split(",")})
    rows = []
    for checkpoint in selected:
        name = f"checkpoint_{checkpoint:03d}.npz"
        comparison = compare_checkpoint(
            Path(args.left) / "states" / name,
            Path(args.right) / "states" / name,
            args.atol,
        )
        rows.append({"checkpoint": checkpoint, **comparison})
    payload = {
        "version": 1, "absolute_tolerance": args.atol,
        "equivalent": all(row["equivalent"] for row in rows),
        "checkpoints": rows,
    }
    rendered = json.dumps(payload, indent=2)
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n")
    print(rendered, flush=True)
    if not payload["equivalent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
