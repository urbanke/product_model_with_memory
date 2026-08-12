#!/usr/bin/env python3
"""Audit split and cumulative checkpoint counts against direct stream counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARRAY_NAMES = (
    "unigram", "keys_ya", "counts_ya", "keys_yb", "counts_yb",
    "keys_ab", "counts_ab",
)


def _sparse(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, counts = np.unique(keys, return_counts=True)
    return unique.astype(np.int64, copy=False), counts.astype(np.int64, copy=False)


def direct_counts(
    stream: np.ndarray, vocabulary_size: int, start: int, stop: int,
) -> dict[str, np.ndarray]:
    """Count one interval, including lagged records born in that interval."""
    if not 0 <= start <= stop <= len(stream):
        raise ValueError("invalid stream interval")
    x = stream
    unigram = np.bincount(
        np.asarray(x[start:stop], dtype=np.int64), minlength=vocabulary_size,
    ).astype(np.int64, copy=False)
    reveal = max(2, start)
    target = np.asarray(x[reveal:stop], dtype=np.int64)
    lag1 = np.asarray(x[reveal - 1:stop - 1], dtype=np.int64)
    lag2 = np.asarray(x[reveal - 2:stop - 2], dtype=np.int64)
    keys_ya, counts_ya = _sparse(lag1 * vocabulary_size + target)
    keys_yb, counts_yb = _sparse(lag2 * vocabulary_size + target)
    keys_ab, counts_ab = _sparse(lag1 * vocabulary_size + lag2)
    return {
        "unigram": unigram,
        "keys_ya": keys_ya, "counts_ya": counts_ya,
        "keys_yb": keys_yb, "counts_yb": counts_yb,
        "keys_ab": keys_ab, "counts_ab": counts_ab,
    }


def _load_arrays(directory: Path) -> dict[str, np.ndarray]:
    return {
        name: np.load(directory / f"{name}.npy", mmap_mode="r")
        for name in ARRAY_NAMES
    }


def _comparison(
    expected: dict[str, np.ndarray], actual: dict[str, np.ndarray],
) -> dict:
    arrays = {}
    equal = True
    for name in ARRAY_NAMES:
        lhs = np.asarray(expected[name])
        rhs = np.asarray(actual[name])
        same = lhs.shape == rhs.shape and np.array_equal(lhs, rhs)
        equal &= same
        row = {
            "equal": bool(same),
            "expected_shape": list(lhs.shape),
            "actual_shape": list(rhs.shape),
        }
        if not same:
            if lhs.shape == rhs.shape and lhs.size:
                mismatch = np.flatnonzero(lhs != rhs)
                row["first_mismatch"] = int(mismatch[0]) if len(mismatch) else None
                row["mismatch_count"] = int(len(mismatch))
            else:
                row["first_mismatch"] = None
                row["mismatch_count"] = None
        arrays[name] = row
    return {"equal": bool(equal), "arrays": arrays}


def audit_checkpoint(
    stream_root: Path, delta_root: Path, cumulative_root: Path,
    checkpoint: int, materialize_cumulative_root: Path | None = None,
) -> dict:
    manifest = json.loads((stream_root / "manifest.json").read_text())
    edges = [int(value) for value in manifest["edges"]]
    if not 0 <= checkpoint < len(edges):
        raise ValueError(f"checkpoint {checkpoint} outside schedule")
    stream = np.load(stream_root / "stream.npy", mmap_mode="r")
    vocabulary_size = int(manifest["vocabulary_size"])
    start = 0 if checkpoint == 0 else edges[checkpoint - 1]
    stop = edges[checkpoint]
    expected_delta = direct_counts(stream, vocabulary_size, start, stop)
    expected_cumulative = direct_counts(stream, vocabulary_size, 0, stop)
    if materialize_cumulative_root is not None:
        destination = (
            materialize_cumulative_root / f"checkpoint_{checkpoint:03d}"
        )
        destination.mkdir(parents=True, exist_ok=True)
        for name, value in expected_cumulative.items():
            np.save(destination / f"{name}.npy", value)
        payload = {
            "version": 1,
            "kind": "direct_audit_cumulative_counts",
            "checkpoint": checkpoint,
            "prefix": stop,
            "vocabulary_size": vocabulary_size,
            "ids": manifest["ids"],
            "n": int(manifest["n"]),
            "top_k": int(manifest["top_k"]),
            "edges": edges,
        }
        (destination / "manifest.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
    delta = _comparison(
        expected_delta,
        _load_arrays(delta_root / f"checkpoint_{checkpoint:03d}"),
    )
    cumulative = _comparison(
        expected_cumulative,
        _load_arrays(cumulative_root / f"checkpoint_{checkpoint:03d}"),
    )
    return {
        "checkpoint": checkpoint, "start": start, "prefix": stop,
        "delta": delta, "cumulative": cumulative,
        "equal": bool(delta["equal"] and cumulative["equal"]),
    }


def _parse_checkpoints(specification: str, count: int) -> list[int]:
    if specification == "representative":
        return sorted({0, min(1, count - 1), count // 2, count - 1})
    if specification == "all":
        return list(range(count))
    return sorted({int(value) for value in specification.split(",")})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--deltas", required=True)
    parser.add_argument("--counts", required=True)
    parser.add_argument(
        "--checkpoints", default="representative",
        help="representative, all, or a comma-separated checkpoint list",
    )
    parser.add_argument("--out")
    parser.add_argument(
        "--materialize-cumulative-root",
        help=(
            "write independently counted cumulative snapshots under this "
            "audit root; never modifies the supplied production counts"
        ),
    )
    args = parser.parse_args()
    stream_root = Path(args.stream)
    manifest = json.loads((stream_root / "manifest.json").read_text())
    selected = _parse_checkpoints(args.checkpoints, len(manifest["edges"]))
    rows = [
        audit_checkpoint(
            stream_root, Path(args.deltas), Path(args.counts), checkpoint,
            (Path(args.materialize_cumulative_root)
             if args.materialize_cumulative_root else None),
        )
        for checkpoint in selected
    ]
    payload = {"version": 1, "equal": all(row["equal"] for row in rows),
               "checkpoints": rows}
    rendered = json.dumps(payload, indent=2)
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n")
    print(rendered, flush=True)
    if not payload["equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
