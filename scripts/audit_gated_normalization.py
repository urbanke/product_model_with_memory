#!/usr/bin/env python3
"""Measure cancellation in the gated scorer without changing production."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration_score_states import load_state


def _corrections(problem, result):
    first: dict[int, dict[int, float]] = {}
    for y, a, value in zip(
        problem.active_ya_y, problem.active_ya_a, result.correction_ya,
    ):
        first.setdefault(int(a), {})[int(y)] = float(value)
    second: dict[int, dict[int, float]] = {}
    for y, b, value in zip(
        problem.active_yb_y, problem.active_yb_b, result.correction_yb,
    ):
        second.setdefault(int(b), {})[int(y)] = float(value)
    return first, second


def audit_checkpoint(root: Path, checkpoint: int) -> dict:
    state = (
        root / "fitted" / f"checkpoint_{checkpoint:03d}" / "states"
        / f"checkpoint_{checkpoint:03d}.npz"
    )
    problem, result, _, _, prefix = load_state(state)
    stream_root = root / "reduced_stream"
    manifest = json.loads((stream_root / "manifest.json").read_text())
    stop = int(manifest["edges"][checkpoint + 1])
    stream = np.load(stream_root / "stream.npy", mmap_mode="r")
    a = np.asarray(stream[prefix - 1:stop - 1], dtype=np.int64)
    b = np.asarray(stream[prefix - 2:stop - 2], dtype=np.int64)
    keys, counts = np.unique(
        a * problem.vocabulary_size + b, return_counts=True,
    )
    supported = set((
        problem.edge_a * problem.vocabulary_size + problem.edge_b
    ).tolist())
    first, second = _corrections(problem, result)
    log_base = result.log_base_y - logsumexp(result.log_base_y)
    base = np.exp(log_base)
    correction_bits = 0.0
    affected_records = 0
    maximum_deficit = 0.0
    worst = None
    for key, count in zip(keys, counts):
        if int(key) not in supported:
            continue
        aa, bb = divmod(int(key), problem.vocabulary_size)
        combined = dict(first.get(aa, {}))
        for y, value in second.get(bb, {}).items():
            combined[y] = combined.get(y, 0.0) + value
        corrected_y = np.fromiter(combined, dtype=np.int64)
        corrected_mass = float(base[corrected_y].sum())
        # This is the production expression under audit.
        old_background = max(0.0, 1.0 - corrected_mass)
        old_log_background = (
            np.log(old_background) if old_background > 0.0 else -np.inf
        )
        # Avoid subtraction when the complement is the small quantity.
        if corrected_mass <= 0.5:
            stable_log_background = np.log1p(-corrected_mass)
        else:
            uncorrected = np.ones(problem.vocabulary_size, dtype=bool)
            uncorrected[corrected_y] = False
            stable_log_background = (
                logsumexp(log_base[uncorrected])
                if uncorrected.any() else -np.inf
            )
        corrected_log_mass = (
            logsumexp(np.fromiter(
                (log_base[y] + value for y, value in combined.items()),
                dtype=np.float64,
            )) if combined else -np.inf
        )
        old_log_normalizer = np.logaddexp(
            old_log_background, corrected_log_mass,
        )
        stable_log_normalizer = np.logaddexp(
            stable_log_background, corrected_log_mass,
        )
        normalization_sum = float(np.exp(
            stable_log_normalizer - old_log_normalizer
        ))
        deficit = abs(normalization_sum - 1.0)
        if deficit > 1e-12:
            affected_records += int(count)
        correction_bits += int(count) * (
            stable_log_normalizer - old_log_normalizer
        ) / np.log(2.0)
        if deficit > maximum_deficit:
            maximum_deficit = deficit
            worst = {
                "a": aa, "b": bb, "records": int(count),
                "corrected_targets": len(combined),
                "old_background_mass": old_background,
                "stable_background_mass": (
                    float(np.exp(stable_log_background))
                    if np.isfinite(stable_log_background) else 0.0
                ),
                "old_probability_sum": normalization_sum,
            }
    score = json.loads((
        root / "scores" / f"checkpoint_{checkpoint:03d}.json"
    ).read_text())
    return {
        "checkpoint": checkpoint,
        "prefix": prefix,
        "records": int(len(a)),
        "affected_records": affected_records,
        "maximum_normalization_deficit": maximum_deficit,
        "candidate_bits_before": float(score["candidate_bits"]),
        "candidate_bit_correction": correction_bits,
        "candidate_bits_after": float(score["candidate_bits"] + correction_bits),
        "worst_context": worst,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--checkpoints", default="0,15,30",
        help="comma-separated fitted checkpoints",
    )
    parser.add_argument("--out")
    args = parser.parse_args()
    rows = [
        audit_checkpoint(Path(args.root), int(value))
        for value in args.checkpoints.split(",")
    ]
    payload = {
        "version": 1,
        "root": args.root,
        "candidate_bits_before": sum(row["candidate_bits_before"] for row in rows),
        "candidate_bit_correction": sum(
            row["candidate_bit_correction"] for row in rows
        ),
        "candidate_bits_after": sum(row["candidate_bits_after"] for row in rows),
        "rows": rows,
    }
    rendered = json.dumps(payload, indent=2)
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
