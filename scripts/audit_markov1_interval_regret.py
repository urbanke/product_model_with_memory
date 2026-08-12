#!/usr/bin/env python3
"""Localize checkpoint staleness using persisted production artifacts.

For every cumulative count checkpoint, evaluate the canonical layered
Markov-1 prefix codelength

    B_k = -sum_a log2 Q_avg(profile_a at boundary k).

The exact sequential cost of the interval between checkpoints k and k+1 is
therefore B_{k+1} - B_k.  Join that oracle to the *actual normalized* pair1
cost already written by the production scorer.  This avoids rebuilding any
frozen pair tables and cleanly reports where checkpoint staleness is paid.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from product_model_with_memory.codelength import default_l_max
from product_model_with_memory.pooled_lags import (
    _LayeredPredictiveBuilder,
    _log2sumexp_arr,
)
from product_model_with_memory.production_coding import (
    PRODUCTION_SEQUENCE_ESTIMATOR,
    require_production_sequence_estimator,
)
from product_model_with_memory.streams import load_stream


def profile_multiplicities(
    keys: np.ndarray,
    counts: np.ndarray,
    vocabulary_size: int,
) -> Counter:
    """Count identical target-count profiles across Markov context rows."""

    keys = np.asarray(keys, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    if keys.shape != counts.shape or (counts <= 0).any():
        raise ValueError("sparse pair keys/counts must match and be positive")
    if len(keys) and (
        keys[0] < 0
        or keys[-1] >= vocabulary_size**2
        or (np.diff(keys) <= 0).any()
    ):
        raise ValueError("sparse pair keys must be unique, sorted, and valid")
    rows = keys // vocabulary_size
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(rows))]
    stops = np.r_[starts[1:], len(rows)]
    answer: Counter = Counter()
    for lo, hi in zip(starts, stops):
        if hi > lo:
            # Use the builder's canonical ascending partition convention.
            answer[tuple(sorted(
                int(value) for value in counts[lo:hi]
            ))] += 1
    return answer


def checkpoint_directories(root: Path) -> list[Path]:
    paths = sorted(root.glob("checkpoint_*"))
    expected = [f"checkpoint_{index:03d}" for index in range(len(paths))]
    if [path.name for path in paths] != expected:
        raise RuntimeError("count checkpoints are not contiguous from zero")
    return paths


def score_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("checkpoint_*.json"))
    expected = [f"checkpoint_{index:03d}.json"
                for index in range(len(paths))]
    if [path.name for path in paths] != expected:
        raise RuntimeError("score checkpoints are not contiguous from zero")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", required=True,
                        help="root containing cumulative checkpoint_NNN counts")
    parser.add_argument("--scores", required=True,
                        help="root containing production checkpoint_NNN scores")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")

    for key, value in (
        ("PMM_UNIVERSAL_TABLES", "tables/anchors_prod"),
        ("PMM_PHI_LADDER_EVERY", "1"),
        ("PMM_PHI_LADDER_DEGREE", "11"),
        ("PMM_PHI_SADDLE_MIN_L", "54"),
    ):
        os.environ.setdefault(key, value)

    started = time.time()
    counts_root = Path(args.counts)
    scores_root = Path(args.scores)
    count_paths = checkpoint_directories(counts_root)
    scores = score_paths(scores_root)
    if len(count_paths) < 2:
        parser.error("at least two cumulative checkpoints are required")
    if len(scores) != len(count_paths) - 1:
        parser.error("production scores must cover every checkpoint interval")

    first_manifest = json.loads(
        (count_paths[0] / "manifest.json").read_text()
    )
    vocabulary_size = int(first_manifest["vocabulary_size"])
    edges = [int(value) for value in first_manifest["edges"]]
    if len(edges) != len(count_paths):
        parser.error("count manifest edge count differs from checkpoint count")
    _, stream_meta = load_stream(first_manifest["ids"])
    original_bytes = int(stream_meta["n_bytes"])

    builder = _LayeredPredictiveBuilder(
        vocabulary_size,
        default_l_max(vocabulary_size),
        None,
        args.jobs,
        None,
    )
    log_depths = math.log2(builder.l_max)
    oracle_prefix_bits = []
    checkpoint_rows = []
    for index, checkpoint in enumerate(count_paths):
        manifest = json.loads((checkpoint / "manifest.json").read_text())
        if int(manifest["checkpoint"]) != index:
            raise RuntimeError(f"wrong checkpoint index in {checkpoint}")
        if int(manifest["prefix"]) != edges[index]:
            raise RuntimeError(f"wrong prefix in {checkpoint}")
        if int(manifest["vocabulary_size"]) != vocabulary_size:
            raise RuntimeError("vocabulary changed across count checkpoints")
        keys = np.load(checkpoint / "keys_ya.npy", mmap_mode="r")
        counts = np.load(checkpoint / "counts_ya.npy", mmap_mode="r")
        multiplicity = profile_multiplicities(
            keys, counts, vocabulary_size
        )
        missing = [profile for profile in multiplicity
                   if profile not in builder.memo]
        batch = 2_000
        for start in range(0, len(missing), batch):
            builder._ensure_families({
                profile: () for profile in missing[start:start + batch]
            })
        bits = -float(sum(
            repeats * (
                _log2sumexp_arr(builder.memo[profile]) - log_depths
            )
            for profile, repeats in multiplicity.items()
        ))
        oracle_prefix_bits.append(bits)
        row = {
            "checkpoint": index,
            "prefix": edges[index],
            "context_rows": int(sum(multiplicity.values())),
            "distinct_profiles": len(multiplicity),
            "oracle_prefix_transition_bits": bits,
        }
        checkpoint_rows.append(row)
        print(json.dumps(row), flush=True)

    intervals = []
    cumulative_regret = 0.0
    for index, path in enumerate(scores):
        score = json.loads(path.read_text())
        require_production_sequence_estimator(
            score.get("sequence_estimator", PRODUCTION_SEQUENCE_ESTIMATOR),
            source=str(path),
        )
        if int(score["checkpoint"]) != index:
            raise RuntimeError(f"wrong checkpoint index in {path}")
        if int(score["fit_prefix"]) != edges[index]:
            raise RuntimeError(f"wrong fit prefix in {path}")
        if int(score["next_prefix"]) != edges[index + 1]:
            raise RuntimeError(f"wrong next prefix in {path}")
        actual = float(score["pair1_bits"])
        oracle = oracle_prefix_bits[index + 1] - oracle_prefix_bits[index]
        regret = actual - oracle
        cumulative_regret += regret
        tokens = edges[index + 1] - edges[index]
        intervals.append({
            "checkpoint": index,
            "start": edges[index],
            "stop": edges[index + 1],
            "tokens": tokens,
            "oracle_sequential_bits": oracle,
            "production_frozen_pair1_bits": actual,
            "regret_bits": regret,
            "regret_bits_per_token": regret / tokens,
            "regret_bpc_contribution": regret / original_bytes,
            "cumulative_regret_bits": cumulative_regret,
        })

    total_actual = float(sum(
        row["production_frozen_pair1_bits"] for row in intervals
    ))
    total_oracle = oracle_prefix_bits[-1] - oracle_prefix_bits[0]
    total_regret = total_actual - total_oracle
    for row in intervals:
        row["fraction_of_scored_regret"] = (
            row["regret_bits"] / total_regret if total_regret else None
        )
    ranked = sorted(intervals, key=lambda row: row["regret_bits"], reverse=True)
    payload = {
        "version": 1,
        "sequence_estimator": PRODUCTION_SEQUENCE_ESTIMATOR,
        "counts": str(counts_root),
        "scores": str(scores_root),
        "vocabulary_size": vocabulary_size,
        "l_max": builder.l_max,
        "edges": edges,
        "original_bytes": original_bytes,
        "unscored_initial_prefix_transition_bits": oracle_prefix_bits[0],
        "scored_production_pair1_bits": total_actual,
        "scored_oracle_sequential_bits": total_oracle,
        "scored_regret_bits": total_regret,
        "scored_regret_bpc": total_regret / original_bytes,
        "largest_regret_checkpoints": [
            row["checkpoint"] for row in ranked[:5]
        ],
        "checkpoint_prefixes": checkpoint_rows,
        "intervals": intervals,
        "seconds": time.time() - started,
    }
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / "results.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(destination / "results.json")
    print(json.dumps({
        "scored_regret_bits": total_regret,
        "scored_regret_bpc": total_regret / original_bytes,
        "largest_regret_checkpoints": payload["largest_regret_checkpoints"],
        "written": str(destination / "results.json"),
        "seconds": payload["seconds"],
    }), flush=True)


if __name__ == "__main__":
    main()
