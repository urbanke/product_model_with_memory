#!/usr/bin/env python3
"""Score one checkpoint interval independently of later fitted states."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration_score_states import load_state

from product_model_with_memory.graphical_calibration import (
    sparse_gated_log_probabilities,
    sparse_pair_log_probabilities,
    sparse_star_log_probabilities,
)
from product_model_with_memory.production_coding import (
    PRODUCTION_SEQUENCE_ESTIMATOR,
)
from product_model_with_memory.streams import load_stream, reduce_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument(
        "--reduced-stream",
        help=(
            "mmap-friendly reduced stream prepared by "
            "prepare_reduced_checkpoint_stream.py; when supplied, avoid "
            "reloading and reducing the complete original token stream"
        ),
    )
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    boundary = parser.add_mutually_exclusive_group(required=True)
    boundary.add_argument("--next-prefix", type=int)
    boundary.add_argument(
        "--next-problem",
        help="constructed checkpoint state supplying only the next boundary",
    )
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    started = time.time()
    problem, result, p_ya, p_yb, prefix = load_state(Path(args.state))
    next_prefix = args.next_prefix
    if args.next_problem:
        with np.load(args.next_problem, allow_pickle=False) as state:
            next_prefix = int(state["prefix"])
    if next_prefix <= prefix:
        parser.error("next prefix must lie after the fitted prefix")
    if args.reduced_stream:
        reduced_source = Path(args.reduced_stream)
        reduced_manifest = json.loads(
            (reduced_source / "manifest.json").read_text()
        )
        if int(reduced_manifest["top_k"]) != args.top_k:
            parser.error("reduced-stream top-k differs from requested top-k")
        if int(reduced_manifest["n"]) != args.n:
            parser.error("reduced-stream length differs from requested n")
        x = np.load(reduced_source / "stream.npy", mmap_mode="r")
    else:
        ids, _ = load_stream(args.ids)
        original = ids[:args.n].astype(np.int64)
        x, _, _ = reduce_ids(original, args.top_k)
    if next_prefix > len(x):
        parser.error("next prefix lies beyond the reduced stream")
    # The shared stream is normally uint16/uint32.  Convert only this
    # interval: the probability routines require integer indexing and the
    # context key products below must not overflow the compact dtype.
    target = np.asarray(x[prefix:next_prefix], dtype=np.int64)
    lag1 = np.asarray(x[prefix - 1:next_prefix - 1], dtype=np.int64)
    lag2 = np.asarray(x[prefix - 2:next_prefix - 2], dtype=np.int64)
    candidate = sparse_gated_log_probabilities(
        problem, result, target, lag1, lag2, p_ya, p_yb
    )
    star = sparse_star_log_probabilities(p_ya, p_yb, target, lag1, lag2)
    pair1 = sparse_pair_log_probabilities(p_ya, target, lag1)
    support = np.sort(
        problem.edge_a * problem.vocabulary_size + problem.edge_b
    )
    keys = lag1 * problem.vocabulary_size + lag2
    positions = np.searchsorted(support, keys)
    covered = (positions < len(support)) & (
        support[np.minimum(positions, len(support) - 1)] == keys
    )
    candidate_bits = -float(candidate.sum()) / np.log(2.0)
    star_bits = -float(star.sum()) / np.log(2.0)
    pair1_bits = -float(pair1.sum()) / np.log(2.0)
    payload = {
        "version": 2,
        "sequence_estimator": PRODUCTION_SEQUENCE_ESTIMATOR,
        "checkpoint": args.checkpoint,
        "fit_prefix": prefix,
        "next_prefix": next_prefix,
        "scored_records": len(target),
        "supported_fraction": float(covered.mean()),
        "candidate_bits": candidate_bits,
        "star_bits": star_bits,
        "pair1_bits": pair1_bits,
        "candidate_bits_per_reduced_token": candidate_bits / len(target),
        "star_bits_per_reduced_token": star_bits / len(target),
        "pair1_bits_per_reduced_token": pair1_bits / len(target),
        "elapsed_seconds": time.time() - started,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(destination)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
