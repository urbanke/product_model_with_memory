#!/usr/bin/env python3
"""Aggregate scheduled interval scores into honest full-file bpc totals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from product_model_with_memory.streams import load_stream


def kt_multinomial_bits(counts: np.ndarray, alphabet_size: int) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    alpha = 0.5
    log_probability = (
        gammaln(alphabet_size * alpha)
        - gammaln(total + alphabet_size * alpha)
        + float(np.sum(gammaln(counts + alpha) - gammaln(alpha)))
    )
    return -float(log_probability) / np.log(2.0)


def chunked_counts(ids: np.ndarray, stop: int, alphabet_size: int) -> np.ndarray:
    counts = np.zeros(alphabet_size, dtype=np.int64)
    chunk_size = 8_000_000
    for start in range(0, stop, chunk_size):
        chunk = np.asarray(
            ids[start:min(start + chunk_size, stop)], dtype=np.int64
        )
        counts += np.bincount(chunk, minlength=alphabet_size)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    reduced_root = root / "reduced_stream"
    reduced_manifest = json.loads(
        (reduced_root / "manifest.json").read_text()
    )
    edges = [int(value) for value in reduced_manifest["edges"]]
    if int(reduced_manifest["n"]) != args.n:
        parser.error("reduced stream length differs from --n")
    if int(reduced_manifest["top_k"]) != args.top_k:
        parser.error("reduced stream top-k differs from --top-k")
    if edges[-1] != args.n:
        parser.error("last checkpoint does not cover the complete stream")

    score_paths = sorted((root / "scores").glob("checkpoint_*.json"))
    if len(score_paths) != len(edges) - 1:
        parser.error(
            f"expected {len(edges) - 1} interval scores, found "
            f"{len(score_paths)}"
        )
    rows = [json.loads(path.read_text()) for path in score_paths]
    for checkpoint, row in enumerate(rows):
        expected_start = edges[checkpoint]
        expected_stop = edges[checkpoint + 1]
        if int(row["checkpoint"]) != checkpoint:
            parser.error(f"score {score_paths[checkpoint]} has wrong checkpoint")
        if int(row["fit_prefix"]) != expected_start:
            parser.error(f"score {checkpoint} has wrong fit prefix")
        if int(row["next_prefix"]) != expected_stop:
            parser.error(f"score {checkpoint} has wrong next prefix")

    reduced = np.load(reduced_root / "stream.npy", mmap_mode="r")
    vocabulary_size = args.top_k + 1
    first_prefix = edges[0]
    initial_counts = np.bincount(
        np.asarray(reduced[:first_prefix], dtype=np.int64),
        minlength=vocabulary_size,
    )
    initial_bits = kt_multinomial_bits(initial_counts, vocabulary_size)
    interval_bits = {
        name: sum(float(row[f"{name}_bits"]) for row in rows)
        for name in ("candidate", "star", "pair1")
    }

    ids, metadata = load_stream(args.ids, mmap_mode="r")
    if args.n != int(metadata["n_tokens"]):
        parser.error("honest full-file accounting requires n == n_tokens")
    tokenizer_alphabet = int(metadata["alphabet"])
    original_counts = chunked_counts(ids, args.n, tokenizer_alphabet)
    order = np.argsort(-original_counts, kind="stable")
    keep = order[:args.top_k][original_counts[order[:args.top_k]] > 0]
    subset_bits = float(
        (
            gammaln(tokenizer_alphabet + 1)
            - gammaln(len(keep) + 1)
            - gammaln(tokenizer_alphabet - len(keep) + 1)
        ) / np.log(2.0)
    )
    tokenizer_vocabulary_bits = float(metadata.get("fixed_bits", 0.0))
    vocabulary_bits = tokenizer_vocabulary_bits + subset_bits
    retained = np.zeros(tokenizer_alphabet, dtype=bool)
    retained[keep] = True
    escaped_counts = original_counts[~retained]
    escaped_total = int(escaped_counts.sum())
    escape_bits = kt_multinomial_bits(
        escaped_counts, tokenizer_alphabet - len(keep)
    )

    denominator = float(metadata["n_bytes"])
    reduced_bits = {
        name: initial_bits + bits for name, bits in interval_bits.items()
    }
    honest_bits = {
        name: bits + vocabulary_bits + escape_bits
        for name, bits in reduced_bits.items()
    }
    payload = {
        "version": 1,
        "source": str(root),
        "stream": str(Path(args.ids)),
        "representation": metadata.get("representation"),
        "n_tokens": args.n,
        "n_bytes": int(metadata["n_bytes"]),
        "V": vocabulary_size,
        "checkpoints": len(edges),
        "initial_prefix_tokens": first_prefix,
        "initial_prefix_reduced_bits": initial_bits,
        "scored_tokens": sum(int(row["scored_records"]) for row in rows),
        "supported_fraction": (
            sum(
                int(row["scored_records"]) * float(row["supported_fraction"])
                for row in rows
            ) / max(sum(int(row["scored_records"]) for row in rows), 1)
        ),
        "reduced_alphabet_predictive_bpc": {
            name: bits / denominator for name, bits in reduced_bits.items()
        },
        "tokenizer_vocabulary_bits": tokenizer_vocabulary_bits,
        "tokenizer_vocabulary_bpc": tokenizer_vocabulary_bits / denominator,
        "selected_subset_description_bits": subset_bits,
        "selected_subset_description_bpc": subset_bits / denominator,
        "vocabulary_description_bpc": vocabulary_bits / denominator,
        "escaped_tokens": escaped_total,
        "escape_fraction": escaped_total / args.n,
        "escaped_token_payload_bpc": escape_bits / denominator,
        "honest_full_stream_bpc": {
            name: bits / denominator for name, bits in honest_bits.items()
        },
        "candidate_gain_over_star_bpc": (
            honest_bits["star"] - honest_bits["candidate"]
        ) / denominator,
        "candidate_gain_over_pair1_bpc": (
            honest_bits["pair1"] - honest_bits["candidate"]
        ) / denominator,
        "rows": rows,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(destination)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
