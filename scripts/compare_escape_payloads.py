#!/usr/bin/env python3
"""Compare KT and layered-memoryless codes for escaped tokenizer IDs.

The reduced stream determines the positions of escape symbols.  At those
positions the original excluded tokenizer IDs form a separate sequence over
the excluded alphabet.  Both codes below are honest sequential codes for
that sequence; the layered result uses the same depth-averaged memoryless
mixture as the paper's memoryless baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from product_model_with_memory.codelength import (
    depth_averaged_codelength_profiles,
    profile_of,
)
from product_model_with_memory.streams import load_stream


def kt_bits(counts: np.ndarray, alphabet_size: int) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    alpha = 0.5
    log_probability = (
        gammaln(alphabet_size * alpha)
        - gammaln(total + alphabet_size * alpha)
        + float(np.sum(gammaln(counts + alpha) - gammaln(alpha)))
    )
    return -float(log_probability) / np.log(2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", required=True)
    parser.add_argument(
        "--top-k", required=True,
        help="comma-separated retained-token counts, e.g. 1023,4095,16383",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    started = time.time()
    ids, metadata = load_stream(args.ids, mmap_mode="r")
    alphabet = int(metadata["alphabet"])
    counts = np.zeros(alphabet, dtype=np.int64)
    chunk = 8_000_000
    for start in range(0, len(ids), chunk):
        values = np.asarray(ids[start:start + chunk], dtype=np.int64)
        counts += np.bincount(values, minlength=alphabet)
    order = np.argsort(-counts, kind="stable")

    rows = []
    for top_k in [int(value) for value in args.top_k.split(",")]:
        keep = order[:top_k][counts[order[:top_k]] > 0]
        retained = np.zeros(alphabet, dtype=bool)
        retained[keep] = True
        escaped_counts = counts[~retained]
        excluded_alphabet = alphabet - len(keep)
        profile = profile_of(escaped_counts)
        if not profile:
            layered_bits = 0.0
            kt_payload_bits = 0.0
        else:
            result = depth_averaged_codelength_profiles(
                {0: profile}, d=excluded_alphabet, jobs=args.jobs,
            )[0]
            layered_bits = -float(result.log2_q_avg)
            kt_payload_bits = kt_bits(escaped_counts, excluded_alphabet)
        row = {
            "V": top_k + 1,
            "retained_tokens": len(keep),
            "excluded_alphabet": excluded_alphabet,
            "observed_excluded_types": len(profile),
            "escaped_tokens": int(escaped_counts.sum()),
            "kt_bits": kt_payload_bits,
            "layered_bits": layered_bits,
            "kt_bpc": kt_payload_bits / float(metadata["n_bytes"]),
            "layered_bpc": layered_bits / float(metadata["n_bytes"]),
            "layered_saving_bpc": (
                kt_payload_bits - layered_bits
            ) / float(metadata["n_bytes"]),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    payload = {
        "version": 1,
        "stream": args.ids,
        "n_tokens": int(metadata["n_tokens"]),
        "n_bytes": int(metadata["n_bytes"]),
        "tokenizer_alphabet": alphabet,
        "rows": rows,
        "seconds": time.time() - started,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    print(f"written: {destination}", flush=True)


if __name__ == "__main__":
    main()
