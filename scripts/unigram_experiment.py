#!/usr/bin/env python3
"""Unigram (memoryless) codelength experiment, as in the paper's Section 5.3.

For each prefix length n of a whitespace-tokenized corpus, computes the
empirical unigram entropy H_n, the exact codelength per token of the
depth-averaged product-simplex predictor with L = 1 .. round(2 c* ln d),
and the redundancy (codelength minus H_n).  Results are written as JSON and
printed as a table.

Moment tables stream to a disk cache and are evaluated one depth at a time,
so full-corpus runs need modest RAM; use --jobs to parallelize the table
build across cores.

Example (text8, alphabet d = 2^18):

    python scripts/unigram_experiment.py --corpus data/text8 --d 262144 \
        --checkpoints 10000,100000,1000000,all --jobs 8 \
        --out output/unigram_text8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from product_model_with_memory import (
    default_l_max,
    empirical_entropy_bits,
    load_tokens,
    prefix_counts,
    profile_of,
)
from product_model_with_memory.codelength import (
    depth_averaged_codelength_profiles,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="path to corpus file")
    parser.add_argument("--d", type=int, required=True, help="alphabet size")
    parser.add_argument(
        "--checkpoints",
        required=True,
        help="comma-separated prefix lengths, e.g. 10000,100000 ('all' allowed)",
    )
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--l-max", type=int, default=None)
    parser.add_argument("--laguerre-order", type=int, default=96)
    parser.add_argument(
        "--cache-dir", default=None, help="moment-table cache (default: OUT/cache)"
    )
    parser.add_argument(
        "--jobs", type=int, default=1, help="parallel table-build workers"
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"

    tokens = load_tokens(args.corpus)
    checkpoints = [
        len(tokens) if c.strip() == "all" else int(c)
        for c in args.checkpoints.split(",")
    ]
    l_max = args.l_max or default_l_max(args.d)

    counts = {n: prefix_counts(tokens, n) for n in checkpoints}
    profiles = {n: profile_of(counts[n]) for n in checkpoints}
    entropies = {n: empirical_entropy_bits(counts[n]) for n in checkpoints}

    t0 = time.time()
    print(
        f"tables + evaluation: L<={l_max}, "
        f"{len(set().union(*[set(p) for p in profiles.values()]))} distinct "
        f"count values, cache: {cache_dir}",
        flush=True,
    )

    def progress(event, _unused) -> None:
        kind, k, total = event
        if kind == "tables" and (k % 100 == 0 or k == total):
            print(
                f"  tables: {k}/{total} orders built ({time.time() - t0:.0f}s)",
                flush=True,
            )
        elif kind == "depth" and (k % 5 == 0 or k == total):
            print(
                f"  evaluation: depth {k}/{total} done ({time.time() - t0:.0f}s)",
                flush=True,
            )

    results = depth_averaged_codelength_profiles(
        profiles,
        d=args.d,
        l_max=l_max,
        cache_dir=cache_dir,
        jobs=args.jobs,
        laguerre_order=args.laguerre_order,
        progress=progress,
    )

    rows = []
    print(
        f"{'n':>10} {'types':>8} {'H_n':>7} {'PS avg':>8} {'redund.':>8} {'L*':>4}",
        flush=True,
    )
    for n in checkpoints:
        result = results[n]
        h_n = entropies[n]
        rows.append(
            {
                "n": n,
                "distinct_types": len(counts[n]),
                "d": args.d,
                "l_max": l_max,
                "empirical_entropy_bits": h_n,
                "ps_avg_bits_per_token": result.bits_per_token,
                "redundancy_bits_per_token": result.bits_per_token - h_n,
                "posterior_mode_depth": result.posterior_mode,
                "posterior_top3": sorted(
                    ((L + 1, w) for L, w in enumerate(result.posterior)),
                    key=lambda t: -t[1],
                )[:3],
                "bits_per_token_by_depth": [
                    result.bits_per_token_at_depth(L) for L in range(1, l_max + 1)
                ],
            }
        )
        print(
            f"{n:>10,} {len(counts[n]):>8,} {h_n:>7.3f} "
            f"{result.bits_per_token:>8.3f} "
            f"{result.bits_per_token - h_n:>8.3f} "
            f"{result.posterior_mode:>4}",
            flush=True,
        )

    payload = {
        "corpus": str(args.corpus),
        "d": args.d,
        "l_max": l_max,
        "c_star_note": "L_max = round(2 c* ln d), c* = 1/(1 - EulerGamma)",
        "laguerre_order": args.laguerre_order,
        "seconds": time.time() - t0,
        "rows": rows,
    }
    out_file = out_dir / "results.json"
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"written: {out_file} ({time.time() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
