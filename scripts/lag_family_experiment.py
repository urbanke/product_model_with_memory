#!/usr/bin/env python3
"""Lag family: the distance profile of memory value (outlook scheme 5, level 0).

One honest member per lag delta (state = token delta steps back, full
vocabulary resolution), all coding the same tokens, mixed uniformly.
The member codelengths trace how predictive information decays with
distance; the posterior over delta locates the effective lag.

Example:

    python scripts/lag_family_experiment.py --corpus data/text8 \
        --top-k 1023 --deltas 0,1,2,3,4,6,8 --jobs 20 \
        --out output/lag_family_v1024
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from product_model_with_memory.corpus import load_tokens
from product_model_with_memory.pairs import empirical_entropies, reduce_vocabulary
from product_model_with_memory.lag_family import lag_family_codelengths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument(
        "--deltas", required=True,
        help="comma-separated lags, e.g. 0,1,2,3,4,6,8 (0 = memoryless)",
    )
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--l-max", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    deltas = [int(x) for x in args.deltas.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"

    t0 = time.time()
    tokens = load_tokens(args.corpus)
    if args.n:
        tokens = tokens[: args.n]
    reduced, vocab = reduce_vocabulary(tokens, args.top_k)
    V = len(vocab)
    ent = empirical_entropies(reduced)
    print(
        f"V={V} n={len(reduced):,} deltas={deltas}  "
        f"targets: H_unigram={ent['unigram_bits']:.3f} "
        f"H(next|prev)={ent['conditional_bits']:.3f}",
        flush=True,
    )

    def progress(event, _unused) -> None:
        kind, k, total = event
        if kind == "profiles":
            print(f"  {total:,} states across members, {k:,} unique profiles "
                  f"({time.time()-t0:.0f}s)", flush=True)
        elif kind == "tables" and (k % 100 == 0 or k == total):
            print(f"  tables: {k}/{total} orders built ({time.time()-t0:.0f}s)",
                  flush=True)
        elif kind == "depth" and (k % 5 == 0 or k == total):
            print(f"  evaluation: depth {k}/{total} done ({time.time()-t0:.0f}s)",
                  flush=True)

    out = lag_family_codelengths(
        reduced,
        vocabulary_size=V,
        deltas=deltas,
        l_max=args.l_max,
        cache_dir=cache_dir,
        jobs=args.jobs,
        progress=progress,
    )

    print("\ndistance profile (bits/token):", flush=True)
    for d in sorted(out["member_bits_per_token"]):
        bits = out["member_bits_per_token"][d]
        post = out["posterior"][d]
        states = out["states_observed"][d]
        label = "memoryless" if d == 0 else f"lag {d}"
        print(f"  {label:>10}: {bits:.4f}  (states {states:,}, "
              f"posterior {post:.3g})", flush=True)
    print(
        f"family mixture: {out['family_bits_per_token']:.4f} bits/token; "
        f"targets H_unigram={ent['unigram_bits']:.3f}, "
        f"H(next|prev)={ent['conditional_bits']:.3f}",
        flush=True,
    )

    payload = {
        "corpus": args.corpus,
        "top_k": args.top_k,
        "vocabulary_size": V,
        "n_tokens": len(reduced),
        "empirical": ent,
        **{
            k: (v if not isinstance(v, dict)
                else {str(kk): vv for kk, vv in v.items()})
            for k, v in out.items()
        },
        "seconds": time.time() - t0,
    }
    out_file = out_dir / "results.json"
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"written: {out_file} ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
