#!/usr/bin/env python3
"""Sliding-window pair experiment (plug-in conditionals from the joint).

Reduces the vocabulary to the top K tokens plus <unk>, estimates the joint
distribution of sliding-window pairs with the depth-averaged product-simplex
mixture (whole corpus, in-sample), derives conditional probabilities, and
compares the conditional log-loss to the empirical conditional entropy
H(next|prev) of the same reduced stream.

Example (first rung of the ladder):

    python scripts/pair_experiment.py --corpus data/text8 --top-k 255 \
        --jobs 20 --out output/pairs_text8_k255
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from product_model_with_memory import default_l_max
from product_model_with_memory.corpus import load_tokens
from product_model_with_memory.pairs import (
    conditional_log_loss,
    depth_averaged_predictive,
    empirical_entropies,
    pair_counts,
    reduce_vocabulary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=None,
                        help="use only the first n tokens (default: all)")
    parser.add_argument("--d-pair", type=int, default=None,
                        help="pair alphabet size (default: (K+1)^2)")
    parser.add_argument("--l-max", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"

    t0 = time.time()
    tokens = load_tokens(args.corpus)
    if args.n:
        tokens = tokens[: args.n]
    reduced, vocab = reduce_vocabulary(tokens, args.top_k)
    V = len(vocab)
    d_pair = args.d_pair or V * V
    l_max = args.l_max or default_l_max(d_pair)

    ent = empirical_entropies(reduced)
    pairs = pair_counts(reduced)
    print(
        f"K={args.top_k} V={V} d_pair={d_pair} L_max={l_max} "
        f"n={len(reduced):,} pairs observed={len(pairs):,}",
        flush=True,
    )
    print(
        f"targets: H_unigram={ent['unigram_bits']:.3f}  "
        f"H(next|prev)={ent['conditional_bits']:.3f} bits/token",
        flush=True,
    )

    def progress(event, _unused) -> None:
        kind, k, total = event
        if kind == "tables" and (k % 100 == 0 or k == total):
            print(f"  tables: {k}/{total} orders built ({time.time()-t0:.0f}s)",
                  flush=True)
        elif kind == "depth" and (k % 5 == 0 or k == total):
            print(f"  predictive: depth {k}/{total} done ({time.time()-t0:.0f}s)",
                  flush=True)

    predictive = depth_averaged_predictive(
        pairs, d=d_pair, l_max=l_max, cache_dir=cache_dir, jobs=args.jobs,
        progress=progress,
    )
    print(
        f"  predictive normalization error before renormalization: "
        f"{predictive.normalization_error:.2e}",
        flush=True,
    )
    losses = conditional_log_loss(pairs, predictive, vocabulary_size=V)

    cond = losses["conditional_bits_per_token"]
    mode_l = 1 + max(range(l_max), key=lambda i: predictive.depth_posterior[i])
    print(
        f"\nplug-in conditional log-loss: {cond:.3f} bits/token  "
        f"(H(next|prev)={ent['conditional_bits']:.3f}, "
        f"gap={cond - ent['conditional_bits']:.3f}); "
        f"unigram entropy {ent['unigram_bits']:.3f}; "
        f"joint posterior mode L={mode_l}",
        flush=True,
    )

    payload = {
        "corpus": args.corpus,
        "top_k": args.top_k,
        "n_tokens": len(reduced),
        "vocabulary_size": V,
        "d_pair": d_pair,
        "l_max": l_max,
        "pairs_observed": len(pairs),
        "empirical": ent,
        "plugin_conditional_bits_per_token": cond,
        "gap_to_conditional_entropy": cond - ent["conditional_bits"],
        "joint_bits_per_pair_in_sample": losses["joint_bits_per_pair"],
        "normalization_error": predictive.normalization_error,
        "depth_posterior": list(predictive.depth_posterior),
        "posterior_mode_depth": mode_l,
        "seconds": time.time() - t0,
    }
    out_file = out_dir / "results.json"
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"written: {out_file} ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
