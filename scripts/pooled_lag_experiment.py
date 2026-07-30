#!/usr/bin/env python3
"""Pooled lag experts: mixtures and tempered products over a lag set.

Combines the single-lag predictors (the lag family of Section 5.3)
into one code, two pooling rules side by side, averaged over parameter
grids, with the posterior over members reported.  Expert tables are
checkpointed (refreshed C times over the corpus).

Example (laptop scale):

    python scripts/pooled_lag_experiment.py --corpus data/text8 \
        --top-k 4095 --lags 1,2,3,4,6,8 --checkpoints 32 \
        --out output/pooled_lags_v4096
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from product_model_with_memory.corpus import load_tokens
from product_model_with_memory.pairs import empirical_entropies, reduce_vocabulary
from product_model_with_memory.pooled_lags import pooled_lag_codelengths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--lags", default="1,2,3,4,6,8")
    parser.add_argument("--checkpoints", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--rules", default="mix,prod",
        help="which pooling rules to evaluate (mix,prod or one of them)",
    )
    parser.add_argument(
        "--expert-model", choices=["layered", "counts"], default="layered",
        help="per-lag predictor refreshed at checkpoints: the layered "
             "mixture predictive (default, the paper's estimator) or "
             "plain smoothed counts (cheap pilot)")
    parser.add_argument("--cache-dir", default=None,
                        help="moment-table cache (default: OUT/cache)")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="truncate the stream (timing/smoke runs)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    t0 = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lags = tuple(int(x) for x in args.lags.split(","))

    tokens = load_tokens(args.corpus)
    reduced, vocab = reduce_vocabulary(tokens, args.top_k)
    V = len(vocab)
    index = {w: i for i, w in enumerate(vocab)}
    ids = [index[w] for w in reduced]
    if args.max_tokens:
        ids = ids[: args.max_tokens]
        reduced = reduced[: args.max_tokens]
    ent = empirical_entropies(reduced)
    print(
        f"V={V} n={len(ids):,} lags={list(lags)} "
        f"checkpoints={args.checkpoints}  "
        f"H_unigram={ent['unigram_bits']:.3f} H(next|prev)={ent['conditional_bits']:.3f}",
        flush=True,
    )

    rules = args.rules.split(",")
    import numpy as np

    from product_model_with_memory.pooled_lags import (
        power_law_mixture_grid,
        power_law_product_grid,
    )

    mix_grid = power_law_mixture_grid(lags)
    prod_grid = power_law_product_grid(lags)
    if "mix" not in rules:
        mix_grid = (["onehot:mem"], np.eye(len(lags) + 1)[:1])
    if "prod" not in rules:
        prod_grid = ([], np.zeros((0, len(lags))))

    last = {}

    def progress(evt, _):
        kind, done, total = evt
        # throttle: table/depth events print at most every 200 items or
        # at completion; checkpoint events always print
        if kind != "checkpoint" and done != total and done - last.get(kind, 0) < 200:
            return
        last[kind] = done
        print(f"  {kind}: {done}/{total} ({time.time() - t0:.0f}s)", flush=True)

    cache_dir = args.cache_dir or (out_dir / "cache")
    result = pooled_lag_codelengths(
        ids,
        resume_path=out_dir / "resume",
        vocabulary_size=V,
        lags=lags,
        checkpoints=args.checkpoints,
        alpha=args.alpha,
        expert_model=args.expert_model,
        cache_dir=cache_dir,
        mix_grid=mix_grid,
        prod_grid=prod_grid,
        jobs=args.jobs,
        progress=progress,
    )

    payload = {
        "corpus": args.corpus,
        "top_k": args.top_k,
        "vocabulary_size": V,
        "n_tokens": len(ids),
        "alpha": args.alpha,
        "expert_model": args.expert_model,
        "rules": rules,
        "empirical": {
            "unigram_bits": ent["unigram_bits"],
            "conditional_bits": ent["conditional_bits"],
        },
        **result.as_dict(),
        "seconds": time.time() - t0,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))

    d = result.as_dict()
    best = d["best_member"]
    print(f"\nfamily: {d['family_bits_per_token']:.4f} bits/token")
    print(f"best member: {best} = {d['member_bits_per_token'][best]:.4f}")
    for name in sorted(
        d["posterior"], key=d["posterior"].get, reverse=True
    )[:5]:
        print(f"  posterior {d['posterior'][name]:.3f}  {name} "
              f"({d['member_bits_per_token'][name]:.4f})")
    print(f"written: {out_dir / 'results.json'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
