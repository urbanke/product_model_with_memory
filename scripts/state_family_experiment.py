#!/usr/bin/env python3
"""Averaging over a nested family of state maps (see paper Section 5/6).

Fixes the emission vocabulary to the top K tokens + <unk> (V = K+1), builds
the nested family of state maps sigma_M (top-M states + backoff, M=0 being
memoryless), computes each member's exact per-state depth-averaged
codelength (the honest share-nothing construction), the uniform-prior family
mixture, and the posterior over M.

Example (proof of concept, laptop):

    python scripts/state_family_experiment.py --corpus data/text8 \
        --top-k 255 --m-grid 0,1,2,4,8,16,32,64,128,256 \
        --jobs 20 --out output/state_family_v256
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from product_model_with_memory.corpus import load_tokens
from product_model_with_memory.pairs import empirical_entropies, reduce_vocabulary
from product_model_with_memory.state_family import state_family_codelengths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--m-grid", required=True,
                        help="comma-separated top-M state counts, e.g. 0,1,2,4,...,256 (0 = memoryless)")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--l-max", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"
    m_grid = [int(x) for x in args.m_grid.split(",")]

    t0 = time.time()
    tokens = load_tokens(args.corpus)
    if args.n:
        tokens = tokens[: args.n]
    reduced, vocab = reduce_vocabulary(tokens, args.top_k)
    V = len(vocab)
    ent = empirical_entropies(reduced)
    print(
        f"V={V} n={len(reduced):,} family M in {m_grid}  "
        f"targets: H_unigram={ent['unigram_bits']:.3f} "
        f"H(next|prev)={ent['conditional_bits']:.3f}",
        flush=True,
    )

    def progress(event, _unused) -> None:
        kind, k, total = event
        if kind == "tables" and (k % 100 == 0 or k == total):
            print(f"  tables: {k}/{total} orders built ({time.time()-t0:.0f}s)",
                  flush=True)
        elif kind == "depth" and (k % 5 == 0 or k == total):
            print(f"  evaluation: depth {k}/{total} done ({time.time()-t0:.0f}s)",
                  flush=True)

    out = state_family_codelengths(
        reduced,
        vocabulary_size=V,
        m_grid=m_grid,
        l_max=args.l_max,
        cache_dir=cache_dir,
        jobs=args.jobs,
        progress=progress,
    )

    print(f"\n{'M':>6} {'states':>7} {'bits/token':>11} {'posterior':>10}",
          flush=True)
    for m in out["m_grid"]:
        print(
            f"{m:>6} {out['member_states_observed'][m]:>7} "
            f"{out['member_bits_per_token'][m]:>11.4f} "
            f"{out['posterior_over_m'][m]:>10.2e}",
            flush=True,
        )
    print(
        f"family mixture: {out['family_bits_per_token']:.4f} bits/token "
        f"(best member M={out['best_member']}: "
        f"{out['member_bits_per_token'][out['best_member']]:.4f}); "
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
        **{k: v for k, v in out.items()},
        "seconds": time.time() - t0,
    }
    # JSON keys must be strings
    payload["member_bits_per_token"] = {
        str(k): v for k, v in payload["member_bits_per_token"].items()
    }
    payload["member_states_observed"] = {
        str(k): v for k, v in payload["member_states_observed"].items()
    }
    payload["posterior_over_m"] = {
        str(k): v for k, v in payload["posterior_over_m"].items()
    }
    out_file = out_dir / "results.json"
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"written: {out_file} ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
