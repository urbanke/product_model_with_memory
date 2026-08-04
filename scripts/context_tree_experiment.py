#!/usr/bin/env python3
"""Context-tree mixture with layered leaves (paper, outlook scheme 1).

Averages over ALL variable-length suffix state maps up to --depth via the
recursive half-half growth prior, with a depth-averaged layered mixture at
every node.  Reports the family codelength, fixed-depth baselines, and the
MAP pruning's leaf-depth histogram (the effective-memory read-out).

Example:

    python scripts/context_tree_experiment.py --corpus data/text8 \
        --top-k 1023 --depth 2 --jobs 20 --out output/ctree_v1024_d2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from product_model_with_memory.corpus import load_tokens
from product_model_with_memory.pairs import empirical_entropies, reduce_vocabulary
from product_model_with_memory.context_tree import context_tree_codelengths
from product_model_with_memory.streams import load_stream, reduce_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=None,
                        help="raw corpus file (tokenized by load_tokens)")
    parser.add_argument("--ids", default=None,
                        help="token-stream directory (e.g. "
                             "output/streams/bpe_text8) --- the LLM "
                             "tokenization; use THIS for paper numbers")
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--l-max", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--leaf-model", choices=["layered", "kt"], default="layered",
        help="per-node estimator: layered mixture (default) or KT "
             "(classical CTW leaf; needs no moment tables)",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"

    t0 = time.time()
    if (args.ids is None) == (args.corpus is None):
        parser.error("give exactly one of --ids or --corpus")
    if args.ids is not None:
        ids, meta = load_stream(args.ids)
        if args.n:
            ids = ids[: args.n]
        reduced, V, _capped = reduce_ids(ids, args.top_k)
    else:
        tokens = load_tokens(args.corpus)
        if args.n:
            tokens = tokens[: args.n]
        reduced, vocab = reduce_vocabulary(tokens, args.top_k)
        V = len(vocab)
    ent = empirical_entropies(reduced)
    print(
        f"V={V} n={len(reduced):,} max depth={args.depth}  "
        f"targets: H_unigram={ent['unigram_bits']:.3f} "
        f"H(next|prev)={ent['conditional_bits']:.3f}",
        flush=True,
    )

    def progress(event, _unused) -> None:
        kind, k, total = event
        if kind == "profiles":
            print(f"  {total:,} contexts observed, {k:,} unique profiles "
                  f"({time.time()-t0:.0f}s)", flush=True)
        elif kind == "tables" and (k % 100 == 0 or k == total):
            print(f"  tables: {k}/{total} orders built ({time.time()-t0:.0f}s)",
                  flush=True)
        elif kind == "depth" and (k % 5 == 0 or k == total):
            print(f"  evaluation: depth {k}/{total} done ({time.time()-t0:.0f}s)",
                  flush=True)

    out = context_tree_codelengths(
        reduced,
        vocabulary_size=V,
        max_depth=args.depth,
        l_max=args.l_max,
        cache_dir=cache_dir,
        jobs=args.jobs,
        leaf_model=args.leaf_model,
        progress=progress,
    )

    print("\nfixed-depth baselines (complete trees):", flush=True)
    for d, bits in sorted(out["fixed_depth_bits_per_token"].items()):
        print(f"  depth {d}: {bits:.4f} bits/token", flush=True)
    print(
        f"context-tree family: {out['family_bits_per_token']:.4f} bits/token; "
        f"MAP pruning: {out['map_bits_per_token']:.4f} with leaves by depth "
        f"{out['map_leaves_by_depth']}; "
        f"targets H_unigram={ent['unigram_bits']:.3f}, "
        f"H(next|prev)={ent['conditional_bits']:.3f}",
        flush=True,
    )

    payload = {
        "corpus": args.corpus or args.ids,
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
