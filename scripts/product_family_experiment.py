#!/usr/bin/env python3
"""Order-two product-state family: states (b_M1(prev), b_M2(prev-prev)).

Fixes the emission vocabulary (top K + <unk>, V = K+1), scores every
member of a grid of (M1, M2) state maps by its honest per-state
depth-averaged codelength, and reports the uniform-mixture family
codelength and the posterior over the grid.  (M,0) members reproduce the
first-order family; (0,0) is memoryless.

Example:

    python scripts/product_family_experiment.py --corpus data/text8 \
        --top-k 1023 \
        --grid 0:0,64:0,256:0,1024:0,1024:4,1024:16,1024:64,256:16,64:64 \
        --jobs 20 --out output/product_family_v1024
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from product_model_with_memory.corpus import load_tokens
from product_model_with_memory.pairs import empirical_entropies, reduce_vocabulary
from product_model_with_memory.product_family import product_family_codelengths
from product_model_with_memory.streams import (
    bits_per_character,
    load_stream,
    reduce_ids,
    state_order_by_id,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=None,
                        help="text8-style whitespace corpus (word tokens)")
    parser.add_argument("--ids", default=None,
                        help="a stream directory from make_stream.py; give "
                             "exactly one of --corpus or --ids")
    parser.add_argument("--state-order", default="id",
                        choices=["id", "frequency"],
                        help="which symbols leave the backoff state, at BOTH "
                             "lags.  'id' (default) takes the smallest "
                             "VOCABULARY ids: fixed before the file is seen, "
                             "so the code is admissible.  'frequency' ranks "
                             "by counts in THIS file and is not admissible "
                             "unless the ranking is paid for")
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument(
        "--grid", required=True,
        help="comma-separated M1:M2 pairs, e.g. 0:0,256:0,1024:16",
    )
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--l-max", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"
    grid = [
        (int(a), int(b))
        for a, b in (pair.split(":") for pair in args.grid.split(","))
    ]

    if (args.corpus is None) == (args.ids is None):
        raise SystemExit("give exactly one of --corpus or --ids")

    t0 = time.time()
    meta = None
    state_order = None
    if args.ids:
        ids, meta = load_stream(args.ids)
        if args.n and args.n < len(ids):
            ids = ids[: args.n]
            meta = dict(meta, truncated_to_tokens=int(args.n))
            meta.pop("fixed_bits", None)
        reduced, V, capped, keep = reduce_ids(ids, args.top_k,
                                              return_keep=True)
        reduced = np.asarray(reduced, dtype=np.int64)
        if args.state_order == "id":
            state_order = state_order_by_id(keep)
        fixed = (f"fixed_bits {meta['fixed_bits']:,.0f}"
                 if "fixed_bits" in meta
                 else "fixed_bits not applicable on a prefix")
        print(f"representation {meta['representation']!r} from "
              f"{meta['source_file']}: {meta['n_tokens']:,} tokens over an "
              f"alphabet of {meta['alphabet']:,}, "
              f"{meta['bytes_per_token']:.2f} bytes/token; {fixed} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if capped:
            print(f"  WARNING: {capped:,} positions fall outside the top "
                  f"{args.top_k} and are coded as <unk>", flush=True)
        print("  state map: symbols leave the backoff state in order of "
              + ("ascending vocabulary id (admissible)"
                 if state_order is not None else
                 "frequency IN THIS FILE (NOT admissible)"), flush=True)
    else:
        tokens = load_tokens(args.corpus)
        if args.n:
            tokens = tokens[: args.n]
        reduced, vocab = reduce_vocabulary(tokens, args.top_k)
        V = len(vocab)
    ent = empirical_entropies(reduced)
    print(
        f"V={V} n={len(reduced):,} members={len(grid)}  "
        f"targets: H_unigram={ent['unigram_bits']:.3f} "
        f"H(next|prev)={ent['conditional_bits']:.3f}",
        flush=True,
    )

    def progress(event, _unused) -> None:
        kind, k, total = event
        if kind == "member":
            print(f"  member {k}/{total} counted ({time.time()-t0:.0f}s)",
                  flush=True)
        elif kind == "profiles":
            print(f"  {k:,} unique successor profiles to evaluate "
                  f"({time.time()-t0:.0f}s)", flush=True)
        elif kind == "tables" and (k % 100 == 0 or k == total):
            print(f"  tables: {k}/{total} orders built ({time.time()-t0:.0f}s)",
                  flush=True)
        elif kind == "depth" and (k % 5 == 0 or k == total):
            print(f"  evaluation: depth {k}/{total} done ({time.time()-t0:.0f}s)",
                  flush=True)

    out = product_family_codelengths(
        reduced,
        vocabulary_size=V,
        grid=grid,
        l_max=args.l_max,
        cache_dir=cache_dir,
        jobs=args.jobs,
        progress=progress,
        state_order=state_order,
    )

    print(f"\n{'M1':>6} {'M2':>6} {'states':>8} {'bits/token':>11} "
          f"{'posterior':>10}", flush=True)
    for k in out["grid"]:
        print(
            f"{k[0]:>6} {k[1]:>6} {out['member_states_observed'][k]:>8,} "
            f"{out['member_bits_per_token'][k]:>11.4f} "
            f"{out['posterior'][k]:>10.2e}",
            flush=True,
        )
    best = out["best_member"]
    print(
        f"family mixture: {out['family_bits_per_token']:.4f} bits/token "
        f"(best member M1={best[0]}, M2={best[1]}: "
        f"{out['member_bits_per_token'][best]:.4f}); "
        f"targets H_unigram={ent['unigram_bits']:.3f}, "
        f"H(next|prev)={ent['conditional_bits']:.3f}",
        flush=True,
    )

    if meta is not None and "fixed_bits" in meta:
        total_bits = -out["family_bits_per_token"] * out["n_coded"]
        best = out["best_member"]
        bpc = bits_per_character(
            out["member_bits_per_token"][best] * out["n_coded"], meta)
        print(f"  in bits per character of the original file: {bpc:.4f} "
              f"(includes fixed_bits {meta['fixed_bits']:,.0f})", flush=True)

    payload = {
        "corpus": args.corpus or args.ids,
        "stream": meta,
        "state_order": args.state_order if args.ids else "frequency",
        "state_order_admissible": bool(args.ids and args.state_order == "id"),
        "top_k": args.top_k,
        "vocabulary_size": V,
        "n_tokens": len(reduced),
        "empirical": ent,
        "n_coded": out["n_coded"],
        "l_max": out["l_max"],
        "unique_profiles": out["unique_profiles"],
        "grid": [f"{a}:{b}" for a, b in out["grid"]],
        "member_bits_per_token": {
            f"{a}:{b}": v for (a, b), v in out["member_bits_per_token"].items()
        },
        "member_states_observed": {
            f"{a}:{b}": v
            for (a, b), v in out["member_states_observed"].items()
        },
        "posterior": {
            f"{a}:{b}": v for (a, b), v in out["posterior"].items()
        },
        "family_bits_per_token": out["family_bits_per_token"],
        "best_member": f"{best[0]}:{best[1]}",
        "seconds": time.time() - t0,
    }
    out_file = out_dir / "results.json"
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"written: {out_file} ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
