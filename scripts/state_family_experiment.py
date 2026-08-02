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

import numpy as np

from product_model_with_memory.corpus import load_tokens
from product_model_with_memory.pairs import empirical_entropies, reduce_vocabulary
from product_model_with_memory.state_family import state_family_codelengths
from product_model_with_memory.streams import (
    bits_per_character,
    load_stream,
    reduce_ids,
    state_order_by_id,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus",
                        help="text8-style whitespace corpus (representation "
                             "2, word tokens)")
    parser.add_argument("--ids",
                        help="a stream directory from make_stream.py: runs "
                             "this experiment on ANY representation "
                             "(bytes, our tokenizer, BPE) instead")
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--m-grid", required=True,
                        help="comma-separated top-M state counts, e.g. 0,1,2,4,...,256 (0 = memoryless)")
    parser.add_argument("--state-order", default="id",
                        choices=["id", "frequency"],
                        help="which symbols member M promotes to states. "
                             "'id' (default) takes the M smallest "
                             "VOCABULARY ids: fixed before the file is "
                             "seen, so the decoder can reproduce it and "
                             "the code is admissible.  'frequency' ranks "
                             "by counts in THIS file, which is not "
                             "admissible unless the ranking is paid for "
                             "(~1.5 Mbit at V = 100,277); it exists to "
                             "reproduce the earlier text8 runs")
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

    if (args.corpus is None) == (args.ids is None):
        raise SystemExit("give exactly one of --corpus or --ids")

    t0 = time.time()
    meta = None
    capped = 0
    if args.ids:
        ids, meta = load_stream(args.ids)
        if args.n and args.n < len(ids):
            # A prefix of the stream covers a prefix of the FILE whose
            # length we no longer know, and `fixed_bits` was measured on
            # the whole file, so bits per character is not defined here.
            # The probe is about cost, not codelength.
            ids = ids[: args.n]
            meta = dict(meta, truncated_to_tokens=int(args.n))
            meta.pop("fixed_bits", None)
        reduced, V, capped, keep = reduce_ids(ids, args.top_k,
                                              return_keep=True)
        reduced = np.asarray(reduced, dtype=np.int64)
        state_order = (state_order_by_id(keep)
                       if args.state_order == "id" else None)
        fixed = (f"fixed_bits {meta['fixed_bits']:,.0f}"
                 if "fixed_bits" in meta
                 else "fixed_bits not applicable on a prefix")
        print(f"representation {meta['representation']!r} from "
              f"{meta['source_file']}: {meta['n_tokens']:,} tokens over an "
              f"alphabet of {meta['alphabet']:,}, "
              f"{meta['bytes_per_token']:.2f} bytes/token; {fixed} "
              f"({time.time()-t0:.0f}s)",
              flush=True)
        if capped:
            print(f"  WARNING: {capped:,} of {len(reduced):,} positions "
                  f"({100*capped/len(reduced):.2f}%) fall outside the top "
                  f"{args.top_k} and are coded as <unk>; the capped model "
                  f"is NOT a complete code until that is accounted for",
                  flush=True)
    else:
        tokens = load_tokens(args.corpus)
        if args.n:
            tokens = tokens[: args.n]
        reduced, vocab = reduce_vocabulary(tokens, args.top_k)
        V = len(vocab)
        state_order = None
    ent = empirical_entropies(reduced)
    print(
        f"V={V} n={len(reduced):,} family M in {m_grid}  "
        f"targets: H_unigram={ent['unigram_bits']:.3f} "
        f"H(next|prev)={ent['conditional_bits']:.3f}",
        flush=True,
    )
    if len(m_grid) > 1 or (m_grid and 0 < m_grid[0] < V - 1):
        print(f"  state map: member M keeps the M "
              + ("smallest vocabulary ids (admissible: fixed by the "
                 "vocabulary, which fixed_bits already pays for)"
                 if state_order is not None else
                 "most frequent symbols IN THIS FILE (NOT admissible "
                 "unless the ranking is transmitted)"),
              flush=True)

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
        state_order=state_order,
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
        "corpus": args.corpus or args.ids,
        "stream": meta,
        "capped_positions": capped,
        "state_order": args.state_order,
        "state_order_admissible": args.state_order == "id",
        "top_k": args.top_k,
        "vocabulary_size": V,
        "n_tokens": len(reduced),
        "empirical": ent,
        **{k: v for k, v in out.items()},
        "seconds": time.time() - t0,
    }
    if meta is not None and "truncated_to_tokens" in meta:
        print("  (prefix run: bits per character is not reported, since "
              "the fixed streams were measured on the whole file)",
              flush=True)
    elif meta is not None:
        # the only figure comparable across representations
        payload["family_bits_per_character"] = bits_per_character(
            out["family_bits_per_token"] * out["n_coded"], meta)
        payload["member_bits_per_character"] = {
            str(m): bits_per_character(b * out["n_coded"], meta)
            for m, b in out["member_bits_per_token"].items()
        }
        print(f"  in bits per character of the original file: "
              f"{payload['family_bits_per_character']:.4f} "
              f"(includes fixed_bits {meta['fixed_bits']:,.0f})", flush=True)
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
