#!/usr/bin/env python3
"""Phase 0 of the roadmap: measure the cost of spelling rare words.

The full-fidelity code spells each distinct word once, at its first
occurrence, using a sequential character-level model; afterwards the
word is an ordinary token.  This script measures that cost exactly:

1. Build the spelling stream: the concatenation, in first-occurrence
   order, of every distinct word followed by a terminator symbol.
   This is precisely the character sequence the escape mechanism
   transmits, in the order it transmits it.
2. Code the stream with the context-tree family over the 27-symbol
   alphabet (a-z + terminator), reporting fixed depths 0..D and the
   family codelength.  These are sequential codes: the character model
   available when word w is spelled has seen exactly the spellings of
   the words that occurred before w.
3. Report the index-assignment correction: the fixed-alphabet token
   code (unigram experiment, d = 2^18) pays -log2 of the mass of one
   SPECIFIC unseen symbol at each first occurrence, but with spellings
   transmitted the decoder only needs the event "a new symbol"; by
   symmetry of the exchangeable mixture over unseen symbols, the
   saving at a first occurrence with u unseen symbols is log2(u).

Outputs: bits/char of the spelling code, total spelling bits, both
amortized over the corpus tokens; the correction in bits/token; and
the resulting full-fidelity memoryless estimate in bits/character of
the original file.

Example:

    python scripts/spelling_experiment.py --corpus data/text8 \
        --depth 3 --alphabet-size 262144 --token-bits 10.895 \
        --out output/spelling_text8
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from product_model_with_memory.corpus import load_tokens
from product_model_with_memory.context_tree import context_tree_codelengths

TERM = "#"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--alphabet-size", type=int, default=262144,
        help="fixed token-alphabet size d of the token-level code "
             "(for the index-assignment correction)",
    )
    parser.add_argument(
        "--token-bits", type=float, default=None,
        help="bits/token of the token-level code to combine with "
             "(e.g. 10.895 for the full-vocabulary memoryless code)",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"

    t0 = time.time()
    tokens = load_tokens(args.corpus)
    if args.n:
        tokens = tokens[: args.n]
    n_tokens = len(tokens)
    total_chars = sum(len(t) for t in tokens) + n_tokens  # + separators

    # spelling stream in first-occurrence order + index correction
    seen: set[str] = set()
    stream: list[str] = []
    correction_bits = 0.0
    d_tok = args.alphabet_size
    for tok in tokens:
        if tok not in seen:
            correction_bits += math.log2(d_tok - len(seen))
            seen.add(tok)
            stream.extend(tok)
            stream.append(TERM)
    n_types = len(seen)
    n_spell = len(stream)
    chars = sorted(set(stream))
    V = len(chars)
    print(
        f"{n_tokens:,} tokens, {n_types:,} distinct words, "
        f"spelling stream {n_spell:,} characters over {V} symbols "
        f"({time.time()-t0:.0f}s)",
        flush=True,
    )

    def progress(event, _unused) -> None:
        kind, k, total = event
        if kind == "profiles":
            print(f"  {total:,} contexts, {k:,} unique profiles "
                  f"({time.time()-t0:.0f}s)", flush=True)
        elif kind == "tables" and (k % 200 == 0 or k == total):
            print(f"  tables: {k}/{total} ({time.time()-t0:.0f}s)",
                  flush=True)
        elif kind == "depth" and (k % 4 == 0 or k == total):
            print(f"  evaluation: depth {k}/{total} ({time.time()-t0:.0f}s)",
                  flush=True)

    out = context_tree_codelengths(
        stream,
        vocabulary_size=V,
        max_depth=args.depth,
        cache_dir=cache_dir,
        jobs=args.jobs,
        progress=progress,
    )

    spell_bpc = out["family_bits_per_token"]  # bits per spelled character
    spell_total = spell_bpc * out["n_coded"]
    per_token = spell_total / n_tokens
    corr_per_token = correction_bits / n_tokens

    print("\nfixed-depth character models (bits/char of the stream):",
          flush=True)
    for dd, bits in sorted(out["fixed_depth_bits_per_token"].items()):
        print(f"  depth {dd}: {bits:.4f}", flush=True)
    print(
        f"context-tree family: {spell_bpc:.4f} bits/char; total spelling "
        f"{spell_total/1e6:.3f} Mbit = {per_token:.4f} bits/token "
        f"amortized", flush=True)
    print(
        f"index-assignment correction: {correction_bits/1e6:.3f} Mbit = "
        f"{corr_per_token:.4f} bits/token", flush=True)

    payload = {
        "corpus": args.corpus,
        "n_tokens": n_tokens,
        "n_types": n_types,
        "total_chars": total_chars,
        "spelling_stream_chars": n_spell,
        "char_alphabet": V,
        "depth": args.depth,
        "spelling_bits_per_char": spell_bpc,
        "fixed_depth_bits_per_char": {
            str(k): v for k, v in out["fixed_depth_bits_per_token"].items()
        },
        "map_leaves_by_depth": {
            str(k): v for k, v in out["map_leaves_by_depth"].items()
        },
        "spelling_total_bits": spell_total,
        "spelling_bits_per_token": per_token,
        "index_correction_bits": correction_bits,
        "index_correction_bits_per_token": corr_per_token,
        "seconds": time.time() - t0,
    }
    if args.token_bits is not None:
        full_bits = (args.token_bits - corr_per_token + per_token) * n_tokens
        payload["token_bits_per_token"] = args.token_bits
        payload["full_fidelity_bits_per_char"] = full_bits / total_chars
        print(
            f"with token code at {args.token_bits} bits/token: "
            f"full-fidelity total = "
            f"({args.token_bits} - {corr_per_token:.4f} + {per_token:.4f}) "
            f"* n = {full_bits/1e6:.2f} Mbit over {total_chars:,} chars "
            f"= {full_bits/total_chars:.4f} bits/char", flush=True)

    out_file = out_dir / "results.json"
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"written: {out_file} ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
