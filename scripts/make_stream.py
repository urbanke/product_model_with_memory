#!/usr/bin/env python3
"""Emit the token stream of one representation, for the memory experiments.

Four representations, one output format (see
`product_model_with_memory.streams`), so the same family machinery runs
on each without knowing which it is:

    bytes   the raw file over d = 256; nothing is paid outside the model
    ours    the token stream S1 of TOKENIZER.md v3.  The spelling, case
            and mask streams are NOT part of S1 and a first-order model
            over S1 leaves them unchanged, so their measured cost is
            carried in `fixed_bits` and added back for every
            bits-per-character figure.  Pass --aux-results pointing at
            the matching token_baseline run to pick that number up.
    bpe     a pretrained subword stream (tiktoken).  fixed_bits is zero
            for the LLM-comparability figure; pass --charge-vocabulary
            with the zipped vocabulary size in bytes for the admissible
            one.
    words   the text8 word tokenization, for continuity with the
            existing results; text8 only.

    python scripts/make_stream.py --representation ours --file data/enwik8 \
        --aux-results output/tok_enwik8_ic/results.json \
        --out output/streams/ours_enwik8
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from product_model_with_memory.streams import save_stream


def _aux_fixed_bits(results_path: str) -> tuple[float, str]:
    """Everything in a token_baseline run except the token stream."""

    r = json.loads(Path(results_path).read_text())
    bits = r["stream_bits"]
    # Runs made before the scheme selection was added do not record which
    # scheme won, and the winner is NOT always the conditioned one: on
    # text8 every word is lowercase, so conditioning the case stream on
    # the token costs 547,004 bits instead of saving, and the pooled
    # model wins.  Recover the winner from the totals rather than
    # assuming one.
    scheme = r.get("selected_scheme")
    if scheme is None:
        by = r.get("total_bits_by_scheme") or {}
        scheme = min(by, key=by.get) if by else "conditioned_conditioned"
    case_variant, mask_variant = (scheme.split("_") + ["pooled"])[:2]
    total = bits.get("S2a_word_spellings", 0.0) + bits.get(
        "S2b_number_spellings", 0.0)
    if "S3_case_independent" in bits:
        total += bits[f"S3_case_{case_variant}"]
    if "S4_masks" in bits:
        total += (bits["S4_masks_conditioned"]
                  if mask_variant == "conditioned"
                  and "S4_masks_conditioned" in bits else bits["S4_masks"])
    total += r.get("overhead_bits", 0) + r.get("scheme_selection_bits", 0.0)
    return total, (f"spellings+case+masks+overhead from {results_path} "
                   f"under scheme {scheme}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--representation", required=True,
                   choices=["bytes", "ours", "bpe", "words"])
    p.add_argument("--file", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bytes", type=int, default=None)
    p.add_argument("--numbers", default="intern")
    p.add_argument("--case", default="conditioned")
    p.add_argument("--encoding", default="cl100k_base")
    p.add_argument("--vocab-dir", default=None)
    p.add_argument("--aux-results", default=None,
                   help="token_baseline results.json, for fixed_bits")
    p.add_argument("--charge-vocabulary", type=int, default=0,
                   help="zipped vocabulary size in BYTES, charged as "
                        "fixed_bits (bpe only)")
    args = p.parse_args()

    raw = np.fromfile(args.file, dtype=np.uint8)
    if args.bytes:
        raw = raw[: args.bytes]
    n_bytes = len(raw)
    fixed, notes = 0.0, ""

    if args.representation == "bytes":
        ids, alphabet = raw.astype(np.int32), 256
        notes = "raw bytes; nothing is paid outside the model"

    elif args.representation == "ours":
        from product_model_with_memory.tokenizer import encode

        tok = encode(raw, args.numbers, args.case)
        ids, alphabet = tok.tokens, tok.alphabet_size
        if args.aux_results:
            fixed, notes = _aux_fixed_bits(args.aux_results)
        else:
            notes = ("WARNING: no --aux-results, so fixed_bits is 0 and "
                     "bits/character will UNDERSTATE the true cost")

    elif args.representation == "bpe":
        import os

        if args.vocab_dir:
            os.environ["TIKTOKEN_CACHE_DIR"] = str(
                Path(args.vocab_dir).expanduser().resolve())
        import tiktoken

        enc = tiktoken.get_encoding(args.encoding)
        text = raw.tobytes().decode("utf-8", errors="surrogateescape")
        toks = enc.encode(text, disallowed_special=())
        back = enc.decode(toks).encode("utf-8", errors="surrogateescape")
        if back != raw.tobytes():
            raise SystemExit("ROUND TRIP FAILED for this encoding")
        ids, alphabet = np.asarray(toks, dtype=np.int32), enc.n_vocab
        fixed = 8.0 * args.charge_vocabulary
        notes = (f"{args.encoding}; vocabulary "
                 + (f"CHARGED at {args.charge_vocabulary} zipped bytes"
                    if args.charge_vocabulary else
                    "NOT charged --- LLM-comparability only, not an entry"))

    else:  # words
        text = Path(args.file).read_text()
        words = text.split()
        index: dict[str, int] = {}
        ids = np.empty(len(words), dtype=np.int32)
        for i, w in enumerate(words):
            j = index.get(w)
            if j is None:
                j = len(index)
                index[w] = j
            ids[i] = j
        alphabet = len(index)
        notes = ("text8 word tokenization; the space is folded into the "
                 "word, so this is NOT comparable to `ours` token for "
                 "token")

    save_stream(args.out, ids, representation=args.representation,
                source_file=args.file, n_bytes=n_bytes, alphabet=alphabet,
                fixed_bits=fixed, notes=notes)
    meta = json.loads((Path(args.out) / "stream.json").read_text())
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print(f"written: {Path(args.out)/'ids.npy'}")


if __name__ == "__main__":
    main()
