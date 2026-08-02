#!/usr/bin/env python3
"""Memoryless token-level baseline over the tokenizer streams.

Encodes a file with the tokenizer of TOKENIZER.md v3, codes every
stream with the depth-averaged layered mixture, and reports the cost of
each stream separately together with the whole-file figure in bits per
character.  Nothing is estimated: the streams reconstruct the file
exactly (the round trip is asserted here as well), so their total plus
the coder overhead IS the codelength of a complete, decodable scheme.

The case stream is reported twice --- modelled independently, and
modelled with the current token as its state (a share-nothing per-state
mixture, the same construction used everywhere else in this project).
The difference between the two is the point of the `case=conditioned`
design and is the number the specification argues about.  The mask
stream is reported the same two ways, for the same reason one level
down: given the token, the capitalisation pattern is nearly determined,
so a repeat of `iPhone` should not pay for its pattern again.  The
totals are therefore given for three complete decodable schemes, not
one, and each is a valid code.

    python scripts/token_baseline.py --file data/enwik8 \
        --numbers intern --case conditioned --out output/tok_enwik8 --jobs 12
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from product_model_with_memory.codelength import (
    default_l_max,
    depth_averaged_codelength_profiles,
)
from product_model_with_memory.tokenizer import CASE_MIXED, decode, encode

FLUSH_BITS_PER_STREAM = 16


def _profile(counts) -> tuple[int, ...]:
    return tuple(sorted((int(c) for c in counts if c > 0), reverse=True))


def _cost(profiles: dict, d: int, jobs: int, progress=None) -> dict:
    """-log2 q_avg for each named profile, in bits."""

    if not profiles:
        return {}
    l_max = default_l_max(d)
    res = depth_averaged_codelength_profiles(
        profiles, d=d, l_max=l_max, jobs=jobs, progress=progress)
    return {k: -v.log2_q_avg for k, v in res.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--numbers", choices=["intern", "compositional"],
                   default="intern")
    p.add_argument("--case", choices=["conditioned", "folded"],
                   default="conditioned")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--bytes", type=int, default=None,
                   help="use only the first N bytes")
    p.add_argument("--skip-round-trip", action="store_true",
                   help="skip the decode check (it is slow on enwik9)")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    raw = np.fromfile(args.file, dtype=np.uint8)
    if args.bytes:
        raw = raw[: args.bytes]
    n = len(raw)
    tok = encode(raw, args.numbers, args.case)
    print(f"{args.file}: {n:,} bytes -> {tok.stats['segments']:,} segments "
          f"({tok.stats['bytes_per_segment']:.2f} bytes/segment), "
          f"vocabulary {tok.stats['vocabulary']:,} "
          f"({time.time()-t0:.0f}s)", flush=True)

    if not args.skip_round_trip:
        assert decode(tok) == raw.tobytes(), "ROUND TRIP FAILED"
        print(f"  round trip verified ({time.time()-t0:.0f}s)", flush=True)

    def progress(evt, _):
        kind, k, total = evt
        if kind == "tables" and (k % 2000 == 0 or k == total):
            print(f"  tables: {k}/{total} ({time.time()-t0:.0f}s)", flush=True)

    bits: dict[str, float] = {}

    # ---- S1: tokens
    d_tok = tok.alphabet_size
    bits["S1_tokens"] = _cost(
        {"S1": _profile(np.bincount(tok.tokens))}, d_tok, args.jobs,
        progress)["S1"]
    print(f"  S1 done ({time.time()-t0:.0f}s)", flush=True)

    # ---- S2a / S2b: spellings.  The letter alphabet depends on the
    # switch: under case=conditioned the spellings are lowercased, so
    # {a..z, END} = 27 symbols; under case=folded they carry their own
    # capitalisation, so {A..Z, a..z, END} = 53.
    if tok.word_spellings:
        d_spell = 27 if args.case == "conditioned" else 53
        end = d_spell - 1
        c = np.zeros(d_spell, dtype=np.int64)
        blob = np.frombuffer(b"".join(tok.word_spellings), dtype=np.uint8)
        if args.case == "conditioned":
            idx = blob.astype(np.int64) - 97
        else:                       # A..Z -> 0..25, a..z -> 26..51
            idx = np.where(blob < 97, blob.astype(np.int64) - 65,
                           blob.astype(np.int64) - 97 + 26)
        if idx.size and (idx.min() < 0 or idx.max() >= end):
            raise ValueError("spelling byte outside the letter alphabet")
        c[:end] = np.bincount(idx, minlength=end)
        c[end] = len(tok.word_spellings)              # one END per word
        bits["S2a_word_spellings"] = _cost(
            {"S2a": _profile(c)}, d_spell, args.jobs)["S2a"]
    if tok.num_spellings:
        c = np.zeros(11, dtype=np.int64)
        blob = np.frombuffer(b"".join(tok.num_spellings), dtype=np.uint8)
        c[:10] = np.bincount(blob.astype(np.int64) - 48, minlength=10)
        c[10] = len(tok.num_spellings)
        bits["S2b_number_spellings"] = _cost(
            {"S2b": _profile(c)}, 11, args.jobs)["S2b"]

    # ---- S3: case, both ways
    mask_states = None
    if args.case == "conditioned" and len(tok.case_classes):
        bits["S3_case_independent"] = _cost(
            {"S3": _profile(np.bincount(tok.case_classes, minlength=4))},
            4, args.jobs)["S3"]

        # Conditioned on the current token: a share-nothing per-state
        # mixture, state = the token whose case class is being coded.
        # State -1 pools every FIRST occurrence (a word seen for the
        # first time has no state yet, and the decoder knows this).
        # All of the below is vectorised: enwik9 has ~10^8 letter runs.
        from product_model_with_memory.tokenizer import (
            ESC_NUM, ESC_WORD, VOCAB_BASE)

        toks = tok.tokens[:-1]                      # drop EOF
        # vocabulary entries are created, in order, by the escape
        # symbols; entry k is a word iff the k-th escape was ESC_WORD
        esc = (toks == ESC_WORD) | (toks == ESC_NUM)
        is_word_vocab = toks[esc] == ESC_WORD
        key = np.where(toks >= VOCAB_BASE, toks - VOCAB_BASE, -1)
        # the letter runs are exactly the segments that consumed a case
        # symbol: an ESC_WORD, or a repeat of a vocabulary word
        sel = ((toks == ESC_WORD)
               | ((key >= 0) & is_word_vocab[np.maximum(key, 0)]))
        states = np.where(toks[sel] == ESC_WORD, -1, key[sel])
        assert len(states) == len(tok.case_classes), "S3 alignment"

        n_states = len(tok.vocabulary) + 1          # +1 for the pool
        flat = np.bincount(
            (states + 1) * 4 + tok.case_classes.astype(np.int64),
            minlength=n_states * 4).reshape(n_states, 4)
        rows = flat[flat.sum(axis=1) > 0]
        # a profile is a row sorted descending; identical rows are one
        # evaluation with a multiplicity
        rows = -np.sort(-rows, axis=1)
        uniq, mult = np.unique(rows, axis=0, return_counts=True)
        profs = {_profile(r): _profile(r) for r in uniq.tolist()}
        costs = _cost(profs, 4, args.jobs)
        bits["S3_case_conditioned"] = float(sum(
            costs[_profile(r)] * m
            for r, m in zip(uniq.tolist(), mult.tolist())))
        bits["S3_states"] = len(rows)

        # the masks belong to the mixed runs, in order
        mask_states = states[tok.case_classes == CASE_MIXED]

    # ---- S4: masks, both ways
    #
    # Pooled: every mask bit from one binary mixture.  Conditioned: the
    # same argument as S3, one level down --- given the token, the
    # capitalisation pattern is nearly determined ("iPhone" is always
    # 0100000), so a repeat should not pay for it again.  The token
    # fixes the word length, so state s contributes one binary stream
    # per letter position; state -1 (first occurrences) has no fixed
    # length and keeps the pooled model.  Decodable: by the time a mask
    # is read the decoder has the token, the class 'mixed' and, for a
    # first occurrence, the spelling --- hence the length.
    if tok.masks:
        flat = np.concatenate(tok.masks)
        bits["S4_masks"] = _cost(
            {"S4": _profile(np.bincount(flat, minlength=2))}, 2,
            args.jobs)["S4"]

        if mask_states is not None:
            assert len(mask_states) == len(tok.masks), "S4 alignment"
            by_state: dict[int, list] = defaultdict(list)
            for st, m in zip(mask_states.tolist(), tok.masks):
                by_state[st].append(m)
            pairs = []                       # (n0, n1) per binary stream
            for st, ms in by_state.items():
                if st < 0:                   # pooled first occurrences
                    b = np.concatenate(ms)
                    pairs.append([len(b) - int(b.sum()), int(b.sum())])
                    continue
                a = np.stack(ms)             # (occurrences, word length)
                ones = a.sum(axis=0, dtype=np.int64)
                pairs.extend(np.stack([len(ms) - ones, ones], axis=1)
                             .tolist())
            arr = -np.sort(-np.asarray(pairs, dtype=np.int64), axis=1)
            uniq, mult = np.unique(arr, axis=0, return_counts=True)
            profs = {_profile(r): _profile(r) for r in uniq.tolist()}
            costs = _cost(profs, 2, args.jobs)
            bits["S4_masks_conditioned"] = float(sum(
                costs[_profile(r)] * m
                for r, m in zip(uniq.tolist(), mult.tolist())))
            bits["S4_streams"] = len(arr)

    # keys that are diagnostics, not codelengths
    COUNTS = {"S3_states", "S4_streams"}
    # one physical stream per group, whichever model is used for it
    n_streams = len({k.split("_")[0] for k in bits if k not in COUNTS})
    overhead = FLUSH_BITS_PER_STREAM * max(n_streams, 1)

    def total(case_variant: str, mask_variant: str) -> float:
        """Codelength of one complete, decodable scheme."""

        t = sum(v for k, v in bits.items()
                if k.startswith(("S1", "S2")))
        if "S3_case_independent" in bits:
            t += bits[f"S3_case_{case_variant}"]
        if "S4_masks" in bits:
            key = f"S4_masks_{mask_variant}" if mask_variant != "pooled" \
                else "S4_masks"
            t += bits.get(key, bits["S4_masks"])
        return t + overhead

    # Three complete schemes.  Which one wins is NOT known in advance:
    # conditioning is not free.  Splitting a stream into per-state
    # mixtures costs each state its own start-up, so when the state is
    # uninformative --- text8, where every letter run is lowercase and
    # the case stream is constant --- the pooled model wins outright.
    # We therefore report all three and take the best, paying the
    # log2(#schemes) bits it costs to say which one was used.
    schemes = {
        "independent_pooled": ("independent", "pooled"),
        "conditioned_pooled": ("conditioned", "pooled"),
        "conditioned_conditioned": ("conditioned", "conditioned"),
    }
    totals = ({name: total(*v) for name, v in schemes.items()}
              if args.case == "conditioned"
              else {"folded": sum(v for k, v in bits.items()
                                  if k not in COUNTS) + overhead})
    selection_bits = math.log2(len(totals))
    best = min(totals, key=totals.get)
    headline = totals[best] + selection_bits

    payload = {
        "file": args.file, "bytes": n,
        "numbers": args.numbers, "case": args.case,
        "stats": tok.stats,
        "alphabet_size": d_tok,
        "stream_bits": bits,
        "overhead_bits": overhead,
        "total_bits_by_scheme": totals,
        "selected_scheme": best,
        "scheme_selection_bits": selection_bits,
        "total_bits": headline,
        "seconds": time.time() - t0,
        "peak_rss_gb": resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 1e9,
    }
    payload["bits_per_character"] = headline / n
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))

    print(f"\n  {args.numbers} / {args.case}")
    for k, v in bits.items():
        if k in COUNTS:
            print(f"    {k:26s} {v:,}")
        else:
            print(f"    {k:26s} {v/1e6:12,.3f} Mbit   "
                  f"{v/n:7.4f} bits/char")
    print(f"    {'overhead':26s} {overhead} bits ({n_streams} streams)")
    print("  complete schemes (case model / mask model):")
    for name, t in totals.items():
        print(f"    {name:26s} {t/1e6:12,.3f} Mbit   "
              f"{t/n:7.4f} bits/char{'   <-- best' if name == best else ''}")
    print(f"    {'scheme selection':26s} {selection_bits:.2f} bits")
    print(f"  TOTAL {headline/1e6:,.3f} Mbit = "
          f"{payload['bits_per_character']:.4f} bits/character")
    print(f"  ({time.time()-t0:.0f}s, peak {payload['peak_rss_gb']:.1f} GB)")
    print(f"written: {out_dir/'results.json'}")


if __name__ == "__main__":
    main()
