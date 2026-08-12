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
from scipy.special import gammaln, logsumexp

from product_model_with_memory.corpus import load_tokens
from product_model_with_memory.pairs import empirical_entropies, reduce_vocabulary
from product_model_with_memory.production_coding import (
    PRODUCTION_SEQUENCE_ESTIMATOR,
    layered_sequence_code,
)
from product_model_with_memory.state_family import state_family_codelengths
from product_model_with_memory.streams import (
    bits_per_character,
    load_stream,
    reduce_ids,
    state_order_by_id,
)


def _chunked_counts(ids: np.ndarray, alphabet_size: int) -> np.ndarray:
    """Count a possibly mmap-backed id stream without a full int64 copy."""

    counts = np.zeros(alphabet_size, dtype=np.int64)
    chunk_size = 8_000_000
    for start in range(0, len(ids), chunk_size):
        chunk = np.asarray(ids[start:start + chunk_size], dtype=np.int64)
        counts += np.bincount(chunk, minlength=alphabet_size)
    return counts


def _enumerative_subset_bits(alphabet_size: int, subset_size: int) -> float:
    if subset_size < 0 or subset_size > alphabet_size:
        raise ValueError("invalid subset size")
    return float(
        (
            gammaln(alphabet_size + 1)
            - gammaln(subset_size + 1)
            - gammaln(alphabet_size - subset_size + 1)
        ) / np.log(2.0)
    )


def _two_part_family_bits(
    member_data_bits: dict[int, float],
    member_description_bits: dict[int, float],
) -> float:
    """Uniform mixture over individually decodable two-part member codes."""

    grid = sorted(member_data_bits)
    if not grid or set(grid) != set(member_description_bits):
        raise ValueError("member and description grids must agree and be nonempty")
    total_bits = np.asarray([
        member_data_bits[m] + member_description_bits[m] for m in grid
    ], dtype=float)
    return float(
        -logsumexp(-total_bits * np.log(2.0)) / np.log(2.0)
        + np.log2(len(grid))
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
                             "the code is admissible.  'frequency' selects "
                             "the M most frequent previous-token symbols "
                             "and, for a complete --ids run, honestly "
                             "charges an enumerative description of that "
                             "subset for every M")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--l-max", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--alphabet-grid-size", type=int, default=1,
        help="number of predeclared retained-alphabet choices; charges "
             "log2(size) bits to identify this run's V (default: 1)",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    if args.alphabet_grid_size < 1:
        raise SystemExit("--alphabet-grid-size must be positive")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"
    m_grid = sorted({int(x) for x in args.m_grid.split(",")})

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
    if not m_grid or any(m < 0 or m > V for m in m_grid):
        raise SystemExit(f"every M must satisfy 0 <= M <= V={V}")
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
                 "most frequent previous-token symbols in this file "
                 "(the complete-code accounting below transmits the "
                 "selected subset)"),
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
        "state_order_admissible_without_description": args.state_order == "id",
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

        # A capped token stream is not by itself a code for the original
        # stream.  Close the accounting here so a reduced-alphabet Markov-1
        # row cannot accidentally be quoted without its hidden-token payload.
        # The retained set is selected by frequency in this file, so it is
        # transmitted as an unordered subset.  Its internal labels are merely
        # a deterministic relabelling and cost nothing.  Every data-bearing
        # sequence -- the first reduced symbol and the escaped identities --
        # uses the same production layered estimator as the state factors.
        tokenizer_alphabet = int(meta["alphabet"])
        full_vocabulary = args.top_k >= tokenizer_alphabet - 1
        if full_vocabulary:
            retained_count = tokenizer_alphabet
            subset_bits = 0.0
            escape_bits = 0.0
            escaped_tokens = 0
            escape_code = None
        else:
            # The declared code retains exactly top_k tokenizer identifiers,
            # including zero-count identifiers when V exceeds the number of
            # types observed in this file.  `reduce_ids` need only materialize
            # positive-count retained ids, so reconstruct the full declared
            # subset here for description and escape-alphabet accounting.
            retained_count = args.top_k
            subset_bits = _enumerative_subset_bits(
                tokenizer_alphabet, retained_count
            )
            original_counts = _chunked_counts(ids, tokenizer_alphabet)
            selected = np.argsort(-original_counts, kind="stable")[
                :retained_count
            ]
            retained = np.zeros(tokenizer_alphabet, dtype=bool)
            retained[selected] = True
            escaped_counts = original_counts[~retained]
            escape_code = layered_sequence_code(
                escaped_counts, tokenizer_alphabet - retained_count,
                jobs=args.jobs,
            )
            escape_bits = escape_code.bits
            escaped_tokens = escape_code.tokens

        first_counts = np.bincount(
            np.asarray(reduced[:1], dtype=np.int64), minlength=V
        )
        first_code = layered_sequence_code(first_counts, V, jobs=args.jobs)
        fixed_bits = float(meta.get("fixed_bits", 0.0))
        alphabet_selection_bits = float(np.log2(args.alphabet_grid_size))
        denominator = float(meta["n_bytes"])
        common_bits = (
            fixed_bits + subset_bits + escape_bits + first_code.bits
            + alphabet_selection_bits
        )
        member_data_bits = {
            m: out["member_bits_per_token"][m] * out["n_coded"]
            for m in out["m_grid"]
        }
        # Frequency-selected states are a legitimate two-part code only after
        # the chosen subset is sent.  No ranking is needed: member M depends
        # only on membership, so an enumerative subset description is exact.
        # The id-ordered comparison needs no state-subset description because
        # that order is fixed by the already transmitted tokenizer vocabulary.
        state_subset_bits = {
            m: (
                _enumerative_subset_bits(V, m)
                if args.state_order == "frequency" else 0.0
            )
            for m in out["m_grid"]
        }
        state_grid_selection_bits = float(np.log2(len(out["m_grid"])))
        family_data_and_state_bits = _two_part_family_bits(
            member_data_bits, state_subset_bits
        )
        honest_family_bpc = (
            family_data_and_state_bits + common_bits
        ) / denominator
        honest_member_bpc = {
            str(m): (
                member_data_bits[m] + state_subset_bits[m]
                + state_grid_selection_bits + common_bits
            ) / denominator
            for m in out["m_grid"]
        }
        payload["honest_accounting"] = {
            "version": 2,
            "sequence_estimator": PRODUCTION_SEQUENCE_ESTIMATOR,
            "alphabet_selection": (
                "full_tokenizer_vocabulary" if full_vocabulary else
                "corpus_frequency_top_k_transmitted_as_unordered_subset"
            ),
            "tie_break": "ascending_tokenizer_id",
            "alphabet_grid_size": args.alphabet_grid_size,
            "alphabet_selection_bits": alphabet_selection_bits,
            "alphabet_selection_bpc": alphabet_selection_bits / denominator,
            "retained_tokenizer_ids": retained_count,
            "selected_subset_description_bits": subset_bits,
            "selected_subset_description_bpc": subset_bits / denominator,
            "tokenizer_vocabulary_bits": fixed_bits,
            "tokenizer_vocabulary_bpc": fixed_bits / denominator,
            "escaped_tokens": escaped_tokens,
            "escape_fraction": escaped_tokens / len(ids),
            "escaped_token_payload_bits": escape_bits,
            "escaped_token_payload_bpc": escape_bits / denominator,
            "escaped_token_payload_code": (
                None if escape_code is None else {
                    "estimator": escape_code.estimator,
                    "alphabet_size": escape_code.alphabet_size,
                    "l_max": escape_code.l_max,
                }
            ),
            "first_symbol_bits": first_code.bits,
            "first_symbol_bpc": first_code.bits / denominator,
            "first_symbol_code": {
                "estimator": first_code.estimator,
                "alphabet_size": first_code.alphabet_size,
                "l_max": first_code.l_max,
            },
            "state_selection": (
                "corpus_frequency_top_m_transmitted_as_unordered_subset"
                if args.state_order == "frequency" else
                "ascending_tokenizer_id_known_from_transmitted_vocabulary"
            ),
            "state_subset_universe_size": V,
            "state_subset_description_bits": {
                str(m): state_subset_bits[m] for m in out["m_grid"]
            },
            "state_subset_description_bpc": {
                str(m): state_subset_bits[m] / denominator
                for m in out["m_grid"]
            },
            "state_grid": out["m_grid"],
            "state_grid_selection_bits": state_grid_selection_bits,
            "state_grid_selection_bpc": (
                state_grid_selection_bits / denominator
            ),
            "state_selection_admissible": True,
            "honest_family_bits_per_character": honest_family_bpc,
            "honest_member_bits_per_character": honest_member_bpc,
        }
        print(
            "  honest original-stream accounting: "
            f"{honest_family_bpc:.4f} bpc "
            f"(subset {subset_bits / denominator:.6f}, "
            f"escape {escape_bits / denominator:.6f}, "
            f"first symbol {first_code.bits / denominator:.9f})",
            flush=True,
        )
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
