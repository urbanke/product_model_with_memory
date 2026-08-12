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
import math
import time
from pathlib import Path

import numpy as np

from product_model_with_memory.corpus import load_tokens
from product_model_with_memory.pairs import empirical_entropies, reduce_vocabulary
from product_model_with_memory.memory2_frontier import (
    MemoryTwoPoint,
    enumerative_subset_bits,
    nested_frequency_subset_bits,
)
from product_model_with_memory.product_family import product_family_codelengths
from product_model_with_memory.production_coding import (
    PRODUCTION_SEQUENCE_ESTIMATOR,
    configure_production_tables,
    layered_sequence_code,
)
from product_model_with_memory.streams import (
    bits_per_character,
    load_stream,
    reduce_ids,
    state_order_by_id,
)


def _chunked_counts(ids: np.ndarray, alphabet_size: int) -> np.ndarray:
    counts = np.zeros(alphabet_size, dtype=np.int64)
    for start in range(0, len(ids), 8_000_000):
        chunk = np.asarray(ids[start:start + 8_000_000], dtype=np.int64)
        counts += np.bincount(chunk, minlength=alphabet_size)
    return counts


def _mixture_bits(total_bits: list[float], declared_members: int) -> float:
    """Contribution of this slice to a declared uniform model mixture."""

    if not total_bits or declared_members < len(total_bits):
        raise ValueError("invalid declared mixture size")
    smallest = min(total_bits)
    mass = sum(2.0 ** (-(value - smallest)) for value in total_bits)
    return smallest - math.log2(mass) + math.log2(declared_members)


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
    parser.add_argument(
        "--alphabet-grid-size", type=int, default=1,
        help="number of declared emission-vocabulary choices",
    )
    parser.add_argument(
        "--triplet-grid-size", type=int, default=None,
        help="global number of declared (V,M1,M2) choices; defaults to the "
             "members in this invocation",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    # This experiment produces paper-facing codelengths.  Refuse the old
    # grow-on-demand exact store: production uses the sealed designed anchor
    # ladder and must never spend hours creating ad-hoc columns.
    configure_production_tables()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"
    grid = [
        (int(a), int(b))
        for a, b in (pair.split(":") for pair in args.grid.split(","))
    ]
    if len(grid) != len(set(grid)):
        raise SystemExit("--grid contains duplicate members")
    triplet_grid_size = args.triplet_grid_size or len(grid)
    if args.alphabet_grid_size < 1 or triplet_grid_size < len(grid):
        raise SystemExit("declared grid sizes are inconsistent")

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
            print(
                f"  escape mapping: {capped:,} positions fall outside the "
                f"top {args.top_k} and map to <unk>; their original token "
                "identities are coded separately in the honest accounting",
                flush=True,
            )
        print("  state map: symbols leave the backoff state in order of "
              + ("ascending vocabulary id (admissible)"
                 if state_order is not None else
                 "frequency in this file (nested subsets are transmitted "
                 "and charged in the honest accounting)"), flush=True)
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
        "sequence_estimator": PRODUCTION_SEQUENCE_ESTIMATOR,
        "corpus": args.corpus or args.ids,
        "stream": meta,
        "state_order": args.state_order if args.ids else "frequency",
        "state_order_known_without_description": bool(
            args.ids and args.state_order == "id"
        ),
        "state_order_admissible": bool(
            args.ids
            and (
                args.state_order == "id"
                or (
                    args.state_order == "frequency"
                    and meta is not None
                    and "fixed_bits" in meta
                )
            )
        ),
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
    if meta is not None and "fixed_bits" in meta:
        tokenizer_alphabet = int(meta["alphabet"])
        full_vocabulary = args.top_k >= tokenizer_alphabet - 1
        if full_vocabulary:
            retained_count = tokenizer_alphabet
            emission_subset_bits = 0.0
            escaped_tokens = 0
            escape_code = None
        else:
            retained_count = args.top_k
            emission_subset_bits = enumerative_subset_bits(
                tokenizer_alphabet, retained_count
            )
            original_counts = _chunked_counts(ids, tokenizer_alphabet)
            selected = np.argsort(-original_counts, kind="stable")[
                :retained_count
            ]
            retained = np.zeros(tokenizer_alphabet, dtype=bool)
            retained[selected] = True
            escape_code = layered_sequence_code(
                original_counts[~retained],
                tokenizer_alphabet - retained_count,
                jobs=args.jobs,
            )
            escaped_tokens = escape_code.tokens
        escape_bits = 0.0 if escape_code is None else escape_code.bits
        first_counts = np.bincount(
            np.asarray(reduced[:1], dtype=np.int64), minlength=V
        )
        first_code = layered_sequence_code(first_counts, V, jobs=args.jobs)
        alphabet_selection_bits = math.log2(args.alphabet_grid_size)
        common_bits = (
            float(meta["fixed_bits"])
            + emission_subset_bits
            + escape_bits
            + first_code.bits
            + alphabet_selection_bits
        )
        state_bits: dict[str, float] = {}
        total_before_triplet_choice: dict[str, float] = {}
        honest_member_bpc: dict[str, float] = {}
        denominator = float(meta["n_bytes"])
        triplet_selection_bits = math.log2(triplet_grid_size)
        for m1, m2 in out["grid"]:
            member = f"{m1}:{m2}"
            point = MemoryTwoPoint(V, m1, m2)
            description = (
                nested_frequency_subset_bits(point)
                if args.state_order == "frequency" else 0.0
            )
            before_choice = (
                out["member_bits_per_token"][(m1, m2)] * out["n_coded"]
                + common_bits + description
            )
            state_bits[member] = description
            total_before_triplet_choice[member] = before_choice
            honest_member_bpc[member] = (
                before_choice + triplet_selection_bits
            ) / denominator
        slice_mixture = _mixture_bits(
            list(total_before_triplet_choice.values()), triplet_grid_size
        )
        payload["honest_accounting"] = {
            "version": 1,
            "sequence_estimator": PRODUCTION_SEQUENCE_ESTIMATOR,
            "alphabet_selection": (
                "full_tokenizer_vocabulary" if full_vocabulary else
                "corpus_frequency_top_k_transmitted_as_unordered_subset"
            ),
            "alphabet_grid_size": args.alphabet_grid_size,
            "alphabet_selection_bits": alphabet_selection_bits,
            "retained_tokenizer_ids": retained_count,
            "selected_subset_description_bits": emission_subset_bits,
            "tokenizer_vocabulary_bits": float(meta["fixed_bits"]),
            "escaped_tokens": escaped_tokens,
            "escaped_token_payload_bits": escape_bits,
            "escaped_token_payload_code": (
                None if escape_code is None else {
                    "estimator": escape_code.estimator,
                    "alphabet_size": escape_code.alphabet_size,
                    "l_max": escape_code.l_max,
                }
            ),
            "first_symbol_bits": first_code.bits,
            "first_symbol_code": {
                "estimator": first_code.estimator,
                "alphabet_size": first_code.alphabet_size,
                "l_max": first_code.l_max,
            },
            "state_selection": (
                "nested_corpus_frequency_subsets_transmitted_enumeratively"
                if args.state_order == "frequency" else
                "ascending_tokenizer_id_known_from_transmitted_vocabulary"
            ),
            "nested_state_subset_description_bits": state_bits,
            "triplet_grid_size": triplet_grid_size,
            "triplet_selection_bits": triplet_selection_bits,
            "total_bits_before_triplet_selection": total_before_triplet_choice,
            "honest_member_bits_per_character": honest_member_bpc,
            "honest_declared_family_slice_bits_per_character": (
                slice_mixture / denominator
            ),
        }
        best_honest = min(honest_member_bpc, key=honest_member_bpc.get)
        print(
            f"  honest best in this slice: {best_honest} "
            f"{honest_member_bpc[best_honest]:.6f} bpc; "
            f"global-grid slice mixture {slice_mixture / denominator:.6f}",
            flush=True,
        )
    out_file = out_dir / "results.json"
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"written: {out_file} ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
