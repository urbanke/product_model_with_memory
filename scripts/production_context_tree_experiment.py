#!/usr/bin/env python3
"""Production CTW experiment with complete original-stream accounting.

This entry point deliberately exposes no estimator choice.  Every
data-bearing stream uses layered_depth_averaged_product_simplex_v1 and the
sealed production table store.  V includes the escape symbol, so a reduced
power-of-two alphabet V retains exactly V-1 tokenizer identifiers.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.special import gammaln

from product_model_with_memory.context_tree import context_tree_codelengths
from product_model_with_memory.production_coding import (
    PRODUCTION_SEQUENCE_ESTIMATOR,
    configure_production_tables,
    layered_sequence_code,
    require_numerical_identity,
    require_production_sequence_estimator,
)
from product_model_with_memory.streams import load_stream, reduce_ids
from product_model_with_memory.state_family import state_family_codelengths


def enumerative_subset_bits(universe: int, subset: int) -> float:
    if subset < 0 or subset > universe:
        raise ValueError("invalid subset size")
    return float(
        (gammaln(universe + 1) - gammaln(subset + 1)
         - gammaln(universe - subset + 1)) / np.log(2.0)
    )


def chunked_counts(ids: np.ndarray, alphabet_size: int) -> np.ndarray:
    counts = np.zeros(alphabet_size, dtype=np.int64)
    for start in range(0, len(ids), 8_000_000):
        chunk = np.asarray(ids[start:start + 8_000_000], dtype=np.int64)
        counts += np.bincount(chunk, minlength=alphabet_size)
    return counts


def load_first_order_gate(path: Path, V: int, M: int) -> dict:
    payload = json.loads(path.read_text())
    accounting = payload["honest_accounting"]
    require_production_sequence_estimator(
        accounting["sequence_estimator"], source=str(path)
    )
    if int(payload["vocabulary_size"]) != V:
        raise RuntimeError(
            f"first-order gate has V={payload['vocabulary_size']}, expected {V}"
        )
    key = str(M)
    if key not in payload["member_bits_per_token"]:
        raise RuntimeError(f"first-order gate lacks member M={M}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--V", type=int, required=True,
                        help="reduced output alphabet including escape")
    parser.add_argument("--M", type=int, default=None,
                        help="common context resolution at every lag; "
                             "default M=V")
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--candidate-grid-size", type=int, required=True,
                        help="number of predeclared (V,D) candidates")
    parser.add_argument("--first-order-results",
                        help="same-(V,M) production Markov-1 results.json; "
                             "required for the M=V identity gate")
    parser.add_argument("--full-v-ctw-results",
                        help="existing same-depth full-V CTW results.json gate")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--universal-path", default=None)
    args = parser.parse_args()

    M = args.V if args.M is None else args.M
    if (args.V < 2 or M < 0 or M > args.V or args.depth < 1
            or args.candidate_grid_size < 1):
        parser.error("require V>=2, depth>=1, and candidate-grid-size>=1")
    if M == args.V and not args.first_order_results:
        parser.error("--first-order-results is required when M=V")

    t0 = time.time()
    table_store = configure_production_tables(args.universal_path)
    ids, meta = load_stream(args.ids, mmap_mode="r")
    tokenizer_alphabet = int(meta["alphabet"])
    if args.V > tokenizer_alphabet:
        parser.error(f"V={args.V} exceeds tokenizer alphabet {tokenizer_alphabet}")
    top_k = args.V - 1
    reduced, V, capped, _keep = reduce_ids(ids, top_k, return_keep=True)
    reduced = np.asarray(reduced, dtype=np.int32)
    if V != args.V:
        raise RuntimeError(f"reducer returned V={V}, requested {args.V}")

    first_order = (
        load_first_order_gate(Path(args.first_order_results), V, M)
        if args.first_order_results else None
    )

    if M == V:
        context_ids = reduced
        context_alphabet_size = V
        context_subset_bits = 0.0
    elif M == 0:
        context_ids = np.zeros(len(reduced), dtype=np.int32)
        context_alphabet_size = 1
        context_subset_bits = 0.0
    else:
        # reduce_ids labels retained symbols in descending corpus-frequency
        # order.  Membership in the selected M-set is transmitted below;
        # its internal labels are a codelength-invariant relabelling.
        context_ids = np.where(reduced < M, reduced, M).astype(np.int32)
        context_alphabet_size = M + 1
        context_subset_bits = enumerative_subset_bits(V, M)

    def progress(event, _unused) -> None:
        kind, k, total = event
        if kind == "profiles":
            print(f"  {total:,} contexts; {k:,} unique profiles", flush=True)
        elif kind == "depth" and (k % 5 == 0 or k == total):
            print(f"  estimator depth {k}/{total}", flush=True)

    tree = context_tree_codelengths(
        reduced,
        vocabulary_size=V,
        max_depth=args.depth,
        cache_dir=Path(args.out) / "cache",
        jobs=args.jobs,
        leaf_model="layered",
        progress=progress,
        tables_source="universal",
        universal_path=table_store,
        context_ids=context_ids,
        context_alphabet_size=context_alphabet_size,
    )

    # Gate 1: run D=1 independently.  A depth-one row inside a deeper CTW
    # run is evaluated only on positions t>D so that all prunings share a
    # common suffix; it therefore cannot be compared to Markov-1 on t>1.
    # The independent D=1 construction uses exactly the Markov-1 positions.
    gate_tree = tree if args.depth == 1 else context_tree_codelengths(
        reduced,
        vocabulary_size=V,
        max_depth=1,
        cache_dir=Path(args.out) / "gate_d1_tree_cache",
        jobs=args.jobs,
        leaf_model="layered",
        tables_source="universal",
        universal_path=table_store,
        context_ids=context_ids,
        context_alphabet_size=context_alphabet_size,
    )
    gate_markov = state_family_codelengths(
        reduced,
        vocabulary_size=V,
        m_grid=[M],
        cache_dir=Path(args.out) / "gate_d1_markov_cache",
        jobs=args.jobs,
        state_order=np.arange(V, dtype=np.int64),
    )
    depth_one_bpt = float(gate_tree["fixed_depth_bits_per_token"][1])
    markov_bpt = float(gate_markov["member_bits_per_token"][M])
    if int(gate_tree["n_coded"]) != len(reduced) - 1:
        raise RuntimeError("D=1 gate does not cover the Markov-1 positions")
    depth_one_error = require_numerical_identity(
        depth_one_bpt,
        markov_bpt,
        gate="D=1/first-order",
    )
    depth_one_gate = {
        "reference": "independent_state_family_codelengths",
        "M": M,
        "error_bits_per_token": depth_one_error,
        "passed": True,
    }
    if first_order is not None:
        artifact_error = require_numerical_identity(
            markov_bpt,
            float(first_order["member_bits_per_token"][str(M)]),
            gate="computed/historical first-order",
        )
        depth_one_gate["historical_reference"] = str(args.first_order_results)
        depth_one_gate["historical_error_bits_per_token"] = artifact_error

    full_v_gate = None
    if args.full_v_ctw_results:
        reference_path = Path(args.full_v_ctw_results)
        reference = json.loads(reference_path.read_text())
        if V != tokenizer_alphabet or M != V:
            raise RuntimeError("the full-V CTW gate requires M=V=full V")
        if int(reference["vocabulary_size"]) != V:
            raise RuntimeError("full-V CTW reference alphabet mismatch")
        if int(reference["max_depth"]) != args.depth:
            raise RuntimeError("full-V CTW reference depth mismatch")
        error = require_numerical_identity(
            float(tree["family_bits_per_token"]),
            float(reference["family_bits_per_token"]),
            gate="full-V CTW reproduction",
        )
        full_v_gate = {
            "reference": str(reference_path),
            "error_bits_per_token": error,
            "passed": True,
        }

    original_counts = chunked_counts(ids, tokenizer_alphabet)
    full_vocabulary = V == tokenizer_alphabet
    if full_vocabulary:
        subset_bits = 0.0
        escape_code = layered_sequence_code(
            np.zeros(0, dtype=np.int64), 0, jobs=args.jobs
        )
        retained_count = tokenizer_alphabet
    else:
        retained_count = top_k
        subset_bits = enumerative_subset_bits(tokenizer_alphabet, retained_count)
        selected = np.argsort(-original_counts, kind="stable")[:retained_count]
        retained = np.zeros(tokenizer_alphabet, dtype=bool)
        retained[selected] = True
        escape_code = layered_sequence_code(
            original_counts[~retained],
            tokenizer_alphabet - retained_count,
            jobs=args.jobs,
            universal_path=table_store,
        )

    prefix_counts = np.bincount(
        reduced[:args.depth].astype(np.int64), minlength=V
    )
    prefix_code = layered_sequence_code(
        prefix_counts, V, jobs=args.jobs, universal_path=table_store
    )
    primary_bits = float(tree["family_bits_per_token"]) * int(tree["n_coded"])
    fixed_bits = float(meta.get("fixed_bits", 0.0))
    model_choice_bits = math.log2(args.candidate_grid_size)
    total_bits = (
        primary_bits + fixed_bits + subset_bits + context_subset_bits
        + escape_code.bits
        + prefix_code.bits + model_choice_bits
    )
    denominator = int(meta["n_bytes"])

    payload = {
        "version": 1,
        "corpus": args.ids,
        "V": V,
        "M": M,
        "context_alphabet_size": context_alphabet_size,
        "top_k_retained_identifiers": retained_count,
        "max_depth": args.depth,
        "candidate_grid_size": args.candidate_grid_size,
        "sequence_estimator": PRODUCTION_SEQUENCE_ESTIMATOR,
        "production_table_store": str(table_store),
        "tree": tree,
        "honest_accounting": {
            "sequence_estimator": PRODUCTION_SEQUENCE_ESTIMATOR,
            "original_bytes": denominator,
            "primary_tree_bits": primary_bits,
            "tokenizer_vocabulary_bits": fixed_bits,
            "selected_subset_description_bits": subset_bits,
            "context_state_selection": (
                "all_reduced_symbols" if M == V else
                "single_pooled_state" if M == 0 else
                "corpus_frequency_top_M_transmitted_as_unordered_subset"
            ),
            "context_subset_description_bits": context_subset_bits,
            "escaped_tokens": escape_code.tokens,
            "escaped_token_payload_bits": escape_code.bits,
            "escaped_token_payload_estimator": escape_code.estimator,
            "boundary_tokens": prefix_code.tokens,
            "boundary_symbol_bits": prefix_code.bits,
            "boundary_symbol_estimator": prefix_code.estimator,
            "model_choice_bits": model_choice_bits,
            "total_bits": total_bits,
            "honest_bits_per_character": total_bits / denominator,
        },
        "numerical_gates": {
            "depth_one_first_order": depth_one_gate,
            "full_v_existing_ctw": full_v_gate,
        },
        "capped_positions": capped,
        "seconds": time.time() - t0,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "results.json"
    target.write_text(json.dumps(payload, indent=2))
    print(
        f"written {target}; honest {total_bits / denominator:.9f} bpc; "
        + (
            f"; D=1 gate error {depth_one_gate['error_bits_per_token']:+.3e} bits/token"
            if depth_one_gate else ""
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
