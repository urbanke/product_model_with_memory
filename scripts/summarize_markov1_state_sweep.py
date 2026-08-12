#!/usr/bin/env python3
"""Validate and summarize the honest Markov-1 V-by-M sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from product_model_with_memory.production_coding import (
    PRODUCTION_SEQUENCE_ESTIMATOR,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default="output/markov1_state_sweep_20260809"
    )
    parser.add_argument(
        "--baseline-root", default="output/markov1_alphabet_sweep_20260809",
        help="M=V control artifacts (empty string disables the comparison)",
    )
    parser.add_argument("--corpora", default="text8,enwik8,enwik9")
    parser.add_argument(
        "--v-grid",
        default="1024,2048,4096,8192,16384,32768,65536,100277",
    )
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    baseline_root = Path(args.baseline_root) if args.baseline_root else None
    corpora = args.corpora.split(",")
    v_grid = [int(value) for value in args.v_grid.split(",")]
    summary: dict[str, object] = {
        "version": 1,
        "root": str(root),
        "sequence_estimator": PRODUCTION_SEQUENCE_ESTIMATOR,
        "corpora": {},
    }
    missing: list[str] = []

    for corpus in corpora:
        rows = []
        for vocabulary_size in v_grid:
            result_path = root / corpus / f"v{vocabulary_size}" / "results.json"
            if not result_path.exists():
                missing.append(str(result_path))
                continue
            result = json.loads(result_path.read_text())
            honest = result["honest_accounting"]
            if honest["sequence_estimator"] != PRODUCTION_SEQUENCE_ESTIMATOR:
                raise RuntimeError(f"wrong estimator in {result_path}")
            if not honest.get("state_selection_admissible", False):
                raise RuntimeError(f"unpaid state selection in {result_path}")

            member_bpc = {
                int(m): float(rate)
                for m, rate in honest[
                    "honest_member_bits_per_character"
                ].items()
            }
            best_m = min(member_bpc, key=member_bpc.get)
            control_difference = None
            if baseline_root is not None:
                baseline_path = (
                    baseline_root / corpus / f"v{vocabulary_size}" / "results.json"
                )
                if baseline_path.exists():
                    baseline = json.loads(baseline_path.read_text())
                    control_difference = (
                        float(result["member_bits_per_token"][str(vocabulary_size)])
                        - float(baseline["member_bits_per_token"][str(vocabulary_size)])
                    )
                    if abs(control_difference) > 1e-12:
                        raise RuntimeError(
                            f"M=V control differs in {result_path}: "
                            f"{control_difference} bits/token"
                        )
            rows.append({
                "V": vocabulary_size,
                "m_grid": result["m_grid"],
                "best_M": best_m,
                "best_member_bpc": member_bpc[best_m],
                "family_bpc": float(
                    honest["honest_family_bits_per_character"]
                ),
                "m_equals_v_data_bits_per_token_difference": control_difference,
            })
        summary["corpora"][corpus] = rows

    if args.require_complete and missing:
        raise SystemExit("missing results:\n" + "\n".join(missing))
    summary["missing"] = missing
    for corpus, rows in summary["corpora"].items():
        print(corpus)
        for row in rows:
            print(
                f"  V={row['V']:>6} best M={row['best_M']:>6} "
                f"member={row['best_member_bpc']:.6f} "
                f"family={row['family_bpc']:.6f}"
            )
    print(f"missing: {len(missing)}")

    out_path = Path(args.out) if args.out else root / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
