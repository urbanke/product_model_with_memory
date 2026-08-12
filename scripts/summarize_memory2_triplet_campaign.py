#!/usr/bin/env python3
"""Apply complete original-stream accounting to a triplet campaign."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from product_model_with_memory.memory2_frontier import (
    MemoryTwoPoint,
    nested_frequency_subset_bits,
)
from product_model_with_memory.production_coding import (
    PRODUCTION_SEQUENCE_ESTIMATOR,
    require_production_sequence_estimator,
)


def _result_assignments(values: list[str]) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for value in values:
        try:
            key, path = value.split("=", 1)
            vocabulary = int(key)
        except ValueError as exc:
            raise SystemExit("--result must have the form V=PATH") from exc
        found.append((vocabulary, Path(path)))
    return found


def _uniform_two_part_mixture_bits(totals: list[float]) -> float:
    if not totals:
        raise ValueError("cannot mix an empty campaign")
    smallest = min(totals)
    relative_mass = sum(2.0 ** (-(value - smallest)) for value in totals)
    return smallest - math.log2(relative_mass) + math.log2(len(totals))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--result", action="append", default=[],
                        help="repeat V=PATH for every result batch")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text())
    result_paths = _result_assignments(args.result)
    vocabularies = set(map(int, plan["vocabulary_grid"]))
    if {vocabulary for vocabulary, _ in result_paths} != vocabularies:
        raise SystemExit("result batches must cover the plan's V grid exactly")

    expected = {
        MemoryTwoPoint(*map(int, values)) for values in plan["triplets"]
    }
    rows: list[dict] = []
    observed: set[MemoryTwoPoint] = set()
    denominators: set[int] = set()
    for vocabulary, result_path in result_paths:
        result = json.loads(result_path.read_text())
        require_production_sequence_estimator(
            result["sequence_estimator"], source=str(result_path)
        )
        if result["state_order"] != "frequency":
            raise RuntimeError("campaign accounting expects frequency-selected states")
        if int(result["vocabulary_size"]) != vocabulary:
            raise RuntimeError("result vocabulary does not match its assignment")
        n_coded = int(result["n_coded"])
        n_bytes = int(result["stream"]["n_bytes"])
        denominators.add(n_bytes)
        accounting = result["honest_accounting"]
        require_production_sequence_estimator(
            accounting["sequence_estimator"], source=str(result_path)
        )
        if int(accounting["triplet_grid_size"]) != int(plan["triplet_count"]):
            raise RuntimeError("result used a different declared triplet grid")
        for member, raw_rate in result["member_bits_per_token"].items():
            m1, m2 = map(int, member.split(":"))
            point = MemoryTwoPoint(vocabulary, m1, m2)
            if point not in expected:
                raise RuntimeError(f"undeclared campaign member {point.as_tuple()}")
            if point in observed:
                raise RuntimeError(f"duplicate campaign member {point.as_tuple()}")
            observed.add(point)
            state_bits = nested_frequency_subset_bits(point)
            recorded_state_bits = float(
                accounting["nested_state_subset_description_bits"][member]
            )
            if abs(state_bits - recorded_state_bits) > 1e-7:
                raise RuntimeError("nested state-subset accounting mismatch")
            total_without_model_choice = float(
                accounting["total_bits_before_triplet_selection"][member]
            )
            rows.append({
                "V": vocabulary,
                "M1": m1,
                "M2": m2,
                "observed_states": int(result["member_states_observed"][member]),
                "raw_bits_per_token": float(raw_rate),
                "nested_state_subset_description_bits": state_bits,
                "total_bits_before_triplet_selection": total_without_model_choice,
                "honest_member_bits_per_character": (
                    float(accounting["honest_member_bits_per_character"][member])
                ),
            })
    missing = expected - observed
    if missing:
        raise RuntimeError(
            "campaign is incomplete; missing "
            + ", ".join(map(str, sorted(p.as_tuple() for p in missing)))
        )
    if len(denominators) != 1:
        raise RuntimeError("all candidates must encode the same original file")
    denominator = float(next(iter(denominators)))
    mixture_bits = _uniform_two_part_mixture_bits([
        row["total_bits_before_triplet_selection"] for row in rows
    ])
    rows.sort(key=lambda row: row["honest_member_bits_per_character"])
    payload = {
        "version": 1,
        "sequence_estimator": PRODUCTION_SEQUENCE_ESTIMATOR,
        "plan": args.plan,
        "triplet_count": len(rows),
        "triplet_selection_bits": float(plan["triplet_selection_bits"]),
        "state_selection": (
            "nested_corpus_frequency_subsets_transmitted_enumeratively"
        ),
        "honest_family_bits_per_character": mixture_bits / denominator,
        "best_triplet": {
            key: rows[0][key] for key in (
                "V", "M1", "M2", "honest_member_bits_per_character"
            )
        },
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"written": str(out), **payload["best_triplet"]}))


if __name__ == "__main__":
    main()
