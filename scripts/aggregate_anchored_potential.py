#!/usr/bin/env python3
"""Aggregate nested-window anchor scores into an eligible-domain potential estimate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from product_model_with_memory.potential_sampling import (
    PotentialAnchor,
    extrapolate_zero_age,
    stratified_delta_estimate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=9173)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--extrapolation-maximum-window", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text())
    anchors = tuple(PotentialAnchor(**row) for row in plan["anchors"])
    values, sensitivities, score_rows = {}, {}, []
    for anchor in anchors:
        path = Path(args.scores) / f"anchor_{anchor.anchor_id:03d}.json"
        score = json.loads(path.read_text())
        if int(score["fit_prefix"]) != anchor.prefix:
            raise RuntimeError(f"score prefix differs for anchor {anchor.anchor_id}")
        rings = score["rings"]
        if args.extrapolation_maximum_window is not None:
            rings = [
                row for row in rings
                if int(row["stop_offset"]) <= args.extrapolation_maximum_window
            ]
        intercept, sensitivity = extrapolate_zero_age(rings, anchor.prefix)
        values[anchor.anchor_id] = intercept
        sensitivities[anchor.anchor_id] = sensitivity
        score_rows.append({
            "anchor_id": anchor.anchor_id, "prefix": anchor.prefix,
            "zero_age_delta_bits_per_token": intercept,
            "quadratic_minus_linear": sensitivity,
        })
    estimate = stratified_delta_estimate(
        anchors, values, bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    quadratic_values = {
        key: values[key] + sensitivities[key] for key in values
    }
    quadratic = stratified_delta_estimate(
        anchors, quadratic_values, bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )["delta_bits_per_token"]
    payload = {
        "version": 1,
        "estimand": "eligible_domain_zero_age_graph_minus_markov1_v1",
        "plan": str(plan_path.resolve()),
        "extrapolation_maximum_window": args.extrapolation_maximum_window,
        **estimate,
        "quadratic_delta_bits_per_token": quadratic,
        "window_extrapolation_sensitivity": quadratic - estimate["delta_bits_per_token"],
        "anchors": score_rows,
        "limitations": (
            "This is the graph-minus-Markov increment on eligible prefix positions; "
            "it is not yet a whole-corpus honest bpc."
        ),
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(destination)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
