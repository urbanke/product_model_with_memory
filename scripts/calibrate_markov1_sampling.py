#!/usr/bin/env python3
"""Compare sampled zero-age Markov-1 loss with its telescoping exact value."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from product_model_with_memory.codelength import (
    default_l_max,
    depth_averaged_codelength_profiles,
)
from product_model_with_memory.potential_sampling import (
    PotentialAnchor,
    extrapolate_zero_age_field,
    stratified_delta_estimate,
)
from product_model_with_memory.production_coding import (
    PRODUCTION_SEQUENCE_ESTIMATOR,
    configure_production_tables,
    require_production_sequence_estimator,
)


def context_profiles(stream: np.ndarray, stop: int, vocabulary_size: int) -> Counter:
    """Return multiplicities of successor-count profiles before ``stop``."""

    counts = np.zeros((vocabulary_size, vocabulary_size), dtype=np.int64)
    if stop > 1:
        np.add.at(counts, (stream[: stop - 1], stream[1:stop]), 1)
    return Counter(
        tuple(sorted((int(x) for x in row if x), reverse=True))
        for row in counts
    )


def context_profiles_at_stops(
    stream: np.ndarray, stops: set[int], vocabulary_size: int,
) -> dict[int, Counter]:
    """Build sparse context-profile multiplicities at several prefix stops."""

    ordered = sorted(stops)
    if not ordered or ordered[0] < 1 or ordered[-1] > len(stream):
        raise ValueError("profile stops must lie within the stream")
    rows: dict[int, Counter] = {}
    answer: dict[int, Counter] = {}
    previous = 1
    for stop in ordered:
        for target_position in range(previous, stop):
            context = int(stream[target_position - 1])
            target = int(stream[target_position])
            row = rows.setdefault(context, Counter())
            row[target] += 1
        answer[stop] = Counter(
            tuple(sorted(row.values(), reverse=True)) for row in rows.values()
        )
        previous = stop
    return answer


def profile_log2_probabilities(profiles: set[tuple[int, ...]], vocabulary_size: int) -> dict:
    nonempty = {index: profile for index, profile in enumerate(sorted(profiles)) if profile}
    result = {(): 0.0}
    if nonempty:
        evaluated = depth_averaged_codelength_profiles(
            nonempty,
            d=vocabulary_size,
            l_max=default_l_max(vocabulary_size),
            jobs=1,
            tables_source="universal",
            universal_path=configure_production_tables(),
        )
        result.update({nonempty[index]: float(value.log2_q_avg) for index, value in evaluated.items()})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=9173)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--maximum-absolute-error", type=float, default=0.02)
    parser.add_argument("--maximum-extrapolation-sensitivity", type=float, default=0.02)
    parser.add_argument("--extrapolation-maximum-window", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    require_production_sequence_estimator(plan.get("sequence_estimator"), source=str(plan_path))
    anchors = tuple(PotentialAnchor(**row) for row in plan["anchors"])
    sampled, sensitivities, scores = {}, {}, {}
    for anchor in anchors:
        score_path = Path(args.scores) / f"anchor_{anchor.anchor_id:03d}.json"
        score = json.loads(score_path.read_text())
        require_production_sequence_estimator(score.get("sequence_estimator"), source=str(score_path))
        if int(score["fit_prefix"]) != anchor.prefix:
            raise RuntimeError(f"score prefix differs for anchor {anchor.anchor_id}")
        rings = score["rings"]
        if args.extrapolation_maximum_window is not None:
            rings = [
                row for row in rings
                if int(row["stop_offset"]) <= args.extrapolation_maximum_window
            ]
        sampled[anchor.anchor_id], sensitivities[anchor.anchor_id] = extrapolate_zero_age_field(
            rings, anchor.prefix, "markov1_bits",
        )
        scores[anchor.anchor_id] = score

    estimate = stratified_delta_estimate(
        anchors, sampled, bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    stream = np.load(plan["stream"], mmap_mode="r", allow_pickle=False)
    vocabulary_size = int(np.max(stream)) + 1
    domain_start = int(plan["minimum_prefix"])
    domain_stop = len(stream) - max(int(x) for x in plan["windows"]) + 1
    start_profiles = context_profiles(stream, domain_start, vocabulary_size)
    stop_profiles = context_profiles(stream, domain_stop, vocabulary_size)
    logq = profile_log2_probabilities(set(start_profiles) | set(stop_profiles), vocabulary_size)
    exact_bits = -sum(n * logq[p] for p, n in stop_profiles.items()) + sum(
        n * logq[p] for p, n in start_profiles.items()
    )
    exact_bpt = exact_bits / (domain_stop - domain_start)
    interval_stops = {anchor.prefix for anchor in anchors}
    for anchor in anchors:
        interval_stops.update(
            anchor.prefix + int(row["stop_offset"])
            for row in scores[anchor.anchor_id]["cumulative"]
        )
    interval_profiles = context_profiles_at_stops(
        stream, interval_stops, vocabulary_size,
    )
    interval_logq = profile_log2_probabilities(
        {profile for profiles in interval_profiles.values() for profile in profiles},
        vocabulary_size,
    )

    def prefix_bits(stop: int) -> float:
        return -sum(
            multiplicity * interval_logq[profile]
            for profile, multiplicity in interval_profiles[stop].items()
        )

    prefix_evidence = {stop: prefix_bits(stop) for stop in interval_stops}
    interval_rows = []
    for anchor in anchors:
        start_bits = prefix_evidence[anchor.prefix]
        for row in scores[anchor.anchor_id]["cumulative"]:
            window = int(row["stop_offset"])
            oracle = prefix_evidence[anchor.prefix + window] - start_bits
            frozen = float(row["markov1_bits"])
            interval_rows.append({
                "anchor_id": anchor.anchor_id,
                "prefix": anchor.prefix,
                "window": window,
                "oracle_continuously_updated_bits": oracle,
                "frozen_checkpoint_bits": frozen,
                "staleness_regret_bits": frozen - oracle,
                "staleness_regret_bits_per_token": (frozen - oracle) / window,
            })
    quadratic_values = {key: sampled[key] + sensitivities[key] for key in sampled}
    quadratic_bpt = stratified_delta_estimate(
        anchors, quadratic_values, bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )["delta_bits_per_token"]
    sampling_error = estimate["delta_bits_per_token"] - exact_bpt
    sensitivity = quadratic_bpt - estimate["delta_bits_per_token"]
    exact_in_analytic_ci = bool(
        estimate["analytic_ci95"][0] <= exact_bpt <= estimate["analytic_ci95"][1]
    )
    accuracy_passed = bool(abs(sampling_error) <= args.maximum_absolute_error)
    sensitivity_passed = bool(abs(sensitivity) <= args.maximum_extrapolation_sensitivity)
    payload = {
        "version": 1,
        "gate": "sampled_markov1_matches_telescoping_exact_v1",
        "sequence_estimator": PRODUCTION_SEQUENCE_ESTIMATOR,
        "plan": str(plan_path),
        "extrapolation_maximum_window": args.extrapolation_maximum_window,
        "domain_start": domain_start,
        "domain_stop": domain_stop,
        "eligible_positions": domain_stop - domain_start,
        "sampled_bits_per_token": estimate["delta_bits_per_token"],
        "sampled_analytic_standard_error": estimate["analytic_standard_error"],
        "sampled_analytic_ci95": estimate["analytic_ci95"],
        "sampled_bootstrap_ci95": estimate["bootstrap_ci95"],
        "sampled_quadratic_bits_per_token": quadratic_bpt,
        "exact_bits": exact_bits,
        "exact_bits_per_token": exact_bpt,
        "sampling_error_bits_per_token": sampling_error,
        "quadratic_error_bits_per_token": quadratic_bpt - exact_bpt,
        "interval_staleness": interval_rows,
        "linear_quadratic_sensitivity_bits_per_token": sensitivity,
        "thresholds": {
            "maximum_absolute_error_bits_per_token": args.maximum_absolute_error,
            "maximum_extrapolation_sensitivity_bits_per_token": args.maximum_extrapolation_sensitivity,
        },
        "checks": {
            "exact_in_analytic_ci95": exact_in_analytic_ci,
            "absolute_accuracy_passed": accuracy_passed,
            "extrapolation_sensitivity_passed": sensitivity_passed,
        },
        "passed": exact_in_analytic_ci and accuracy_passed and sensitivity_passed,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(destination)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
