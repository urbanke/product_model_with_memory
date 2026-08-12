"""Immutable stratified sampling design for continuous-update potential."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


POTENTIAL_SAMPLING_DESIGN = "hybrid_stratified_anchors_v1"


@dataclass(frozen=True)
class PotentialAnchor:
    anchor_id: int
    stratum: int
    region: str
    prefix: int
    stratum_start: int
    stratum_stop: int
    inclusion_probability: float
    expansion_weight: float


def extrapolate_zero_age_field(
    rings: list[dict], prefix: int, field: str,
) -> tuple[float, float]:
    """Fit a ring-average loss field versus relative age.

    The intercept estimates the loss at age zero and the quadratic-minus-linear
    difference is retained as an extrapolation-sensitivity diagnostic.
    """

    usable = [row for row in rings if int(row["tokens"]) > 0]
    if len(usable) < 2:
        raise ValueError("at least two nonempty rings are required")
    x = np.asarray([
        (float(row["start_offset"]) + float(row["stop_offset"]) - 1.0)
        / (2.0 * prefix)
        for row in usable
    ])
    y = np.asarray([float(row[field]) / int(row["tokens"]) for row in usable])
    weights = np.sqrt(np.asarray([int(row["tokens"]) for row in usable]))
    linear = np.polyfit(x, y, 1, w=weights)
    intercept = float(linear[-1])
    sensitivity = 0.0
    if len(usable) >= 3:
        quadratic = np.polyfit(x, y, 2, w=weights)
        sensitivity = float(quadratic[-1] - intercept)
    return intercept, sensitivity


def extrapolate_zero_age(rings: list[dict], prefix: int) -> tuple[float, float]:
    """Fit ring-average graph-minus-control loss at zero age."""

    return extrapolate_zero_age_field(rings, prefix, "delta_bits")


def stratified_delta_estimate(
    anchors: tuple[PotentialAnchor, ...],
    values: dict[int, float],
    *,
    bootstrap_seed: int = 0,
    bootstrap_replicates: int = 2000,
) -> dict:
    """Aggregate anchor values with exact stratum weights and uncertainty."""

    groups: dict[int, list[PotentialAnchor]] = {}
    for anchor in anchors:
        if anchor.anchor_id not in values:
            raise ValueError(f"missing value for anchor {anchor.anchor_id}")
        groups.setdefault(anchor.stratum, []).append(anchor)
    total_population = sum(
        group[0].stratum_stop - group[0].stratum_start for group in groups.values()
    )
    total = 0.0
    variance_total = 0.0
    rows = []
    for stratum, group in sorted(groups.items()):
        observed = np.asarray([values[anchor.anchor_id] for anchor in group])
        population = group[0].stratum_stop - group[0].stratum_start
        mean = float(np.mean(observed))
        total += population * mean
        sample_variance = float(np.var(observed, ddof=1)) if len(group) > 1 else 0.0
        variance_total += population ** 2 * (1.0 - len(group) / population) * sample_variance / len(group)
        rows.append((stratum, population, observed, mean))
    estimate = total / total_population
    standard_error = np.sqrt(variance_total) / total_population
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = np.empty(bootstrap_replicates)
    for replicate in range(bootstrap_replicates):
        value = 0.0
        for _, population, observed, _ in rows:
            value += population * float(np.mean(rng.choice(observed, len(observed), replace=True)))
        bootstrap[replicate] = value / total_population
    jackknife = []
    for omitted, (_, population, _, _) in enumerate(rows):
        remaining_population = total_population - population
        remaining_total = sum(
            other_population * mean
            for index, (_, other_population, _, mean) in enumerate(rows)
            if index != omitted
        )
        jackknife.append(remaining_total / remaining_population)
    return {
        "eligible_positions": total_population,
        "delta_bits_per_token": estimate,
        "estimated_delta_bits": estimate * total_population,
        "analytic_standard_error": float(standard_error),
        "analytic_ci95": [estimate - 1.96 * standard_error, estimate + 1.96 * standard_error],
        "bootstrap_ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "delete_one_stratum_range": [float(np.min(jackknife)), float(np.max(jackknife))],
    }


def plan_potential_anchors(
    n: int,
    *,
    minimum_prefix: int,
    maximum_window: int,
    seed: int,
    early_strata: int = 8,
    late_strata: int = 24,
    samples_per_stratum: int = 2,
    early_fraction: float = 0.05,
) -> tuple[PotentialAnchor, ...]:
    """Return a fixed hybrid stratified sample of eligible prefix positions."""

    if not 0.0 < early_fraction < 1.0:
        raise ValueError("early_fraction must lie in (0, 1)")
    if min(early_strata, late_strata, samples_per_stratum) < 1:
        raise ValueError("stratum counts and samples must be positive")
    stop = n - maximum_window + 1
    if minimum_prefix < 2 or stop <= minimum_prefix:
        raise ValueError("corpus is too short for the requested prefix and window")
    early_stop = min(stop - samples_per_stratum, max(
        minimum_prefix + early_strata * samples_per_stratum,
        int(np.floor(early_fraction * n)),
    ))
    if early_stop <= minimum_prefix:
        raise ValueError("no room for the early sampling region")

    early = np.rint(np.geomspace(
        minimum_prefix, early_stop, early_strata + 1,
    )).astype(np.int64)
    early[0], early[-1] = minimum_prefix, early_stop
    late = np.rint(np.linspace(
        early_stop, stop, late_strata + 1,
    )).astype(np.int64)
    late[0], late[-1] = early_stop, stop
    boundaries = [
        ("early", int(lo), int(hi)) for lo, hi in zip(early, early[1:])
    ] + [
        ("late", int(lo), int(hi)) for lo, hi in zip(late, late[1:])
    ]
    if any(hi - lo < samples_per_stratum for _, lo, hi in boundaries):
        raise ValueError("a sampling stratum is too small")
    if any(a[2] != b[1] for a, b in zip(boundaries, boundaries[1:])):
        raise RuntimeError("sampling strata do not form a contiguous partition")
    rng = np.random.default_rng(seed)
    anchors: list[PotentialAnchor] = []
    for stratum, (region, lo, hi) in enumerate(boundaries):
        population = hi - lo
        selected = np.sort(
            rng.choice(population, size=samples_per_stratum, replace=False) + lo
        )
        inclusion = samples_per_stratum / float(population)
        for prefix in selected:
            anchors.append(PotentialAnchor(
                anchor_id=len(anchors), stratum=stratum, region=region,
                prefix=int(prefix), stratum_start=lo, stratum_stop=hi,
                inclusion_probability=inclusion,
                expansion_weight=1.0 / inclusion,
            ))
    return tuple(anchors)
