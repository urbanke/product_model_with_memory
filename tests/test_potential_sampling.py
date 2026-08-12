import numpy as np

from product_model_with_memory.potential_sampling import (
    extrapolate_zero_age,
    extrapolate_zero_age_field,
    plan_potential_anchors,
    stratified_delta_estimate,
)


def test_zero_age_extrapolation_accepts_an_explicit_loss_field():
    rings = [
        {"start_offset": 0, "stop_offset": 4, "tokens": 4, "loss_bits": 6.0},
        {"start_offset": 4, "stop_offset": 12, "tokens": 8, "loss_bits": 14.0},
        {"start_offset": 12, "stop_offset": 28, "tokens": 16, "loss_bits": 34.0},
    ]
    intercept, sensitivity = extrapolate_zero_age_field(rings, 100, "loss_bits")
    assert np.isfinite(intercept)
    assert np.isfinite(sensitivity)


def test_hybrid_anchor_design_is_reproducible_contiguous_and_honestly_weighted():
    kwargs = dict(
        n=1_000_000, minimum_prefix=2050, maximum_window=4096,
        seed=81, early_strata=8, late_strata=24, samples_per_stratum=2,
    )
    first = plan_potential_anchors(**kwargs)
    second = plan_potential_anchors(**kwargs)
    assert first == second
    assert len(first) == 64
    assert sum(anchor.region == "early" for anchor in first) == 16
    assert sum(anchor.region == "late" for anchor in first) == 48
    by_stratum = {}
    for anchor in first:
        by_stratum.setdefault(anchor.stratum, []).append(anchor)
        assert anchor.stratum_start <= anchor.prefix < anchor.stratum_stop
    assert len(by_stratum) == 32
    assert all(len(values) == 2 for values in by_stratum.values())
    intervals = [values[0] for values in by_stratum.values()]
    assert intervals[0].stratum_start == kwargs["minimum_prefix"]
    assert all(a.stratum_stop == b.stratum_start for a, b in zip(intervals, intervals[1:]))
    assert intervals[-1].stratum_stop == kwargs["n"] - kwargs["maximum_window"] + 1
    # Horvitz--Thompson weights cover every eligible prefix exactly once.
    expected = kwargs["n"] - kwargs["maximum_window"] + 1 - kwargs["minimum_prefix"]
    np.testing.assert_allclose(sum(anchor.expansion_weight for anchor in first), expected)


def test_zero_age_extrapolation_and_stratified_estimate_recover_linear_signal():
    anchors = plan_potential_anchors(
        100_000, minimum_prefix=100, maximum_window=1000, seed=4,
        early_strata=2, late_strata=3, samples_per_stratum=2,
    )
    values = {}
    for anchor in anchors:
        intercept = 0.2 + 1e-6 * anchor.prefix
        rings = []
        for lo, hi in ((0, 10), (10, 40), (40, 100)):
            midpoint_relative = (lo + hi - 1) / (2 * anchor.prefix)
            mean = intercept + 0.7 * midpoint_relative
            rings.append({
                "start_offset": lo, "stop_offset": hi, "tokens": hi - lo,
                "delta_bits": mean * (hi - lo),
            })
        recovered, sensitivity = extrapolate_zero_age(rings, anchor.prefix)
        np.testing.assert_allclose(recovered, intercept, atol=1e-12)
        np.testing.assert_allclose(sensitivity, 0.0, atol=1e-12)
        values[anchor.anchor_id] = recovered
    result = stratified_delta_estimate(
        anchors, values, bootstrap_seed=2, bootstrap_replicates=100,
    )
    assert result["eligible_positions"] == 100_000 - 1000 + 1 - 100
    assert result["analytic_standard_error"] >= 0.0
    assert result["bootstrap_ci95"][0] <= result["delta_bits_per_token"] <= result["bootstrap_ci95"][1]
