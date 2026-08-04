"""The windowed scan must return the SAME numbers as the full sweep.

`PMM_SCAN=sparse` evaluates the log-integrand on a coarse stride of the
grid, then finely only inside windows around the coarse maxima, instead
of sweeping all 30,608 points.  Peak refinement is untouched: the same
bracketed solves on the same brackets, so the saddles, the curvatures
and the Laplace contributions are computed from identical inputs and
the result is bit-identical, not merely close.  These tests fix that.

They also cover the two ways the window could hide a peak: a peak
sitting at the extreme left of the grid (the sparse-profile case, where
the mass is in the analytic left tail) and a profile heavy enough to
push the saddle far to the right.
"""

import math
import os

from product_model_with_memory.codelength import (
    default_l_max,
    depth_averaged_codelength_profiles,
)


PROFILES = {
    "sparse": (3, 2, 1, 1, 1, 1, 1, 1),
    "one_heavy": (400, 3, 2, 1, 1, 1),
    "flat": tuple([2] * 40),
    "heavy": (5000, 900, 400, 120, 60, 30, 10, 4, 2, 1, 1, 1),
    "singleton": (1,),
    "pair": (1, 1),
}


def _both(profiles, d, l_max):
    """(full sweep, windowed) results for the same profiles."""

    keep = os.environ.get("PMM_SCAN")
    try:
        os.environ["PMM_SCAN"] = "full"
        full = depth_averaged_codelength_profiles(profiles, d=d, l_max=l_max)
        os.environ["PMM_SCAN"] = "sparse"
        windowed = depth_averaged_codelength_profiles(
            profiles, d=d, l_max=l_max)
    finally:
        if keep is None:
            os.environ.pop("PMM_SCAN", None)
        else:
            os.environ["PMM_SCAN"] = keep
    return full, windowed


def test_windowed_scan_is_bit_identical():
    d = 256
    full, windowed = _both(PROFILES, d, default_l_max(d))
    for key in PROFILES:
        a, b = full[key].log2_q_avg, windowed[key].log2_q_avg
        assert a == b, f"{key}: full {a!r} windowed {b!r}"


def test_windowed_scan_identical_at_every_depth():
    """Not just the average: every level's term must match, since the
    level truncation compares them against a threshold."""

    d = 256
    full, windowed = _both(PROFILES, d, default_l_max(d))
    for key in PROFILES:
        fa = full[key].log2_q_by_depth
        wa = windowed[key].log2_q_by_depth
        assert len(fa) == len(wa), key
        for level, (x, y) in enumerate(zip(fa, wa), start=1):
            assert x == y, f"{key} L={level}: full {x!r} windowed {y!r}"


def test_windowed_scan_large_alphabet():
    """The regime the paper runs in: a sparse profile over a subword
    alphabet, where the dominant saddle sits left of the grid."""

    d = 100_277
    profiles = {"state": (7, 4, 3, 2, 2, 1, 1, 1, 1, 1)}
    full, windowed = _both(profiles, d, 20)
    assert full["state"].log2_q_avg == windowed["state"].log2_q_avg


def test_windowed_scan_still_a_probability():
    d = 256
    _, windowed = _both(PROFILES, d, default_l_max(d))
    for key, res in windowed.items():
        assert res.log2_q_avg <= 0.0, f"{key}: log2 q = {res.log2_q_avg}"
        assert math.isfinite(res.log2_q_avg), key
