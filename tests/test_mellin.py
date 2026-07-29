"""Tests for the Mellin--Barnes prototype (module mellin).

The contour integral is held to the exact closed forms; the saddle
approximation is held to the contour integral, with error decreasing
in r; structural identities (log-convexity in r, the derivative
identity phi_{r+1} = -d phi_r / dt) are checked on contour values.
"""

import math

import numpy as np

from product_model_with_memory.mellin import (
    log_phi_closed_l1,
    log_phi_contour,
    log_phi_limit_t0,
    log_phi_saddle,
)


def test_contour_matches_closed_form_at_level_one():
    for r in [1, 5, 50, 1000]:
        for u in [-3.0, 0.0, 2.0]:
            exact = log_phi_closed_l1(r, u)
            got = log_phi_contour(r, 1, u)
            assert abs(got - exact) < 1e-8 * max(1.0, abs(exact))


def test_contour_matches_t_to_zero_limit():
    # far to the left, phi approaches Gamma(r+1)^L
    for r, L in [(3, 4), (20, 8)]:
        exact = log_phi_limit_t0(r, L)
        got = log_phi_contour(r, L, -60.0)
        assert abs(got - exact) < 1e-6 * max(1.0, abs(exact))


def test_saddle_error_decreases_in_r():
    L, u = 8, 0.5
    errs = []
    for r in [10, 100, 1000, 10_000]:
        ref = log_phi_contour(r, L, u)
        approx, _ = log_phi_saddle(r, L, u, order=2)
        errs.append(abs(approx - ref))
    assert errs[-1] < errs[0]
    assert errs[-1] < 1e-3  # nats, absolute, at r = 10^4


def test_log_convexity_in_r():
    # r -> ln phi_r(t) is convex (moments of a positive measure)
    L, u = 6, 1.0
    rs = [2, 4, 8, 16, 32]
    vals = [log_phi_contour(r, L, u) for r in rs]
    # convexity on the geometric ladder in the r variable:
    for i in range(1, len(rs) - 1):
        lam = (rs[i + 1] - rs[i]) / (rs[i + 1] - rs[i - 1])
        chord = lam * vals[i - 1] + (1 - lam) * vals[i + 1]
        assert vals[i] <= chord + 1e-9


def test_derivative_identity():
    # phi_{r+1}(t) = - d/dt phi_r(t); check with a central difference
    r, L, u = 6, 5, 0.3
    t = math.exp(u)
    h = 1e-5 * t
    lp = log_phi_contour(r, L, math.log(t + h))
    lm = log_phi_contour(r, L, math.log(t - h))
    p0 = log_phi_contour(r, L, u)
    # d phi/dt = phi * d(ln phi)/dt
    dln = (lp - lm) / (2 * h)
    got = p0 + math.log(-dln)  # ln( -d phi/dt ) = ln phi + ln(-dln)
    want = log_phi_contour(r + 1, L, u)
    assert abs(got - want) < 1e-5 * max(1.0, abs(want))
