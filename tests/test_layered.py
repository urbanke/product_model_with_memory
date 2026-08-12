import math
from collections import Counter

import numpy as np
import pytest

from product_model_with_memory import (
    default_l_max,
    depth_averaged_codelength,
    needed_r_values,
    profile_of,
)
from product_model_with_memory.fast_tables import build_tables_fast
from product_model_with_memory.layered import (
    log_q_lambda_scan,
    build_selected_product_moment_tables,
    log_q_lambda_closed_l1,
)


def test_fast_tables_match_reference():
    r_values = [0, 1, 2, 3, 5, 6, 7]
    reference = build_selected_product_moment_tables(
        max_L=7, r_values=r_values, u_points=2_001
    )
    fast = build_tables_fast(max_L=7, r_values=r_values, u_points=2_001)
    for key, ref in reference.log_phi.items():
        got = fast.log_phi[key]
        finite = np.isfinite(ref)
        assert np.all(np.isfinite(got) == finite)
        assert np.allclose(got[finite], ref[finite], rtol=1e-9, atol=1e-9), key


def test_l1_closed_form_is_add_one_dirichlet():
    # q_m = prod(m_i!) Gamma(d) / Gamma(d + N) for L = 1.
    part = (3, 2, 1)
    d, n = 20, 6
    expected = (
        math.lgamma(d)
        - math.lgamma(d + n)
        + sum(math.lgamma(m + 1) for m in part)
    )
    assert log_q_lambda_closed_l1(d=d, partition=part).log_q == pytest.approx(
        expected
    )


def test_depth_average_against_monte_carlo():
    # Q^(L)(x^N) = E_theta prod theta_i^{m_i}; check the production scan
    # evaluator against a direct Monte Carlo estimate on a small alphabet.
    rng = np.random.default_rng(7)
    d, L, part = 8, 3, (3, 1)
    samples = 400_000
    y = np.ones((samples, d))
    for _ in range(L):
        y *= rng.exponential(size=(samples, d))
    theta = y / y.sum(axis=1, keepdims=True)
    mc = float(np.mean(theta[:, 0] ** 3 * theta[:, 1]))
    tables = build_tables_fast(max_L=L, r_values=needed_r_values(part))
    result = log_q_lambda_scan(d=d, L=L, partition=part, tables=tables)
    assert result.log_q == pytest.approx(math.log(mc), abs=0.05)


def test_deep_layer_heavy_count_never_positive():
    # Regression: with a grid cut at the reference default -70, deep layers
    # with heavy counts inflate near the left edge and can fabricate a
    # spurious peak with log q > 0.  The adaptive grid must keep log q <= 0
    # (a sequence probability) at every depth.
    part = (521, 236, 118, 60, 30) + (8,) * 40 + (2,) * 300 + (1,) * 1200
    tables = build_tables_fast(max_L=59, r_values=needed_r_values(part))
    for L in (2, 15, 30, 59):
        r = log_q_lambda_scan(d=262_144, L=L, partition=part, tables=tables)
        assert r.converged, (L, r.message)
        assert r.log_q <= 0.0, (L, r.log_q)


def test_piecewise_grid_matches_uniform():
    # The adaptive piecewise grid (coarse far-left segment) must agree with
    # a fully fine uniform grid on a deep-layer heavy-count case.
    from product_model_with_memory.codelength import (
        depth_averaged_codelength_profiles,
    )
    import tempfile

    d, L, part = 2_000, 20, (100, 5, 2, 1)
    r_needed = needed_r_values(part)
    uniform = build_tables_fast(
        max_L=L, r_values=r_needed, u_min=-160.0, u_points=28_001
    )
    piecewise = build_tables_fast(max_L=L, r_values=r_needed)  # adaptive default
    for depth in (2, 10, 20):
        a = log_q_lambda_scan(d=d, L=depth, partition=part, tables=uniform)
        b = log_q_lambda_scan(d=d, L=depth, partition=part, tables=piecewise)
        assert abs(a.log_q - b.log_q) < 0.02, depth
    # and the memory-frugal streaming path gives the same depth-average
    with tempfile.TemporaryDirectory() as tmp:
        lazy = depth_averaged_codelength_profiles(
            {108: part}, d=d, l_max=L, cache_dir=tmp
        )[108]
    eager = depth_averaged_codelength(
        dict(zip("abcd", part)), d=d, l_max=L
    )
    assert abs(lazy.log2_q_avg - eager.log2_q_avg) < 0.05


def test_depth_averaged_codelength_smoke():
    counts = Counter("abrakadabra")
    result = depth_averaged_codelength(counts, d=64, l_max=8)
    assert result.n == 11
    assert result.l_max == 8
    # average must lie between best and worst single depth
    per_depth = [result.bits_per_token_at_depth(L) for L in range(1, 9)]
    assert min(per_depth) <= result.bits_per_token
    assert result.bits_per_token <= max(per_depth) + math.log2(8) / 11
    assert abs(sum(result.posterior) - 1.0) < 1e-9


def test_default_l_max():
    # d = 262144: 2 * c* * ln d = 59.02...
    assert default_l_max(262_144) == 59


def test_profile_of():
    assert profile_of(Counter("aabbbc")) == (3, 2, 1)
    assert profile_of({"x": 2, "y": 0}) == (2,)
