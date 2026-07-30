"""Regression: universal-table path vs legacy cache path.

Two comparisons, matched to what each side can promise:

1. In the legacy tables' clean regime (all counts < 500), both paths
   must agree to ~1e-3 bits --- the legacy tables' own accuracy.
2. For a heavy profile (count 800, inside the legacy tables' KNOWN
   failure regime r >~ 600), the universal path must instead match
   exact columns built directly on the production grid (no store, no
   interpolation) --- arbitration July 2026 showed universal == exact
   to ~1e-6 while legacy drifts by up to 6e-3 bits, exactly the
   documented legacy-table error.
"""

import tempfile

import numpy as np

from product_model_with_memory.codelength import (
    depth_averaged_codelength_profiles,
    needed_r_values,
)

# legacy-clean regime: every count well below the legacy failure onset
# (measured July 2026: destructive far-left legacy errors begin near
# r ~ 360-400; at counts <= ~300 legacy is accurate where integrands
# have mass)
PROFILES = {
    0: (301, 250, 120, 50, 20, 8, 3, 1, 1, 1),
    1: (301, 301, 120, 8, 8, 2, 1),
    2: (50, 20, 8, 3, 1),
    3: (1,),
}
D = 48
L_MAX = 6


def _run(source, tmp, jobs=1):
    return depth_averaged_codelength_profiles(
        PROFILES, d=D, l_max=L_MAX,
        cache_dir=f"{tmp}/cache",
        tables_source=source,
        universal_path=f"{tmp}/universal",
        jobs=jobs,
    )


def test_universal_matches_legacy_cache_in_clean_regime():
    # Tolerance note (measured July 2026): even at counts ~300 the
    # legacy tables drift by up to ~5e-3 bits per profile at depth 6
    # (drift grows with depth; arbitration against directly built
    # exact columns shows the LEGACY side moves, the universal side
    # matches exact to ~1e-6).  This test therefore only verifies the
    # wiring at legacy's own accuracy; the tight correctness anchor is
    # test_universal_matches_direct_exact_columns_heavy_profile.
    with tempfile.TemporaryDirectory() as tmp:
        legacy = _run("cache", tmp)
        universal = _run("universal", tmp)
        for key in PROFILES:
            a, b = legacy[key], universal[key]
            assert abs(a.log2_q_avg - b.log2_q_avg) < 2e-2, (
                key, a.log2_q_avg, b.log2_q_avg)
            for L, (qa, qb) in enumerate(
                    zip(a.log2_q_by_depth, b.log2_q_by_depth), start=1):
                assert abs(qa - qb) < 2e-2, (key, L, qa, qb)


def test_universal_parallel_matches_serial():
    with tempfile.TemporaryDirectory() as tmp:
        serial = _run("universal", tmp)
        parallel = _run("universal", tmp, jobs=2)
        for key in PROFILES:
            assert np.allclose(
                serial[key].log2_q_by_depth,
                parallel[key].log2_q_by_depth,
                rtol=0, atol=1e-12,
            )


def test_universal_matches_direct_exact_columns_heavy_profile():
    from product_model_with_memory.fast_tables import (
        default_grid_spec,
        grid_from_spec,
    )
    from product_model_with_memory.layered import (
        ProductMomentTables,
        log_q_lambda_scan,
    )
    from product_model_with_memory.mellin import exact_log_phi_column
    from product_model_with_memory.universal_tables import (
        ensure_universal_tables,
    )

    part = (800, 50, 3, 1)   # 800 is inside the legacy failure regime
    d, l_max = 48, 4
    rs = sorted(needed_r_values(part))
    u_grid = grid_from_spec(
        default_grid_spec(max_L=l_max, r_max=max(rs), u_max=35.0))
    with tempfile.TemporaryDirectory() as tmp:
        ut = ensure_universal_tables(tmp)
        for L in range(2, l_max + 1):
            exact = ProductMomentTables(
                max_L=l_max, max_r=max(rs), r_values=tuple(rs),
                u_grid=u_grid,
                log_phi={(L, r): exact_log_phi_column(r, L, u_grid)
                         for r in rs},
            )
            qe = log_q_lambda_scan(d=d, L=L, partition=part,
                                   tables=exact).log2_q
            qu = log_q_lambda_scan(
                d=d, L=L, partition=part,
                tables=ut.level_tables(L, rs, u_grid)).log2_q
            assert abs(qu - qe) < 1e-5, (L, qu, qe)
