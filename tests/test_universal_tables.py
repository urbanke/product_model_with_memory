"""Tests for the universal table store (v2)."""

import tempfile

import numpy as np

from product_model_with_memory.mellin import log_phi_contour
from product_model_with_memory.universal_tables import (
    UniversalTables,
    ensure_universal_tables,
)


def test_grow_persist_reopen():
    with tempfile.TemporaryDirectory() as tmp:
        ut = ensure_universal_tables(tmp)
        ut.ensure_columns(5, [1, 100, 1000])
        i0a, ca = ut.column(5, 1000)
        ut2 = UniversalTables(tmp)  # reopen from disk
        i0b, cb = ut2.column(5, 1000)
        assert i0a == i0b and np.array_equal(ca, cb)


def test_series_handoff_left_of_column():
    with tempfile.TemporaryDirectory() as tmp:
        ut = ensure_universal_tables(tmp)
        v = ut.log_phi(5, 3, [-200.0, 0.0])
        # far left equals the t->0 limit region value: 5*lgamma(4)
        from scipy.special import loggamma
        assert abs(v[0] - 5 * float(loggamma(4.0))) < 1e-6
        assert np.isfinite(v[1])


def test_served_values_match_reference_all_regimes():
    # OFF-GRID queries (interpolation included) against the reference,
    # covering left handoff, interior, and right of the grid edge
    with tempfile.TemporaryDirectory() as tmp:
        ut = ensure_universal_tables(tmp)
        for r in [3, 100, 5000]:
            for u in [-5.003, 0.0137, 5.011, 34.507, 36.2]:
                got = float(ut.log_phi(8, r, [u])[0])
                ref = log_phi_contour(r, 8, u)
                assert abs(got - ref) < 1e-6, (r, u, got - ref)


def test_certify_records_to_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        ut = ensure_universal_tables(tmp)
        ut.ensure_columns(4, [1, 50, 2000])
        res = ut.certify(samples=10)
        assert res["n"] > 0 and res["max"] < 1e-6
        ut2 = UniversalTables(tmp)
        assert len(ut2.manifest["certifications"]) == 1
