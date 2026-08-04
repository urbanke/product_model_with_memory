"""The compiled stencil must return EXACTLY what the numpy path does.

`universal_tables.log_phi_matrix` serves stored columns onto a query
grid, either through the compiled 8-point stencil or through numpy.
These compare the two value by value, across several levels, several
query grids and a wide span of r, and require equality — not closeness.

That strictness is the point.  The kernel shipped once with a defect a
benchmark could not see: a local array named `vals` inside
`_interp_leftovers` shadowed the parameter holding the stored column, so
the edge branch interpolated the wrong array.  It was wrong by up to 622
nats at about 1% of query points, and every end-to-end check still
printed the right codelength, because the profiles in those runs never
read the affected points.  Comparing codelengths is not enough; the
comparison has to be at the values.

The three regions the query grid splits into all have to be covered,
since the defect lived in the handoff between them: points served by the
stencil, points left of the stored column (answered by the certified
series), and points where the stencil window would run off the stored
values (the clamped-weight path).
"""

import os

import numpy as np

import product_model_with_memory.universal_tables as UT
from product_model_with_memory.codelength import (
    _provision_tables,
    needed_r_values,
)


PROFILES = [
    (5, 3, 2, 1, 1, 1),
    (400, 90, 17, 3, 2, 1, 1),
    (17,) * 30,
    (67465, 4001, 251, 17, 3, 1),
]


def _both_paths(L, rs, u, ut):
    """(compiled, numpy) matrices for the same request."""

    if UT._INTERP is None:
        return None, None
    keep = os.environ.get("PMM_INTERP_KERNEL")
    try:
        os.environ["PMM_INTERP_KERNEL"] = "1"
        compiled = ut.log_phi_matrix(L, rs, u)
        saved, UT._INTERP = UT._INTERP, None
        try:
            numpy_path = ut.log_phi_matrix(L, rs, u)
        finally:
            UT._INTERP = saved
    finally:
        if keep is None:
            os.environ.pop("PMM_INTERP_KERNEL", None)
        else:
            os.environ["PMM_INTERP_KERNEL"] = keep
    return compiled, numpy_path


def _case(l_max, levels):
    all_r = set()
    for p in PROFILES:
        all_r.update(needed_r_values(p))
    prov = _provision_tables("universal", l_max, all_r, None, None, 1, 96,
                             None)
    u = np.asarray(prov["u_grid"])
    ut = prov["ut"]
    rs = sorted(all_r)
    for L in levels:
        compiled, numpy_path = _both_paths(L, rs, u, ut)
        if compiled is None:
            return                      # no compiler on this machine
        bad = np.argwhere(compiled != numpy_path)
        if len(bad):
            i, j = bad[0]
            raise AssertionError(
                f"L={L}, r={rs[i]}, u={u[j]!r}: compiled "
                f"{compiled[i, j]!r} vs numpy {numpy_path[i, j]!r} "
                f"({len(bad)} of {compiled.size} values differ, "
                f"max |delta| = {np.abs(compiled - numpy_path).max()})")


def test_stencil_matches_numpy_shallow():
    _case(2, (2,))


def test_stencil_matches_numpy_deep():
    _case(26, (2, 7, 17, 26))


def test_stencil_matches_numpy_middle_grid():
    _case(12, (2, 5, 12))


def test_series_tail_shortcut_is_exact():
    """`PMM_SERIES_TAIL` replaces the certified series with its t -> 0
    limit far left of a stored column.  The limit is reached EXACTLY in
    float64 thirty nats out, so the shortcut must not move a single
    value — this compares it against evaluating the series everywhere."""

    all_r = set()
    for p in PROFILES:
        all_r.update(needed_r_values(p))
    rs = sorted(all_r)
    keep = UT.SERIES_TAIL_NATS
    try:
        for l_max, levels in ((2, (2,)), (12, (2, 5, 12)), (26, (2, 17, 26))):
            prov = _provision_tables("universal", l_max, all_r, None, None,
                                     1, 96, None)
            u = np.asarray(prov["u_grid"])
            ut = prov["ut"]
            for L in levels:
                UT.SERIES_TAIL_NATS = float("inf")
                full = ut.log_phi_matrix(L, rs, u)
                UT.SERIES_TAIL_NATS = 40.0
                short = ut.log_phi_matrix(L, rs, u)
                bad = np.argwhere(full != short)
                if len(bad):
                    i, j = bad[0]
                    raise AssertionError(
                        f"L={L}, r={rs[i]}, u={u[j]!r}: series "
                        f"{full[i, j]!r} vs limit {short[i, j]!r} "
                        f"({len(bad)} of {full.size} differ)")
    finally:
        UT.SERIES_TAIL_NATS = keep


def test_all_three_regions_are_exercised():
    """A guard on the guard: if every query point were served by the
    stencil, the tests above would never touch the series branch or the
    clamped-weight branch, where the shipped defect lived."""

    all_r = set()
    for p in PROFILES:
        all_r.update(needed_r_values(p))
    prov = _provision_tables("universal", 12, all_r, None, None, 1, 96, None)
    u = np.asarray(prov["u_grid"])
    ut = prov["ut"]
    n_left = n_clamped = n_plain = 0
    t = (UT.U_MAX - u) / UT.H
    k0 = np.floor(-t).astype(np.int64) - (UT._STENCIL // 2 - 1)
    for r in sorted(all_r):
        i0, vals = ut.column(5, r)
        grid0 = UT.U_MAX - UT.H * i0
        left = u < grid0
        start = i0 + k0
        clamped = (~left) & ((start < 0) | (start > len(vals) - UT._STENCIL))
        n_left += int(left.sum())
        n_clamped += int(clamped.sum())
        n_plain += int((~left & ~clamped).sum())
    assert n_left > 0, "no query point exercises the series branch"
    assert n_clamped > 0, "no query point exercises the clamped branch"
    assert n_plain > 0, "no query point exercises the stencil itself"
