"""The accuracy-calibration hooks must do exactly what they claim.

`scripts/phi_sensitivity.py` measures how many bits per character one nat
of error in ln phi is worth, by injecting a controlled error field.  The
whole measurement is worthless if the field is not what it says it is, so
these tests pin the three properties the interpretation depends on.

  * OFF BY DEFAULT.  An unperturbed run must be bit-identical to one with
    the hooks compiled in but unset, or every number in the paper would
    silently depend on this code.

  * A FIXED FIELD, not a fresh draw.  The same (L, r, u) is read many
    times in a run.  If the error were resampled per call it would
    average away and every scheme would look better than it is --- the
    measurement would systematically under-report the risk.

  * BOUNDED AND CENTRED.  The wave must lie in [-eps, +eps] and take
    both signs; bias must be exactly +eps.  Otherwise `delta/eps` is not
    the conversion factor it is read as.

  * SMOOTH IN u.  This is the one the first version failed, and the
    failure was expensive: a field decorrelated between adjacent grid
    points is amplified by 1/H and 1/H^2 when the evaluator
    differentiates the column to find the Laplace peak, so the sweep
    reported an amplification of 1e5 that was an artefact of the error
    model.  `test_the_field_is_smooth_in_u` pins it by bounding the
    second difference, which is precisely the quantity that blew up.
"""

import importlib
import os

import numpy as np
import pytest


def _fresh(env: dict[str, str]):
    """Re-import the store module under a given environment: the hooks
    are read at import time, which is what the sensitivity script relies
    on when it launches each run as a subprocess."""

    # every hook must be listed here.  A hook left out is not merely
    # untested: it LEAKS into every later test in the process, which
    # would quietly run them against a perturbed or laddered store.
    old = {k: os.environ.get(k) for k in
           ("PMM_PHI_WAVE", "PMM_PHI_WAVE_SCALE", "PMM_PHI_BIAS",
            "PMM_PHI_SADDLE_MIN_L", "PMM_PHI_LADDER",
            "PMM_PHI_LADDER_EVERY", "PMM_PHI_LADDER_DEGREE")}
    try:
        for k in old:
            os.environ.pop(k, None)
        os.environ.update(env)
        import product_model_with_memory.universal_tables as ut
        return importlib.reload(ut)
    finally:
        for k, v in old.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


@pytest.fixture(autouse=True)
def _restore():
    yield
    _fresh({})          # leave the module in its unperturbed state


U = np.linspace(-6.0, 12.0, 257)


def test_off_by_default():
    ut = _fresh({})
    assert ut._phi_perturbation(7, 13, U) is None


def test_bias_is_exactly_eps():
    ut = _fresh({"PMM_PHI_BIAS": "1e-3"})
    p = ut._phi_perturbation(7, 13, U)
    assert np.allclose(p, 1e-3, rtol=0, atol=0)


def test_wave_is_bounded_and_two_sided():
    ut = _fresh({"PMM_PHI_WAVE": "1e-3"})
    p = ut._phi_perturbation(7, 13, U)
    assert np.all(np.abs(p) <= 1e-3 + 1e-18)
    assert (p > 0).any() and (p < 0).any()


def test_the_field_is_smooth_in_u():
    """The property whose absence invalidated the first sweep.

    The evaluator differentiates the column twice (peak location, then
    curvature), so the quantity that matters is the second difference on
    the H = 0.02 grid.  For a field of amplitude eps varying on a scale
    lambda it must be of order eps * (H/lambda)^2, not of order eps.
    """

    eps, lam = 1e-3, 4.0
    ut = _fresh({"PMM_PHI_WAVE": repr(eps), "PMM_PHI_WAVE_SCALE": repr(lam)})
    grid = 35.0 - 0.02 * np.arange(4000)[::-1]
    p = ut._phi_perturbation(11, 501, grid)
    d2 = np.abs(np.diff(p, n=2))
    assert d2.max() < 4.0 * eps * (0.02 / lam) ** 2, d2.max()
    # For scale: the decorrelated field the first version used had a
    # second difference of order eps itself.  Here it is of order
    # eps * (H/lambda)^2 = 2.5e-5 * eps, about forty thousand times
    # smaller, and that ratio is the whole difference between a
    # measurement and an artefact.
    assert d2.max() < 1e-4 * eps, d2.max()


def test_the_field_is_fixed_not_resampled():
    """The property the whole measurement rests on."""

    ut = _fresh({"PMM_PHI_WAVE": "1e-3"})
    a = ut._phi_perturbation(7, 13, U)
    b = ut._phi_perturbation(7, 13, U)
    assert np.array_equal(a, b)


def test_distinct_columns_get_distinct_errors():
    """Different (L, r) must get different phases, or the field would be
    one global wave and would cancel far more than a real scheme's."""

    ut = _fresh({"PMM_PHI_WAVE": "1e-3"})
    same_u = [ut._phi_perturbation(L, 13, U[:1])[0] for L in (5, 6, 7)]
    assert len(set(same_u)) == 3
    same_L = [ut._phi_perturbation(7, r, U[:1])[0] for r in (12, 13, 14)]
    assert len(set(same_L)) == 3


def test_saddle_substitution_switches_on_at_the_cutoff():
    """The model-free measurement: levels at or above the cutoff must
    come from the expansion, and levels below must still come from the
    store, byte for byte."""

    ut = _fresh({})
    import os as _os
    from pathlib import Path
    root = Path(_os.environ.get("PMM_UNIVERSAL_TABLES",
                                "tables/universal_v2"))
    if not (root / "manifest.json").exists():
        return                      # no store here; nothing to compare
    u = np.linspace(-3.0, 8.0, 41)
    plain = ut.UniversalTables(root, read_only=True)
    below = plain.log_phi(6, 251, u).copy()
    above = plain.log_phi(20, 251, u).copy()

    ut2 = _fresh({"PMM_PHI_SADDLE_MIN_L": "12"})
    sub = ut2.UniversalTables(root, read_only=True)
    assert np.array_equal(sub.log_phi(6, 251, u), below)     # untouched
    got = sub.log_phi(20, 251, u)
    assert not np.array_equal(got, above)                    # substituted
    # and it is the store-free evaluator, not something else
    from product_model_with_memory.mellin import (log_phi_column,
                                                  log_phi_contour)
    want = log_phi_column(251.0, 20, u)
    assert np.allclose(got, want, rtol=0, atol=0)
    # the substitution is only worth measuring if it is actually
    # accurate; pin that against the independent reference rather than
    # trusting the two approximations to agree with each other
    ex = np.array([log_phi_contour(251.0, 20, float(x)) for x in u])
    assert np.abs(got - ex).max() < 1e-4, np.abs(got - ex).max()


def test_the_perturbation_actually_reaches_a_codelength():
    """A hook that never fires would make every amplitude read as
    harmless --- the most dangerous possible failure of this script."""

    from product_model_with_memory.codelength import (
        default_l_max,
        depth_averaged_codelength_profiles,
    )

    d = 12
    profiles = {"a": (5, 3, 2, 1, 1), "b": (9, 1), "c": (2, 2, 2)}

    def run():
        res = depth_averaged_codelength_profiles(
            profiles, d=d, l_max=default_l_max(d), jobs=1)
        return {k: res[k].log2_q_avg for k in profiles}

    base = run()
    _fresh({"PMM_PHI_BIAS": "1e-3"})
    import product_model_with_memory.codelength as cl
    importlib.reload(cl)
    from product_model_with_memory.codelength import (
        depth_averaged_codelength_profiles as pert_fn,
    )
    res = pert_fn(profiles, d=d, l_max=default_l_max(d), jobs=1)
    moved = {k: abs(res[k].log2_q_avg - base[k]) for k in profiles}
    assert max(moved.values()) > 1e-6, moved
    _fresh({})
    importlib.reload(cl)


def test_ladder_serves_anchors_exactly_and_interpolates_between():
    """The ladder hook: PMM_PHI_LADDER=f interpolates across r.

    Two properties, and the first is the one that would fail silently.
    An anchor must be served EXACTLY --- if the ladder interpolated at
    its own anchors it would be measuring interpolation error at points
    where there is none to make, and every end-to-end number would come
    out optimistic.  Between anchors it must actually interpolate, i.e.
    differ from the exact column but stay close to it.
    """

    import json
    from pathlib import Path

    root = Path(os.environ.get("PMM_ANCHOR_TABLES", ""))
    if not (root / "anchors.json").exists():
        return                      # no designed store here; nothing to test
    meta = json.loads((root / "anchors.json").read_text())
    L = int(next(iter(meta["levels"])))
    anchors = meta["levels"][str(L)]["anchors"]
    targets = meta["levels"][str(L)]["targets"]
    if not targets:
        return
    u = np.linspace(-6.0, 6.0, 9)

    ut = _fresh({"PMM_PHI_LADDER_EVERY": "1"})
    tab = ut.UniversalTables(root, read_only=True)
    a = anchors[len(anchors) // 2]
    assert np.array_equal(tab.log_phi(L, a, u), tab.log_phi_exact(L, a, u))

    t = targets[0]
    got, ex = tab.log_phi(L, t, u), tab.log_phi_exact(L, t, u)
    assert not np.array_equal(got, ex)          # it really interpolated
    assert np.abs(got - ex).max() < 1e-3        # and did not wander off


def test_ladder_off_by_default_leaves_values_untouched():
    import json
    from pathlib import Path

    root = Path(os.environ.get("PMM_ANCHOR_TABLES", ""))
    if not (root / "anchors.json").exists():
        return
    meta = json.loads((root / "anchors.json").read_text())
    L = int(next(iter(meta["levels"])))
    t = (meta["levels"][str(L)]["targets"] or [None])[0]
    if t is None:
        return
    u = np.linspace(-6.0, 6.0, 9)
    ut = _fresh({})
    tab = ut.UniversalTables(root, read_only=True)
    assert np.array_equal(tab.log_phi(L, t, u), tab.log_phi_exact(L, t, u))
