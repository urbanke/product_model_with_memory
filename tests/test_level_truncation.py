"""Truncating the sum over levels must not change any number.

The rule (codelength._LevelWindow) stops evaluating a profile once its
per-level term has fallen far below its own running maximum and has been
decreasing for several levels.  These tests check the two things that
could go wrong: that the answer moves, and that a family stops
normalizing because its members were truncated at different levels.
"""

import math
import os

import numpy as np

from product_model_with_memory.codelength import (
    _LevelWindow,
    default_l_max,
    depth_averaged_codelength_families,
    depth_averaged_codelength_profiles,
)
from product_model_with_memory.layered import augmented_partition

D = 64
L_MAX = default_l_max(D)


def _zipf(d: int, n: int) -> tuple[int, ...]:
    w = np.array([1.0 / (i + 1) for i in range(d)])
    w /= w.sum()
    return tuple(sorted(np.maximum(1, np.round(w * n)).astype(int).tolist(),
                        reverse=True))


# Both regimes have to survive, and they are genuinely different.
# `dense` fills the alphabet (k = d) with heavy counts: its level curve
# peaks at L = 1 and falls ~96 bits, which is the shape the real corpus
# profiles have and the only one truncation can exploit.  The others are
# sparse relative to d, their curves RISE to L_max, and truncation must
# correctly decline to fire on them.
PROFILES = {
    "dense": _zipf(D, 20_000),
    "sparse": tuple(sorted((900, 500, 300, 120, 60, 30, 12, 5, 3, 1),
                           reverse=True)),
    "light": (4, 3, 2, 1, 1),
    "single": (11,),
}

# the family test needs unseen symbols to exist, so its base must leave
# room in the alphabet (a saturated row has no c = 0 augmentation)
FAMILY_BASE = _zipf(D - 16, 20_000)


# These profiles are small enough to evaluate in a unit test, so their
# level curves never fall the production 80 bits within L_max.  The
# tests therefore force an aggressive threshold: truncation that is
# exact at 20 bits is a fortiori exact at 80.
DROP = "20"


def _both(fn):
    """Run fn() with truncation on (aggressively) and off."""

    saved = {k: os.environ.get(k)
             for k in ("PMM_NO_TRUNCATE", "PMM_LEVEL_DROP")}
    try:
        os.environ.pop("PMM_NO_TRUNCATE", None)
        os.environ["PMM_LEVEL_DROP"] = DROP
        on = fn()
        os.environ["PMM_NO_TRUNCATE"] = "1"
        off = fn()
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
    return on, off


def test_profiles_identical_to_full_sweep():
    on, off = _both(lambda: depth_averaged_codelength_profiles(
        PROFILES, d=D, l_max=L_MAX, jobs=1))
    for k in PROFILES:
        a, b = on[k].log2_q_avg, off[k].log2_q_avg
        assert abs(a - b) <= 1e-9 * max(1.0, abs(b)), (k, a, b)
        assert on[k].posterior_mode == off[k].posterior_mode, k


def test_truncation_actually_fires():
    """A test that passes because nothing was skipped proves nothing."""

    res, _ = _both(lambda: depth_averaged_codelength_profiles(
        PROFILES, d=D, l_max=L_MAX, jobs=1))
    skipped = {k: sum(1 for v in r.log2_q_by_depth if v == -math.inf)
               for k, r in res.items()}
    # the dense profile must be truncated
    assert skipped["dense"] >= L_MAX // 3, skipped
    # and the sparse ones, whose curves rise to L_max, must not be
    assert skipped["light"] == 0 and skipped["single"] == 0, skipped


def test_families_identical_and_normalized():
    base = FAMILY_BASE
    cs = tuple(sorted(set(base) | {0}))
    fam = {"f": (base, cs)}

    on, off = _both(lambda: depth_averaged_codelength_families(
        fam, d=D, l_max=L_MAX, jobs=1))

    b_on, a_on = on["f"]
    b_off, a_off = off["f"]
    assert abs(b_on.log2_q_avg - b_off.log2_q_avg) <= 1e-9 * abs(
        b_off.log2_q_avg)
    for c in cs:
        assert abs(a_on[c].log2_q_avg - a_off[c].log2_q_avg) <= 1e-9 * abs(
            a_off[c].log2_q_avg), c

    # The predictive row must still sum to one: every symbol of the
    # alphabet is either one of the s observed ones (count c, with
    # multiplicity mu_c) or one of the d - s unseen ones.  The identity
    # holds level by level, so it survives truncation exactly PROVIDED
    # the whole family stops at the same level.  It does not hold to
    # machine precision even without truncation --- the saddle
    # refinement has its own floor --- so the test is that truncation
    # does not make the residual worse.
    def _residual(res, b):
        mult = {c: base.count(c) for c in set(base)}
        terms = [math.log2(mult[c]) + res[c].log2_q_avg for c in mult]
        terms.append(math.log2(D - len(base)) + res[0].log2_q_avg)
        m = max(terms)
        total = m + math.log2(sum(2.0 ** (t - m) for t in terms))
        return abs(total - b.log2_q_avg)

    r_on, r_off = _residual(a_on, b_on), _residual(a_off, b_off)
    assert r_off < 1e-4 * abs(b_off.log2_q_avg), r_off      # sanity
    assert r_on <= r_off + 1e-12 * abs(b_off.log2_q_avg), (r_on, r_off)


def test_window_rule():
    """The stopping rule itself: falling for `patience` levels AND more
    than `drop` bits below the running maximum."""

    w = _LevelWindow(["a"], drop=10.0, patience=3)
    for L, v in enumerate([-100.0, -90.0, -95.0, -99.0], start=1):
        w.observe("a", v, L)
    assert "a" in w                       # only two falling steps so far
    w.observe("a", -105.0, 5)
    assert "a" not in w                   # three falling, 15 bits down
    assert w.stopped_at["a"] == 5

    # a curve that falls but never far enough is never dropped
    w2 = _LevelWindow(["b"], drop=10.0, patience=2)
    for L, v in enumerate([-100.0, -101.0, -102.0, -103.0], start=1):
        w2.observe("b", v, L)
    assert "b" in w2


def test_disabled_by_environment():
    os.environ["PMM_NO_TRUNCATE"] = "1"
    try:
        w = _LevelWindow(["a"], drop=1.0, patience=1)
        for L, v in enumerate([-1.0, -100.0, -200.0, -300.0], start=1):
            w.observe("a", v, L)
        assert "a" in w and w.any_active()
    finally:
        os.environ.pop("PMM_NO_TRUNCATE", None)
