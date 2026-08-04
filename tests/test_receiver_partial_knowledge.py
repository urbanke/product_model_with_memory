"""Can a receiver holding only PARTIAL counts reproduce our codelength?

We report -log2 of a product over states of q_avg evaluated at each
state's FINAL profile.  A receiver never sees a final profile: at time
t it holds only what it has decoded.  So the question is not whether
our expression is a probability distribution -- it is whether a
receiver with partial knowledge can realise it.

These tests answer it by running the receiver.  A sequence is walked
symbol by symbol; at each step the predictor is formed from the running
profile alone, its cost is accumulated, and the total is compared with
the closed form evaluated at the final profile.  They must agree
exactly.

Why they do, and why the averaging does not break it:

  * q_avg is a FUNCTION OF A PROFILE, not of the file.  The receiver
    evaluates the same function at partial profiles; we evaluate it once
    at the final one.

  * the per-step costs telescope.  The numerator at step i is
    q_avg(lambda_i + x_i) = q_avg(lambda_{i+1}), which is the next
    step's denominator, so the product collapses to q_avg(final).  This
    is an algebraic identity about any function, and averaging does not
    touch it.

  * the ratio of averages is NOT the uniform average of the per-depth
    predictors.  It is their POSTERIOR-weighted average, with the
    posterior formed from the running profile.  That is what makes it
    both computable at time t and a valid distribution; the uniform
    average of ratios is a different object and does not sum to one.
    `test_uniform_average_of_ratios_is_not_a_code` pins that down,
    because it is the natural wrong guess.
"""

import math
import random
from collections import Counter

import numpy as np

from product_model_with_memory.codelength import (
    default_l_max,
    depth_averaged_codelength_profiles,
    profile_of,
)


# ---------------------------------------------------------------- exact
# Dirichlet: an exchangeable, consistent q in closed form, so the tests
# below carry no numerical-integration error at all.
ALPHAS = (1.0, 0.3, 0.05)


def _q_dirichlet(counts, alpha, d):
    n = sum(counts.values())
    lg = sum(math.lgamma(alpha + c) - math.lgamma(alpha)
             for c in counts.values())
    return math.exp(lg - (math.lgamma(d * alpha + n) - math.lgamma(d * alpha)))


def _q_avg(counts, d, alphas=ALPHAS):
    return sum(_q_dirichlet(counts, a, d) for a in alphas) / len(alphas)


def test_receiver_with_partial_counts_matches_the_closed_form():
    """The receiver's accumulated bits equal -log2 q_avg(final profile)."""

    d = 5
    random.seed(1)
    seq = [random.choice([0, 0, 0, 1, 1, 2, 3, 4]) for _ in range(30)]
    lam = Counter()
    bits = 0.0
    for y in seq:
        p = _q_avg(lam + Counter([y]), d) / _q_avg(lam, d)
        bits += -math.log2(p)
        lam[y] += 1
    assert abs(bits + math.log2(_q_avg(lam, d))) < 1e-9, bits


def test_the_receivers_predictor_sums_to_one_at_every_step():
    """Validity: with only partial knowledge, the receiver still emits a
    probability distribution over the next symbol."""

    d = 5
    random.seed(2)
    lam = Counter()
    for _ in range(30):
        base = _q_avg(lam, d)
        total = sum(_q_avg(lam + Counter([y]), d) / base for y in range(d))
        assert abs(total - 1.0) < 1e-12, (dict(lam), total)
        lam[random.randrange(d)] += 1


def test_uniform_average_of_ratios_is_not_a_code():
    """Uniformly averaging the per-depth PREDICTORS also sums to one, so
    it too is a code -- just not ours.  This is the heart of the matter.

    Two receivers are available.  (i) weights the depths by the
    POSTERIOR given the running profile; its per-step probability is
    q_avg(lam+y)/q_avg(lam), and its accumulated cost telescopes to
    -log2 q_avg(final profile), which is the number we report.  (ii)
    weights them UNIFORMLY at every step; it is normalised by the same
    interchange of two finite sums, but its per-step probability is not
    a ratio of one function at consecutive profiles, so nothing
    telescopes and its total differs.

    So the closed form is NOT "the codelength of any receiver that
    averages its estimators".  It is the codelength of (i), and (i) is
    the one carrying the guarantee: a mixture is at least any one of its
    terms, so (i) costs at most log2 of the number of components more
    than the best component, on every file.  (ii) has no such bound."""

    d = 5
    lam = Counter({0: 7, 1: 3, 2: 1})
    base = _q_avg(lam, d)
    ours = uniform = 0.0
    for y in range(d):
        l2 = lam + Counter([y])
        ours += _q_avg(l2, d) / base
        uniform += sum(_q_dirichlet(l2, a, d) / _q_dirichlet(lam, a, d)
                       for a in ALPHAS) / len(ALPHAS)
    assert abs(ours - 1.0) < 1e-12, ours
    assert abs(uniform - 1.0) < 1e-12, uniform        # ALSO a distribution

    # ... and yet the two schemes charge different totals for one file
    random.seed(5)
    seq = [random.choice([0, 0, 0, 1, 1, 2, 3, 4]) for _ in range(30)]
    lam, bits_i, bits_ii = Counter(), 0.0, 0.0
    for y in seq:
        l2 = lam + Counter([y])
        bits_i += -math.log2(_q_avg(l2, d) / _q_avg(lam, d))
        bits_ii += -math.log2(
            sum(_q_dirichlet(l2, a, d) / _q_dirichlet(lam, a, d)
                for a in ALPHAS) / len(ALPHAS))
        lam = l2
    assert abs(bits_i + math.log2(_q_avg(lam, d))) < 1e-9    # (i) telescopes
    assert abs(bits_ii - bits_i) > 0.05, (bits_i, bits_ii)   # (ii) does not


def test_ratio_of_averages_is_the_posterior_weighted_mixture():
    """And the receiver's predictor is exactly the per-depth predictors
    weighted by the posterior over depth given ITS OWN running profile,
    which is why partial knowledge suffices."""

    d = 5
    lam = Counter({0: 7, 1: 3, 2: 1})
    base = _q_avg(lam, d)
    post = [_q_dirichlet(lam, a, d) / (base * len(ALPHAS)) for a in ALPHAS]
    assert abs(sum(post) - 1.0) < 1e-12
    for y in range(d):
        l2 = lam + Counter([y])
        ratio = _q_avg(l2, d) / base
        weighted = sum(
            p * (_q_dirichlet(l2, a, d) / _q_dirichlet(lam, a, d))
            for p, a in zip(post, ALPHAS))
        assert abs(ratio - weighted) < 1e-12, (y, ratio, weighted)


# ------------------------------------------------- the real estimator
def test_receiver_matches_the_closed_form_for_the_layered_mixture():
    """The same walk with the depth-averaged layered mixture actually
    used in the paper, rather than the Dirichlet stand-in."""

    d = 12
    l_max = default_l_max(d)
    random.seed(3)
    seq = [random.choice([0, 0, 0, 1, 1, 2, 3, 4, 5, 7, 11])
           for _ in range(24)]

    need, lam = {}, Counter()
    for i, y in enumerate(seq):
        need[("at", i)] = profile_of(lam)
        lam[y] += 1
        need[("next", i)] = profile_of(lam)
    res = depth_averaged_codelength_profiles(
        {k: v for k, v in need.items() if sum(v) > 0},
        d=d, l_max=l_max, jobs=1)

    def lq(key):
        pr = need[key]
        return 0.0 if sum(pr) == 0 else res[key].log2_q_avg

    bits = -sum(lq(("next", i)) - lq(("at", i)) for i in range(len(seq)))
    assert abs(bits + lq(("next", len(seq) - 1))) < 1e-9, bits


def test_two_states_walk_matches_the_product_over_states():
    """With memory: the receiver keeps one running profile per state and
    the totals still agree with the product over final profiles."""

    d = 6
    random.seed(4)
    seq = [random.randrange(d) for _ in range(40)]
    lam = {}
    bits = 0.0
    for t in range(1, len(seq)):
        s = seq[t - 1] % 2                      # a two-state map of the past
        cur = lam.setdefault(s, Counter())
        p = _q_avg(cur + Counter([seq[t]]), d) / _q_avg(cur, d)
        bits += -math.log2(p)
        cur[seq[t]] += 1
    closed = -sum(math.log2(_q_avg(c, d)) for c in lam.values())
    assert abs(bits - closed) < 1e-9, (bits, closed)
