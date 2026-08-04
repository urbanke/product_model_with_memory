"""Can a receiver realise the FAMILY average with partial knowledge?

The family codelength we report is

    Q(x) = sum_M (1/|G|) * prod_{s in S(M)} q_avg(lambda_s^(M)),

evaluated at the FINAL profiles of each member's states.  The objection
this file answers: the receiver never sees a final profile, and the
average of products is not the product of averages, so how can the two
be the same number?

They are the same number, and no interchange of the sum over M with the
product over states is used anywhere.  The receiver:

  * keeps one bookkeeping per MEMBER --- that member's own partition
    and its own running profiles;
  * predicts with each member from that member's running profile;
  * weights the members by the POSTERIOR, w_M proportional to
    (1/|G|) * Q^(M)(prefix), which it accumulates as it goes;
  * emits the weighted mixture.

Its per-step probability is then R(prefix + y) / R(prefix) with
R = sum_M (1/|G|) Q^(M), because the member joints inside w_M cancel
against the ones in the member predictions.  Those ratios telescope, so
the accumulated cost is -log2 R(whole file) --- the closed form.

Note what the receiver's per-step prediction is NOT: it is not a product
over states.  It is a ratio of two sums of products.  A code has to be
computable from the past and sum to one over the next symbol; it does
not have to factorise.  `test_mixture_prediction_does_not_factorise`
makes that explicit, since it is the point of confusion.

The estimator here is a closed-form Dirichlet mixture, so nothing in
these tests depends on numerical integration.
"""

import math
import random
from collections import Counter

D = 4
ALPHAS = (1.0, 0.3)          # stands in for the depth mixture inside a state
PRIOR = None                 # set per test from the member list


def _q(counts, alpha):
    n = sum(counts.values())
    lg = sum(math.lgamma(alpha + c) - math.lgamma(alpha)
             for c in counts.values())
    return math.exp(lg - (math.lgamma(D * alpha + n) - math.lgamma(D * alpha)))


def _q_avg(counts):
    return sum(_q(counts, a) for a in ALPHAS) / len(ALPHAS)


def _state_of(prev, m):
    """member 0 has a single state; member 1 is full first order."""

    return 0 if m == 0 else prev


def _sequence(seed=7, n=25):
    random.seed(seed)
    return [random.choice([0, 0, 1, 1, 2, 3]) for _ in range(n)]


def _closed_form(seq, members):
    """log2 of each member's joint, from its FINAL profiles."""

    out = {}
    for m in members:
        lam = {}
        for t in range(1, len(seq)):
            lam.setdefault(_state_of(seq[t - 1], m), Counter())[seq[t]] += 1
        out[m] = sum(math.log2(_q_avg(c)) for c in lam.values())
    return out


def _receiver(seq, members):
    """Walk the sequence holding only the decoded prefix.  Returns the
    accumulated bits."""

    prior = 1.0 / len(members)
    prof = {m: {} for m in members}
    log_joint = {m: 0.0 for m in members}
    bits = 0.0
    for t in range(1, len(seq)):
        y = seq[t]
        pred = {}
        for m in members:
            s = _state_of(seq[t - 1], m)
            lam = prof[m].setdefault(s, Counter())
            pred[m] = _q_avg(lam + Counter([y])) / _q_avg(lam)
        z = sum(prior * 2.0 ** log_joint[m] for m in members)
        w = {m: prior * 2.0 ** log_joint[m] / z for m in members}
        bits += -math.log2(sum(w[m] * pred[m] for m in members))
        for m in members:
            prof[m][_state_of(seq[t - 1], m)][y] += 1
            log_joint[m] += math.log2(pred[m])
    return bits


def test_receiver_matches_the_family_closed_form():
    members = [0, 1]
    seq = _sequence()
    closed = _closed_form(seq, members)
    mix = math.log2(sum(2.0 ** closed[m] / len(members) for m in members))
    assert abs(_receiver(seq, members) + mix) < 1e-9


def test_members_really_differ():
    """A guard: if the two members assigned the same probability the test
    above would pass for the wrong reason."""

    closed = _closed_form(_sequence(), [0, 1])
    assert abs(closed[0] - closed[1]) > 1.0, closed


def test_mixture_prediction_sums_to_one():
    """Validity, at a prefix chosen part-way through."""

    members = [0, 1]
    seq = _sequence()
    prior = 1.0 / len(members)
    prof = {m: {} for m in members}
    log_joint = {m: 0.0 for m in members}
    for t in range(1, 12):
        y = seq[t]
        for m in members:
            prof[m].setdefault(_state_of(seq[t - 1], m), Counter())
        z = sum(prior * 2.0 ** log_joint[m] for m in members)
        w = {m: prior * 2.0 ** log_joint[m] / z for m in members}
        total = 0.0
        for cand in range(D):
            p = 0.0
            for m in members:
                lam = prof[m][_state_of(seq[t - 1], m)]
                p += w[m] * (_q_avg(lam + Counter([cand])) / _q_avg(lam))
            total += p
        assert abs(total - 1.0) < 1e-12, (t, total)
        for m in members:
            s = _state_of(seq[t - 1], m)
            lam = prof[m][s]
            log_joint[m] += math.log2(
                _q_avg(lam + Counter([y])) / _q_avg(lam))
            lam[y] += 1


def test_mixture_prediction_does_not_factorise():
    """The mixture's one-step prediction is NOT a product over states,
    and it does not need to be.  Here the two members disagree, so the
    mixture's prediction differs from either member's --- it is a ratio
    of sums of products, computable from the prefix, summing to one."""

    members = [0, 1]
    seq = _sequence()
    prof = {m: {} for m in members}
    log_joint = {m: 0.0 for m in members}
    for t in range(1, 15):
        y = seq[t]
        for m in members:
            s = _state_of(seq[t - 1], m)
            lam = prof[m].setdefault(s, Counter())
            log_joint[m] += math.log2(
                _q_avg(lam + Counter([y])) / _q_avg(lam))
            lam[y] += 1
    prior = 1.0 / len(members)
    z = sum(prior * 2.0 ** log_joint[m] for m in members)
    w = {m: prior * 2.0 ** log_joint[m] / z for m in members}
    assert 0.01 < w[0] < 0.99, w      # both members still live
    cand = 1
    per_member = []
    for m in members:
        lam = prof[m][_state_of(seq[14], m)]
        per_member.append(_q_avg(lam + Counter([cand])) / _q_avg(lam))
    mixed = sum(w[m] * p for m, p in zip(members, per_member))
    assert abs(per_member[0] - per_member[1]) > 1e-3, per_member
    assert min(per_member) < mixed < max(per_member), (per_member, mixed)
