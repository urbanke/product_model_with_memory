"""The vectorised order-two state counter must match the reference.

`product_family.member_profile_multiset` walks the stream one position
at a time in Python and is the definition of what member (M1, M2)
means.  `member_profile_multiset_ids` computes the same thing by
sorting packed keys, because the Python path cannot process the 2.7e8
tokens of enwik9.  These check the two agree exactly, over a grid of
(M1, M2) that includes every degenerate corner: M2 = 0, which must
reproduce the first-order family; M1 = 0, which keeps only the second
lag; and (0, 0), which is memoryless.

They also check the promotion order.  As at order one, which symbols
leave the backoff state is part of the code, so it has to be fixed by
the vocabulary rather than by counts in the file being compressed --- at
BOTH lags.
"""

import numpy as np

from product_model_with_memory.product_family import (
    member_profile_multiset,
    member_profile_multiset_ids,
)
from product_model_with_memory.streams import reduce_ids, state_order_by_id


GRID = [(0, 0), (1, 0), (2, 0), (4, 0), (0, 1), (0, 4),
        (1, 1), (2, 1), (2, 4), (4, 2), (4, 4), (6, 6)]


def _stream(seed=3, n=3000, d=6):
    rng = np.random.default_rng(seed)
    w = np.arange(d, 0, -1, dtype=np.float64)
    return rng.choice(d, size=n, p=w / w.sum()).astype(np.int64)


def _frequency_rank(ids, d):
    first = np.bincount(ids[1:-1], minlength=d)
    order = np.argsort(-first, kind="stable")
    rank = np.empty(d, dtype=np.int64)
    rank[order] = np.arange(d)
    return order, rank


def test_matches_the_reference_implementation():
    d = 6
    ids = _stream(d=d)
    order, rank = _frequency_rank(ids, d)
    tokens = [str(i) for i in ids.tolist()]
    vocab = [str(w) for w in order]
    for m1, m2 in GRID:
        a = member_profile_multiset(tokens, vocab, m1, m2)
        b = member_profile_multiset_ids(ids, rank, m1, m2)
        assert a == b, (m1, m2)


def test_m2_zero_is_the_first_order_family():
    """The (M, 0) slice must be the first-order family, computed by
    completely different code, on the same complete transition stream."""

    from collections import Counter

    from product_model_with_memory.state_family import (
        member_state_profiles_ids,
    )

    d = 6
    ids = _stream(d=d)
    order, rank = _frequency_rank(ids, d)
    for m1 in (1, 2, 4, 6):
        two = member_profile_multiset_ids(ids, rank, m1, 0)
        one = Counter(member_state_profiles_ids(ids, order, m1).values())
        assert two == one, m1


def test_memoryless_corner():
    d = 6
    ids = _stream(d=d)
    _, rank = _frequency_rank(ids, d)
    ms = member_profile_multiset_ids(ids, rank, 0, 0)
    assert sum(ms.values()) == 1, "(0, 0) must have exactly one state"
    profile = next(iter(ms))
    assert sum(profile) == len(ids) - 1


def test_state_counts_grow_with_resolution():
    d = 6
    ids = _stream(d=d)
    _, rank = _frequency_rank(ids, d)
    counts = [sum(member_profile_multiset_ids(ids, rank, m1, m2).values())
              for m1, m2 in [(0, 0), (2, 0), (4, 0), (4, 2), (4, 4), (6, 6)]]
    assert counts == sorted(counts), counts


def test_id_promotion_order_is_independent_of_the_data():
    """Same vocabulary, different counts: the map must not move.  This is
    the property the admissibility argument rests on, and it has to hold
    at both lags."""

    a = np.repeat(np.arange(6), [5, 20, 60, 200, 400, 900]).astype(np.int64)
    b = np.repeat(np.arange(6), [900, 400, 200, 60, 20, 5]).astype(np.int64)
    orders = []
    for ids in (a, b):
        _reduced, _V, _capped, keep = reduce_ids(ids, 6, return_keep=True)
        order = state_order_by_id(keep)
        orders.append([int(keep[j]) for j in order])
    assert orders[0] == orders[1] == [0, 1, 2, 3, 4, 5]
