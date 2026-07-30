"""Tests for the sorted-key counting (complexity notes, T4).

The new counting must EQUAL the old hash-table counting: same number
of contexts per depth, same multiset of profiles, and parent links
whose totals telescope exactly.
"""

import numpy as np

from product_model_with_memory.codelength import profile_of
from product_model_with_memory.context_tree import build_context_nodes
from product_model_with_memory.counting import context_profile_tables


def _random_case(seed, V, n, D):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, V, size=n)
    return ids, [str(x) for x in ids]


def test_counts_equal_hash_counting():
    for seed, V, n, D in [(0, 5, 4000, 2), (1, 12, 9000, 3),
                          (2, 3, 500, 4), (3, 50, 20000, 2)]:
        ids, toks = _random_case(seed, V, n, D)
        old = build_context_nodes(toks, D)
        new = context_profile_tables(ids, V, D)
        for d in range(D + 1):
            old_d = sorted(profile_of(c) for ctx, c in old.items()
                           if len(ctx) == d)
            new_d = sorted(new.profiles[i] for i in new.profile_id[d])
            assert len(old_d) == new.n_contexts[d]
            assert old_d == new_d


def test_parent_totals_telescope():
    ids, _ = _random_case(7, 20, 30000, 3)
    new = context_profile_tables(ids, 20, 3)
    for d in range(1, 4):
        p = new.parent[d]
        assert p.min() >= 0 and p.max() < new.n_contexts[d - 1]
        child_N = np.array([sum(new.profiles[i])
                            for i in new.profile_id[d]])
        got = np.zeros(new.n_contexts[d - 1], dtype=np.int64)
        np.add.at(got, p, child_N)
        want = np.array([sum(new.profiles[i])
                         for i in new.profile_id[d - 1]])
        assert np.array_equal(got, want)


def test_multiword_keys_match_singleword_regime():
    # deep enough that keys need more than one 64-bit word
    ids, toks = _random_case(11, 1000, 8000, 6)  # 7 digits x ~10 bits
    old = build_context_nodes(toks, 6)
    new = context_profile_tables(ids, 1000, 6)
    for d in range(7):
        old_d = sorted(profile_of(c) for ctx, c in old.items()
                       if len(ctx) == d)
        new_d = sorted(new.profiles[i] for i in new.profile_id[d])
        assert old_d == new_d
