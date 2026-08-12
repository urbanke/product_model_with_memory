import math
import tempfile
from collections import Counter

from product_model_with_memory.pairs import reduce_vocabulary
from product_model_with_memory.product_family import (
    member_profile_multiset,
    product_family_codelengths,
)
from product_model_with_memory.state_family import state_family_codelengths


def _order2_stream():
    # x_t is determined by x_{t-2}: after (b, a) comes c, after (c, a)
    # comes b; given only the previous token 'a' the successor is ambiguous.
    return ["a", "b", "a", "c"] * 600 + ["a"]


def test_profiles_partition_the_coded_tokens():
    reduced, _ = reduce_vocabulary(_order2_stream(), 10)
    freq = Counter(reduced)
    order = [w for w, _ in freq.most_common()]
    for m1, m2 in [(0, 0), (2, 0), (4, 4)]:
        multiset = member_profile_multiset(reduced, order, m1, m2)
        total = sum(sum(p) * mult for p, mult in multiset.items())
        assert total == len(reduced) - 1


def test_m0_slice_matches_first_order_family():
    # (M, 0) members must equal the first-order family on the complete
    # stream, including the x_1 -> x_2 transition.
    reduced, vocab = reduce_vocabulary(_order2_stream(), 10)
    V = len(vocab)
    with tempfile.TemporaryDirectory() as tmp:
        prod = product_family_codelengths(
            reduced, vocabulary_size=V, grid=[(4, 0)], l_max=6,
            cache_dir=tmp + "/a",
        )
        first = state_family_codelengths(
            reduced, vocabulary_size=V, m_grid=[4], l_max=6,
            cache_dir=tmp + "/b",
        )
    assert abs(
        prod["member_bits_per_token"][(4, 0)]
        - first["member_bits_per_token"][4]
    ) < 1e-6


def test_order_two_memory_wins_on_order_two_source():
    reduced, vocab = reduce_vocabulary(_order2_stream(), 10)
    V = len(vocab)
    with tempfile.TemporaryDirectory() as tmp:
        out = product_family_codelengths(
            reduced, vocabulary_size=V,
            grid=[(0, 0), (4, 0), (4, 4)], l_max=6, cache_dir=tmp,
        )
    bits = out["member_bits_per_token"]
    # first-order helps little on this source; second order is decisive
    assert bits[(4, 4)] < bits[(4, 0)] - 0.3
    assert out["posterior"][(4, 4)] > 0.999
    best = min(bits.values())
    fam = out["family_bits_per_token"]
    assert best <= fam <= best + math.log2(3) / out["n_coded"] + 1e-12
