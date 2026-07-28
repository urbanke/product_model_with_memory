import math
import tempfile

from product_model_with_memory.lag_family import (
    lag_family_codelengths,
    member_profile_multiset,
)
from product_model_with_memory.pairs import reduce_vocabulary
from product_model_with_memory.state_family import state_family_codelengths


def _order2_stream():
    # x_t is determined by x_{t-2}; given only x_{t-1} = 'a' the successor
    # is ambiguous, so lag 2 is decisive and lag 1 is weak.
    return ["a", "b", "a", "c"] * 600 + ["a"]


def test_profiles_partition_the_coded_tokens():
    reduced, _ = reduce_vocabulary(_order2_stream(), 10)
    for deltas in [(0, 1), (1, 2), (0, 1, 2, 3)]:
        dmax = max(deltas)
        for d in deltas:
            multiset = member_profile_multiset(reduced, d, dmax)
            total = sum(sum(p) * mult for p, mult in multiset.items())
            assert total == len(reduced) - dmax


def test_lag1_matches_first_order_family():
    # deltas=[1] codes x_2..x_n with state = previous token, which is the
    # M=V member of the context-partition family.
    reduced, vocab = reduce_vocabulary(_order2_stream(), 10)
    V = len(vocab)
    with tempfile.TemporaryDirectory() as tmp:
        lag = lag_family_codelengths(
            reduced, vocabulary_size=V, deltas=[1], l_max=6,
            cache_dir=tmp + "/a",
        )
        first = state_family_codelengths(
            reduced, vocabulary_size=V, m_grid=[V], l_max=6,
            cache_dir=tmp + "/b",
        )
    assert abs(
        lag["member_bits_per_token"][1] - first["member_bits_per_token"][V]
    ) < 1e-6


def test_lag_two_wins_on_order_two_source():
    reduced, vocab = reduce_vocabulary(_order2_stream(), 10)
    V = len(vocab)
    with tempfile.TemporaryDirectory() as tmp:
        out = lag_family_codelengths(
            reduced, vocabulary_size=V, deltas=[0, 1, 2, 3], l_max=6,
            cache_dir=tmp,
        )
    bits = out["member_bits_per_token"]
    assert bits[2] < bits[1] - 0.3
    assert bits[2] < bits[0] - 0.3
    assert out["posterior"][2] > 0.999
    best = min(bits.values())
    fam = out["family_bits_per_token"]
    assert best <= fam <= best + math.log2(4) / out["n_coded"] + 1e-12
