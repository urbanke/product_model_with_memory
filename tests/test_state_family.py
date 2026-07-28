import math
import tempfile
from collections import Counter

from product_model_with_memory.codelength import (
    depth_averaged_codelength,
)
from product_model_with_memory.pairs import reduce_vocabulary
from product_model_with_memory.state_family import (
    member_state_profiles,
    state_family_codelengths,
)


def _tiny_stream():
    # periodic-with-noise stream over a small vocabulary
    base = ["a", "b", "c", "a", "b", "d"] * 400
    return base + ["a"]


def test_m0_member_is_memoryless():
    # The M=0 member has a single state whose successor profile is the
    # unigram profile of x_2..x_n; its codelength must equal the plain
    # depth-averaged unigram codelength of that stream.
    reduced, vocab = reduce_vocabulary(_tiny_stream(), 10)
    V = len(vocab)
    with tempfile.TemporaryDirectory() as tmp:
        out = state_family_codelengths(
            reduced, vocabulary_size=V, m_grid=[0], l_max=6,
            cache_dir=tmp + "/a",
        )
        direct = depth_averaged_codelength(
            Counter(reduced[1:]), d=V, l_max=6
        )
    n = len(reduced) - 1
    assert abs(out["member_bits_per_token"][0] - direct.bits_per_token) < 1e-6
    assert out["member_states_observed"][0] == 1
    assert abs(sum(out["posterior_over_m"].values()) - 1.0) < 1e-12


def test_family_tracks_best_member():
    reduced, vocab = reduce_vocabulary(_tiny_stream(), 10)
    V = len(vocab)
    grid = [0, 2, 4]
    with tempfile.TemporaryDirectory() as tmp:
        out = state_family_codelengths(
            reduced, vocabulary_size=V, m_grid=grid, l_max=6, cache_dir=tmp,
        )
    best = min(out["member_bits_per_token"].values())
    fam = out["family_bits_per_token"]
    n = out["n_coded"]
    assert best <= fam + 1e-12
    assert fam <= best + math.log2(len(grid)) / n + 1e-12


def test_memory_helps_on_structured_stream():
    # On the periodic stream, the finer members must beat memoryless and the
    # posterior must not sit on M=1.
    reduced, vocab = reduce_vocabulary(_tiny_stream(), 10)
    V = len(vocab)
    with tempfile.TemporaryDirectory() as tmp:
        out = state_family_codelengths(
            reduced, vocabulary_size=V, m_grid=[0, 4], l_max=6, cache_dir=tmp,
        )
    assert (
        out["member_bits_per_token"][4]
        < out["member_bits_per_token"][0] - 0.5
    )
    assert out["posterior_over_m"][4] > 0.999


def test_profiles_partition_the_pairs():
    reduced, vocab = reduce_vocabulary(_tiny_stream(), 10)
    first = Counter(reduced[:-1])
    state_vocab = [w for w, _ in first.most_common()]
    for m in (0, 1, 4):
        profs = member_state_profiles(reduced, state_vocab, m)
        assert sum(sum(p) for p in profs.values()) == len(reduced) - 1
