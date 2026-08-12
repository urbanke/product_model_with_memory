import math
import tempfile
from collections import Counter

import numpy as np

from product_model_with_memory.codelength import (
    depth_averaged_codelength_profiles,
    profile_of,
)
from product_model_with_memory.context_tree import (
    build_context_nodes,
    context_tree_codelengths,
)
from product_model_with_memory.pairs import reduce_vocabulary
from product_model_with_memory.state_family import state_family_codelengths


def _stream():
    return (["a", "b", "a", "c"] * 500) + ["a"]


def test_nodes_partition_coded_tokens():
    reduced, _ = reduce_vocabulary(_stream(), 10)
    D = 2
    nodes = build_context_nodes(reduced, D)
    for d in range(D + 1):
        total = sum(
            sum(c.values()) for ctx, c in nodes.items() if len(ctx) == d
        )
        assert total == len(reduced) - D


def test_beta_matches_bruteforce_enumeration():
    # Depth-1 tree over a small alphabet: the mixture is exactly
    # 1/2 q(root) + 1/2 prod_a q(context a), enumerable by hand.
    reduced, vocab = reduce_vocabulary(_stream(), 10)
    V = len(vocab)
    D = 1
    nodes = build_context_nodes(reduced, D)
    profiles = {ctx: profile_of(c) for ctx, c in nodes.items()}
    with tempfile.TemporaryDirectory() as tmp:
        res = depth_averaged_codelength_profiles(
            {p: p for p in set(profiles.values())},
            d=V, l_max=6, cache_dir=tmp + "/q",
        )
        out = context_tree_codelengths(
            reduced, vocabulary_size=V, max_depth=D, l_max=6,
            cache_dir=tmp + "/t",
        )
    lq = {ctx: res[p].log2_q_avg for ctx, p in profiles.items()}
    root = lq[()]
    split = sum(v for ctx, v in lq.items() if len(ctx) == 1)
    m = max(root, split)
    expected = -(
        m + math.log2(2.0 ** (root - 1 - m) + 2.0 ** (split - 1 - m))
    ) / (len(reduced) - D)
    assert abs(out["family_bits_per_token"] - expected) < 1e-9


def test_map_tree_finds_order_two_structure():
    # On the order-two source the MAP pruning must go deep and the family
    # must approach the (near-zero) conditional entropy of depth 2.
    reduced, vocab = reduce_vocabulary(_stream(), 10)
    V = len(vocab)
    with tempfile.TemporaryDirectory() as tmp:
        out = context_tree_codelengths(
            reduced, vocabulary_size=V, max_depth=2, l_max=6, cache_dir=tmp,
        )
    fixed = out["fixed_depth_bits_per_token"]
    assert fixed[2] < fixed[1] - 0.3  # depth 2 is decisively better
    assert out["family_bits_per_token"] < fixed[1]
    assert out["family_bits_per_token"] <= fixed[2] + 0.01
    # MAP pruning uses depth-2 leaves where it matters
    assert max(out["map_leaves_by_depth"]) == 2


def test_family_never_worse_than_best_fixed_depth_plus_prior():
    reduced, vocab = reduce_vocabulary(_stream(), 10)
    V = len(vocab)
    with tempfile.TemporaryDirectory() as tmp:
        out = context_tree_codelengths(
            reduced, vocabulary_size=V, max_depth=2, l_max=6, cache_dir=tmp,
        )
    best_fixed = min(out["fixed_depth_bits_per_token"].values())
    # every complete-depth tree is one pruning; its prior cost is at most
    # (#nodes) bits, tiny per token here
    assert out["family_bits_per_token"] <= best_fixed + 0.05


def test_complete_depth_one_member_matches_full_state_first_order():
    reduced, vocab = reduce_vocabulary(_stream(), 10)
    V = len(vocab)
    with tempfile.TemporaryDirectory() as tmp:
        tree = context_tree_codelengths(
            reduced, vocabulary_size=V, max_depth=1, l_max=6,
            cache_dir=tmp + "/tree",
        )
        first_order = state_family_codelengths(
            reduced, vocabulary_size=V, m_grid=[V], l_max=6,
            cache_dir=tmp + "/first-order",
        )
    assert tree["n_coded"] + 1 == len(reduced)
    assert first_order["n_coded"] + 1 == len(reduced)
    assert abs(
        tree["fixed_depth_bits_per_token"][1]
        - first_order["member_bits_per_token"][V]
    ) < 1e-12


def test_pooled_common_m_depth_one_matches_same_markov_state_map():
    ids = np.asarray(([0, 1, 4, 0, 2, 1, 3] * 300) + [0], dtype=np.int64)
    V, M = 5, 2
    contexts = np.where(ids < M, ids, M)
    with tempfile.TemporaryDirectory() as tmp:
        tree = context_tree_codelengths(
            ids, vocabulary_size=V, max_depth=1, l_max=6,
            cache_dir=tmp + "/tree", context_ids=contexts,
            context_alphabet_size=M + 1,
        )
        first_order = state_family_codelengths(
            ids, vocabulary_size=V, m_grid=[M], l_max=6,
            cache_dir=tmp + "/first-order",
            state_order=np.arange(V, dtype=np.int64),
        )
    assert abs(
        tree["fixed_depth_bits_per_token"][1]
        - first_order["member_bits_per_token"][M]
    ) < 1e-12


def test_common_m_equals_v_is_exact_original_ctw_path():
    ids = np.asarray(([0, 1, 4, 0, 2, 1, 3] * 200) + [0], dtype=np.int64)
    V = 5
    with tempfile.TemporaryDirectory() as tmp:
        original = context_tree_codelengths(
            ids, vocabulary_size=V, max_depth=2, l_max=6,
            cache_dir=tmp + "/original",
        )
        separated = context_tree_codelengths(
            ids, vocabulary_size=V, max_depth=2, l_max=6,
            cache_dir=tmp + "/separate", context_ids=ids,
            context_alphabet_size=V,
    )
    assert original["n_coded"] == separated["n_coded"]
    assert abs(
        original["family_bits_per_token"]
        - separated["family_bits_per_token"]
    ) < 1e-12
    assert original["fixed_depth_bits_per_token"] == separated[
        "fixed_depth_bits_per_token"
    ]


def test_kt_leaf_matches_sequential_product():
    # closed-form KT profile probability == telescoping product of the
    # KT sequential predictor (m_i + 1/2)/(t + V/2)
    from product_model_with_memory.context_tree import kt_log2_q
    seq = ["a", "b", "a", "a", "c", "b"]
    V = 4
    counts = {}
    log2p = 0.0
    for t, x in enumerate(seq):
        p = (counts.get(x, 0) + 0.5) / (t + V / 2.0)
        log2p += math.log2(p)
        counts[x] = counts.get(x, 0) + 1
    profile = tuple(sorted(counts.values()))
    assert abs(kt_log2_q(profile, V) - log2p) < 1e-12


def test_kt_tree_runs_and_finds_order_two():
    reduced, vocab = reduce_vocabulary(_stream(), 10)
    V = len(vocab)
    with tempfile.TemporaryDirectory() as tmp:
        out = context_tree_codelengths(
            reduced, vocabulary_size=V, max_depth=2, cache_dir=tmp,
            leaf_model="kt",
        )
    fixed = out["fixed_depth_bits_per_token"]
    assert fixed[2] < fixed[1] - 0.3
    assert out["family_bits_per_token"] <= min(fixed.values()) + 0.05
    assert max(out["map_leaves_by_depth"]) == 2
