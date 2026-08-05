"""Tests for the layered-expert pooled evaluator.

The decisive check is the telescoping identity: with the memoryless
expert refreshed at EVERY step, the per-step predictives of the layered
mixture telescope, so the pooled evaluator's total must equal the
layered codelength of the final unigram profile computed by the core
machinery --- an exact end-to-end agreement between the new evaluator
and everything the paper is built on.
"""

import math
import random
import tempfile

import numpy as np

from product_model_with_memory.codelength import (
    default_l_max,
    depth_averaged_codelength_profiles,
)
from product_model_with_memory.pooled_lags import (
    _LayeredPredictiveBuilder,
    _layered_log_sparse_tables,
    _SparseLagTables,
    SparseCountRows,
    pooled_lag_codelengths,
)


def _ids(n, V, seed=13):
    rng = random.Random(seed)
    return np.array([rng.randrange(V) for _ in range(n)])


def test_predictive_rows_normalize():
    V = 7
    with tempfile.TemporaryDirectory() as tmp:
        b = _LayeredPredictiveBuilder(V, default_l_max(V), tmp, 1, None)
        for row in [
            np.zeros(V),
            np.array([3.0, 0, 1, 0, 1, 0, 0]),
            np.array([10.0, 5, 5, 2, 1, 1, 1]),
        ]:
            wanted = set()
            nz = np.flatnonzero(row)
            base = tuple(sorted(int(row[i]) for i in nz))
            log_row = b.row_log_table(row)
            sparse_ids, sparse_logp, sparse_unseen = b.row_log_sparse(row)
            direct_ids, direct_logp, direct_unseen = (
                b.row_log_sparse_entries(nz, row[nz])
            )
            assert np.array_equal(direct_ids, sparse_ids)
            assert np.allclose(direct_logp, sparse_logp)
            assert direct_unseen == sparse_unseen
            total = np.exp2(log_row).sum()
            assert abs(total - 1.0) < 1e-9
            # empty row must be exactly uniform
            if not base:
                assert np.allclose(log_row, -math.log2(V))


def test_sparse_count_rows_build_same_layered_tables_as_dense_counts():
    v = 7
    counts = np.array([
        [0, 3, 0, 1, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 0],
        [2, 0, 1, 0, 4, 0, 0],
        [0, 0, 0, 2, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 5, 0],
        [0, 2, 0, 0, 3, 0, 0],
    ], dtype=np.int64)
    row, column = np.nonzero(counts)
    keys = row * v + column
    sparse_counts = SparseCountRows.from_sorted_keys(
        v, keys, counts[row, column]
    )
    unigram = np.array([9, 7, 5, 3, 2, 1, 1], dtype=np.float64)
    with tempfile.TemporaryDirectory() as tmp:
        builder = _LayeredPredictiveBuilder(
            v, default_l_max(v), tmp, 1, None
        )
        sparse_q0, (sparse_table,) = _layered_log_sparse_tables(
            builder, unigram, [sparse_counts]
        )
        dense_q0 = builder.row_log_table(unigram)
        dense_table = _SparseLagTables(builder, [counts]).lags[0]

    assert np.allclose(sparse_q0, dense_q0)
    for key in ("ptr", "idx", "val", "unseen", "rho"):
        assert np.allclose(sparse_table[key], dense_table[key])


def test_telescoping_identity_memoryless():
    # checkpoint at every step => the layered predictives telescope:
    # sum of per-step bits == log2 q_avg(profile of ids[:1]) -
    #                          log2 q_avg(profile of ids[:n])
    V, n = 6, 80
    ids = _ids(n, V)
    start = 1  # lags=(1,)
    with tempfile.TemporaryDirectory() as tmp:
        res = pooled_lag_codelengths(
            ids,
            vocabulary_size=V,
            lags=(1,),
            checkpoints=n - start,
            expert_model="layered",
            cache_dir=tmp,
            mix_grid=(["mem"], np.array([[1.0, 0.0]])),
            prod_grid=([], np.zeros((0, 1))),
        )
        evaluator_total = res.member_bits[0] * res.n_coded

        prof_full = tuple(sorted(int(c) for c in np.bincount(ids) if c > 0))
        prof_first = (1,)
        q = depth_averaged_codelength_profiles(
            {0: prof_full, 1: prof_first},
            d=V,
            l_max=default_l_max(V),
            cache_dir=tmp,
        )
        expected = -(q[0].log2_q_avg - q[1].log2_q_avg)
    # the identity is exact in exact arithmetic; computed q values carry
    # the moment-table integration error (~5e-5 per step), which the
    # per-row renormalization absorbs --- so assert per-token accuracy
    assert abs(evaluator_total - expected) / res.n_coded < 1e-4


def test_onehot_mix_equals_onehot_prod_layered():
    V, n = 6, 120
    ids = _ids(n, V, seed=3)
    with tempfile.TemporaryDirectory() as tmp:
        res = pooled_lag_codelengths(
            ids,
            vocabulary_size=V,
            lags=(1, 2),
            checkpoints=4,
            expert_model="layered",
            cache_dir=tmp,
        )
    by = {n_: b for n_, b in zip(res.member_names, res.member_bits)}
    for d in (1, 2):
        assert abs(by[f"onehot:lag{d}"] - by[f"prod-onehot:lag{d}"]) < 1e-9


def test_layered_beats_counts_experts_on_markov_source():
    # on a sparse first-order source the layered expert should code
    # shorter than the alpha-smoothed count expert, at equal checkpoints
    rng = np.random.default_rng(9)
    V, n = 16, 6_000
    T = rng.dirichlet(np.full(V, 0.15), size=V)
    ids = np.zeros(n, dtype=np.int64)
    for t in range(1, n):
        ids[t] = rng.choice(V, p=T[ids[t - 1]])
    grid = (["lag1"], np.array([[0.0, 1.0]]))
    empty_prod = (["x"], np.array([[1.0]]))
    with tempfile.TemporaryDirectory() as tmp:
        layered = pooled_lag_codelengths(
            ids, vocabulary_size=V, lags=(1,), checkpoints=8,
            expert_model="layered", cache_dir=tmp,
            mix_grid=grid, prod_grid=empty_prod,
        ).member_bits[0]
    counts = pooled_lag_codelengths(
        ids, vocabulary_size=V, lags=(1,), checkpoints=8,
        expert_model="counts",
        mix_grid=grid, prod_grid=empty_prod,
    ).member_bits[0]
    assert layered < counts


def test_sparse_product_eval_equals_dense():
    # T7: the sparse normalization must reproduce the dense evaluation
    # exactly (both rules, every member), including saturated rows and
    # unseen-target steps
    rng = np.random.default_rng(4)
    V, n = 10, 3000
    T = rng.dirichlet(np.full(V, 0.3), size=V)
    ids = np.zeros(n, dtype=np.int64)
    for t in range(1, n):
        ids[t] = rng.choice(V, p=T[ids[t - 1]])
    kw = dict(vocabulary_size=V, lags=(1, 2), checkpoints=4,
              expert_model="layered", cache_dir=None)
    dense = pooled_lag_codelengths(ids, eval_mode="dense", **kw)
    sparse = pooled_lag_codelengths(ids, eval_mode="sparse", **kw)
    diff = np.abs(np.array(dense.member_bits) - np.array(sparse.member_bits))
    assert diff.max() < 1e-9
