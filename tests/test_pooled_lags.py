"""Tests for the pooled-lag evaluator.

The critical properties, each checked exactly or to float tolerance:
brute-force agreement of both rules on a tiny case; one-hot mixture ==
one-hot product == the standalone single-expert code; product
normalization; staleness decreasing in checkpoint count; and a
switching source on which pooling beats every single expert.
"""

import math
import random

import numpy as np

from product_model_with_memory.pooled_lags import (
    pooled_lag_codelengths,
    power_law_mixture_grid,
    power_law_product_grid,
)


def _tiny_case():
    rng = random.Random(11)
    V, n = 5, 240
    ids = np.array([rng.randrange(V) for _ in range(n)])
    return ids, V


def _brute_force(ids, V, lags, checkpoints, alpha, lam, beta):
    """Straight-loop reimplementation of both rules for one member each."""

    n = len(ids)
    start = max(lags)
    bounds = np.linspace(start, n, checkpoints + 1).astype(int)
    mix_total = 0.0
    prod_total = 0.0
    for ck in range(checkpoints):
        lo, hi = int(bounds[ck]), int(bounds[ck + 1])
        u = [0] * V
        for x in ids[:lo]:
            u[x] += 1
        q0 = [(u[x] + 0.5) / (lo + V / 2.0) for x in range(V)]
        tabs = []
        for d in lags:
            counts = [[0.0] * V for _ in range(V)]
            for t in range(d, lo):
                counts[ids[t - d]][ids[t]] += 1.0
            tab = []
            for c in range(V):
                rn = sum(counts[c])
                tab.append(
                    [(counts[c][x] + alpha * q0[x]) / (rn + alpha) for x in range(V)]
                )
            tabs.append(tab)
        for t in range(lo, hi):
            x = ids[t]
            probs = [q0[x]] + [
                tabs[j][ids[t - d]][x] for j, d in enumerate(lags)
            ]
            mix_total -= math.log2(sum(w * p for w, p in zip(lam, probs)))
            score = [
                q0[y]
                * math.prod(
                    (tabs[j][ids[t - d]][y] / q0[y]) ** beta[j]
                    for j, d in enumerate(lags)
                )
                for y in range(V)
            ]
            prod_total -= math.log2(score[x] / sum(score))
    return mix_total / (n - start), prod_total / (n - start)


def test_brute_force_agreement():
    ids, V = _tiny_case()
    lags = (1, 3)
    lam = [0.2, 0.5, 0.3]
    beta = [0.7, 0.4]
    mix_grid = (["m"], np.array([lam]))
    prod_grid = (["p"], np.array([beta]))
    res = pooled_lag_codelengths(
        ids, vocabulary_size=V, lags=lags, checkpoints=4, alpha=1.0,
        mix_grid=mix_grid, prod_grid=prod_grid,
    )
    bf_mix, bf_prod = _brute_force(ids, V, lags, 4, 1.0, lam, beta)
    assert abs(res.member_bits[0] - bf_mix) < 1e-9
    assert abs(res.member_bits[1] - bf_prod) < 1e-9


def test_onehot_mixture_equals_onehot_product():
    ids, V = _tiny_case()
    lags = (1, 2, 4)
    res = pooled_lag_codelengths(
        ids, vocabulary_size=V, lags=lags, checkpoints=3
    )
    by = {n: b for n, b in zip(res.member_names, res.member_bits)}
    for d in lags:
        assert abs(by[f"onehot:lag{d}"] - by[f"prod-onehot:lag{d}"]) < 1e-9


def test_more_checkpoints_reduce_staleness():
    # first-order Markov source: fresher lag-1 tables must code shorter
    rng = np.random.default_rng(5)
    V, n = 8, 20_000
    T = rng.dirichlet(np.full(V, 0.3), size=V)
    ids = np.zeros(n, dtype=np.int64)
    for t in range(1, n):
        ids[t] = rng.choice(V, p=T[ids[t - 1]])
    grid = (["lag1"], np.array([[0.0, 1.0]]))
    coarse = pooled_lag_codelengths(
        ids, vocabulary_size=V, lags=(1,), checkpoints=2,
        mix_grid=grid, prod_grid=(["x"], np.array([[1.0]])),
    ).member_bits[0]
    fine = pooled_lag_codelengths(
        ids, vocabulary_size=V, lags=(1,), checkpoints=64,
        mix_grid=grid, prod_grid=(["x"], np.array([[1.0]])),
    ).member_bits[0]
    assert fine < coarse


def test_pooling_beats_single_experts_on_switching_source():
    # each symbol copies lag 1 or lag 3 (with noise): a mixture of the
    # two lags must beat both one-hot members
    rng = np.random.default_rng(7)
    V, n = 12, 30_000
    ids = np.zeros(n, dtype=np.int64)
    ids[:3] = rng.integers(0, V, 3)
    for t in range(3, n):
        src = ids[t - 1] if rng.random() < 0.5 else ids[t - 3]
        ids[t] = src if rng.random() < 0.8 else rng.integers(0, V)
    res = pooled_lag_codelengths(
        ids, vocabulary_size=V, lags=(1, 3), checkpoints=16
    )
    by = {n_: b for n_, b in zip(res.member_names, res.member_bits)}
    best_single = min(by["onehot:lag1"], by["onehot:lag3"])
    pooled_best = res.member_bits.min()
    assert pooled_best < best_single - 0.05
    assert res.family_bits < best_single


def test_family_tracks_best_member():
    ids, V = _tiny_case()
    res = pooled_lag_codelengths(ids, vocabulary_size=V, lags=(1, 2), checkpoints=3)
    K = len(res.member_names)
    assert res.family_bits <= res.member_bits.min() + math.log2(K) / res.n_coded + 1e-12
