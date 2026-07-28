import math
import tempfile
from collections import Counter

import numpy as np

from product_model_with_memory.codelength import needed_r_values, profile_of
from product_model_with_memory.fast_tables import build_tables_fast
from product_model_with_memory.layered import log_q_lambda_scan
from product_model_with_memory.pairs import (
    conditional_log_loss,
    depth_averaged_predictive,
    empirical_entropies,
    pair_counts,
    reduce_vocabulary,
)


def test_predictive_matches_exact_ratio():
    # The saddle-rho posterior mean must match the exact predictive ratio
    # q(profile with one count c -> c+1) / q(profile) at each depth.
    d, L = 500, 6
    part = (5, 3, 2, 1, 1)
    orders = sorted(set(needed_r_values(part)) | {6, 7, 8})
    tables = build_tables_fast(max_L=L, r_values=orders)
    base = log_q_lambda_scan(d=d, L=L, partition=part, tables=tables)

    from product_model_with_memory.layered import _rho

    peak_logs = np.array([lc for _, lc in base.peaks])
    w = np.exp(peak_logs - peak_logs.max())
    w /= w.sum()
    n = sum(part)
    for c, bumped in [(5, (6, 3, 2, 1, 1)), (1, (5, 3, 2, 2, 1)),
                      (0, (5, 3, 2, 1, 1, 1))]:
        exact = math.exp(
            log_q_lambda_scan(d=d, L=L, partition=bumped, tables=tables).log_q
            - base.log_q
        )
        approx = sum(
            wi * _rho(L=L, r=c, tables=tables, u=u) / n
            for (u, _), wi in zip(base.peaks, w)
        )
        assert abs(approx - exact) / exact < 2e-2, (c, approx, exact)


def test_predictive_add_one_at_depth_one():
    # With l_max = 1 the predictive is exactly Laplace add-one, which
    # normalizes exactly, so renormalization is a no-op.
    counts = Counter({("a", "b"): 3, ("b", "a"): 1})
    with tempfile.TemporaryDirectory() as tmp:
        pred = depth_averaged_predictive(counts, d=25, l_max=1, cache_dir=tmp)
    assert abs(pred.prob(3) - 4.0 / 29.0) < 1e-12
    assert abs(pred.prob(1) - 2.0 / 29.0) < 1e-12
    assert abs(pred.prob(0) - 1.0 / 29.0) < 1e-12
    assert pred.normalization_error < 1e-12


def test_predictive_normalizes():
    rng = np.random.default_rng(3)
    zipf = (1000.0 / np.arange(1, 400) ** 1.3).astype(int)
    counts = Counter({i: int(c) for i, c in enumerate(zipf) if c > 0})
    with tempfile.TemporaryDirectory() as tmp:
        pred = depth_averaged_predictive(
            counts, d=10_000, l_max=12, cache_dir=tmp
        )
    assert pred.normalization_error < 5e-2
    # after renormalization the total is exactly 1
    sizes = Counter(profile_of(counts))
    total = sum(k * pred.prob(c) for c, k in sizes.items())
    total += (10_000 - sum(sizes.values())) * pred.prob(0)
    assert abs(total - 1.0) < 1e-9


def test_pair_pipeline_end_to_end():
    # Deterministic-ish periodic text: conditional structure is strong, and
    # the plug-in conditional loss must land near the (near-zero) empirical
    # conditional entropy, far below the unigram entropy.
    tokens = (["a", "b", "c", "d"] * 2_000) + ["a"]
    reduced, vocab = reduce_vocabulary(tokens, 10)
    ent = empirical_entropies(reduced)
    assert ent["conditional_bits"] < 0.01 < ent["unigram_bits"]
    pairs = pair_counts(reduced)
    with tempfile.TemporaryDirectory() as tmp:
        pred = depth_averaged_predictive(
            pairs, d=len(vocab) ** 2, l_max=10, cache_dir=tmp
        )
    losses = conditional_log_loss(pairs, pred, vocabulary_size=len(vocab))
    cond = losses["conditional_bits_per_token"]
    assert ent["conditional_bits"] <= cond + 1e-9
    assert cond < 0.15  # close to zero, nowhere near H_unigram = 2 bits
