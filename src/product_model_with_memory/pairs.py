"""Sliding-window pair experiment: plug-in conditionals from a joint mixture.

The scheme (plug-in, in-sample - the whole corpus informs the estimate):

1. Reduce the vocabulary to the top ``K`` tokens plus one ``<unk>`` bucket
   (vocabulary size V = K + 1, pair alphabet ``d_pair = V**2``).
2. Count the n-1 sliding-window pairs and form their count profile.
3. Estimate the joint pair distribution by the depth-averaged product-simplex
   mixture's posterior-mean predictive: for a pair currently at count ``c``,

       p_hat(c)  =  sum_L w_L * p_L(c),      w_L propto Q^(L)(pairs),

   where ``p_L(c) = E_L[theta_pair | data]`` is evaluated per count class at
   the dominant saddle(s) of the outer integral: at a saddle ``u*`` the
   posterior mean of a coordinate with count ``c`` is ``rho(L, c, u*) / N``
   (and the saddle equation ``sum_i rho_i = N`` makes the class-weighted sum
   normalize exactly at saddle order).  Multi-peak profiles combine peaks by
   their scan weights; a final numerical renormalization absorbs the
   remaining O(1/N)-level error, which is also reported as a diagnostic.
4. Conditional probabilities by row normalization,
   ``p_hat(b|a) = p_hat(a,b) / sum_b' p_hat(a,b')``, and the plug-in
   conditional log-loss  ``-(1/(n-1)) sum_t log2 p_hat(x_t | x_{t-1})``,
   computable from counts and per-row count histograms.

The comparison targets are the empirical conditional entropy H(next|prev)
and the empirical unigram entropy of the same reduced stream.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from product_model_with_memory.codelength import (
    default_l_max,
    needed_r_values,
    profile_of,
)
from product_model_with_memory.fast_tables import TableCache, build_tables_fast
from product_model_with_memory.layered import (
    log_q_lambda_scan,
    _rho,
)

UNK = "<unk>"


def reduce_vocabulary(
    tokens: Sequence[str], top_k: int
) -> tuple[list[str], list[str]]:
    """Map all but the ``top_k`` most frequent tokens to ``<unk>``."""

    counts = Counter(tokens)
    keep = {w for w, _ in counts.most_common(top_k)}
    vocab = sorted(keep) + [UNK]
    reduced = [t if t in keep else UNK for t in tokens]
    return reduced, vocab


def pair_counts(reduced: Sequence[str]) -> Counter:
    return Counter(zip(reduced[:-1], reduced[1:]))


def empirical_entropies(reduced: Sequence[str]) -> dict[str, float]:
    """Unigram entropy of the stream and conditional entropy H(next|prev)."""

    n_pairs = len(reduced) - 1
    pc = pair_counts(reduced)
    first = Counter(reduced[:-1])
    uni = Counter(reduced)
    h_pair = -sum(v / n_pairs * math.log2(v / n_pairs) for v in pc.values())
    h_first = -sum(v / n_pairs * math.log2(v / n_pairs) for v in first.values())
    h_uni = -sum(
        v / len(reduced) * math.log2(v / len(reduced)) for v in uni.values()
    )
    return {
        "unigram_bits": h_uni,
        "pair_bits": h_pair,
        "conditional_bits": h_pair - h_first,
    }


@dataclass(frozen=True)
class PredictiveClasses:
    """Depth-averaged predictive probability per count class."""

    d: int
    n: int
    l_max: int
    class_prob: dict[int, float]  # count value (incl. 0) -> predictive prob
    normalization_error: float  # |sum - 1| before final renormalization
    depth_posterior: tuple[float, ...]

    def prob(self, count: int) -> float:
        return self.class_prob[count]


def depth_averaged_predictive(
    counts: Mapping,
    *,
    d: int,
    l_max: int | None = None,
    cache_dir: str | Path,
    jobs: int = 1,
    progress=None,
) -> PredictiveClasses:
    """Posterior-mean predictive probability for each count class."""

    partition = profile_of(counts)
    n = sum(partition)
    if l_max is None:
        l_max = default_l_max(d)
    class_sizes = Counter(partition)  # count value -> number of symbols
    classes = sorted(class_sizes)  # positive counts
    s = len(partition)

    cache = build_tables_fast(
        max_L=l_max,
        r_values=needed_r_values(partition),
        cache_dir=cache_dir,
        jobs=jobs,
        materialize=False,
        progress=(
            None
            if progress is None
            else lambda k, t: progress(("tables", k, t), None)
        ),
    )
    assert isinstance(cache, TableCache)

    log_q_by_depth: list[float] = []
    # per depth: predictive probability per class (0 and each positive count)
    pred_by_depth: list[dict[int, float]] = []

    for L in range(1, l_max + 1):
        if L == 1:
            # add-one rule, exact
            from product_model_with_memory.layered import log_q_lambda_closed_l1

            log_q_by_depth.append(
                log_q_lambda_closed_l1(d=d, partition=partition).log_q
            )
            pred_by_depth.append(
                {c: (c + 1.0) / (n + d) for c in [0, *classes]}
            )
        else:
            tables = cache.level_tables(L, cache.r_values)
            result = log_q_lambda_scan(
                d=d, L=L, partition=partition, tables=tables
            )
            if not result.converged:
                raise RuntimeError(f"L={L}: {result.message}")
            log_q_by_depth.append(result.log_q)
            # peak weights within this depth
            peak_logs = np.array([lc for _, lc in result.peaks])
            peak_w = np.exp(peak_logs - peak_logs.max())
            peak_w /= peak_w.sum()
            pred: dict[int, float] = {}
            for c in [0, *classes]:
                value = 0.0
                for (u_peak, _), w in zip(result.peaks, peak_w):
                    value += w * _rho(L=L, r=c, tables=tables, u=u_peak) / n
                pred[c] = value
            pred_by_depth.append(pred)
            del tables
        if progress is not None:
            progress(("depth", L, l_max), None)

    log_q = np.array(log_q_by_depth)
    w = np.exp(log_q - log_q.max())
    w /= w.sum()

    class_prob = {
        c: float(sum(w[i] * pred_by_depth[i][c] for i in range(l_max)))
        for c in [0, *classes]
    }
    total = sum(class_sizes[c] * class_prob[c] for c in classes)
    total += (d - s) * class_prob[0]
    error = abs(total - 1.0)
    class_prob = {c: p / total for c, p in class_prob.items()}
    return PredictiveClasses(
        d=d,
        n=n,
        l_max=l_max,
        class_prob=class_prob,
        normalization_error=float(error),
        depth_posterior=tuple(float(x) for x in w),
    )


def conditional_log_loss(
    pairs: Mapping,
    predictive: PredictiveClasses,
    *,
    vocabulary_size: int,
) -> dict[str, float]:
    """Plug-in conditional log-loss from pair counts and class predictives.

    ``p_hat(b|a) = p_hat(count(a,b)) / [sum over the row a]``, with unseen
    successors carrying the class-0 predictive.  The loss is evaluated on the
    counts themselves (in-sample):  -(1/n) sum_ab m_ab log2 p_hat(b|a).
    """

    rows: dict = defaultdict(Counter)
    for (a, b), m in pairs.items():
        rows[a][m] += 1  # per-row histogram of successor counts

    p0 = predictive.prob(0)
    n = sum(pairs.values())
    loss = 0.0
    joint_loss = 0.0
    for a, hist in rows.items():
        support = sum(hist.values())
        denom = sum(k * predictive.prob(c) for c, k in hist.items())
        denom += (vocabulary_size - support) * p0
        for c, k in hist.items():
            p_cond = predictive.prob(c) / denom
            loss += k * c * (-math.log2(p_cond))
            joint_loss += k * c * (-math.log2(predictive.prob(c)))
    return {
        "conditional_bits_per_token": loss / n,
        "joint_bits_per_pair": joint_loss / n,
    }
