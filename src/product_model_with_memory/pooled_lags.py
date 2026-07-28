"""Pooled lag predictors: combining single-lag experts across distances.

Experts: a memoryless expert and one expert per lag delta, each a
conditional row of counts smoothed toward the (checkpointed) unigram
distribution.  Expert tables are FROZEN AT CHECKPOINTS: the corpus is
cut into C equal blocks, and the table used inside a block is built
from all positions strictly before the block.  A code that uses stale
(past-measurable) tables is still a valid sequential code, just
slightly longer; the staleness cost is measurable by varying C.

Two pooling rules, evaluated side by side over parameter grids:

* mixture (latent-switch story):
    p(x) = sum_e  lambda_e  p_e(x | ctx_e)
* tempered product (conditional-independence story, log-linear):
    p(x)  proportional to  q0(x) * prod_d [p_d(x | ctx_d)/q0(x)]^beta_d
  normalized over the vocabulary at each step.

All members (both rules, all grid points) code the same positions
t = max_lag .. n-1, so they are mixable: the family codelength is the
uniform mixture over all members, and the posterior over members is
reported.  A one-hot mixture weight and a one-hot product exponent
describe the SAME single-expert code, which the tests exploit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FloatArray = np.ndarray

LOG2E = math.log2(math.e)


# ----------------------------------------------------------------- grids


def power_law_mixture_grid(
    lags,
    a_grid=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
    s_grid=(0.2, 0.4, 0.6, 0.8, 1.0),
    m_grid=(0.0, 0.02, 0.1),
) -> tuple[list[str], FloatArray]:
    """Mixture weight vectors over [memoryless, lag_1, ..., lag_D].

    lambda over the lag experts is proportional to (1+i)^(-a) for lag
    index i (0-based, so the SHORTEST lag in the set gets the largest
    weight), scaled by s; the remaining 1-s goes to the first lag; a
    fraction m of everything is diverted to the memoryless expert.
    Includes all one-hot vectors.
    """

    n_lags = len(lags)
    names: list[str] = []
    rows: list[np.ndarray] = []
    for e in range(n_lags + 1):
        v = np.zeros(n_lags + 1)
        v[e] = 1.0
        names.append("onehot:mem" if e == 0 else f"onehot:lag{lags[e - 1]}")
        rows.append(v)
    for a in a_grid:
        shape = np.array([(1.0 + i) ** (-a) for i in range(n_lags)])
        shape /= shape.sum()
        for s in s_grid:
            lag_part = (1.0 - s) * np.eye(n_lags)[0] + s * shape
            for m in m_grid:
                v = np.concatenate(([m], (1.0 - m) * lag_part))
                names.append(f"mix:a={a},s={s},m={m}")
                rows.append(v)
    return names, np.array(rows)


def power_law_product_grid(
    lags,
    b_grid=(0.33, 0.66, 1.0),
    c_grid=(0.0, 0.5, 1.0, 2.0),
) -> tuple[list[str], FloatArray]:
    """Tempered-product exponent vectors beta over the lag experts.

    beta_i = b * (1+i)^(-c); includes the one-hot exponents (which
    reproduce single experts) and excludes the all-zero vector (that is
    the memoryless one-hot of the mixture grid).
    """

    n_lags = len(lags)
    names: list[str] = []
    rows: list[np.ndarray] = []
    for e in range(n_lags):
        v = np.zeros(n_lags)
        v[e] = 1.0
        names.append(f"prod-onehot:lag{lags[e]}")
        rows.append(v)
    for b in b_grid:
        for c in c_grid:
            v = np.array([b * (1.0 + i) ** (-c) for i in range(n_lags)])
            names.append(f"prod:b={b},c={c}")
            rows.append(v)
    return names, np.array(rows)


# ------------------------------------------------------------- evaluator


@dataclass
class PooledLagResult:
    n_coded: int
    lags: tuple[int, ...]
    checkpoints: int
    member_names: list[str]
    member_bits: FloatArray  # bits/token per member
    family_bits: float
    posterior: FloatArray

    def as_dict(self) -> dict:
        order = np.argsort(self.member_bits)
        return {
            "n_coded": self.n_coded,
            "lags": list(self.lags),
            "checkpoints": self.checkpoints,
            "family_bits_per_token": self.family_bits,
            "best_member": self.member_names[int(order[0])],
            "member_bits_per_token": {
                self.member_names[i]: float(self.member_bits[i])
                for i in range(len(self.member_names))
            },
            "posterior": {
                self.member_names[i]: float(self.posterior[i])
                for i in range(len(self.member_names))
                if self.posterior[i] > 1e-12
            },
        }


def _smoothed_log_tables(
    ids: FloatArray,
    V: int,
    lags: tuple[int, ...],
    upto: int,
    alpha: float,
) -> tuple[FloatArray, list[FloatArray]]:
    """log2 tables from positions < upto: unigram q0 and one V x V table
    per lag (rows: context id; cols: next id), each row smoothed toward
    q0 with pseudo-count alpha."""

    u = np.bincount(ids[:upto], minlength=V).astype(np.float64)
    q0 = (u + 0.5) / (upto + V / 2.0)
    log_q0 = np.log2(q0)
    tables = []
    for d in lags:
        counts = np.zeros((V, V), dtype=np.float64)
        if upto > d:
            np.add.at(counts, (ids[: upto - d], ids[d:upto]), 1.0)
        row_n = counts.sum(axis=1, keepdims=True)
        tables.append(np.log2((counts + alpha * q0[None, :]) / (row_n + alpha)))
    return log_q0, tables


def pooled_lag_codelengths(
    ids,
    *,
    vocabulary_size: int,
    lags: tuple[int, ...] = (1, 2, 3, 4, 6, 8),
    checkpoints: int = 32,
    alpha: float = 1.0,
    mix_grid: tuple[list[str], FloatArray] | None = None,
    prod_grid: tuple[list[str], FloatArray] | None = None,
    step_chunk: int = 65_536,
    jobs: int = 1,
    progress=None,
) -> PooledLagResult:
    """Evaluate mixture and tempered-product pooling over the corpus."""

    ids = np.asarray(ids, dtype=np.int64)
    V = int(vocabulary_size)
    n = len(ids)
    D = len(lags)
    start = max(lags)
    n_coded = n - start
    if mix_grid is None:
        mix_grid = power_law_mixture_grid(lags)
    if prod_grid is None:
        prod_grid = power_law_product_grid(lags)
    mix_names, mix_w = mix_grid
    prod_names, prod_b = prod_grid
    assert mix_w.shape[1] == D + 1 and prod_b.shape[1] == D

    mix_bits = np.zeros(len(mix_names))
    prod_bits = np.zeros(len(prod_names))

    # keep the cached per-chunk gather (D, steps, V) around ~2 * 10^8 floats
    if len(prod_names) and D * V:
        step_chunk = max(1024, min(step_chunk, int(2e8) // (D * V)))

    from concurrent.futures import ThreadPoolExecutor

    def _product_member(k, G, log_q0_row, tgt, steps_idx):
        beta = prod_b[k]
        logits = np.tensordot(beta, G, axes=1)
        logits += (1.0 - beta.sum()) * log_q0_row
        mx = logits.max(axis=1)
        z = mx + np.log2(np.exp2(logits - mx[:, None]).sum(axis=1))
        return k, float((logits[steps_idx, tgt] - z).sum())

    bounds = np.linspace(start, n, checkpoints + 1).astype(int)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for ck in range(checkpoints):
            lo, hi = int(bounds[ck]), int(bounds[ck + 1])
            if hi <= lo:
                continue
            log_q0, log_tabs = _smoothed_log_tables(ids, V, lags, lo, alpha)
            for c0 in range(lo, hi, step_chunk):
                c1 = min(c0 + step_chunk, hi)
                tgt = ids[c0:c1]
                # per-expert log2 prob of the true next token, (steps, D+1)
                lp = np.empty((c1 - c0, D + 1))
                lp[:, 0] = log_q0[tgt]
                for j, d in enumerate(lags):
                    lp[:, j + 1] = log_tabs[j][ids[c0 - d : c1 - d], tgt]
                # mixture rule: log2 sum_e lambda_e 2^lp
                m = lp.max(axis=1, keepdims=True)
                p = np.exp2(lp - m)
                mix_bits -= (
                    np.log2(np.clip(p @ mix_w.T, 1e-300, None)) + m
                ).sum(axis=0)
                if len(prod_names):
                    # product rule: gather each lag's full rows ONCE,
                    # then every exponent vector reuses them (threads;
                    # numpy releases the GIL in tensordot/exp)
                    G = np.stack(
                        [
                            log_tabs[j][ids[c0 - d : c1 - d]]
                            for j, d in enumerate(lags)
                        ]
                    )
                    steps_idx = np.arange(c1 - c0)
                    for k, val in pool.map(
                        lambda k: _product_member(
                            k, G, log_q0[None, :], tgt, steps_idx
                        ),
                        range(len(prod_names)),
                    ):
                        prod_bits[k] -= val
                    del G
            if progress is not None:
                progress(("checkpoint", ck + 1, checkpoints), None)

    names = list(mix_names) + list(prod_names)
    totals = np.concatenate([mix_bits, prod_bits])
    K = len(names)
    # uniform mixture over all members (log-domain)
    neg = -totals  # log2-likelihood totals
    m = neg.max()
    family_total = -(m + np.log2(np.exp2(neg - m).sum())) + math.log2(K)
    post = np.exp2(neg - m)
    post /= post.sum()
    return PooledLagResult(
        n_coded=n_coded,
        lags=tuple(lags),
        checkpoints=checkpoints,
        member_names=names,
        member_bits=totals / n_coded,
        family_bits=family_total / n_coded,
        posterior=post,
    )
