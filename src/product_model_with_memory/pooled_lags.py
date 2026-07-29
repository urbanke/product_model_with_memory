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


def _log2sumexp_arr(v: FloatArray) -> float:
    m = float(np.max(v))
    return m + math.log2(float(np.exp2(v - m).sum()))


def _augmented_profile(base: tuple, c: int) -> tuple:
    """The profile after one more observation of a symbol with count c."""

    if c == 0:
        return tuple(sorted(base + (1,)))
    lst = list(base)
    lst[lst.index(c)] = c + 1
    return tuple(sorted(lst))


class _LayeredPredictiveBuilder:
    """Builds per-row predictive tables of the layered per-state mixture.

    The predictive probability of a symbol whose current count in the
    row is c is the ratio  q_avg(profile + one more c) / q_avg(profile),
    which depends on the row only through its count profile and on the
    symbol only through c --- so one evaluation per DISTINCT count value
    per row suffices (plus one for the unseen symbols).  Evaluated
    per-level q values are memoized across rows, lags, and checkpoints.
    """

    def __init__(self, V, l_max, cache_dir, jobs, progress):
        self.V = V
        self.l_max = l_max
        self.cache_dir = cache_dir
        self.jobs = jobs
        self.progress = progress
        self.memo: dict[tuple, FloatArray] = {}

    def _ensure(self, profiles: set[tuple]) -> None:
        from product_model_with_memory.codelength import (
            depth_averaged_codelength_profiles,
        )

        missing = {p for p in profiles if p and p not in self.memo}
        if missing:
            results = depth_averaged_codelength_profiles(
                {p: p for p in missing},
                d=self.V,
                l_max=self.l_max,
                cache_dir=self.cache_dir,
                jobs=self.jobs,
                progress=self.progress,
            )
            for p, res in results.items():
                self.memo[p] = np.asarray(res.log2_q_by_depth)

    def _log2_ratio(self, base: tuple, aug: tuple) -> float:
        # q_avg(empty profile) = 1 (no observations)
        top = _log2sumexp_arr(self.memo[aug])
        bot = _log2sumexp_arr(self.memo[base]) if base else math.log2(self.l_max)
        return top - bot

    def row_log_table(self, counts_row: FloatArray) -> FloatArray:
        """log2 predictive over the alphabet for one count row
        (renormalized; deviation from 1 is numerical only)."""

        V = self.V
        nz = np.flatnonzero(counts_row)
        base = tuple(sorted(int(counts_row[i]) for i in nz))
        distinct = sorted(set(base))
        saturated = len(base) >= V  # no unseen symbol exists
        cs = distinct if saturated else distinct + [0]
        wanted = {base} if base else set()
        aug_of = {c: _augmented_profile(base, c) for c in cs}
        wanted.update(aug_of.values())
        self._ensure(wanted)
        log_unseen = (
            -np.inf if saturated else self._log2_ratio(base, aug_of[0])
        )
        row = np.full(V, log_unseen)
        if len(nz):
            distinct_arr = np.array(distinct)
            lut = np.array([self._log2_ratio(base, aug_of[c]) for c in distinct])
            vals = counts_row[nz].astype(np.int64)
            row[nz] = lut[np.searchsorted(distinct_arr, vals)]
        row -= _log2sumexp_arr(row)
        return row


def _layered_log_tables(
    builder: _LayeredPredictiveBuilder,
    uni_counts: FloatArray,
    lag_counts: list[FloatArray],
) -> tuple[FloatArray, list[FloatArray]]:
    """Assemble q0 and per-lag dense log tables from cumulative counts.

    Pre-collects every needed profile across all rows so the layered
    evaluations happen in ONE batched (parallel) call per refresh.
    """

    wanted: set[tuple] = set()
    all_rows = [uni_counts[None, :]] + lag_counts
    for block in all_rows:
        for r in range(block.shape[0]):
            nz = np.flatnonzero(block[r])
            base = tuple(sorted(int(block[r][i]) for i in nz))
            if base:
                wanted.add(base)
            cs = set(base) if len(base) >= block.shape[1] else set(base) | {0}
            for c in cs:
                wanted.add(_augmented_profile(base, c))
    builder._ensure(wanted)

    log_q0 = builder.row_log_table(uni_counts)
    tables = [
        np.vstack([builder.row_log_table(cnt[r]) for r in range(cnt.shape[0])])
        for cnt in lag_counts
    ]
    return log_q0, tables


def pooled_lag_codelengths(
    ids,
    *,
    vocabulary_size: int,
    lags: tuple[int, ...] = (1, 2, 3, 4, 6, 8),
    checkpoints: int = 32,
    alpha: float = 1.0,
    expert_model: str = "layered",
    l_max: int | None = None,
    cache_dir=None,
    mix_grid: tuple[list[str], FloatArray] | None = None,
    prod_grid: tuple[list[str], FloatArray] | None = None,
    step_chunk: int = 65_536,
    jobs: int = 1,
    progress=None,
) -> PooledLagResult:
    """Evaluate mixture and tempered-product pooling over the corpus.

    expert_model selects the per-lag predictor refreshed at each
    checkpoint: "layered" (the per-state layered mixture predictive,
    the estimator of this paper; requires cache_dir) or "counts"
    (count tables smoothed toward the unigram; the cheap pilot)."""

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

    if expert_model == "layered":
        if cache_dir is None:
            raise ValueError("expert_model='layered' requires a cache_dir")
        from product_model_with_memory.codelength import default_l_max

        if l_max is None:
            l_max = default_l_max(V)
        builder = _LayeredPredictiveBuilder(V, l_max, cache_dir, jobs, progress)
    elif expert_model != "counts":
        raise ValueError(f"unknown expert_model: {expert_model!r}")

    bounds = np.linspace(start, n, checkpoints + 1).astype(int)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for ck in range(checkpoints):
            lo, hi = int(bounds[ck]), int(bounds[ck + 1])
            if hi <= lo:
                continue
            if expert_model == "layered":
                uni = np.bincount(ids[:lo], minlength=V).astype(np.float64)
                lag_counts = []
                for d in lags:
                    cnt = np.zeros((V, V), dtype=np.float64)
                    if lo > d:
                        np.add.at(cnt, (ids[: lo - d], ids[d:lo]), 1.0)
                    lag_counts.append(cnt)
                log_q0, log_tabs = _layered_log_tables(builder, uni, lag_counts)
            else:
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
