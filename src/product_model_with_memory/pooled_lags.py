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

import gc
import time
import os

import math
from dataclasses import dataclass

import numpy as np

FloatArray = np.ndarray

LOG2E = math.log2(math.e)


def _timing_on() -> bool:
    """Per-checkpoint [phase] lines, opt-in via PMM_TIMING=1.  The
    instrumentation was added to diagnose the quadratic memo (fixed) and
    is kept behind a flag so ordinary logs stay clean."""

    return os.environ.get("PMM_TIMING", "") not in ("", "0")


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


def _evaluate_layered_family_shard(args):
    """Evaluate independent profile families serially inside one worker.

    Sharding by family, rather than synchronizing all workers after every
    depth, lets different families progress through their own certified
    depth windows independently.  Each family is still evaluated by the
    unchanged production codelength routine.
    """

    families, d, l_max, cache_dir = args
    from product_model_with_memory.codelength import (
        depth_averaged_codelength_families,
    )
    return depth_averaged_codelength_families(
        families, d=d, l_max=l_max, cache_dir=cache_dir, jobs=1,
    )


def _balanced_family_shards(families: dict, shards: int) -> list[dict]:
    """Greedily balance families by profile and augmentation size."""

    if shards < 1:
        raise ValueError("shards must be positive")
    groups: list[dict] = [{} for _ in range(min(shards, len(families)))]
    loads = [0] * len(groups)
    weighted = sorted(
        families.items(),
        key=lambda row: len(row[1][0]) + len(row[1][1]),
        reverse=True,
    )
    for key, value in weighted:
        target = min(range(len(groups)), key=loads.__getitem__)
        groups[target][key] = value
        loads[target] += len(value[0]) + len(value[1])
    return groups


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

    def _ensure_families(self, fams: dict[tuple, tuple]) -> None:
        """Evaluate whatever is missing, grouped as base + augmented
        families so each family shares its grid integrand (complexity
        notes, T3)."""

        from product_model_with_memory.codelength import (
            depth_averaged_codelength_families,
        )

        todo: dict[tuple, tuple] = {}
        for base, cs in fams.items():
            missing_cs = tuple(sorted(
                c for c in set(cs)
                if _augmented_profile(base, c) not in self.memo
            ))
            base_missing = bool(base) and base not in self.memo
            if missing_cs or base_missing:
                todo[base] = missing_cs
        if not todo:
            return
        families = {base: (base, cs) for base, cs in todo.items()}
        if self.jobs > 1 and len(families) > 1:
            import multiprocessing as mp
            # More shards than workers let the pool repair imperfect static
            # cost estimates and avoid a single long family group forming a
            # serial tail.  The worker count remains exactly ``self.jobs``.
            shards = _balanced_family_shards(families, 4 * self.jobs)
            with mp.get_context("spawn").Pool(
                min(self.jobs, len(shards))
            ) as pool:
                parts = list(pool.imap_unordered(
                    _evaluate_layered_family_shard,
                    [
                        (shard, self.V, self.l_max, self.cache_dir)
                        for shard in shards
                    ],
                    chunksize=1,
                ))
            results = {
                key: value for part in parts for key, value in part.items()
            }
        else:
            results = depth_averaged_codelength_families(
                families,
                d=self.V,
                l_max=self.l_max,
                cache_dir=self.cache_dir,
                jobs=1,
                progress=self.progress,
            )
        for base, (b_res, a_res) in results.items():
            if b_res is not None:
                self.memo[base] = np.asarray(b_res.log2_q_by_depth)
            for c, res in a_res.items():
                self.memo[_augmented_profile(base, c)] = np.asarray(
                    res.log2_q_by_depth)

    def _log2_ratio(self, base: tuple, aug: tuple) -> float:
        # q_avg(empty profile) = 1 (no observations)
        top = _log2sumexp_arr(self.memo[aug])
        bot = _log2sumexp_arr(self.memo[base]) if base else math.log2(self.l_max)
        return top - bot

    def row_log_sparse(
        self, counts_row: FloatArray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Sparse form of the row's predictive: (observed symbol ids,
        their log2 probabilities, the shared log2 probability of every
        unobserved symbol).  Renormalized identically to the dense
        form; the two agree exactly (complexity notes, T7)."""

        nz = np.flatnonzero(counts_row)
        return self.row_log_sparse_entries(nz, counts_row[nz])

    def row_log_sparse_entries(
        self, symbol_ids: np.ndarray, counts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Sparse row predictive directly from observed ids and counts."""

        V = self.V
        nz = np.asarray(symbol_ids, dtype=np.int64)
        vals = np.asarray(counts, dtype=np.int64)
        if nz.shape != vals.shape:
            raise ValueError("symbol ids and counts must have matching shapes")
        if ((nz < 0) | (nz >= V)).any() or (vals <= 0).any():
            raise ValueError("sparse row entries must be valid and positive")
        if len(np.unique(nz)) != len(nz):
            raise ValueError("sparse row symbol ids must be unique")
        base = tuple(sorted(int(value) for value in vals))
        distinct = sorted(set(base))
        saturated = len(base) >= V
        cs = distinct if saturated else distinct + [0]
        aug_of = {c: _augmented_profile(base, c) for c in cs}
        self._ensure_families({base: tuple(cs)})
        log_unseen = (
            -np.inf if saturated else self._log2_ratio(base, aug_of[0])
        )
        if len(nz):
            distinct_arr = np.array(distinct)
            lut = np.array([self._log2_ratio(base, aug_of[c])
                            for c in distinct])
            logp = lut[np.searchsorted(distinct_arr, vals)]
        else:
            logp = np.empty(0)
        # renormalize exactly as the dense path does: total mass =
        # observed entries + (V - k) copies of the unseen value
        parts = [logp]
        if not saturated:
            parts.append(np.array([log_unseen + math.log2(V - len(nz))]))
        total = _log2sumexp_arr(np.concatenate(parts))
        return nz, logp - total, float(log_unseen - total)

    def row_log_table(self, counts_row: FloatArray) -> FloatArray:
        """log2 predictive over the alphabet for one count row
        (renormalized; deviation from 1 is numerical only)."""

        V = self.V
        nz = np.flatnonzero(counts_row)
        base = tuple(sorted(int(counts_row[i]) for i in nz))
        distinct = sorted(set(base))
        saturated = len(base) >= V  # no unseen symbol exists
        cs = distinct if saturated else distinct + [0]
        aug_of = {c: _augmented_profile(base, c) for c in cs}
        self._ensure_families({base: tuple(cs)})
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

    fams: dict[tuple, set] = {}
    all_rows = [uni_counts[None, :]] + lag_counts
    for block in all_rows:
        for r in range(block.shape[0]):
            nz = np.flatnonzero(block[r])
            base = tuple(sorted(int(block[r][i]) for i in nz))
            cs = set(base) if len(base) >= block.shape[1] else set(base) | {0}
            fams.setdefault(base, set()).update(cs)
    builder._ensure_families(
        {b: tuple(sorted(cs)) for b, cs in fams.items()})

    log_q0 = builder.row_log_table(uni_counts)
    tables = [
        np.vstack([builder.row_log_table(cnt[r]) for r in range(cnt.shape[0])])
        for cnt in lag_counts
    ]
    return log_q0, tables


class _SparseLagTables:
    """Per-lag sparse predictive tables (complexity notes, T7): for
    each state, the observed symbols with their log2 probabilities,
    plus the one shared log2 probability of every unobserved symbol.
    CSR layout across states."""

    def __init__(self, builder, lag_counts):
        self.lags = []
        for cnt in lag_counts:
            V = cnt.shape[0]
            ptr = np.zeros(V + 1, dtype=np.int64)
            idx_parts, val_parts = [], []
            unseen = np.empty(V)
            for s in range(V):
                nz, logp, log_unseen = builder.row_log_sparse(cnt[s])
                ptr[s + 1] = ptr[s] + len(nz)
                idx_parts.append(nz)
                val_parts.append(logp)
                unseen[s] = log_unseen
            self.lags.append({
                "ptr": ptr,
                "idx": np.concatenate(idx_parts) if idx_parts else
                np.empty(0, dtype=np.int64),
                "val": np.concatenate(val_parts) if val_parts else
                np.empty(0),
                "unseen": unseen,
                # reference value for the support corrections: the true
                # unseen value where one exists; for SATURATED rows
                # (every symbol observed) no unseen symbol exists and
                # any finite reference is algebraically equivalent
                "rho": np.where(np.isfinite(unseen), unseen, 0.0),
            })


@dataclass(frozen=True)
class SparseCountRows:
    """CSR transition counts with rows=context and columns=target."""

    vocabulary_size: int
    ptr: np.ndarray
    idx: np.ndarray
    count: np.ndarray

    @classmethod
    def from_sorted_keys(
        cls, vocabulary_size: int, keys: np.ndarray, counts: np.ndarray
    ) -> "SparseCountRows":
        keys = np.asarray(keys, dtype=np.int64)
        counts = np.asarray(counts, dtype=np.int64)
        if keys.shape != counts.shape or (counts <= 0).any():
            raise ValueError("sorted keys/counts must match and be positive")
        if len(keys) and (
            keys[0] < 0 or keys[-1] >= vocabulary_size**2
            or (np.diff(keys) <= 0).any()
        ):
            raise ValueError("transition keys must be unique and sorted")
        rows = keys // vocabulary_size
        ptr = np.zeros(vocabulary_size + 1, dtype=np.int64)
        ptr[1:] = np.cumsum(np.bincount(rows, minlength=vocabulary_size))
        return cls(
            vocabulary_size,
            ptr,
            keys % vocabulary_size,
            counts,
        )


def _layered_log_sparse_tables(
    builder: _LayeredPredictiveBuilder,
    uni_counts: FloatArray,
    lag_counts: list[SparseCountRows],
) -> tuple[FloatArray, list[dict[str, np.ndarray]]]:
    """Layered unigram and sparse conditional rows without V-by-V counts."""

    v = builder.V
    if len(uni_counts) != v or any(table.vocabulary_size != v
                                   for table in lag_counts):
        raise ValueError("all sparse counts must use the builder vocabulary")
    families: dict[tuple, set[int]] = {}

    def add_family(values: np.ndarray) -> None:
        base = tuple(sorted(int(value) for value in values))
        cs = set(base) if len(base) >= v else set(base) | {0}
        families.setdefault(base, set()).update(cs)

    add_family(np.asarray(uni_counts)[np.asarray(uni_counts) > 0])
    add_family(np.empty(0, dtype=np.int64))
    for table in lag_counts:
        nonempty = np.flatnonzero(np.diff(table.ptr))
        for row in nonempty:
            add_family(table.count[table.ptr[row]:table.ptr[row + 1]])
    builder._ensure_families({
        base: tuple(sorted(cs)) for base, cs in families.items()
    })

    log_q0 = builder.row_log_table(np.asarray(uni_counts))
    tables = []
    for counts in lag_counts:
        unseen = np.full(v, -math.log2(v), dtype=np.float64)
        values = np.empty(len(counts.idx), dtype=np.float64)
        nonempty = np.flatnonzero(np.diff(counts.ptr))
        for row in nonempty:
            lo, hi = counts.ptr[row], counts.ptr[row + 1]
            ids, logp, log_unseen = builder.row_log_sparse_entries(
                counts.idx[lo:hi], counts.count[lo:hi]
            )
            if not np.array_equal(ids, counts.idx[lo:hi]):
                raise RuntimeError("sparse layered row changed symbol order")
            values[lo:hi] = logp
            unseen[row] = log_unseen
        tables.append({
            "ptr": counts.ptr.copy(),
            "idx": counts.idx.copy(),
            "val": values,
            "unseen": unseen,
            "rho": np.where(np.isfinite(unseen), unseen, 0.0),
        })
    return log_q0, tables


def _layered_log_sparse_conditionals(
    builder: _LayeredPredictiveBuilder,
    lag_counts: list[SparseCountRows],
) -> list[dict[str, np.ndarray]]:
    """Sparse conditional rows without also evaluating the unigram.

    This is the split-construction counterpart of
    :func:`_layered_log_sparse_tables`: a separately persisted unigram can be
    shared by several lag estimators, while each lag family is evaluated
    exactly once.  Family provisioning and row normalization are otherwise
    identical to the monolithic path.
    """

    v = builder.V
    if any(table.vocabulary_size != v for table in lag_counts):
        raise ValueError("all sparse counts must use the builder vocabulary")
    families: dict[tuple, set[int]] = {}
    for table in lag_counts:
        for row in np.flatnonzero(np.diff(table.ptr)):
            values = table.count[table.ptr[row]:table.ptr[row + 1]]
            base = tuple(sorted(int(value) for value in values))
            cs = set(base) if len(base) >= v else set(base) | {0}
            families.setdefault(base, set()).update(cs)
    builder._ensure_families({
        base: tuple(sorted(cs)) for base, cs in families.items()
    })

    tables = []
    for counts in lag_counts:
        unseen = np.full(v, -math.log2(v), dtype=np.float64)
        values = np.empty(len(counts.idx), dtype=np.float64)
        for row in np.flatnonzero(np.diff(counts.ptr)):
            lo, hi = counts.ptr[row], counts.ptr[row + 1]
            ids, logp, log_unseen = builder.row_log_sparse_entries(
                counts.idx[lo:hi], counts.count[lo:hi]
            )
            if not np.array_equal(ids, counts.idx[lo:hi]):
                raise RuntimeError("sparse layered row changed symbol order")
            values[lo:hi] = logp
            unseen[row] = log_unseen
        tables.append({
            "ptr": counts.ptr.copy(), "idx": counts.idx.copy(),
            "val": values, "unseen": unseen,
            "rho": np.where(np.isfinite(unseen), unseen, 0.0),
        })
    return tables


def _gather_entries(table, states):
    """All support entries of the given per-step states, flattened:
    (step index, symbol, value, unseen value of that step's row)."""

    ptr, idx, val = table["ptr"], table["idx"], table["val"]
    starts = ptr[states]
    lens = ptr[states + 1] - starts
    total = int(lens.sum())
    if total == 0:
        z = np.empty(0, dtype=np.int64)
        return z, z, np.empty(0), np.empty(0)
    t_e = np.repeat(np.arange(len(states)), lens)
    base = np.repeat(starts - np.concatenate([[0], np.cumsum(lens)[:-1]]),
                     lens)
    pos = np.arange(total) + base
    rho_e = table["rho"][states][t_e]
    return t_e, idx[pos], val[pos], rho_e


def _eval_chunk_sparse(sparse_tabs, log_q0, prod_b, gammas, L_gamma,
                       mix_w, sd, tgt):
    """One chunk of positions, BOTH rules, from sparse tables.

    Product-rule normalizer per position (T7): with gamma = 1 - sum
    beta and delta(x) the support corrections,
      Z = 2^B [ T_gamma + sum_{x in S} q0(x)^gamma (2^delta - 1) ],
    and the common factor 2^B cancels against the numerator, so only
    T_gamma (precomputed per refresh) and the sparse union S enter.
    """

    D, m = sd.shape
    V = len(log_q0)
    # ---- entry list over all lags: (step, symbol, lag, logp - unseen)
    t_parts, x_parts, v_parts, d_parts = [], [], [], []
    lp = np.empty((m, D + 1))          # per-expert log2 p(target), mixture
    lp[:, 0] = log_q0[tgt]
    for j in range(D):
        t_e, x_e, val_e, unseen_e = _gather_entries(sparse_tabs.lags[j],
                                                    sd[j])
        v_e = val_e - unseen_e
        t_parts.append(t_e)
        x_parts.append(x_e)
        v_parts.append(v_e)
        d_parts.append(np.full(len(t_e), j, dtype=np.int64))
        # mixture gather: unseen by default, corrected where target is
        # in the row's support
        lp[:, j + 1] = sparse_tabs.lags[j]["unseen"][sd[j]]
        hit = x_e == tgt[t_e]
        lp[t_e[hit], j + 1] = val_e[hit]
    t_all = np.concatenate(t_parts)
    x_all = np.concatenate(x_parts)
    v_all = np.concatenate(v_parts)
    d_all = np.concatenate(d_parts)

    # ---- group equal (step, symbol) pairs
    key = t_all * V + x_all
    order = np.argsort(key, kind="stable")
    key_s = key[order]
    new = np.concatenate([[True], key_s[1:] != key_s[:-1]])
    g_of_entry = np.empty(len(order), dtype=np.int64)
    g_of_entry[order] = np.cumsum(new) - 1
    n_g = int(new.sum())
    first = order[np.flatnonzero(new)]
    t_g, x_g = t_all[first], x_all[first]
    tgt_hit = x_g == tgt[t_g]          # groups that ARE the coded symbol

    prod_bits = np.zeros(len(prod_b))
    lq0_tgt = log_q0[tgt]
    for k in range(len(prod_b)):
        beta = prod_b[k]
        gamma = gammas[k]
        delta_g = np.bincount(g_of_entry, weights=beta[d_all] * v_all,
                              minlength=n_g)
        contrib = np.exp2(gamma * log_q0[x_g]) * (np.exp2(delta_g) - 1.0)
        Z_lin = np.full(m, L_gamma[k])
        np.add.at(Z_lin, t_g, contrib)
        delta_tgt = np.zeros(m)
        delta_tgt[t_g[tgt_hit]] = delta_g[tgt_hit]
        prod_bits[k] = float(
            (gamma * lq0_tgt + delta_tgt
             - np.log2(np.maximum(Z_lin, 1e-300))).sum())

    # ---- mixture rule from lp, as in the dense path
    mx = lp.max(axis=1, keepdims=True)
    p = np.exp2(lp - mx)
    mix_vals = np.log2(np.clip(p @ mix_w.T, 1e-300, None)) + mx
    return mix_vals.sum(axis=0), prod_bits


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
    eval_mode: str = "auto",
    resume_path=None,
) -> PooledLagResult:
    """Evaluate mixture and tempered-product pooling over the corpus.

    expert_model selects the per-lag predictor refreshed at each
    checkpoint: "layered" (the per-state layered mixture predictive,
    the estimator of this paper) or "counts" (count tables smoothed
    toward the unigram; the cheap pilot).

    eval_mode selects the evaluation of the two pooling rules for the
    layered expert: "dense" (materialize V x V tables; the original
    path), "sparse" (T7: per-row supports plus one shared unseen
    value; identical numbers, cost independent of the alphabet size),
    or "auto" (sparse for V > 4096).

    resume_path (layered expert only): a directory where the run
    saves its state after every checkpoint --- accumulated bits and
    the profile memo --- so a killed run RESUMES at the next
    checkpoint instead of starting over.  The saved state is bound to
    a fingerprint of the inputs; a mismatch starts fresh."""

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
        from product_model_with_memory.codelength import (
            _resolve_tables_source,
            default_l_max,
        )

        if cache_dir is None and _resolve_tables_source(None) == "cache":
            raise ValueError(
                "expert_model='layered' requires a cache_dir when the "
                "legacy cache table source is selected")
        if l_max is None:
            l_max = default_l_max(V)
        builder = _LayeredPredictiveBuilder(V, l_max, cache_dir, jobs, progress)
    elif expert_model != "counts":
        raise ValueError(f"unknown expert_model: {expert_model!r}")

    ck_done = -1
    resume_dir = None
    if resume_path is not None and expert_model == "layered":
        import json as _json
        import pickle
        from pathlib import Path as _Path

        resume_dir = _Path(resume_path)
        resume_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = [V, list(lags), checkpoints, n,
                       int(ids[:1000].sum()), int(ids[-1000:].sum()),
                       len(mix_names), len(prod_names)]
        state_f = resume_dir / "state.json"
        if state_f.exists():
            state = _json.loads(state_f.read_text())
            if state.get("fingerprint") == fingerprint:
                ck_done = int(state["ck_done"])
                mix_bits[:] = np.array(state["mix_bits"])
                prod_bits[:] = np.array(state["prod_bits"])
                for f in sorted(resume_dir.glob("memo_*.pkl")):
                    with open(f, "rb") as fh:
                        builder.memo.update(pickle.load(fh))
                if progress is not None:
                    progress(("resume", ck_done + 1, checkpoints), None)
            else:
                for f in sorted(resume_dir.glob("memo_*.pkl")):
                    f.unlink()

    def _memo_n() -> int:
        """Memo size, or 0 for experts that have no builder."""

        return len(builder.memo) if expert_model == "layered" else 0

    def _save_resume(ck: int, n_before: int) -> None:
        """Persist the entries added since the last checkpoint.

        The previous version took `known_before = set(builder.memo)` at
        the top of every checkpoint --- a fresh set holding every key in
        the memo, millions of them --- and then filtered the whole memo
        against it here.  Two O(total memo) passes per checkpoint, to
        compute a delta that is known exactly: the memo is append-only,
        so the new entries are simply the ones after position n_before.
        The old form also allocated a multi-million-element set each
        checkpoint, which is the kind of spike that triggers a full
        generation-2 collection --- cost O(live heap) --- and that is
        the most likely source of the 78s / 1886s oscillation in the
        checkpoint times.
        """

        if resume_dir is None:
            return
        import json as _json
        import pickle
        from itertools import islice

        new_entries = dict(islice(builder.memo.items(), n_before, None))
        if new_entries:
            with open(resume_dir / f"memo_{ck:03d}.pkl", "wb") as fh:
                pickle.dump(new_entries, fh, protocol=4)
        tmp = resume_dir / "state.json.tmp"
        tmp.write_text(_json.dumps({
            "fingerprint": fingerprint, "ck_done": ck,
            "mix_bits": list(mix_bits), "prod_bits": list(prod_bits),
        }))
        tmp.replace(resume_dir / "state.json")

    if eval_mode not in ("auto", "dense", "sparse"):
        raise ValueError(f"unknown eval_mode {eval_mode!r}")
    sparse = (eval_mode == "sparse"
              or (eval_mode == "auto" and V > 4096))
    if sparse and expert_model != "layered":
        raise ValueError("sparse evaluation requires the layered expert")

    bounds = np.linspace(start, n, checkpoints + 1).astype(int)
    # The profile memo grows monotonically across rows, lags and
    # checkpoints and is never discarded, so every cyclic-GC pass walks
    # the whole of it.  Sampled on a live run (2 August, V=1024, 21
    # checkpoints in): 3678 of 4041 stack samples were inside
    # gc_collect_main traversing dicts and sets --- 91% of the process's
    # time was garbage collection, on one core, while the machine looked
    # idle.  That is almost certainly why this run has never finished.
    #
    # gc.freeze() moves everything currently alive into a permanent
    # generation that the collector never visits again.  Calling it once
    # per checkpoint keeps the accumulated memo out of every subsequent
    # pass while leaving collection working normally for whatever the
    # next checkpoint allocates.  The gain scales with memo size
    # (measured on a synthetic memo: 1.2x at 200k entries, 1.6x at 800k,
    # 3.1x at 3.2M), so it is largest exactly where the problem is.
    gc.freeze()

    # INSTRUMENTATION, not a fix.  The checkpoint times oscillate --- 78s
    # and 1886s alternating on a heap that grows smoothly --- and a
    # sampled profile showed 91% of the time inside the collector.  Those
    # two facts do not reconcile unless the expensive collections are
    # threshold-triggered rather than steady: a full (generation-2) pass
    # costs O(live heap) and fires only when the long-lived allocation
    # ratio crosses a threshold, so a checkpoint that crosses it twice
    # pays minutes and its neighbour pays nothing.  That is a hypothesis.
    # PMM_GC_TRACE=1 logs every collection with its generation and
    # duration, which confirms or kills it from one run's log.
    if os.environ.get("PMM_GC_TRACE", "") not in ("", "0"):
        _gc_t0 = {}

        def _gc_cb(phase, info):
            gen = info.get("generation")
            if phase == "start":
                _gc_t0[gen] = time.time()
            else:
                dt = time.time() - _gc_t0.get(gen, time.time())
                if dt > 0.5 or gen == 2:
                    print(f"  [gc] gen{gen} {dt:8.2f}s "
                          f"collected={info.get('collected')} "
                          f"uncollectable={info.get('uncollectable')}",
                          flush=True)

        gc.callbacks.append(_gc_cb)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for ck in range(checkpoints):
            if ck <= ck_done:
                continue
            # a COUNT, not a copy of every key (see _save_resume).
            # `builder` exists only for the layered expert, which is
            # also the only one that resumes --- keep the guard.
            known_before = 0
            if resume_dir is not None:
                known_before = len(builder.memo)
            _t_ck = time.time()
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
                if sparse:
                    # dense q0 row + sparse per-lag tables (T7)
                    fams: dict[tuple, set] = {}
                    for block in [uni[None, :]] + lag_counts:
                        for rr in range(block.shape[0]):
                            nzr = np.flatnonzero(block[rr])
                            b = tuple(sorted(int(block[rr][i]) for i in nzr))
                            cset = (set(b) if len(b) >= block.shape[1]
                                    else set(b) | {0})
                            fams.setdefault(b, set()).update(cset)
                    builder._ensure_families(
                        {b: tuple(sorted(cc)) for b, cc in fams.items()})
                    log_q0 = builder.row_log_table(uni)
                    sparse_tabs = _SparseLagTables(builder, lag_counts)
                    gammas = 1.0 - prod_b.sum(axis=1)
                    L_gamma = np.array([
                        float(np.exp2(g * log_q0).sum()) for g in gammas
                    ])
                else:
                    log_q0, log_tabs = _layered_log_tables(
                        builder, uni, lag_counts)
            else:
                log_q0, log_tabs = _smoothed_log_tables(ids, V, lags, lo, alpha)
            if sparse:
                for c0 in range(lo, hi, step_chunk):
                    c1 = min(c0 + step_chunk, hi)
                    tgt = ids[c0:c1]
                    sd = np.stack([ids[c0 - d:c1 - d] for d in lags])
                    mix_vals, prod_vals = _eval_chunk_sparse(
                        sparse_tabs, log_q0, prod_b, gammas, L_gamma,
                        mix_w, sd, tgt)
                    mix_bits -= mix_vals
                    prod_bits -= prod_vals
                _t_work = time.time()
                _save_resume(ck, known_before)
                gc.freeze()
                if progress is not None:
                    progress(("checkpoint", ck + 1, checkpoints), None)
                if _timing_on():
                    print(f"  [phase] ck {ck + 1}: work "
                          f"{_t_work - _t_ck:7.1f}s  "
                          f"save {time.time() - _t_work:6.1f}s  "
                          f"memo {_memo_n():,}", flush=True)
                continue
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
            _t_work = time.time()
            _save_resume(ck, known_before)
            gc.freeze()          # the normal completion path; see above
            if progress is not None:
                progress(("checkpoint", ck + 1, checkpoints), None)
            if _timing_on():
                print(f"  [phase] ck {ck + 1}: work "
                      f"{_t_work - _t_ck:7.1f}s  "
                      f"save {time.time() - _t_work:6.1f}s  "
                      f"memo {_memo_n():,}", flush=True)

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
