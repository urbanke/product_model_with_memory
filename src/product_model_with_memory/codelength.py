"""Exact codelength of the depth-averaged product-simplex predictor.

For an exchangeable mixture the probability of a sequence depends only on its
count profile, so the codelength of a corpus prefix under the depth-averaged
predictor ``Q_avg = (1/L_max) sum_L Q^(L)`` is computed exactly from the
counts:  ``-log2 Q_avg(x^n) = -log2 [ (1/L_max) sum_L q_lambda(L) ]`` with
``lambda`` the profile of the prefix (paper, Sections 3.1 and 5.3).

``L_max = round(2 c* ln d)`` with ``c* = 1/(1 - gamma)`` (Euler-Mascheroni
``gamma``), the depth range used for the King James Bible experiment.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from product_model_with_memory.layered import (
    ProductMomentTables,
    log_q_lambda_closed_l1,
    log_q_lambda_scan,
    partition_multiplicities,
)
from product_model_with_memory.fast_tables import TableCache, build_tables_fast

EULER_GAMMA = 0.5772156649015329
C_STAR = 1.0 / (1.0 - EULER_GAMMA)  # ~2.3653, the paper's critical depth coefficient


def default_l_max(d: int, *, c_star: float = C_STAR, factor: float = 2.0) -> int:
    """``round(factor * c_star * ln d)`` — the paper's depth range."""

    return max(1, round(factor * c_star * math.log(d)))


def profile_of(counts: Mapping[str, int] | Iterable[int]) -> tuple[int, ...]:
    """Sorted (descending) nonzero count profile of a count table."""

    values = counts.values() if isinstance(counts, Mapping) else counts
    return tuple(sorted((v for v in values if v > 0), reverse=True))


def needed_r_values(partition: tuple[int, ...]) -> tuple[int, ...]:
    """Moment orders required for Laplace evaluation of ``partition``."""

    needed = {0, 1, 2}
    for part, _ in partition_multiplicities(partition):
        needed.update((part, part + 1, part + 2))
    return tuple(sorted(needed))


@dataclass(frozen=True)
class DepthAveragedCodelength:
    """Codelength of one count profile under the depth-averaged predictor."""

    d: int
    n: int
    l_max: int
    log2_q_by_depth: tuple[float, ...]  # index L-1 -> log2 q_lambda(L)
    log2_q_avg: float

    @property
    def bits_per_token(self) -> float:
        return -self.log2_q_avg / self.n

    @property
    def posterior(self) -> tuple[float, ...]:
        """Posterior weight of each depth given the data (uniform prior)."""

        log_q = np.array(self.log2_q_by_depth)
        w = np.exp((log_q - np.max(log_q)) * math.log(2.0))
        w /= w.sum()
        return tuple(float(x) for x in w)

    @property
    def posterior_mode(self) -> int:
        return 1 + int(np.argmax(self.log2_q_by_depth))

    def bits_per_token_at_depth(self, L: int) -> float:
        return -self.log2_q_by_depth[L - 1] / self.n


def depth_averaged_codelength(
    counts: Mapping[str, int] | Iterable[int],
    *,
    d: int,
    l_max: int | None = None,
    tables: ProductMomentTables | None = None,
    cache_dir: str | Path | None = None,
    u_points: int = 16_001,
    laguerre_order: int = 96,
    strict: bool = True,
) -> DepthAveragedCodelength:
    """Exact codelength of ``counts`` under the depth-averaged predictor.

    ``tables`` may be passed to reuse moment tables across checkpoints; they
    must cover ``l_max`` and every order in ``needed_r_values(profile)``.
    """

    partition = profile_of(counts)
    n = sum(partition)
    if n == 0:
        raise ValueError("counts must contain at least one observation")
    if len(partition) > d:
        raise ValueError(
            f"profile has {len(partition)} distinct symbols but d={d}"
        )
    if l_max is None:
        l_max = default_l_max(d)

    if tables is None and l_max > 1:
        tables = build_tables_fast(
            max_L=l_max,
            r_values=needed_r_values(partition),
            u_points=u_points,
            laguerre_order=laguerre_order,
            cache_dir=cache_dir,
        )

    log2_q: list[float] = []
    for L in range(1, l_max + 1):
        if L == 1:
            result = log_q_lambda_closed_l1(d=d, partition=partition)
        else:
            result = log_q_lambda_scan(
                d=d, L=L, partition=partition, tables=tables
            )
        if strict and not result.converged:
            raise RuntimeError(f"L={L}: {result.message}")
        if result.log2_q > 1e-9:
            raise RuntimeError(
                f"L={L}: log2 q = {result.log2_q:.6g} > 0 is impossible for a "
                "sequence probability - numerical tables are outside their "
                "domain of validity"
            )
        log2_q.append(result.log2_q)

    log2_avg = _log2sumexp(log2_q) - math.log2(l_max)
    return DepthAveragedCodelength(
        d=d,
        n=n,
        l_max=l_max,
        log2_q_by_depth=tuple(log2_q),
        log2_q_avg=float(log2_avg),
    )


def depth_averaged_codelength_profiles(
    profiles: Mapping[int, tuple[int, ...]],
    *,
    d: int,
    l_max: int | None = None,
    cache_dir: str | Path,
    jobs: int = 1,
    laguerre_order: int = 96,
    progress=None,
) -> dict[int, DepthAveragedCodelength]:
    """Depth-averaged codelength for several count profiles, memory-frugally.

    Builds (or completes) the on-disk moment-table cache without holding the
    tables in RAM, then evaluates one depth at a time: for each ``L`` only
    that level's rows are loaded (memory-mapped), all profiles are evaluated
    against them, and the rows are dropped.  This is the API for full-corpus
    runs, where the complete table set exceeds memory.
    """

    if not profiles:
        return {}
    if l_max is None:
        l_max = default_l_max(d)
    all_r: set[int] = set()
    for partition in profiles.values():
        all_r.update(needed_r_values(partition))

    cache = build_tables_fast(
        max_L=l_max,
        r_values=all_r,
        laguerre_order=laguerre_order,
        cache_dir=cache_dir,
        jobs=jobs,
        materialize=False,
        progress=(
            None
            if progress is None
            else lambda done, total: progress(("tables", done, total), None)
        ),
    )
    assert isinstance(cache, TableCache)

    log2_q: dict[int, list[float]] = {n: [] for n in profiles}
    if jobs <= 1 or len(profiles) < 2 * jobs:
        from product_model_with_memory.layered import log_q_lambda_scan

        for L in range(1, l_max + 1):
            tables = None if L == 1 else cache.level_tables(L, all_r)
            for n, partition in profiles.items():
                if L == 1:
                    result = log_q_lambda_closed_l1(d=d, partition=partition)
                else:
                    result = log_q_lambda_scan(
                        d=d, L=L, partition=partition, tables=tables
                    )
                _check_eval(result, L, n)
                log2_q[n].append(result.log2_q)
            if progress is not None:
                progress(("depth", L, l_max), None)
            del tables
    else:
        # Parallel evaluation: profiles are independent, so each level's
        # work is split into one chunk per worker.  Workers open the (fully
        # built) cache themselves in the initializer and memory-map only
        # the level being evaluated, so parent and workers share pages.
        import multiprocessing as mp

        items = sorted(
            profiles.items(), key=lambda kv: len(kv[1]), reverse=True
        )
        chunks = [items[w::jobs] for w in range(jobs)]
        chunks = [c for c in chunks if c]
        # default start method, matching build_tables_fast (fork on Linux;
        # spawn on macOS, where the experiment scripts' __main__ guards
        # make re-import safe)
        with mp.Pool(
            processes=len(chunks),
            initializer=_init_eval_worker,
            initargs=(str(cache_dir), l_max, sorted(all_r), laguerre_order),
        ) as pool:
            for L in range(1, l_max + 1):
                for part in pool.starmap(
                    _eval_level_chunk,
                    [(d, L, chunk) for chunk in chunks],
                    chunksize=1,
                ):
                    for n, value in part:
                        log2_q[n].append(value)
                if progress is not None:
                    progress(("depth", L, l_max), None)

    results = {}
    for n, partition in profiles.items():
        values = log2_q[n]
        results[n] = DepthAveragedCodelength(
            d=d,
            n=sum(partition),
            l_max=l_max,
            log2_q_by_depth=tuple(values),
            log2_q_avg=float(_log2sumexp(values) - math.log2(l_max)),
        )
    return results


def _check_eval(result, L: int, n: int) -> None:
    if not result.converged:
        raise RuntimeError(f"L={L}, n={n}: {result.message}")
    if result.log2_q > 1e-9:
        raise RuntimeError(
            f"L={L}, n={n}: log2 q = {result.log2_q:.6g} > 0 is "
            "impossible for a sequence probability"
        )


_EVAL_CACHE = None
_EVAL_ALL_R = None


def _init_eval_worker(cache_dir: str, l_max: int, all_r, laguerre_order: int):
    """Open the (already fully built) table cache in a worker process."""

    global _EVAL_CACHE, _EVAL_ALL_R
    cache = build_tables_fast(
        max_L=l_max,
        r_values=set(all_r),
        laguerre_order=laguerre_order,
        cache_dir=cache_dir,
        jobs=1,
        materialize=False,
    )
    _EVAL_CACHE = cache
    _EVAL_ALL_R = set(all_r)


def _eval_level_chunk(d: int, L: int, chunk):
    """Evaluate one level for a chunk of (id, partition) pairs."""

    from product_model_with_memory.layered import log_q_lambda_scan

    tables = None if L == 1 else _EVAL_CACHE.level_tables(L, _EVAL_ALL_R)
    out = []
    for n, partition in chunk:
        if L == 1:
            result = log_q_lambda_closed_l1(d=d, partition=partition)
        else:
            result = log_q_lambda_scan(d=d, L=L, partition=partition, tables=tables)
        _check_eval(result, L, n)
        out.append((n, result.log2_q))
    return out


def _log2sumexp(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    m = float(np.max(arr))
    return m + math.log2(float(np.sum(np.exp((arr - m) * math.log(2.0)))))
