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
        source = _resolve_tables_source(None)
        if source == "universal":
            from product_model_with_memory.layered import (
                ProductMomentTables,
            )

            rs = sorted(needed_r_values(partition))
            prov = _provision_tables(
                source, l_max, set(rs), cache_dir, None, 1,
                laguerre_order, None,
            )
            log_phi = {}
            for L in range(2, l_max + 1):
                M = _provision_level_rows(prov, L, rs)
                log_phi.update({(L, r): M[i] for i, r in enumerate(rs)})
            tables = ProductMomentTables(
                max_L=l_max, max_r=rs[-1], r_values=tuple(rs),
                u_grid=np.asarray(prov["u_grid"], dtype=np.float64),
                log_phi=log_phi,
            )
        else:
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


def _resolve_tables_source(tables_source: str | None) -> str:
    """The moment-table source: "universal" (the permanent certified
    store; the default) or "cache" (the legacy per-run recursion
    cache, kept for regression comparison).  Overridable through the
    environment variable PMM_TABLES_SOURCE."""

    import os

    source = tables_source or os.environ.get("PMM_TABLES_SOURCE", "universal")
    if source not in ("universal", "cache"):
        raise ValueError(f"unknown tables_source {source!r}")
    return source


def _provision_tables(source, l_max, all_r, cache_dir, universal_path,
                      jobs, laguerre_order, progress):
    """Prepare the moment-table source: ensure universal columns exist
    (built in parallel with ``jobs``), or build/complete the legacy
    cache.  Returns a provider dict used by the evaluation loops."""

    if source == "universal":
        from product_model_with_memory.fast_tables import (
            default_grid_spec,
            grid_from_spec,
        )
        from product_model_with_memory.universal_tables import (
            ensure_universal_tables,
        )

        ut = ensure_universal_tables(universal_path)
        u_grid = grid_from_spec(default_grid_spec(
            max_L=l_max, r_max=max(all_r), u_max=35.0
        ))
        ut.build_columns(
            [(L, r) for L in range(2, l_max + 1) for r in sorted(all_r)],
            jobs=jobs, progress=progress,
        )
        return {"source": source, "ut": ut, "u_grid": u_grid, "cache": None}
    if cache_dir is None:
        raise ValueError('tables_source="cache" requires a cache_dir')
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
    return {"source": source, "ut": None,
            "u_grid": np.asarray(cache.u_grid, dtype=np.float64),
            "cache": cache}


def _provision_level_rows(prov, L, rs) -> np.ndarray:
    """The (len(rs), G) matrix of level-L log phi columns on the
    provider's u grid."""

    if prov["source"] == "universal":
        return np.stack([prov["ut"].log_phi(L, r, prov["u_grid"])
                         for r in rs])
    from product_model_with_memory.fast_tables import _cache_file

    G = len(prov["u_grid"])
    M = np.empty((len(rs), G))
    for i, r in enumerate(rs):
        arr = np.load(_cache_file(prov["cache"].cache_path, r), mmap_mode="r")
        M[i] = arr[L - 1]
    return M


def depth_averaged_codelength_profiles(
    profiles: Mapping[int, tuple[int, ...]],
    *,
    d: int,
    l_max: int | None = None,
    cache_dir: str | Path | None = None,
    jobs: int = 1,
    laguerre_order: int = 96,
    progress=None,
    tables_source: str | None = None,
    universal_path: str | Path | None = None,
) -> dict[int, DepthAveragedCodelength]:
    """Depth-averaged codelength for several count profiles, memory-frugally.

    Moment tables come from the permanent universal store by default
    (``tables_source="universal"``: certified values, built once ever,
    grown on demand at ``universal_path`` or the store's default
    location).  The legacy per-run cache path (``"cache"``, requires
    ``cache_dir``) remains available for regression comparison.
    Either way evaluation proceeds one depth at a time: only one
    level's rows are in memory, all profiles are evaluated against
    them, and the rows are dropped.
    """

    if not profiles:
        return {}
    if l_max is None:
        l_max = default_l_max(d)
    all_r: set[int] = set()
    for partition in profiles.values():
        all_r.update(needed_r_values(partition))
    source = _resolve_tables_source(tables_source)

    prov = _provision_tables(
        source, l_max, all_r, cache_dir, universal_path, jobs,
        laguerre_order, progress,
    )
    ut = prov["ut"]
    u_grid_universal = prov["u_grid"] if source == "universal" else None
    cache = prov["cache"]

    def _level_tables(L: int):
        if source == "universal":
            return ut.level_tables(L, all_r, u_grid_universal)
        return cache.level_tables(L, all_r)

    log2_q: dict[int, list[float]] = {n: [] for n in profiles}
    if jobs <= 1 or len(profiles) < 2 * jobs:
        from product_model_with_memory.layered import log_q_lambda_scan

        for L in range(1, l_max + 1):
            tables = None if L == 1 else _level_tables(L)
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
        # work is split into one chunk per worker.  The PARENT alone reads
        # the cache (one pass per level, as the serial path does) and
        # exposes the level's rows to workers through POSIX shared memory;
        # workers never touch the filesystem.  This matters on network
        # filesystems (cluster homes), where per-worker file access is
        # pathologically slow.
        import multiprocessing as mp
        from multiprocessing import shared_memory

        items = sorted(
            profiles.items(), key=lambda kv: len(kv[1]), reverse=True
        )
        chunks = [items[w::jobs] for w in range(jobs)]
        chunks = [c for c in chunks if c]
        rs = sorted(all_r)
        u_grid = np.asarray(
            u_grid_universal if source == "universal" else cache.u_grid,
            dtype=np.float64,
        )
        # default start method, matching build_tables_fast (fork on Linux;
        # spawn on macOS, where the experiment scripts' __main__ guards
        # make re-import safe)
        with mp.Pool(
            processes=len(chunks),
            initializer=_init_eval_worker,
            initargs=(u_grid, rs, l_max),
        ) as pool:
            for L in range(1, l_max + 1):
                shm = None
                try:
                    if L == 1:
                        args = [(d, L, None, None, chunk) for chunk in chunks]
                    else:
                        G = len(u_grid)
                        shm = shared_memory.SharedMemory(
                            create=True, size=len(rs) * G * 8
                        )
                        M = np.ndarray(
                            (len(rs), G), dtype=np.float64, buffer=shm.buf
                        )
                        M[:] = _provision_level_rows(prov, L, rs)
                        args = [
                            (d, L, shm.name, (len(rs), G), chunk)
                            for chunk in chunks
                        ]
                    for part in pool.starmap(
                        _eval_level_chunk, args, chunksize=1
                    ):
                        for n, value in part:
                            log2_q[n].append(value)
                finally:
                    if shm is not None:
                        shm.close()
                        shm.unlink()
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


def depth_averaged_codelength_families(
    families: Mapping,
    *,
    d: int,
    l_max: int | None = None,
    cache_dir: str | Path | None = None,
    jobs: int = 1,
    laguerre_order: int = 96,
    progress=None,
    tables_source: str | None = None,
    universal_path: str | Path | None = None,
) -> dict:
    """Codelengths for FAMILIES of profiles: each family is a base
    profile plus one-observation augmentations (one per count value c;
    c = 0 means a previously unseen symbol).  These are exactly the
    evaluations behind predictive rows (q_avg(profile + c)/q_avg), and
    within a family the grid integrand is shared (complexity notes,
    T3): each member costs O(G) on top of the base's O(G k).

    ``families`` maps a key to ``(base_partition, cs)``; the base may
    be empty (then every augmentation is the one-symbol profile, known
    in closed form).  Returns ``{key: (base_result_or_None,
    {c: aug_result})}`` with DepthAveragedCodelength values.
    """

    from product_model_with_memory.layered import (
        augmented_partition,
        log_q_lambda_scan_family,
    )

    if not families:
        return {}
    if l_max is None:
        l_max = default_l_max(d)
    clean: dict = {}
    all_r: set[int] = set()
    for key, (base, cs) in families.items():
        base = tuple(base)
        cs = tuple(sorted(set(int(c) for c in cs)))
        for c in cs:
            if c != 0 and c not in base:
                raise ValueError(f"family {key!r}: count {c} not in base")
        clean[key] = (base, cs)
        if base:
            all_r.update(needed_r_values(base))
        for c in cs:
            all_r.update(needed_r_values(augmented_partition(base, c)))

    base_logs: dict = {k: [] for k in clean}
    aug_logs: dict = {k: {c: [] for c in cs} for k, (b, cs) in clean.items()}
    any_scan = any(base for base, _ in clean.values()) and l_max > 1
    prov = (
        _provision_tables(
            _resolve_tables_source(tables_source), l_max, all_r,
            cache_dir, universal_path, jobs, laguerre_order, progress,
        )
        if any_scan else None
    )

    def _serial_level(L: int, tables) -> None:
        for key, (base, cs) in clean.items():
            if L == 1:
                if base:
                    base_logs[key].append(
                        log_q_lambda_closed_l1(d=d, partition=base).log2_q)
                for c in cs:
                    aug_logs[key][c].append(log_q_lambda_closed_l1(
                        d=d, partition=augmented_partition(base, c)).log2_q)
                continue
            if not base:  # every augmentation is the one-symbol profile
                for c in cs:
                    aug_logs[key][c].append(-math.log2(d))
                continue
            b_res, a_res = log_q_lambda_scan_family(
                d=d, L=L, base_partition=base, cs=cs, tables=tables)
            _check_eval(b_res, L, key)
            base_logs[key].append(b_res.log2_q)
            for c in cs:
                _check_eval(a_res[c], L, key)
                aug_logs[key][c].append(a_res[c].log2_q)

    if jobs <= 1 or len(clean) < 2 * jobs:
        for L in range(1, l_max + 1):
            tables = (
                None if (L == 1 or prov is None)
                else _family_level_tables(prov, L, sorted(all_r))
            )
            _serial_level(L, tables)
            if progress is not None:
                progress(("depth", L, l_max), None)
            del tables
    else:
        import multiprocessing as mp
        from multiprocessing import shared_memory

        items = sorted(
            clean.items(),
            key=lambda kv: len(kv[1][0]) + len(kv[1][1]), reverse=True,
        )
        chunks = [items[w::jobs] for w in range(jobs)]
        chunks = [c for c in chunks if c]
        rs = sorted(all_r)
        u_grid = (np.asarray(prov["u_grid"], dtype=np.float64)
                  if prov is not None else np.zeros(1))
        with mp.Pool(
            processes=len(chunks),
            initializer=_init_eval_worker,
            initargs=(u_grid, rs, l_max),
        ) as pool:
            for L in range(1, l_max + 1):
                shm = None
                try:
                    if L == 1 or prov is None:
                        args = [(d, L, None, None, chunk) for chunk in chunks]
                    else:
                        G = len(u_grid)
                        shm = shared_memory.SharedMemory(
                            create=True, size=len(rs) * G * 8)
                        M = np.ndarray((len(rs), G), dtype=np.float64,
                                       buffer=shm.buf)
                        M[:] = _provision_level_rows(prov, L, rs)
                        args = [(d, L, shm.name, (len(rs), G), chunk)
                                for chunk in chunks]
                    for part in pool.starmap(
                            _eval_family_level_chunk, args, chunksize=1):
                        for key, b_val, a_vals in part:
                            if b_val is not None:
                                base_logs[key].append(b_val)
                            for c, v in a_vals.items():
                                aug_logs[key][c].append(v)
                finally:
                    if shm is not None:
                        shm.close()
                        shm.unlink()
                if progress is not None:
                    progress(("depth", L, l_max), None)

    out: dict = {}
    for key, (base, cs) in clean.items():
        def _mk(partition, values):
            return DepthAveragedCodelength(
                d=d, n=sum(partition), l_max=l_max,
                log2_q_by_depth=tuple(values),
                log2_q_avg=float(_log2sumexp(values) - math.log2(l_max)),
            )
        base_res = _mk(base, base_logs[key]) if base else None
        out[key] = (base_res, {
            c: _mk(augmented_partition(base, c), aug_logs[key][c])
            for c in cs
        })
    return out


def _family_level_tables(prov, L: int, rs):
    """ProductMomentTables for one level from the provider."""

    from product_model_with_memory.layered import ProductMomentTables

    M = _provision_level_rows(prov, L, rs)
    return ProductMomentTables(
        max_L=L, max_r=rs[-1], r_values=tuple(rs),
        u_grid=np.asarray(prov["u_grid"], dtype=np.float64),
        log_phi={(L, r): M[i] for i, r in enumerate(rs)},
    )


def _eval_family_level_chunk(d: int, L: int, shm_name, shape, chunk):
    """Worker: one level for a chunk of (key, (base, cs)) families."""

    from product_model_with_memory.layered import (
        ProductMomentTables,
        augmented_partition,
        log_q_lambda_scan_family,
    )

    shm = None
    tables = None
    try:
        if L > 1 and shm_name is not None:
            shm = _attach_shm_untracked(shm_name)
            M = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)
            tables = ProductMomentTables(
                max_L=_EVAL_LMAX,
                max_r=_EVAL_RS[-1],
                r_values=tuple(_EVAL_RS),
                u_grid=_EVAL_UGRID,
                log_phi={(L, r): M[i] for i, r in enumerate(_EVAL_RS)},
            )
        out = []
        for key, (base, cs) in chunk:
            if L == 1:
                b_val = (log_q_lambda_closed_l1(d=d, partition=base).log2_q
                         if base else None)
                a_vals = {c: log_q_lambda_closed_l1(
                    d=d, partition=augmented_partition(base, c)).log2_q
                    for c in cs}
            elif not base:
                b_val = None
                a_vals = {c: -math.log2(d) for c in cs}
            else:
                b_res, a_res = log_q_lambda_scan_family(
                    d=d, L=L, base_partition=base, cs=cs, tables=tables)
                _check_eval(b_res, L, key)
                for c in cs:
                    _check_eval(a_res[c], L, key)
                b_val = b_res.log2_q
                a_vals = {c: a_res[c].log2_q for c in cs}
            out.append((key, b_val, a_vals))
        return out
    finally:
        if shm is not None:
            del tables
            shm.close()


def _check_eval(result, L: int, n) -> None:
    if not result.converged:
        raise RuntimeError(f"L={L}, n={n}: {result.message}")
    if result.log2_q > 1e-9:
        raise RuntimeError(
            f"L={L}, n={n}: log2 q = {result.log2_q:.6g} > 0 is "
            "impossible for a sequence probability"
        )


_EVAL_UGRID = None
_EVAL_RS = None
_EVAL_LMAX = None


def _init_eval_worker(u_grid, rs, l_max: int):
    """Store the (small) shared evaluation context in the worker process."""

    global _EVAL_UGRID, _EVAL_RS, _EVAL_LMAX
    _EVAL_UGRID = u_grid
    _EVAL_RS = rs
    _EVAL_LMAX = l_max


def _attach_shm_untracked(name):
    """Attach to an existing shared-memory block WITHOUT registering it
    with this process's resource tracker.

    The parent owns the block and unlinks it; if workers also register
    it, every worker's tracker attempts a second cleanup at shutdown and
    warns.  Python >= 3.13 has track=False for exactly this; on older
    versions the registration is reverted manually.
    """

    from multiprocessing import shared_memory

    try:
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:  # Python < 3.13
        shm = shared_memory.SharedMemory(name=name)
        try:
            from multiprocessing import resource_tracker

            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:  # noqa: BLE001 - cosmetic only; never fail eval
            pass
        return shm


def _eval_level_chunk(d: int, L: int, shm_name, shape, chunk):
    """Evaluate one level for a chunk of (id, partition) pairs.

    For L >= 2 the level's moment-table rows are read from the shared
    memory block the parent filled; this function does no file I/O.
    """

    from product_model_with_memory.fast_tables import ProductMomentTables
    from product_model_with_memory.layered import log_q_lambda_scan

    shm = None
    tables = None
    try:
        if L > 1:
            shm = _attach_shm_untracked(shm_name)
            M = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)
            tables = ProductMomentTables(
                max_L=_EVAL_LMAX,
                max_r=_EVAL_RS[-1],
                r_values=tuple(_EVAL_RS),
                u_grid=_EVAL_UGRID,
                log_phi={(L, r): M[i] for i, r in enumerate(_EVAL_RS)},
            )
        out = []
        for n, partition in chunk:
            if L == 1:
                result = log_q_lambda_closed_l1(d=d, partition=partition)
            else:
                result = log_q_lambda_scan(
                    d=d, L=L, partition=partition, tables=tables
                )
            _check_eval(result, L, n)
            out.append((n, result.log2_q))
        return out
    finally:
        if shm is not None:
            del tables
            shm.close()


def _log2sumexp(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    m = float(np.max(arr))
    return m + math.log2(float(np.sum(np.exp((arr - m) * math.log(2.0)))))
