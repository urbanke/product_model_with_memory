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
import os
import time
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
        return prov["ut"].log_phi_matrix(L, rs, prov["u_grid"])
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
    for key, partition in profiles.items():
        if len(partition) > d:
            raise ValueError(
                f"profile {key!r} has {len(partition)} distinct symbols "
                f"but the alphabet size is d={d}; the model is undefined "
                "for d smaller than the number of observed symbols"
            )
        all_r.update(needed_r_values(partition))
    source = _resolve_tables_source(tables_source)

    prov = _provision_tables(
        source, l_max, all_r, cache_dir, universal_path, jobs,
        laguerre_order, progress,
    )
    ut = prov["ut"]
    u_grid_universal = prov["u_grid"] if source == "universal" else None
    cache = prov["cache"]

    r_of = {n: set(needed_r_values(p)) for n, p in profiles.items()}

    def _level_r(active_keys) -> list[int]:
        """Only the orders the still-active profiles need.  Once the
        heavy profiles have collapsed the survivors are the light rows,
        whose counts are small, so the column set shrinks sharply --- and
        provisioning is what a level costs when many profiles share it."""

        want: set[int] = set()
        for k in active_keys:
            want |= r_of[k]
        return sorted(want) if want else sorted(all_r)

    def _level_tables(L: int, rs_L):
        if source == "universal":
            return ut.level_tables(L, rs_L, u_grid_universal)
        return cache.level_tables(L, rs_L)

    log2_q: dict[int, list[float]] = {n: [] for n in profiles}
    window = _LevelWindow(profiles)
    # Wall time per phase of the level loop, printed under PMM_TIMING.
    # Three scheduling changes were made against guesses about where the
    # time goes and all three did nothing (1 August); the scaling fit
    # says a third of the run is serial, and this says which third.
    _phase = {"plan": 0.0, "shm": 0.0, "fill": 0.0, "eval": 0.0,
              "collect": 0.0, "shm_bytes": 0}
    # The parallel path pays off whenever the universal store is the
    # source, EVEN for few profiles: preparing each level's rows
    # (thousands of column reads + interpolations) dominates and is
    # split over the pool regardless of the profile count.
    if jobs <= 1 or (len(profiles) < 2 * jobs
                     and prov["source"] != "universal"):
        from product_model_with_memory.layered import log_q_lambda_scan

        for L in range(1, l_max + 1):
            if L > 1 and not window.any_active():
                break
            live = [n for n in profiles if n in window]
            tables = None if L == 1 else _level_tables(L, _level_r(live))
            for n, partition in profiles.items():
                if n not in window:
                    log2_q[n].append(-math.inf)
                    continue
                if L == 1:
                    result = log_q_lambda_closed_l1(d=d, partition=partition)
                else:
                    result = log_q_lambda_scan(
                        d=d, L=L, partition=partition, tables=tables
                    )
                _check_eval(result, L, n)
                log2_q[n].append(result.log2_q)
                window.observe(n, result.log2_q, L)
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
        rs = sorted(all_r)
        u_grid = np.asarray(
            u_grid_universal if source == "universal" else cache.u_grid,
            dtype=np.float64,
        )
        # default start method, matching build_tables_fast (fork on Linux;
        # spawn on macOS, where the experiment scripts' __main__ guards
        # make re-import safe)
        with mp.Pool(
            processes=jobs,
            initializer=_init_eval_worker,
            initargs=(u_grid, rs, l_max,
                      str(prov["ut"].path) if prov and prov.get("ut")
                      else None,
                      dict(profiles)),
        ) as pool:
            shm = None
            shm_cap = 0
            for L in range(1, l_max + 1):
                if L > 1 and not window.any_active():
                    break
                # rebuild the chunks each level: profiles whose terms
                # have collapsed are no longer evaluated.  The chunks
                # carry KEYS; the workers hold the partitions from
                # pool startup.
                #
                # There are several times more chunks than workers so
                # the pool can hand work out dynamically.  With one
                # chunk per worker a level ended when the SLOWEST chunk
                # did, and the parent was measured idle in posix.read
                # for 32 of 44 seconds (1 August); profile costs vary
                # by orders of magnitude and no static split fixes that.
                nchunks = jobs * _CHUNKS_PER_JOB
                _t = time.perf_counter()
                active = [kv[0] for kv in items if kv[0] in window]
                chunks = [c for c in
                          (active[w::nchunks] for w in range(nchunks)) if c]
                rs_L = _level_r(active)
                _phase["plan"] += time.perf_counter() - _t
                try:
                    if L == 1:
                        args = [(d, L, None, None, chunk, None, True)
                                for chunk in chunks]
                    else:
                        G = len(u_grid)
                        nr = len(rs_L)
                        _t = time.perf_counter()
                        # ONE block for the whole run, grown when a level
                        # needs more.  Creating and unlinking it per level
                        # meant 39 GiB of fresh anonymous pages over 54
                        # levels of the enwik8 subword run (1 August),
                        # every page faulted in by the parent and faulted
                        # again by each of twelve workers; page-fault
                        # handling serialises on the address space, which
                        # is why the run scaled as though a third of it
                        # were single-threaded.
                        need = nr * G * 8
                        if shm is None or shm_cap < need:
                            if shm is not None:
                                shm.close()
                                shm.unlink()
                            shm = shared_memory.SharedMemory(
                                create=True, size=need)
                            shm_cap = need
                            _phase["shm_bytes"] += need
                        M = np.ndarray(
                            (nr, G), dtype=np.float64, buffer=shm.buf
                        )
                        _phase["shm"] += time.perf_counter() - _t
                        _t = time.perf_counter()
                        if prov["source"] == "universal":
                            # likewise for the fill: a column whose
                            # stored start lies right of the query grid
                            # costs a series evaluation while its
                            # neighbours cost nothing, so equal row
                            # bands are NOT equal work
                            nb = min(nr, jobs * _CHUNKS_PER_JOB)
                            bnds = np.linspace(0, nr, nb + 1).astype(int)
                            pool.starmap(_fill_level_chunk, [
                                (shm.name, (nr, G), L,
                                 int(bnds[w]), int(bnds[w + 1]), rs_L)
                                for w in range(nb) if bnds[w + 1] > bnds[w]
                            ], chunksize=1)
                        else:
                            M[:] = _provision_level_rows(prov, L, rs_L)
                        _phase["fill"] += time.perf_counter() - _t
                        args = [
                            (d, L, shm.name, (nr, G), chunk, rs_L, True)
                            for chunk in chunks
                        ]
                    _t = time.perf_counter()
                    parts = pool.starmap(_eval_level_chunk, args, chunksize=1)
                    _phase["eval"] += time.perf_counter() - _t
                    _t = time.perf_counter()
                    for part in parts:
                        for n, value in part:
                            log2_q[n].append(value)
                            window.observe(n, value, L)
                    for n in profiles:            # levels not evaluated
                        if len(log2_q[n]) < L:
                            log2_q[n].append(-math.inf)
                    _phase["collect"] += time.perf_counter() - _t
                finally:
                    pass
                if progress is not None:
                    progress(("depth", L, l_max), None)
            if shm is not None:
                _t = time.perf_counter()
                shm.close()
                shm.unlink()
                _phase["shm"] += time.perf_counter() - _t

    if os.environ.get("PMM_TIMING"):
        gib = _phase.pop("shm_bytes", 0) / 2 ** 30
        print("  timing: " + "  ".join(f"{k}={v:.1f}s"
                                       for k, v in _phase.items())
              + f"   ({gib:.1f} GiB of shared blocks created)", flush=True)

    results = {}
    for n, partition in profiles.items():
        values = _pad(log2_q[n], l_max)
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
        if len(base) > d:
            raise ValueError(
                f"family {key!r}: base profile has {len(base)} distinct "
                f"symbols but the alphabet size is d={d}"
            )
        for c in cs:
            if c != 0 and c not in base:
                raise ValueError(f"family {key!r}: count {c} not in base")
            if c == 0 and len(base) >= d:
                raise ValueError(
                    f"family {key!r}: no unseen symbol exists (saturated "
                    f"row, {len(base)} of d={d} symbols observed)")
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

    # a family stays active while its base does (or, when it has no
    # base, while any augmentation does): the members share an integrand
    # and their peaks coincide to under a grid step, so they collapse
    # together
    window = _LevelWindow([k for k, (b, _) in clean.items() if b])

    r_of: dict = {}
    for key, (base, cs) in clean.items():
        want = set(needed_r_values(base)) if base else set()
        for c in cs:
            want |= set(needed_r_values(augmented_partition(base, c)))
        r_of[key] = want

    def _level_r(active_keys) -> list[int]:
        want: set[int] = set()
        for k in active_keys:
            want |= r_of[k]
        return sorted(want) if want else sorted(all_r)

    def _observe(key, b_val, a_vals, L) -> None:
        v = b_val if b_val is not None else (
            max(a_vals.values()) if a_vals else -math.inf)
        window.observe(key, v, L)

    def _skip(key, L) -> None:
        base, cs = clean[key]
        if base:
            base_logs[key].append(-math.inf)
        for c in cs:
            aug_logs[key][c].append(-math.inf)

    def _serial_level(L: int, tables) -> None:
        for key, (base, cs) in clean.items():
            if key not in window:
                _skip(key, L)
                continue
            if L == 1:
                b_val = (log_q_lambda_closed_l1(d=d, partition=base).log2_q
                         if base else None)
                if b_val is not None:
                    base_logs[key].append(b_val)
                a_vals = {}
                for c in cs:
                    a_vals[c] = log_q_lambda_closed_l1(
                        d=d, partition=augmented_partition(base, c)).log2_q
                    aug_logs[key][c].append(a_vals[c])
                _observe(key, b_val, a_vals, L)
                continue
            if not base:  # every augmentation is the one-symbol profile
                for c in cs:
                    aug_logs[key][c].append(-math.log2(d))
                continue
            b_res, a_res = log_q_lambda_scan_family(
                d=d, L=L, base_partition=base, cs=cs, tables=tables)
            _check_eval(b_res, L, key)
            base_logs[key].append(b_res.log2_q)
            a_vals = {}
            for c in cs:
                _check_eval(a_res[c], L, key)
                a_vals[c] = a_res[c].log2_q
                aug_logs[key][c].append(a_vals[c])
            _observe(key, b_res.log2_q, a_vals, L)

    if jobs <= 1 or (len(clean) < 2 * jobs
                     and (prov is None or prov["source"] != "universal")):
        for L in range(1, l_max + 1):
            if L > 1 and not window.any_active():
                break
            tables = (
                None if (L == 1 or prov is None)
                else _family_level_tables(
                    prov, L, _level_r([k for k in clean if k in window]))
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
        rs = sorted(all_r)
        u_grid = (np.asarray(prov["u_grid"], dtype=np.float64)
                  if prov is not None else np.zeros(1))
        with mp.Pool(
            processes=jobs,
            initializer=_init_eval_worker,
            initargs=(u_grid, rs, l_max,
                      str(prov["ut"].path) if prov and prov.get("ut")
                      else None),
        ) as pool:
            for L in range(1, l_max + 1):
                if L > 1 and not window.any_active():
                    break
                active = [kv for kv in items if kv[0] in window]
                chunks = [c for c in (active[w::jobs] for w in range(jobs))
                          if c]
                rs_L = _level_r([kv[0] for kv in active])
                shm = None
                try:
                    if L == 1 or prov is None:
                        args = [(d, L, None, None, chunk) for chunk in chunks]
                    else:
                        G = len(u_grid)
                        nr = len(rs_L)
                        shm = shared_memory.SharedMemory(
                            create=True, size=nr * G * 8)
                        M = np.ndarray((nr, G), dtype=np.float64,
                                       buffer=shm.buf)
                        if prov["source"] == "universal":
                            bnds = np.linspace(0, nr, jobs + 1).astype(int)
                            pool.starmap(_fill_level_chunk, [
                                (shm.name, (nr, G), L,
                                 int(bnds[w]), int(bnds[w + 1]), rs_L)
                                for w in range(jobs)
                            ], chunksize=1)
                        else:
                            M[:] = _provision_level_rows(prov, L, rs_L)
                        args = [(d, L, shm.name, (nr, G), chunk, rs_L)
                                for chunk in chunks]
                    for part in pool.starmap(
                            _eval_family_level_chunk, args, chunksize=1):
                        for key, b_val, a_vals in part:
                            if b_val is not None:
                                base_logs[key].append(b_val)
                            for c, v in a_vals.items():
                                aug_logs[key][c].append(v)
                            _observe(key, b_val, a_vals, L)
                    for key, (base, cs) in clean.items():   # not evaluated
                        miss = -math.inf if base else -math.log2(d)
                        if base and len(base_logs[key]) < L:
                            base_logs[key].append(miss)
                        for c in cs:
                            if len(aug_logs[key][c]) < L:
                                aug_logs[key][c].append(miss)
                finally:
                    if shm is not None:
                        shm.close()
                        shm.unlink()
                if progress is not None:
                    progress(("depth", L, l_max), None)

    out: dict = {}
    for key, (base, cs) in clean.items():
        # A family with no base is CONSTANT in the level (every
        # augmentation is the one-symbol profile, -log2 d for L >= 2),
        # so if the loop stopped early those levels are filled with the
        # value they would have had, not with -inf.  Truncation must
        # never change a value, only skip computing one.
        fill = -math.inf if base else -math.log2(d)

        def _mk(partition, values, fill=fill):
            if len(values) < l_max:
                values = values + [fill] * (l_max - len(values))
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
    return ProductMomentTables.from_matrix(
        max_L=L, L=L, r_values=rs,
        u_grid=np.asarray(prov["u_grid"], dtype=np.float64), matrix=M)


def _eval_family_level_chunk(d: int, L: int, shm_name, shape, chunk,
                             rs=None):
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
            tables = ProductMomentTables.from_matrix(
                max_L=_EVAL_LMAX, L=L,
                r_values=_EVAL_RS if rs is None else rs,
                u_grid=_EVAL_UGRID, matrix=M)
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


# ---------------------------------------------------------------- levels
# Truncating the sum over depths (complexity notes, T2(1)).  Measured on
# text8 at V = 1024, L_max = 33: the depth average is carried by ONE
# level of 32 for the heaviest profiles and by at most 15 for the
# lightest, and the level sets are contiguous --- so walking L upward
# and stopping once a profile's terms have collapsed loses nothing.
#
# The stopping rule is deliberately crude.  A profile is dropped only
# after its term has fallen DROP bits below its own running maximum AND
# has decreased for PATIENCE consecutive levels.  With DROP = 80 the
# worst-case bound on the discarded tail --- every remaining level as
# large as the last one evaluated --- is L_max * 2^-80, beneath float64
# resolution and far beneath the 1e-7 nat accuracy of the moment tables,
# so the truncation is exact for every purpose this code has.  The cost
# of that caution is small: on the measured curves the terms fall by
# hundreds of bits within a few levels of the mode, so DROP = 80 stops
# at the same level as DROP = 40 for the heavy profiles.
#
# PMM_NO_TRUNCATE=1 restores the full sweep (use it to verify);
# PMM_LEVEL_DROP overrides the threshold.
LEVEL_DROP_BITS = 80.0
LEVEL_PATIENCE = 3


class _LevelWindow:
    """Which keys still need evaluating as the level increases."""

    def __init__(self, keys, *, drop: float | None = None,
                 patience: int = LEVEL_PATIENCE):
        import os

        self.enabled = os.environ.get(
            "PMM_NO_TRUNCATE", "").lower() not in ("1", "true", "yes")
        self.drop = float(os.environ.get(
            "PMM_LEVEL_DROP",
            LEVEL_DROP_BITS if drop is None else drop))
        self.patience = patience
        self.active = set(keys)
        self._best = {k: -math.inf for k in keys}
        self._last = {k: math.inf for k in keys}
        self._falling = {k: 0 for k in keys}
        self.stopped_at: dict = {}

    def __contains__(self, key) -> bool:
        # keys that were never registered are never truncated: their
        # terms do not decay with the level (a family with no base is a
        # constant -log2 d), so they must not hold the loop open either
        return ((not self.enabled) or key not in self._best
                or key in self.active)

    def any_active(self) -> bool:
        return (not self.enabled) or bool(self.active)

    def observe(self, key, value: float, L: int) -> None:
        if not self.enabled or key not in self.active:
            return
        if value < self._last[key]:
            self._falling[key] += 1
        else:
            self._falling[key] = 0
        self._last[key] = value
        if value > self._best[key]:
            self._best[key] = value
        if (self._falling[key] >= self.patience
                and self._best[key] - value > self.drop):
            self.active.discard(key)
            self.stopped_at[key] = L


def _pad(values: list, l_max: int) -> list:
    """A skipped level contributes nothing, and -inf keeps index L-1
    meaningful for posterior/mode/per-depth queries."""

    if len(values) < l_max:
        return values + [-math.inf] * (l_max - len(values))
    return values


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
_EVAL_STORE_PATH = None
_EVAL_STORE = None


_EVAL_PROFILES = None


# Chunks per worker per level.  Six was tried on the enwik8 subword run
# (1 August) on the theory that a level ended when its slowest chunk
# did; wall time did not move and CPU rose from 388 s to 416 s, so the
# extra shared-memory attach per task costs more than the balancing
# saves.  Back to one; the knob stays for machines with different core
# counts.
_CHUNKS_PER_JOB = int(os.environ.get("PMM_CHUNKS_PER_JOB", "1"))


def _init_eval_worker(u_grid, rs, l_max: int, store_path=None,
                      profiles=None):
    """Store the shared evaluation context in the worker process.

    ``profiles`` is sent ONCE here rather than per level.  The chunks
    used to carry the partitions themselves, so a run re-pickled every
    profile at every one of the 54 levels to every one of the twelve
    workers --- tens of millions of integers per run, all of it data
    the worker had already seen.  With the profiles resident the level
    chunks are lists of keys."""

    global _EVAL_UGRID, _EVAL_RS, _EVAL_LMAX, _EVAL_STORE_PATH, _EVAL_STORE
    global _EVAL_PROFILES
    _EVAL_UGRID = u_grid
    _EVAL_RS = rs
    _EVAL_LMAX = l_max
    _EVAL_STORE_PATH = store_path
    _EVAL_STORE = None
    _EVAL_PROFILES = profiles


def _eval_store():
    """Lazily opened read-only view of the universal store in a worker."""

    global _EVAL_STORE
    if _EVAL_STORE is None:
        from product_model_with_memory.universal_tables import UniversalTables

        _EVAL_STORE = UniversalTables(_EVAL_STORE_PATH, read_only=True)
    return _EVAL_STORE


def _fill_level_chunk(shm_name, shape, L: int, i0: int, i1: int,
                      rs=None):
    """Worker: fill rows i0..i1 of the level's shared-memory matrix
    from the universal store (read-only; concurrent-safe).  Added
    July 30: the parent used to do ALL of this serially, leaving the
    workers idle for most of each level --- the single-busy-thread
    symptom during evaluation."""

    shm = _attach_shm_cached(shm_name)
    M = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)
    ut = _eval_store()
    if i1 > i0:
        use = _EVAL_RS if rs is None else rs
        M[i0:i1] = ut.log_phi_matrix(L, use[i0:i1], _EVAL_UGRID)
    return i1 - i0


def _nbytes(shape) -> int:
    return int(shape[0]) * int(shape[1]) * 8


_ATTACHED: dict = {}


def _attach_shm_cached(name):
    """Attach once per worker process and keep the mapping.

    The parent now keeps ONE block for the whole run, so a worker that
    re-attached per task re-mapped and re-faulted the same pages 54
    times per run.  Any stale mapping is dropped when the name changes
    (the parent grows the block when a level needs more rows)."""

    shm = _ATTACHED.get(name)
    if shm is None:
        for old_name, old in list(_ATTACHED.items()):
            old.close()
            del _ATTACHED[old_name]
        shm = _attach_shm_untracked(name)
        _ATTACHED[name] = shm
    return shm


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


def _eval_level_chunk(d: int, L: int, shm_name, shape, chunk,
                      rs=None, keys_only=False):
    """Evaluate one level for a chunk of (id, partition) pairs.

    For L >= 2 the level's moment-table rows are read from the shared
    memory block the parent filled; this function does no file I/O.
    """

    from product_model_with_memory.fast_tables import ProductMomentTables
    from product_model_with_memory.layered import log_q_lambda_scan

    tables = None
    if True:
        if L > 1:
            shm = _attach_shm_cached(shm_name)
            M = np.ndarray(shape, dtype=np.float64,
                           buffer=shm.buf)
            tables = ProductMomentTables.from_matrix(
                max_L=_EVAL_LMAX, L=L,
                r_values=_EVAL_RS if rs is None else rs,
                u_grid=_EVAL_UGRID, matrix=M)
        out = []
        pairs = (((n, _EVAL_PROFILES[n]) for n in chunk) if keys_only
                 else chunk)
        for n, partition in pairs:
            if L == 1:
                result = log_q_lambda_closed_l1(d=d, partition=partition)
            else:
                result = log_q_lambda_scan(
                    d=d, L=L, partition=partition, tables=tables
                )
            _check_eval(result, L, n)
            out.append((n, result.log2_q))
        del tables            # drop the views before the caller returns
        return out


def _log2sumexp(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    m = float(np.max(arr))
    return m + math.log2(float(np.sum(np.exp((arr - m) * math.log(2.0)))))
