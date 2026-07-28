"""Fast batched construction of the product-exponential moment tables.

Same recursion as :func:`product_model_with_memory.layered.
build_selected_product_moment_tables` (Appendix B layer recursion
``phi_r^(l)(t) = E[E^r phi_r^(l-1)(t E)]`` on the grid ``u = log t``), with
three practical extensions for real-corpus profiles, all validated against
the reference implementation in ``tests/test_layered.py``:

1. The interpolation indices/weights for the Gauss-Laguerre quadrature
   points are precomputed once (via ``searchsorted``, so any strictly
   increasing grid works) and reused for every layer and moment order.

2. The grid adapts to the heavy-count / deep-layer regime (the analogue of
   the paper's Appendix B.2): the left edge sits at
   ``u_min = min(-70, -(max_L * ln(r_max + 1)) - 40)``, where the analytic
   ``t -> 0`` fill ``Gamma(r+1)^(l-1)`` is genuinely valid, and the segment
   left of ``-75`` uses a coarser spacing (the functions there are smooth
   transitions on the scale of whole units in ``u``), which shrinks tables
   several-fold.

3. Tables stream to an on-disk cache (one array of shape
   ``(max_L, len(grid))`` per moment order ``r``).  ``materialize=False``
   builds the cache without holding anything in RAM - essential at
   full-corpus scale, where the complete table set exceeds memory - and
   :class:`TableCache` then serves single-depth views via ``numpy`` memory
   mapping.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.polynomial.laguerre import laggauss
from numpy.typing import NDArray

from product_model_with_memory.layered import ProductMomentTables

FloatArray = NDArray[np.float64]

DEFAULT_GRID_STEP = 105.0 / 16_000.0  # the reference grid's spacing
COARSE_BREAK = -75.0
COARSE_STEP = 0.04

# A grid spec is a plain tuple so worker processes can rebuild it cheaply:
#   ("uniform", u_min, u_max, u_points)
#   ("piecewise", u_min, coarse_break, coarse_step, u_max, fine_step)
GridSpec = tuple


def grid_from_spec(spec: GridSpec) -> FloatArray:
    if spec[0] == "uniform":
        _, u_min, u_max, u_points = spec
        return np.linspace(u_min, u_max, int(u_points), dtype=np.float64)
    if spec[0] == "piecewise":
        _, u_min, coarse_break, coarse_step, u_max, fine_step = spec
        fine_points = int(math.ceil((u_max - coarse_break) / fine_step)) + 1
        fine = np.linspace(coarse_break, u_max, fine_points, dtype=np.float64)
        if u_min >= coarse_break:
            return fine
        coarse_points = int(math.ceil((coarse_break - u_min) / coarse_step))
        coarse = np.linspace(
            u_min, coarse_break, coarse_points + 1, dtype=np.float64
        )[:-1]
        return np.concatenate([coarse, fine])
    raise ValueError(f"unknown grid spec {spec!r}")


def default_grid_spec(
    *,
    max_L: int,
    r_max: int,
    u_max: float = 35.0,
    fine_step: float = DEFAULT_GRID_STEP,
) -> GridSpec:
    u_min = min(-70.0, -(max_L * math.log(r_max + 1.0)) - 40.0)
    if u_min >= COARSE_BREAK:
        u_points = int(math.ceil((u_max - u_min) / fine_step)) + 1
        return ("uniform", u_min, u_max, u_points)
    return ("piecewise", u_min, COARSE_BREAK, COARSE_STEP, u_max, fine_step)


def _spec_key(spec: GridSpec, max_L: int, laguerre_order: int) -> str:
    text = f"v2|{max_L}|{laguerre_order}|" + "|".join(str(x) for x in spec)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class TableCache:
    """Handle to a fully built on-disk table cache."""

    cache_path: Path
    grid_spec: GridSpec
    max_L: int
    r_values: tuple[int, ...]

    @property
    def u_grid(self) -> FloatArray:
        return grid_from_spec(self.grid_spec)

    def level_tables(
        self, L: int, r_values: Iterable[int] | None = None
    ) -> ProductMomentTables:
        """Moment tables holding ONLY level ``L`` rows (loaded via mmap)."""

        selected = tuple(sorted(set(r_values))) if r_values else self.r_values
        u_grid = self.u_grid
        log_phi: dict[tuple[int, int], FloatArray] = {}
        for r in selected:
            arr = np.load(_cache_file(self.cache_path, r), mmap_mode="r")
            log_phi[(L, r)] = np.array(arr[L - 1])
        return ProductMomentTables(
            max_L=self.max_L,
            max_r=max(selected),
            r_values=selected,
            u_grid=u_grid,
            log_phi=log_phi,
        )


def build_tables_fast(
    *,
    max_L: int,
    r_values: Iterable[int],
    u_min: float | None = None,
    u_max: float = 35.0,
    u_points: int | None = None,
    laguerre_order: int = 96,
    chunk_size: int = 4_096,
    cache_dir: str | Path | None = None,
    jobs: int = 1,
    materialize: bool = True,
    progress=None,
) -> ProductMomentTables | TableCache:
    """Build (or complete) moment tables; batched, cached, parallel.

    With explicit ``u_min``/``u_points`` a uniform grid is used (matching the
    reference builder's conventions); otherwise the adaptive piecewise grid
    of :func:`default_grid_spec` applies.  ``jobs > 1`` builds independent
    per-``r`` recursions in worker processes.  ``materialize=False``
    (requires ``cache_dir``) streams everything to disk and returns a
    :class:`TableCache`; nothing is held in RAM.
    """

    if max_L < 1:
        raise ValueError("max_L must be positive")
    selected_r = tuple(sorted({int(r) for r in r_values}))
    if not selected_r or any(r < 0 for r in selected_r):
        raise ValueError("r_values must be non-empty and non-negative")

    if u_min is not None or u_points is not None:
        if u_min is None:
            u_min = min(
                -70.0, -(max_L * math.log(max(selected_r) + 1.0)) - 40.0
            )
        if u_points is None:
            u_points = int(math.ceil((u_max - u_min) / DEFAULT_GRID_STEP)) + 1
        spec: GridSpec = ("uniform", float(u_min), float(u_max), int(u_points))
    else:
        spec = default_grid_spec(max_L=max_L, r_max=max(selected_r), u_max=u_max)

    if not materialize and cache_dir is None:
        raise ValueError("materialize=False requires a cache_dir")

    cache_path = None
    if cache_dir is not None:
        key = _spec_key(spec, max_L, laguerre_order)
        cache_path = Path(cache_dir) / f"moment_tables_{key}"
        cache_path.mkdir(parents=True, exist_ok=True)

    missing = [
        r
        for r in selected_r
        if (cached := _load_cached(cache_path, r)) is None
        or cached.shape[0] < max_L
    ]

    if missing:
        worker_args = (
            (r, max_L, spec, laguerre_order, chunk_size,
             str(cache_path) if cache_path else None)
            for r in missing
        )
        done = 0
        if jobs > 1 and cache_path is not None:
            import concurrent.futures

            with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
                # consume lazily; workers persist results to the cache
                for _ in pool.map(_build_worker_cached, worker_args, chunksize=4):
                    done += 1
                    if progress is not None:
                        progress(done, len(missing))
        else:
            setup = _interp_setup(grid_from_spec(spec), laguerre_order)
            for args in worker_args:
                stack = _build_one_r_with_setup(args[0], max_L, setup, chunk_size,
                                                _load_cached(cache_path, args[0]))
                if cache_path is not None:
                    _store_cached(cache_path, args[0], stack)
                else:
                    _MEMORY_ONLY.setdefault(id(setup), {})[args[0]] = stack
                done += 1
                if progress is not None:
                    progress(done, len(missing))

    if not materialize:
        return TableCache(
            cache_path=cache_path,
            grid_spec=spec,
            max_L=max_L,
            r_values=selected_r,
        )

    u_grid = grid_from_spec(spec)
    tables: dict[tuple[int, int], FloatArray] = {}
    for r in selected_r:
        stack = _load_cached(cache_path, r)
        if stack is None:  # memory-only build (no cache dir)
            for store in _MEMORY_ONLY.values():
                if r in store:
                    stack = store[r]
                    break
        if stack is None:
            raise RuntimeError(f"table for r={r} missing after build")
        for ell in range(1, max_L + 1):
            tables[(ell, r)] = stack[ell - 1]
    _MEMORY_ONLY.clear()
    return ProductMomentTables(
        max_L=max_L,
        max_r=max(selected_r),
        r_values=selected_r,
        u_grid=u_grid,
        log_phi=tables,
    )


_MEMORY_ONLY: dict[int, dict[int, FloatArray]] = {}


def _interp_setup(u_grid: FloatArray, laguerre_order: int):
    nodes, weights = laggauss(laguerre_order)
    log_nodes = np.log(nodes)
    log_quadrature_weights = np.log(weights)
    query = u_grid[:, None] + log_nodes[None, :]
    idx0 = np.searchsorted(u_grid, query, side="right") - 1
    idx0 = np.clip(idx0, 0, u_grid.size - 2)
    denom = u_grid[idx0 + 1] - u_grid[idx0]
    frac = np.clip((query - u_grid[idx0]) / denom, 0.0, 1.0)
    below = query < u_grid[0]
    above = query > u_grid[-1]
    return (u_grid, log_nodes, log_quadrature_weights, idx0, frac, below, above)


def _build_one_r_with_setup(
    r: int,
    max_L: int,
    setup,
    chunk_size: int,
    previous: FloatArray | None,
) -> FloatArray:
    (u_grid, log_nodes, log_quadrature_weights, idx0, frac, below, above) = setup
    u_points = u_grid.size
    stack = np.empty((max_L, u_points), dtype=np.float64)
    start_level = 1
    if previous is not None and previous.shape[1] == u_points:
        stack[: previous.shape[0]] = previous
        start_level = previous.shape[0]
        current = previous[-1].copy()
    else:
        current = math.lgamma(r + 1) - (r + 1) * np.logaddexp(0.0, u_grid)
        stack[0] = current

    quadrature_prefactor = log_quadrature_weights + r * log_nodes

    for ell in range(start_level + 1, max_L + 1):
        left_log_moment = (ell - 1) * math.lgamma(r + 1)
        next_values = np.empty(u_points, dtype=np.float64)
        for start in range(0, u_points, chunk_size):
            stop = min(start + chunk_size, u_points)
            i0 = idx0[start:stop]
            f = frac[start:stop]
            interpolated = current[i0] * (1.0 - f) + current[i0 + 1] * f
            interpolated = np.where(below[start:stop], left_log_moment, interpolated)
            interpolated = np.where(above[start:stop], -np.inf, interpolated)
            log_terms = quadrature_prefactor[None, :] + interpolated
            max_terms = np.max(log_terms, axis=1)
            finite = np.isfinite(max_terms)
            safe_max = np.where(finite, max_terms, 0.0)
            sums = np.sum(np.exp(log_terms - safe_max[:, None]), axis=1)
            next_values[start:stop] = np.where(
                finite, safe_max + np.log(sums), -np.inf
            )
        current = next_values
        stack[ell - 1] = current

    return stack


_WORKER_SETUP: dict[tuple, object] = {}


def _build_worker_cached(args) -> int:
    """Build one order and persist it; returns the order (no array payload)."""

    r, max_L, spec, laguerre_order, chunk_size, cache = args
    cache_path = Path(cache)
    cached = _load_cached(cache_path, r)
    if cached is not None and cached.shape[0] >= max_L:
        return r
    key = (tuple(spec), laguerre_order)
    setup = _WORKER_SETUP.get(key)
    if setup is None:
        setup = _interp_setup(grid_from_spec(spec), laguerre_order)
        _WORKER_SETUP.clear()  # keep at most one grid's setup in memory
        _WORKER_SETUP[key] = setup
    stack = _build_one_r_with_setup(r, max_L, setup, chunk_size, cached)
    _store_cached(cache_path, r, stack)
    return r


def _cache_file(cache_path: Path | None, r: int) -> Path | None:
    if cache_path is None:
        return None
    return cache_path / f"r{r}.npy"


def _load_cached(cache_path: Path | None, r: int) -> FloatArray | None:
    f = _cache_file(cache_path, r)
    if f is None or not f.exists():
        return None
    try:
        return np.load(f)
    except Exception:  # noqa: BLE001 - treat unreadable cache as missing
        return None


def _store_cached(cache_path: Path | None, r: int, stack: FloatArray) -> None:
    f = _cache_file(cache_path, r)
    if f is None:
        return
    tmp = f.with_suffix(".tmp.npy")
    np.save(tmp, stack)
    tmp.replace(f)
