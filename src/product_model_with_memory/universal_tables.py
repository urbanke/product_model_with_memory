"""The universal moment-function table (v2).

One permanent, corpus-independent store of the moment functions
ln phi_r^(L) (complexity notes, T1), grown on demand and reused by
every experiment.  Layout: one file per level under a table directory,
each file a packed array of columns plus an index; a manifest records
the design constants and certification results.

v2 (2026-07-29): stored values are built by the CERTIFIED methods only
(small-t series / pole-aware contour integration / right-pole series,
module mellin), replacing v1's mix of level recursion and order-2
saddle approximation; queries interpolate with a degree-7
stencil between grid points (linear interpolation alone costs whole
nats at large r even on exact stored values); queries right of the
grid are answered
exactly on the fly, so the whole axis is covered and certifiable.

Design:
  * master u-grid: uniform spacing H on (-infty, U_MAX], addressed by
    integer index; each stored column (L, r) covers
    [series boundary - margin, U_MAX] --- everything left of a
    column's start is answered exactly by the certified small-t
    series, everything right of U_MAX by the certified large-t
    machinery, both on the fly.
  * certification: random OFF-GRID spot checks of the values the
    table actually serves (i.e. after interpolation) against the
    independent contour/series reference, over the full axis,
    recorded in the manifest.
  * the store only grows; files are never rewritten.  Levels are
    independent files, so partial builds are valid stores.

Entry point for experiments:  ensure_universal_tables(path) returns a
UniversalTables object, creating the directory and manifest on first
use; columns are computed and persisted the first time they are
requested, so "check if the table exists, build if not" is automatic
and incremental.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
from scipy.special import loggamma

from product_model_with_memory.mellin import (
    exact_log_phi_column,
    log_phi_contour,
    log_phi_right_series,
    series_column,
)

VERSION = "v2"
H = 0.02             # master grid spacing on the u axis
U_MAX = 35.0         # right edge of the master grid
L_MAX = 70           # levels covered by the design
TAU_LOG_MARGIN = -8.0  # starting point for the boundary search
SERIES_CERT_LOG = -30.0  # a column starts only where the small-t
                         # series' own certificate is below e^-30

_ACCURACY_NOTE = (
    "v2 accuracy: stored values built by certified methods only "
    "(series / pole-aware contour / right-pole series); degree-7 "
    "interpolation between grid points; off-grid certification over "
    "the FULL axis against the independent contour reference.  "
    "Measured accuracy is recorded in the certification entries "
    "below."
)


def _grid_index(u: float) -> int:
    """Index of the master grid point at or below u (grid point i is
    at u = U_MAX - i * H, counting DOWN from the right edge)."""

    return int(math.ceil((U_MAX - u) / H - 1e-9))


def _grid_points(i_start: int) -> np.ndarray:
    """Master grid points from index i_start (leftmost) up to U_MAX,
    in increasing u order."""

    return U_MAX - H * np.arange(i_start, -1, -1)


def _series_cert_log(L: int, r: int, u: float) -> float:
    """ln of the small-t series certificate at u: the series is
    asymptotic, summed over its decreasing prefix only, and its error
    bound is the term at the first local minimum.  (For small r and
    deep L the terms start growing almost immediately --- a fixed
    tau-margin rule places the boundary far too late; found by
    certification, July 2026.)"""

    j = np.arange(1, 61)
    tl = (j * u + L * (loggamma(r + j + 1.0) - loggamma(r + 1.0))
          - loggamma(j + 1.0))
    inc = np.flatnonzero(np.diff(tl) >= 0.0)
    return float(tl[inc[0]] if len(inc) else tl[-1])


def column_start_index(L: int, r: int) -> int:
    """Leftmost stored index for column (L, r): the rightmost grid
    point where the certified small-t series can take over."""

    u = TAU_LOG_MARGIN - L * float(loggamma(r + 2.0) - loggamma(r + 1.0))
    while _series_cert_log(L, r, u) > SERIES_CERT_LOG:
        u -= 5.0
    return _grid_index(min(u, U_MAX - H))


_STENCIL = 8  # 8-point (degree-7) Lagrange interpolation
_BARY_W = np.array([(-1.0) ** k * math.comb(_STENCIL - 1, k)
                    for k in range(_STENCIL)])


def _interp_column(grid: np.ndarray, vals: np.ndarray,
                   u: np.ndarray) -> np.ndarray:
    """8-point (degree-7) barycentric Lagrange interpolation on the
    uniform grid.

    Every u-derivative of ln phi scales like the saddle location
    z*(u), which reaches ~r in the transition region --- so on the
    H = 0.02 grid linear interpolation costs up to ~1 nat at r = 1e6,
    cubic ~1e-5 (both measured July 2026, on EXACT stored values).
    Each additional degree buys roughly a factor H, so degree 7 puts
    the served error at ~1e-10 for the r range in use.
    """

    s = (u - grid[0]) / H
    i = np.clip(np.floor(s).astype(np.int64) - (_STENCIL // 2 - 1),
                0, len(grid) - _STENCIL)
    x = s - i  # offset within the window, in [0, STENCIL-1]
    dx = x[:, None] - np.arange(_STENCIL)[None, :]
    exact = np.abs(dx) < 1e-12
    dx = np.where(exact, 1.0, dx)
    w = _BARY_W[None, :] / dx
    window = vals[i[:, None] + np.arange(_STENCIL)[None, :]]
    out = (w * window).sum(axis=1) / w.sum(axis=1)
    hit = exact.any(axis=1)
    if hit.any():  # query lies (numerically) on a grid point
        out[hit] = window[np.arange(len(u)), np.argmax(exact, axis=1)][hit]
    return out


class UniversalTables:
    """Grow-on-demand universal store of ln phi columns."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.path / "manifest.json"
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())
            for key, want in [("version", VERSION), ("H", H),
                              ("U_MAX", U_MAX)]:
                if self.manifest.get(key) != want:
                    raise RuntimeError(
                        f"universal table at {self.path} has {key}="
                        f"{self.manifest.get(key)}, this code wants {want}; "
                        "use a fresh directory or matching code version"
                    )
        else:
            self.manifest = {
                "version": VERSION, "H": H, "U_MAX": U_MAX,
                "L_MAX": L_MAX,
                "tau_log_margin": TAU_LOG_MARGIN,
                "builder": "exact (series / contour / right series)",
                "accuracy": _ACCURACY_NOTE,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "certifications": [],
            }
            self._save_manifest()
        self._levels: dict[int, dict] = {}  # L -> {"index": {r: (off, n)},
        #                                          "data": np.ndarray}

    # ---------------------------------------------------------- files

    def _level_files(self, L: int) -> tuple[Path, Path]:
        return (self.path / f"level_{L:02d}.bin",
                self.path / f"level_{L:02d}.index.json")

    def _load_level(self, L: int) -> dict:
        if L in self._levels:
            return self._levels[L]
        data_f, idx_f = self._level_files(L)
        if idx_f.exists():
            index = {int(k): tuple(v)
                     for k, v in json.loads(idx_f.read_text()).items()}
            data = np.fromfile(data_f, dtype=np.float64)
        else:
            index, data = {}, np.empty(0)
        level = {"index": index, "data": data}
        self._levels[L] = level
        return level

    def _append_column(self, L: int, r: int, values: np.ndarray) -> None:
        level = self._load_level(L)
        data_f, idx_f = self._level_files(L)
        with open(data_f, "ab") as f:
            values.astype(np.float64).tofile(f)
        level["index"][r] = (len(level["data"]), len(values))
        level["data"] = np.concatenate([level["data"], values])
        idx_f.write_text(json.dumps(
            {str(k): list(v) for k, v in level["index"].items()}))

    def _save_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2))

    # -------------------------------------------------------- columns

    def column(self, L: int, r: int) -> tuple[int, np.ndarray]:
        """(start index, values) of the stored column for (L, r) on the
        master grid; computes and persists it on first request."""

        if L < 1 or L > L_MAX:
            raise ValueError(f"L={L} outside design range 1..{L_MAX}")
        if L == 1:  # closed form; never stored
            i0 = column_start_index(1, r)
            u = _grid_points(i0)
            vals = float(loggamma(r + 1.0)) - (r + 1.0) * np.log1p(np.exp(u))
            return i0, vals
        level = self._load_level(L)
        if r in level["index"]:
            off, n = level["index"][r]
            return column_start_index(L, r), level["data"][off:off + n]
        values = self._build_column(L, r)
        self._append_column(L, r, values)
        return column_start_index(L, r), values

    def ensure_columns(self, L: int, r_values) -> None:
        """Build (persist) any missing columns among r_values at level
        L."""

        if L == 1:
            return
        level = self._load_level(L)
        missing = sorted(set(int(r) for r in r_values) - set(level["index"]))
        for r in missing:
            self._append_column(L, r, self._build_column(L, r))

    def _build_column(self, L: int, r: int) -> np.ndarray:
        return _build_column_values(L, r)

    def build_columns(self, pairs, *, jobs: int = 1, progress=None) -> int:
        """Build (persist) every missing column among the (L, r) pairs,
        optionally in parallel; returns how many were built.

        Workers compute columns (pure math, no store access); the
        parent alone appends to the level files, so the store stays
        consistent regardless of jobs."""

        missing = []
        seen = set()
        for L, r in pairs:
            L, r = int(L), int(r)
            if L < 2 or (L, r) in seen:
                continue
            seen.add((L, r))
            if r not in self._load_level(L)["index"]:
                missing.append((L, r))
        if not missing:
            return 0
        if jobs <= 1 or len(missing) < 4:
            for i, (L, r) in enumerate(missing):
                self._append_column(L, r, self._build_column(L, r))
                if progress is not None:
                    progress(("tables", i + 1, len(missing)), None)
        else:
            import multiprocessing as mp

            with mp.Pool(processes=min(jobs, len(missing))) as pool:
                done = 0
                for L, r, values in pool.imap_unordered(
                        _column_worker, missing, chunksize=1):
                    self._append_column(L, r, values)
                    done += 1
                    if progress is not None:
                        progress(("tables", done, len(missing)), None)
        return len(missing)

    # ------------------------------------------------------ evaluation

    def log_phi(self, L: int, r: int, u) -> np.ndarray:
        """ln phi_r^(L)(e^u) for scalar or array u: certified series
        left of the stored column, cubic interpolation inside it,
        certified large-t evaluation right of the grid."""

        u = np.atleast_1d(np.asarray(u, dtype=np.float64))
        i0, vals = self.column(L, r)
        grid = _grid_points(i0)
        out = np.empty_like(u)
        inside = (u >= grid[0]) & (u <= grid[-1])
        if inside.any():
            out[inside] = _interp_column(grid, vals, u[inside])
        left = u < grid[0]
        if left.any():
            v, cert = series_column(r, L, u[left])
            if np.any(cert > 1e-10):
                bad = u[left][cert > 1e-10]
                raise RuntimeError(
                    f"series not certified left of column (L={L}, r={r}, "
                    f"u={bad[:3]}); store margin violated")
            out[left] = v
        for j in np.flatnonzero(u > grid[-1]):
            v, cert = log_phi_right_series(r, L, float(u[j]))
            if cert < 1e-10:
                out[j] = v
            else:  # rare (large L): fall back to the exact contour
                out[j] = log_phi_contour(r, L, float(u[j]))
        return out

    def level_tables(self, L: int, r_values, u_grid):
        """Scan-compatible ProductMomentTables for one level, served
        from the store on an arbitrary u grid (the production scan's
        adaptive grid, typically).

        Values are exactly what log_phi serves: certified series left
        of the stored column, degree-7 interpolation inside it,
        certified large-t evaluation right of the grid edge.  This is
        the adapter that lets log_q_lambda_scan run unchanged on
        universal-table values.
        """

        from product_model_with_memory.layered import ProductMomentTables

        rs = tuple(sorted(int(r) for r in set(r_values)))
        u = np.asarray(u_grid, dtype=np.float64)
        self.ensure_columns(L, rs)
        return ProductMomentTables(
            max_L=max(L, 1),
            max_r=rs[-1],
            r_values=rs,
            u_grid=u,
            log_phi={(L, r): self.log_phi(L, r, u) for r in rs},
        )

    # --------------------------------------------------- certification

    def certify(self, samples: int = 60, seed: int = 0) -> dict:
        """Spot-check the values the table SERVES (off-grid queries,
        i.e. including interpolation) against the independent
        contour/series reference, over the full axis; append the
        result to the manifest."""

        rng = np.random.default_rng(seed)
        errs = []
        checked = []
        on_disk = [int(p.name.split("_")[1].split(".")[0])
                   for p in self.path.glob("level_*.index.json")]
        levels = [L for L in sorted(on_disk)
                  if self._load_level(L)["index"]]
        if not levels:
            return {"n": 0}
        per_level = max(1, samples // len(levels))
        for L in levels:
            level = self._load_level(L)
            rs = sorted(level["index"])
            for _ in range(per_level):
                r = int(rng.choice(rs))
                i0, _vals = self.column(L, r)
                grid = _grid_points(i0)
                # off-grid u across the whole axis: left of the column,
                # inside it, and right of the grid edge
                uu = float(rng.uniform(grid[0] - 3.0, U_MAX + 2.0))
                got = float(self.log_phi(L, r, uu)[0])
                ref = log_phi_contour(r, L, uu)
                errs.append(abs(got - ref))
                checked.append({"L": L, "r": r, "u": uu,
                                "abs_err_nats": errs[-1]})
        result = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n": len(errs),
            "median": float(np.median(errs)) if errs else None,
            "max": float(np.max(errs)) if errs else None,
            "worst_cases": sorted(checked,
                                  key=lambda x: -x["abs_err_nats"])[:5],
        }
        self.manifest["certifications"].append(result)
        self._save_manifest()
        return result


def _build_column_values(L: int, r: int) -> np.ndarray:
    """Column values on the master grid: exact evaluation on a strided
    sub-lattice, degree-7 fill in between, SPOT-CHECKED against exact
    values at random filled points (full exact rebuild on failure).

    The strided build exists because exact evaluation of every H=0.02
    point costs ~0.3-3 s per column and experiments need thousands of
    columns; the fill reuses the same interpolation the store serves
    with, and the spot check keeps the certification honest (measured
    July 2026: fill agrees with exact at ~1e-9; threshold 2e-8).
    """

    i0 = column_start_index(L, r)
    grid = _grid_points(i0)
    n = len(grid)
    stride = 4 if r < 1e5 else 2
    if n < 8 * stride:
        return exact_log_phi_column(r, L, grid)
    lattice_idx = np.arange(0, n, stride)
    lattice = grid[lattice_idx]
    coarse = exact_log_phi_column(r, L, lattice)
    m = len(coarse)
    values = np.empty(n)
    values[lattice_idx] = coarse
    fill_idx = np.setdiff1d(np.arange(n), lattice_idx)
    interior = fill_idx[fill_idx < lattice_idx[-1]]
    tail = fill_idx[fill_idx >= lattice_idx[-1]]

    # The degree-7 interpolation error is governed by the 8th
    # derivative, whose finite-difference estimate on the lattice is
    # the 8th difference of the coarse values.  Windows where it is
    # large cannot be filled to ~1e-8 --- evaluate those points
    # exactly instead of interpolating (random probes alone missed a
    # localized 3e-6 miss; measured July 2026).
    rough_lattice = np.zeros(m, dtype=bool)
    if m > 8:
        d8 = np.abs(np.diff(coarse, n=8))
        for k in np.flatnonzero(d8 > 1e-7):
            rough_lattice[k:k + 9] = True
    if len(interior):
        s = (grid[interior] - lattice[0]) / (stride * H)
        starts = np.clip(np.floor(s).astype(np.int64) - (_STENCIL // 2 - 1),
                         0, m - _STENCIL)
        stencil_rough = np.array([
            rough_lattice[i:i + _STENCIL].any() for i in starts
        ]) if rough_lattice.any() else np.zeros(len(interior), dtype=bool)
        smooth = interior[~stencil_rough]
        rough = interior[stencil_rough]
        if len(smooth):
            values[smooth] = _lagrange_uniform(
                lattice[0], stride * H, coarse, grid[smooth])
        if len(rough):
            values[rough] = exact_log_phi_column(r, L, grid[rough])
    if len(tail):  # beyond the last lattice point: evaluate exactly
        values[tail] = exact_log_phi_column(r, L, grid[tail])

    # final certificate: random probes among the interpolated points
    smooth_check = (interior[~stencil_rough]
                    if len(interior) and rough_lattice.any()
                    else interior)
    if len(smooth_check):
        rng = np.random.default_rng(hash((L, r)) & 0xFFFFFFFF)
        probe = rng.choice(smooth_check, size=min(16, len(smooth_check)),
                           replace=False)
        ref = exact_log_phi_column(r, L, grid[probe])
        if np.max(np.abs(values[probe] - ref)) > 2e-8:
            return exact_log_phi_column(r, L, grid)  # certified fallback
    return values


def _lagrange_uniform(x0: float, dx: float, vals: np.ndarray,
                      xq: np.ndarray) -> np.ndarray:
    """Degree-7 barycentric Lagrange interpolation on the uniform grid
    x0 + k dx (same stencil as the store's serving interpolation)."""

    s = (xq - x0) / dx
    i = np.clip(np.floor(s).astype(np.int64) - (_STENCIL // 2 - 1),
                0, len(vals) - _STENCIL)
    x = s - i
    d = x[:, None] - np.arange(_STENCIL)[None, :]
    exact = np.abs(d) < 1e-12
    d = np.where(exact, 1.0, d)
    w = _BARY_W[None, :] / d
    window = vals[i[:, None] + np.arange(_STENCIL)[None, :]]
    out = (w * window).sum(axis=1) / w.sum(axis=1)
    hit = exact.any(axis=1)
    if hit.any():
        out[hit] = window[np.arange(len(xq)), np.argmax(exact, axis=1)][hit]
    return out


def _column_worker(pair):
    """Top-level (picklable) column builder for build_columns."""

    L, r = pair
    return L, r, _build_column_values(L, r)


def ensure_universal_tables(path: str | Path | None = None) -> UniversalTables:
    """Open (or initialize) the universal table store.

    Resolution order for the location: explicit argument; environment
    variable PMM_UNIVERSAL_TABLES; ./tables/universal_v2 relative to
    the current working directory.
    """

    if path is None:
        path = os.environ.get("PMM_UNIVERSAL_TABLES",
                              Path.cwd() / "tables" / f"universal_{VERSION}")
    return UniversalTables(path)
