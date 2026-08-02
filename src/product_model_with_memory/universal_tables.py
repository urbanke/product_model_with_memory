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

import contextlib
import fcntl
import json
import math
import os
import time
import zlib
from pathlib import Path

import numpy as np

from product_model_with_memory.kernel import interp_column as _INTERP
from scipy.special import loggamma

from product_model_with_memory.mellin import (
    exact_log_phi_column,
    log_phi_column,
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
SERIES_TAIL_NATS = float(os.environ.get("PMM_SERIES_TAIL", "40"))
# Left of (column start - SERIES_TAIL_NATS), ln phi_r IS its t -> 0
# limit L * lgamma(r + 1): measured deviation 3.4e-4 nats AT a column
# start, 1.5e-8 ten nats left of it, and EXACTLY zero -- bit for bit in
# float64 -- thirty nats left, over L in 2..70 and r up to 67,465.  On
# average 37% of a query grid lies left of a stored column, so
# evaluating the certified series out there was running it at some
# 7,400 points per column per level for values already known.  Skipping
# it took the fill phase of the enwik8 subword run from 16.0 s to 10.8 s
# (1 August), moved no value (tests/test_interp_kernel.py) and left the
# codelength unchanged.  Forty nats carries ten nats of margin beyond
# the measured zero.  PMM_SERIES_TAIL=inf restores the full series.

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


# ------------------------------------------------- accuracy calibration
#
# Every accuracy threshold in this project has so far been chosen by
# looking at errors in ln phi and guessing what they are worth.  Nothing
# connected them to the quantity actually reported, bits per character.
# These hooks make that connection measurable: they inject a controlled
# error into every value the store serves, so a run can be repeated at a
# sequence of amplitudes and the slope read off.
#
#   PMM_PHI_WAVE=eps    add eps * cos(u / lambda + phase(L, r)): an
#                       error SMOOTH in u, oscillating in sign, and
#                       different in every column.  lambda is
#                       PMM_PHI_WAVE_SCALE, default 4 nats.
#   PMM_PHI_BIAS=eps    add exactly +eps nats everywhere.  Nothing
#                       cancels anywhere; the pessimistic end.
#
# SMOOTHNESS IN u IS NOT OPTIONAL, and the first version of this code
# got it wrong.  It used a field decorrelated between adjacent grid
# points, on the reasoning that independent signs were the conservative
# choice.  They are not.  The evaluator does not merely read a column,
# it DIFFERENTIATES it, to locate the Laplace peak and estimate the
# curvature there.  A field that jumps between neighbouring points is
# amplified by 1/H and 1/H^2 on the way through, and H = 0.02.
# Measured on bpe_text8: 1e-8 nats of such jitter moved the codelength
# by 7.5e-4 bits/token --- amplification of order 1e5 --- and at 1e-3
# nats it produced a log2 q above zero, caught by the
# sequence-probability check.
#
# Every scheme we would actually deploy (ladder interpolation across r,
# the saddlepoint expansion, a Chebyshev fit) has an error that is a
# smooth function of u.  A perturbation study is informative only if its
# field has the same character as the error it stands in for.
#
# The field must also be FIXED, not a fresh draw per call: the same
# (L, r, u) is read many times in a run, and resampling would average
# the error away and make any scheme look better than it is.  Only the
# phase is hashed; the u dependence is analytic.
_PHI_WAVE = float(os.environ.get("PMM_PHI_WAVE", "0") or 0.0)
_PHI_WAVE_SCALE = float(os.environ.get("PMM_PHI_WAVE_SCALE", "4") or 4.0)
_PHI_BIAS = float(os.environ.get("PMM_PHI_BIAS", "0") or 0.0)


def _phi_perturbation(L: int, r: int, u: np.ndarray) -> np.ndarray | None:
    """The fixed error field at these query points, or None."""

    if not _PHI_WAVE and not _PHI_BIAS:
        return None
    u = np.asarray(u, dtype=np.float64)
    if not _PHI_WAVE:
        return np.full(len(u), _PHI_BIAS)
    # wraparound is the point of a hash, so numpy's overflow warnings on
    # uint64 multiplication are noise here, not a diagnostic
    with np.errstate(over="ignore"):
        key = (np.uint64(int(L) + 1) * np.uint64(0x9E3779B97F4A7C15)
               ^ np.uint64(int(r) + 1) * np.uint64(0xBF58476D1CE4E5B9))
        key ^= key >> np.uint64(30)
        key *= np.uint64(0xBF58476D1CE4E5B9)
        key ^= key >> np.uint64(27)
        key *= np.uint64(0x94D049BB133111EB)
        key ^= key >> np.uint64(31)
    phase = 2.0 * math.pi * float(key >> np.uint64(11)) / float(1 << 53)
    return _PHI_BIAS + _PHI_WAVE * np.cos(u / _PHI_WAVE_SCALE + phase)


# PMM_PHI_SADDLE_MIN_L=<L>: serve levels at or above L from the
# second-order saddlepoint expansion instead of the store.
#
# This is the measurement that needs no error model at all, and after
# getting the error model wrong once it is the one to trust.  The
# perturbation hooks above ask "what would an error of size eps cost?",
# which requires guessing the shape of the error.  This asks the
# question directly: run the experiment with a real candidate evaluator
# substituted, and see whether any digit we report moves.
#
# The cutoff is on L because that is what governs the expansion's
# accuracy: measured against stored columns, the worst error over a
# column is ~5e-3 nats at L=2 for every r from 1 to 1e6, falling to
# ~2e-6 by L>=35 (scripts/saddle_accuracy.py).  Sweeping the cutoff
# down from L_MAX therefore prices the hybrid of the note's Section 4
# one level at a time, and tells us where to put the boundary.
_SADDLE_MIN_L = int(os.environ.get("PMM_PHI_SADDLE_MIN_L", "0") or 0)
_LADDER_F = float(os.environ.get("PMM_PHI_LADDER", "0") or 0.0)
_LADDER_EVERY = int(os.environ.get("PMM_PHI_LADDER_EVERY", "0") or 0)
_LADDER_DEGREE = int(os.environ.get("PMM_PHI_LADDER_DEGREE", "7") or 7)


def _saddle_row(L: int, r: int, u: np.ndarray) -> np.ndarray:
    """ln phi without the store: certified small-t series where it
    applies, order-2 saddle elsewhere.

    This is `log_phi_column`, which already solves the saddle by
    simultaneous bisection across the whole grid and is the store's own
    builder for large r.  The first version looped the scalar
    `log_phi_saddle` per point instead, which cost 3019 s to serve a
    single level and made sweeping the cutoff impossible; reusing the
    vectorised path is 41x faster and adds no new arithmetic to get
    wrong.

    It is also strictly MORE accurate, which matters for what the
    experiment claims.  In the far-left tail the pure expansion drifts
    to ~9e-3 nats while the series there is exact to the last bit
    (measured at L=43, r=157: saddle 2.00e-02, column 0.00e+00 against
    contour); everywhere else the two agree exactly.  So the
    substitution measures the evaluator we would actually deploy --- a
    hybrid that uses each method inside its own regime --- rather than
    an expansion pushed into a regime no sane implementation would use
    it in.
    """

    return log_phi_column(float(r), int(L),
                          np.asarray(u, dtype=np.float64))


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

    def __init__(self, path: str | Path, *, read_only: bool = False):
        self.read_only = read_only
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
        self._levels: dict[int, dict] = {}  # L -> {"index": {r: (off, n, crc)},
        #                                          "data": np.ndarray}
        self._handles: dict[int, object] = {}   # L -> open read handle
        self._crc_ok: set[tuple[int, int]] = set()   # verified this process
        self._ladder_cache: dict[int, np.ndarray] = {}

        # A DESIGNED store is immutable.  Its whole value is that its
        # contents are a stated function of (factor, levels, r_max) and
        # therefore identical on any machine; one on-demand append
        # destroys that and turns it back into a cache with a
        # reproducible-looking name.  It happened immediately in
        # practice: an unladdered run pointed at the anchor store began
        # rebuilding the entire 86 GB cache inside it, and the only
        # symptom was that the job appeared to hang.
        #
        # anchors.json is the seal, and build_anchor_store.py writes it
        # LAST --- so the store is mutable exactly while it is being
        # constructed and sealed forever after.  To change the grid,
        # build a new store; that is the point.
        self.sealed = (self.path / "anchors.json").exists()

    # ---------------------------------------------------------- files

    def _level_files(self, L: int) -> tuple[Path, Path]:
        return (self.path / f"level_{L:02d}.bin",
                self.path / f"level_{L:02d}.index.json")

    def _load_level(self, L: int) -> dict:
        """Level bookkeeping ONLY (index + file size); column data
        stays on disk and is read per column.  (The first version kept
        every level's data in one in-memory array and re-concatenated
        it per append --- quadratic, and it strangled parallel builds
        once the store reached gigabytes; found July 2026.)"""

        if L in self._levels:
            return self._levels[L]
        data_f, idx_f = self._level_files(L)
        if idx_f.exists():
            index = {int(k): tuple(v)
                     for k, v in json.loads(idx_f.read_text()).items()}
        else:
            index = {}
        size = data_f.stat().st_size // 8 if data_f.exists() else 0
        if index:
            # entries are (off, n) in stores written before checksums and
            # (off, n, crc) after; both forms must load, or a checksummed
            # store cannot be opened at all
            need = max(e[0] + e[1] for e in index.values())
            if size < need:
                raise RuntimeError(
                    f"universal table level {L} is INCOMPLETE: "
                    f"{data_f.name} holds {size} values but the index "
                    f"claims {need} (a partial copy or an interrupted "
                    f"build).  Re-transfer or delete level_{L:02d}.* and "
                    "let it rebuild."
                )
        level = {"index": index, "size": size, "dirty": False}
        self._levels[L] = level
        return level

    def _data_handle(self, L: int):
        """A read handle for one level's data file, kept open.

        Every column used to open, seek, read and close: 32,453 opens
        for a single first-order run on the enwik8 subword stream (31
        July), 5% of it in the open/close alone.  A handle opened "rb"
        sees bytes appended afterwards through the separate "ab"
        handle, so this stays correct while a level grows."""

        fh = self._handles.get(L)
        if fh is None or fh.closed:
            fh = open(self._level_files(L)[0], "rb")
            self._handles[L] = fh
        return fh

    def close(self) -> None:
        for fh in self._handles.values():
            if not fh.closed:
                fh.close()
        self._handles.clear()

    def _read_column(self, L: int, r: int) -> np.ndarray:
        level = self._load_level(L)
        entry = level["index"][r]
        off, n = entry[0], entry[1]
        crc = entry[2] if len(entry) > 2 else None
        data_f, _ = self._level_files(L)
        f = self._data_handle(L)
        f.seek(off * 8)
        raw = f.read(n * 8)
        if len(raw) != n * 8:
            raise RuntimeError(
                f"universal table: column (L={L}, r={r}) is short "
                f"({len(raw) // 8} of {n} values) --- {data_f.name} is "
                "truncated (partial copy or interrupted build)")
        # The checksum is the only guard that catches a column which is
        # well-formed but is not the column the index claims.  Verified
        # once per (L, r) per process: the failure is a property of the
        # bytes on disk, so re-checking every read buys nothing.
        if crc is not None and (L, r) not in self._crc_ok:
            if (zlib.crc32(raw) & 0xFFFFFFFF) != crc:
                raise RuntimeError(
                    f"universal table: column (L={L}, r={r}) fails its "
                    f"checksum --- {data_f.name} is corrupted.  The bytes "
                    "are intact but belong to a different column; this is "
                    "what concurrent writers to one level produce.  Delete "
                    f"level_{L:02d}.* and let it rebuild, and recheck any "
                    "result computed from this level.")
            self._crc_ok.add((L, r))
        vals = np.frombuffer(raw, dtype=np.float64).copy()
        if not np.all(np.isfinite(vals)):
            raise RuntimeError(
                f"universal table: column (L={L}, r={r}) contains "
                "non-finite values --- the store is damaged")
        return vals

    def verify_level(self, L: int) -> dict:
        """Check every column of a level against its stored checksum.

        I/O bound and needs no numerics, so a whole store verifies in
        minutes.  Columns written before checksums existed report as
        `unchecked` --- absence of a checksum is not evidence of health.
        """

        level = self._load_level(L)
        bad, unchecked = [], []
        f = self._data_handle(L)
        for r, entry in sorted(level["index"].items()):
            if len(entry) < 3:
                unchecked.append(r)
                continue
            off, n, crc = entry
            f.seek(off * 8)
            raw = f.read(n * 8)
            if len(raw) != n * 8 or (zlib.crc32(raw) & 0xFFFFFFFF) != crc:
                bad.append(r)
        return {"L": L, "columns": len(level["index"]),
                "bad": bad, "unchecked": unchecked}

    # ------------------------------------------------- writer integrity
    #
    # On 1 August 285 of 594 columns at L=43 were found to be wrong by up
    # to 47,000 nats, and the cause was here.  The old code appended in
    # "ab" mode but recorded the offset from level["size"], its own
    # IN-MEMORY counter.  Two processes appending to one level each keep
    # their own counter, so both write real bytes and both record
    # plausible offsets, but each one's offsets are short by whatever the
    # other wrote --- and _flush_index then overwrote the whole JSON with
    # a single process's view.  The result reads back as a well-formed
    # column of exactly the right length whose contents belong somewhere
    # else: smooth, finite, and wrong.  Nothing downstream could notice.
    #
    # Three changes, and all three are needed:
    #
    #   1. The offset comes from the file itself, under an exclusive
    #      lock, never from in-memory state.
    #   2. The index is re-read and merged under that same lock, so a
    #      concurrent writer's entries survive instead of being clobbered.
    #   3. Every column carries a CRC of its own bytes, verified on read.
    #
    # (3) is what makes the failure detectable rather than silent.  It
    # also turns whole-store verification into an I/O-bound pass of a few
    # minutes instead of hours of contour integration.

    @contextlib.contextmanager
    def _level_lock(self, L: int):
        """Exclusive across processes for the duration of an append."""

        lock_f = self.path / f"level_{L:02d}.lock"
        with open(lock_f, "w") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def _append_column(self, L: int, r: int, values: np.ndarray,
                       flush_index: bool = True) -> None:
        if self.sealed:
            raise RuntimeError(
                f"universal table {self.path.name} is SEALED (it has an "
                f"anchors.json) and must not grow: refusing to append "
                f"column (L={L}, r={r}).  A designed store's contents are "
                "a stated function of its grid; appending to it makes it "
                "unreproducible.  If this r is genuinely needed, either "
                "serve it through the ladder (PMM_PHI_LADDER_EVERY) or "
                "build a new store with a finer grid.")
        level = self._load_level(L)
        data_f, _idx_f = self._level_files(L)
        values = np.ascontiguousarray(values, dtype=np.float64)
        payload = values.tobytes()
        crc = zlib.crc32(payload) & 0xFFFFFFFF

        with self._level_lock(L):
            # the offset is whatever the file actually is, right now
            with open(data_f, "ab") as f:
                f.flush()
                offset = os.fstat(f.fileno()).st_size // 8
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())      # data durable BEFORE it is indexed
            level["index"][r] = (offset, len(values), crc)
            level["size"] = offset + len(values)
            level["dirty"] = True
            if flush_index:
                self._flush_index(L, _locked=True)

    def _flush_index(self, L: int, *, _locked: bool = False) -> None:
        level = self._levels.get(L)
        if not level or not level["dirty"]:
            return
        if not _locked:
            with self._level_lock(L):
                return self._flush_index(L, _locked=True)

        _, idx_f = self._level_files(L)
        # merge rather than clobber: another writer may have added
        # entries since this process loaded the level
        merged = {}
        if idx_f.exists():
            try:
                merged = json.loads(idx_f.read_text())
            except json.JSONDecodeError:
                merged = {}           # a torn index; ours is the good copy
        merged.update({str(k): list(v) for k, v in level["index"].items()})
        tmp = idx_f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(merged))
        os.replace(tmp, idx_f)        # atomic: readers see old or new
        level["dirty"] = False

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
            return column_start_index(L, r), self._read_column(L, r)
        if self.sealed:
            raise RuntimeError(
                f"universal table {self.path.name} is a SEALED anchor "
                f"store and has no column (L={L}, r={r}).  It holds only "
                "its designed grid, so this run is asking for a value it "
                "was never meant to serve directly --- set "
                "PMM_PHI_LADDER_EVERY to interpolate from the anchors, or "
                "point at the cache for exact columns.  It will not build "
                "one: a store that grows on demand is not a designed "
                "store.")
        if self.read_only:
            raise RuntimeError(
                f"universal table opened read-only is missing column "
                f"(L={L}, r={r}); the parent process must build all "
                "required columns before workers read them (concurrent "
                "appends would corrupt the level file)")
        values = self._build_column(L, r)
        self._append_column(L, r, values)
        return column_start_index(L, r), values

    def ensure_columns(self, L: int, r_values) -> None:
        """Build (persist) any missing columns among r_values at level
        L."""

        if L == 1:
            return
        # Levels at or above the substitution cutoff are served by the
        # expansion and read no column at all, so provisioning must skip
        # them entirely.  Without this the ladder asks for an anchor grid
        # at a level the anchor store does not even cover --- which is
        # exactly the configuration the whole design calls for: anchors
        # below the cutoff, expansion above.
        if _SADDLE_MIN_L and L >= _SADDLE_MIN_L:
            return
        level = self._load_level(L)
        rs = set(int(r) for r in r_values)

        # Under a ladder the columns actually READ are the anchors, not
        # the requested r.  Provisioning r itself would build (or, in a
        # sealed store, demand) tens of thousands of columns that the
        # run will never touch --- which is precisely how the first
        # end-to-end attempt failed.
        if _LADDER_F or _LADDER_EVERY:
            needed = set()
            for r in rs:
                needed.update(self.ladder_anchors_for(L, r))
            missing = sorted(needed - set(level["index"]))
            if missing:
                raise RuntimeError(
                    f"ladder at level {L} needs anchor columns that are "
                    f"not in {self.path.name}: {missing[:5]}"
                    f"{' ...' if len(missing) > 5 else ''}.  The grid and "
                    "the store disagree; rebuild the store or check "
                    "PMM_PHI_LADDER_EVERY.")
            return

        missing = sorted(rs - set(level["index"]))
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

        if _LADDER_F or _LADDER_EVERY:
            # same reasoning as ensure_columns: under a ladder these are
            # not the columns that will be read
            by_level: dict[int, set[int]] = {}
            for L, r in pairs:
                if int(L) >= 2:
                    by_level.setdefault(int(L), set()).add(int(r))
            for L, rs in by_level.items():
                self.ensure_columns(L, rs)
            return 0

        missing = []
        seen = set()
        for L, r in pairs:
            L, r = int(L), int(r)
            if L < 2 or (L, r) in seen:
                continue
            if _SADDLE_MIN_L and L >= _SADDLE_MIN_L:
                continue              # served by the expansion; no column
            seen.add((L, r))
            if r not in self._load_level(L)["index"]:
                missing.append((L, r))
        if not missing:
            return 0
        try:
            if jobs <= 1 or len(missing) < 4:
                for i, (L, r) in enumerate(missing):
                    self._append_column(L, r, self._build_column(L, r),
                                        flush_index=False)
                    if (i + 1) % 50 == 0:
                        self._flush_index(L)
                    if progress is not None:
                        progress(("tables", i + 1, len(missing)), None)
            else:
                import multiprocessing as mp

                with mp.Pool(processes=min(jobs, len(missing))) as pool:
                    done = 0
                    for L, r, values in pool.imap_unordered(
                            _column_worker, missing, chunksize=1):
                        self._append_column(L, r, values, flush_index=False)
                        done += 1
                        if done % 50 == 0:
                            self._flush_index(L)
                        if progress is not None:
                            progress(("tables", done, len(missing)), None)
        finally:
            for L in {p[0] for p in missing}:
                self._flush_index(L)
        return len(missing)

    # ------------------------------------------------------ evaluation

    def log_phi(self, L: int, r: int, u) -> np.ndarray:
        """ln phi_r^(L)(e^u) for scalar or array u: certified series
        left of the stored column, cubic interpolation inside it,
        certified large-t evaluation right of the grid."""

        u = np.atleast_1d(np.asarray(u, dtype=np.float64))
        if _SADDLE_MIN_L and L >= _SADDLE_MIN_L:
            return _saddle_row(L, r, u)
        if (_LADDER_F or _LADDER_EVERY) and L < (_SADDLE_MIN_L or L_MAX + 1):
            return self._ladder_log_phi(L, r, u)
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
        pert = _phi_perturbation(L, r, u)
        return out if pert is None else out + pert

    # ------------------------------------------------------- the ladder
    #
    # PMM_PHI_LADDER=f serves every column by interpolating across r
    # from anchors on the grid round((1+f)^k), instead of reading the
    # exact column.  This is the OTHER half of the small-store design:
    # the substitution hook removes deep levels entirely, this one
    # subsamples the levels that remain.
    #
    # It exists to be measured end to end.  The intrinsic error of the
    # interpolation, in nats, does not translate into a codelength
    # without an amplification factor, and the two estimates we have for
    # that factor differ by an order of magnitude.  Serving a whole run
    # off the ladder and reading the bits is the only honest answer, and
    # it is what settled the deep-level question.
    #
    # Anchors are evaluated through the ordinary certified read path and
    # only then combined, so the ladder adds exactly one approximation
    # to what the store already does --- no second interpolation scheme
    # to account for.

    def _ladder_grid(self, L: int) -> np.ndarray:
        """The anchor set this level is served from.

        PMM_PHI_LADDER_EVERY=m decimates the store's OWN grid from
        anchors.json, and that is the form to use.  PMM_PHI_LADDER=f
        recomputes round((1+f)^k), which does NOT coincide with any
        decimation --- every 2nd point of a 0.5% grid lies on
        1.005^2 = 1.010025^k, not on 1.01^k --- so a nominal factor
        silently selects a sparse irregular subset of what is stored.
        That mistake invalidated an entire accuracy sweep before it was
        caught; here it would have quietly corrupted the end-to-end
        number instead, with nothing in the output to show for it.
        """

        g = self._ladder_cache.get(L)
        if g is not None:
            return g
        have = self._load_level(L)["index"]

        if _LADDER_EVERY:
            meta_f = self.path / "anchors.json"
            if not meta_f.exists():
                raise RuntimeError(
                    f"PMM_PHI_LADDER_EVERY needs {meta_f}: this store was "
                    "not built by scripts/build_anchor_store.py, so it has "
                    "no designed grid to decimate.  Serving a ladder off a "
                    "cache measures nothing.")
            meta = json.loads(meta_f.read_text())
            entry = meta["levels"].get(str(L))
            if entry is None:
                raise RuntimeError(
                    f"anchors.json has no level {L}; the ladder cannot be "
                    "served here")
            # decimate only above the dense floor.  Every integer below
            # it is an anchor and must stay one: small r is where the
            # counts actually are and where ln phi is steepest in
            # ln(r+1), and interpolating there produced log2 q = +1321
            # bits on a real run.
            _all = [int(r) for r in entry["anchors"]]
            _floor = int(meta.get("dense_below", 0))
            out = ([r for r in _all if r < _floor]
                   + [r for r in _all if r >= _floor][::_LADDER_EVERY])
            # Do NOT filter to what happens to be present.  Dropping a
            # missing anchor silently widens the ladder and makes it
            # irregular, which is the failure that has invalidated this
            # measurement twice already --- and it would do so with the
            # label still reading "every 8".  The grid is a stated
            # object; if the store cannot supply it, say so.
            absent = [r for r in out if r not in have]
            if absent:
                raise RuntimeError(
                    f"ladder grid at level {L} needs {len(absent)} anchor "
                    f"column(s) missing from {self.path.name}: "
                    f"{absent[:5]}{' ...' if len(absent) > 5 else ''}.  "
                    "Serving the remainder would silently be a coarser, "
                    "irregular ladder; refusing.")
        else:
            r_max = max(have) if have else 1
            out, k = [], 0
            while True:
                r = int(round((1.0 + _LADDER_F) ** k))
                if r > r_max:
                    break
                if (not out or r > out[-1]) and r in have:
                    out.append(r)
                k += 1

        g = np.asarray(out, dtype=np.int64)
        self._ladder_cache[L] = g
        return g

    def ladder_anchors_for(self, L: int, r: int) -> list[int]:
        """The anchors the ladder would read to serve (L, r).

        Public because the provisioning path needs it: callers
        pre-build every column they are about to read, and under a
        ladder the columns actually read are these, not r itself.
        """

        grid = self._ladder_grid(L)
        if len(grid) < _LADDER_DEGREE + 1:
            return [int(r)]
        # EXTRAPOLATION IS NOT ALLOWED.  Outside the anchor span the
        # Lagrange form diverges fast and silently: r=0 against a grid
        # starting at 1 returned a value that made log2 q = +91 bits.
        # Refuse instead --- a grid that does not cover the queries is a
        # grid that needs rebuilding, not a case to paper over.
        if r < grid[0] or r > grid[-1]:
            raise RuntimeError(
                f"ladder cannot serve (L={L}, r={r}): the anchor grid "
                f"spans {int(grid[0])}..{int(grid[-1])} and this query is "
                "outside it, so interpolating would be extrapolation.  "
                "Rebuild the store to cover this r.")
        j = int(np.searchsorted(grid, r))
        if j < len(grid) and int(grid[j]) == r:
            return [int(r)]                         # r IS an anchor
        m = _LADDER_DEGREE + 1
        lo = max(0, min(j - m // 2, len(grid) - m))
        return [int(a) for a in grid[lo:lo + m]]

    def _ladder_log_phi(self, L: int, r: int, u: np.ndarray) -> np.ndarray:
        anch = self.ladder_anchors_for(L, r)
        if len(anch) == 1:
            return self.log_phi_exact(L, anch[0], u)

        xs = np.log(np.asarray(anch, dtype=np.float64) + 1.0)
        m = len(anch)
        Y = np.array([self.log_phi_exact(L, a, u)
                      - L * float(loggamma(a + 1.0)) for a in anch])
        w = np.ones(m)
        for i in range(m):
            for k in range(m):
                if k != i:
                    w[i] /= (xs[i] - xs[k])
        d = math.log(r + 1.0) - xs
        c = w / d
        # no perturbation term: the anchors were read through
        # log_phi_exact, which applied it already, and under a ladder
        # that is where the error physically enters.  Adding one for r
        # too would count it twice.  (log_phi_matrix does the same.)
        return (c[:, None] * Y).sum(0) / c.sum() + L * float(
            loggamma(r + 1.0))

    def log_phi_exact(self, L: int, r: int, u) -> np.ndarray:
        """log_phi with the ladder bypassed: the stored column itself.

        Used by the ladder to evaluate its own anchors, and by tests
        that need the unapproximated value while a ladder is active.
        """

        global _LADDER_F, _LADDER_EVERY
        saved = (_LADDER_F, _LADDER_EVERY)
        _LADDER_F, _LADDER_EVERY = 0.0, 0
        try:
            return self.log_phi(L, r, u)
        finally:
            _LADDER_F, _LADDER_EVERY = saved

    def log_phi_matrix(self, L: int, r_values, u) -> np.ndarray:
        """Many columns of one level served onto the SAME query grid,
        as a contiguous (len(r_values), len(u)) matrix.

        All stored columns live on the one master grid and differ only
        by an integer start offset, so the degree-7 stencil weights
        are identical for every column: computed once here instead of
        per column.  (Measured 31 July: per-column weight construction
        was ~90% of level provisioning, which itself dominated a
        refresh --- 7.9 s per level against 0.6 s of evaluation.)
        """

        rs = [int(r) for r in r_values]
        u = np.asarray(u, dtype=np.float64)
        n = len(u)
        out = np.empty((len(rs), n))

        if _SADDLE_MIN_L and L >= _SADDLE_MIN_L:
            for m, r in enumerate(rs):
                out[m] = _saddle_row(L, r, u)
            return out

        # The ladder must be applied HERE too, not only in log_phi.
        # This is the hot path the scan adapter calls, so a ladder that
        # exists only in log_phi silently keeps reading exact columns ---
        # which reads as "sealed store has no column (L, r)" rather than
        # as the missing branch it is.
        #
        # Every r in this call shares one u grid and neighbouring r share
        # most of their anchors, so each anchor is evaluated once for the
        # whole matrix instead of once per row: with degree 11 that is
        # twelve column reads per r reduced to roughly one.
        if _LADDER_F or _LADDER_EVERY:
            per_r = [self.ladder_anchors_for(L, r) for r in rs]
            raw, resid = {}, {}
            for anch in per_r:
                for a in anch:
                    if a not in raw:
                        raw[a] = self.log_phi_exact(L, a, u)
                        resid[a] = raw[a] - L * float(loggamma(a + 1.0))
            for m, (r, anch) in enumerate(zip(rs, per_r)):
                base = L * float(loggamma(r + 1.0))
                if len(anch) == 1:
                    # an anchor must come back bit-identical; going out
                    # through the residual and back loses the last bits
                    out[m] = raw[anch[0]]
                    continue
                xs = np.log(np.asarray(anch, dtype=np.float64) + 1.0)
                w = np.ones(len(anch))
                for i in range(len(anch)):
                    for k in range(len(anch)):
                        if k != i:
                            w[i] /= (xs[i] - xs[k])
                c = w / (math.log(r + 1.0) - xs)
                Y = np.array([resid[a] for a in anch])
                # no perturbation term here: the anchors were read
                # through log_phi_exact, which already applied it, and
                # under a ladder that is where the error physically
                # enters.  Adding one for r as well would count it twice
                # and inflate any measurement that combines the two.
                out[m] = (c[:, None] * Y).sum(0) / c.sum() + base
            return out

        # position of each query point in ANY column's local indexing:
        # s = i0 - t with t = (U_MAX - u) / H, so the fractional part
        # (hence the weights) is column-independent.
        t = (U_MAX - u) / H
        k0 = np.floor(-t).astype(np.int64) - (_STENCIL // 2 - 1)
        x = (-t) - k0                       # offset within the window
        dx = x[:, None] - np.arange(_STENCIL)[None, :]
        exact = np.abs(dx) < 1e-12
        w = _BARY_W[None, :] / np.where(exact, 1.0, dx)
        w_sum = w.sum(axis=1)
        hit = exact.any(axis=1)
        hit_col = np.argmax(exact, axis=1)
        offs = np.arange(_STENCIL)[None, :]

        # The compiled stencil is checked against the numpy path VALUE
        # BY VALUE in tests/test_interp_kernel.py -- 2.9 million of them
        # across several levels and query grids, requiring exact
        # equality.  It shipped once with a defect that a benchmark
        # could not see (a shadowed variable in _interp_leftovers, wrong
        # by up to 622 nats at ~1% of points, 1 August), which is why
        # the test compares values rather than codelengths.
        # PMM_INTERP_KERNEL=0 falls back to numpy.
        if _INTERP is not None and os.environ.get(
                "PMM_INTERP_KERNEL", "1") != "0":
            hit_u8 = np.ascontiguousarray(hit, dtype=np.uint8)
            hit_col64 = np.ascontiguousarray(hit_col, dtype=np.int64)
            k0_c = np.ascontiguousarray(k0, dtype=np.int64)
            w_c = np.ascontiguousarray(w, dtype=np.float64)
            wsum_c = np.ascontiguousarray(w_sum, dtype=np.float64)
            u_c = np.ascontiguousarray(u, dtype=np.float64)
            todo = np.empty(n, dtype=np.uint8)
            ptr = (u_c.ctypes.data, k0_c.ctypes.data, w_c.ctypes.data,
                   wsum_c.ctypes.data, hit_u8.ctypes.data,
                   hit_col64.ctypes.data, todo.ctypes.data)
            for m, r in enumerate(rs):
                i0, vals = self.column(L, r)
                vals = np.ascontiguousarray(vals, dtype=np.float64)
                row = np.empty(n)
                _INTERP(vals.ctypes.data, len(vals), int(i0),
                        ptr[0], n, U_MAX - H * i0, U_MAX,
                        ptr[1], ptr[2], ptr[3], ptr[4], ptr[5],
                        row.ctypes.data, ptr[6])
                rest = np.flatnonzero(todo)
                if rest.size:
                    row[rest] = self._interp_leftovers(L, r, i0, vals,
                                                       u[rest])
                pert = _phi_perturbation(L, r, u)
                out[m] = row if pert is None else row + pert
            return out

        for m, r in enumerate(rs):
            i0, vals = self.column(L, r)
            grid0 = U_MAX - H * i0
            inside = (u >= grid0) & (u <= U_MAX)
            row = np.empty(n)
            if inside.any():
                start_raw = i0 + k0[inside]
                lim = len(vals) - _STENCIL
                plain = (start_raw >= 0) & (start_raw <= lim)
                idx_in = np.flatnonzero(inside)
                if plain.any():
                    st = start_raw[plain]
                    window = vals[st[:, None] + offs]
                    ww = w[inside][plain]
                    v = (ww * window).sum(axis=1) / w_sum[inside][plain]
                    h = hit[inside][plain]
                    if h.any():
                        v[h] = window[np.arange(len(window)),
                                      hit_col[inside][plain]][h]
                    row[idx_in[plain]] = v
                if (~plain).any():
                    # the few edge points where the stencil is clamped:
                    # weights differ there, so use the per-point path
                    edge = idx_in[~plain]
                    row[edge] = _interp_column(
                        _grid_points(i0), vals, u[edge])
            left = u < grid0
            if left.any():
                sv, cert = series_column(r, L, u[left])
                if np.any(cert > 1e-10):
                    raise RuntimeError(
                        f"series not certified left of column (L={L}, "
                        f"r={r}); store margin violated")
                row[left] = sv
            right = u > U_MAX
            for j in np.flatnonzero(right):
                v, cert = log_phi_right_series(r, L, float(u[j]))
                row[j] = v if cert < 1e-10 else log_phi_contour(
                    r, L, float(u[j]))
            pert = _phi_perturbation(L, r, u)
            out[m] = row if pert is None else row + pert
        return out

    def _interp_leftovers(self, L: int, r: int, i0: int,
                          vals: np.ndarray, u: np.ndarray) -> np.ndarray:
        """The query points the compiled stencil declines: left of the
        column's own start (series), right of U_MAX (right series or
        contour), and the few whose stencil would run off the stored
        values (clamped weights, so the per-point path)."""

        grid0 = U_MAX - H * i0
        row = np.empty(len(u))
        left = u < grid0
        if left.any():
            ul = u[left]
            # ln phi_r -> L * lgamma(r + 1) as u -> -infinity, and it
            # gets there fast: measured over L in 2..70 and r up to
            # 67,465, the series value is 3.4e-4 nats from the limit AT
            # the column start, 1.5e-8 ten nats left of it, and EXACTLY
            # the limit --- bit for bit in float64 --- thirty nats left
            # (1 August).  Evaluating it out there is pure waste, and on
            # average 37% of the query grid lies left of a stored
            # column, so the series was being run at some 7,400 points
            # per column per level.  SERIES_TAIL_NATS carries a further
            # ten nats of margin beyond the measured zero.
            far = ul < grid0 - SERIES_TAIL_NATS
            # NOT named `vals`: that is the stored column, and shadowing
            # it here made the edge branch below interpolate the series
            # values instead of the column --- wrong by up to 622 nats
            # at ~1% of query points (1 August).
            lv = np.empty(len(ul))
            if far.any():
                lv[far] = L * float(loggamma(r + 1.0))
            near = ~far
            if near.any():
                sv, cert = series_column(r, L, ul[near])
                if np.any(cert > 1e-10):
                    raise RuntimeError(
                        f"series not certified left of column (L={L}, "
                        f"r={r}); store margin violated")
                lv[near] = sv
            row[left] = lv
        right = u > U_MAX
        for j in np.flatnonzero(right):
            v, cert = log_phi_right_series(r, L, float(u[j]))
            row[j] = v if cert < 1e-10 else log_phi_contour(r, L, float(u[j]))
        edge = ~(left | right)
        if edge.any():
            row[edge] = _interp_column(_grid_points(i0), vals, u[edge])
        return row

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
        M = self.log_phi_matrix(L, rs, u)
        return ProductMomentTables.from_matrix(
            max_L=max(L, 1), L=L, r_values=rs, u_grid=u, matrix=M)

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


def verify_store(path=None) -> dict:
    """Check every level file for index/data consistency.  Returns a
    summary; raises on the first inconsistent level.  Run this after
    copying a store between machines."""

    ut = ensure_universal_tables(path)
    levels, columns, values = 0, 0, 0
    for p in sorted(ut.path.glob("level_*.index.json")):
        L = int(p.name.split("_")[1].split(".")[0])
        lv = ut._load_level(L)          # raises if inconsistent
        levels += 1
        columns += len(lv["index"])
        values += lv["size"]
    return {"path": str(ut.path), "levels": levels, "columns": columns,
            "values": values, "gigabytes": round(values * 8 / 1e9, 2)}


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
