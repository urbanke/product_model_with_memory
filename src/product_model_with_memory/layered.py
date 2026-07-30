"""Layered product-simplex mixture numerics.

Faithful port of ``product_model.mixture_weights`` (and
``partition_multiplicities`` from ``product_model.pattern_weights``) from the
companion repository ``product_model`` (paper: "A Layered Simplex Architecture
for Large Alphabets", Appendix B).  The math and defaults are unchanged; only
the imports were flattened so this package is self-contained.
"""

from __future__ import annotations

import math
import operator
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from numpy.polynomial.laguerre import laggauss
from numpy.typing import NDArray
from scipy.optimize import brentq



FloatArray = NDArray[np.float64]
QMethod = Literal["auto", "closed_l1", "grid", "laplace"]


def partition_multiplicities(partition: Iterable[int]) -> tuple[tuple[int, int], ...]:
    """Return ``(r, c_r)`` pairs for a positive integer partition."""

    try:
        parts = tuple(operator.index(part) for part in partition)
    except TypeError as exc:
        raise ValueError("partition entries must be integers") from exc
    if any(part <= 0 for part in parts):
        raise ValueError("partition entries must be positive integers")
    return tuple(sorted(Counter(parts).items()))



@dataclass(frozen=True)
class ProductMomentTables:
    """Tables for ``log E[Y^r exp(-tY)]`` with ``Y`` a product of exponentials."""

    max_L: int
    max_r: int
    r_values: tuple[int, ...]
    u_grid: FloatArray
    log_phi: dict[tuple[int, int], FloatArray]

    def log_phi_value(self, *, L: int, r: int, u: float) -> float:
        if L < 1 or L > self.max_L:
            raise ValueError(f"L={L} is outside the table range 1..{self.max_L}")
        if r not in self.r_values:
            raise ValueError(f"r={r} is not available in this moment table")
        return float(
            np.interp(
                u,
                self.u_grid,
                self.log_phi[(L, r)],
                left=L * math.lgamma(r + 1),
                right=-math.inf,
            )
        )


@dataclass(frozen=True)
class QLambdaResult:
    """Natural-log value and diagnostics for one mixture profile probability."""

    log_q: float
    method: str
    d: int
    L: int
    N: int
    partition: tuple[int, ...]
    saddle_u: float | None = None
    curvature: float | None = None
    left_gap: float | None = None
    right_gap: float | None = None
    converged: bool = True
    message: str = ""
    peaks: tuple[tuple[float, float], ...] = ()  # (u, log contribution)

    @property
    def log2_q(self) -> float:
        return self.log_q / math.log(2.0)


def build_product_moment_tables(
    *,
    max_L: int,
    max_r: int,
    u_grid: FloatArray | None = None,
    u_min: float = -70.0,
    u_max: float = 35.0,
    u_points: int = 16_001,
    laguerre_order: int = 96,
    chunk_size: int = 512,
) -> ProductMomentTables:
    """Precompute product-exponential moment tables on ``u = log(t)``.

    The recurrence is

    ``phi_r^(ell)(t) = E[E^r phi_r^(ell-1)(tE)]``,

    with ``E`` exponential with mean one.  This keeps the finite-``L`` model
    intact, including the subcritical ``L = c log(d)`` regimes.
    """

    if max_L < 1:
        raise ValueError("max_L must be positive")
    if max_r < 0:
        raise ValueError("max_r must be non-negative")
    return build_selected_product_moment_tables(
        max_L=max_L,
        r_values=range(max_r + 1),
        u_grid=u_grid,
        u_min=u_min,
        u_max=u_max,
        u_points=u_points,
        laguerre_order=laguerre_order,
        chunk_size=chunk_size,
    )


def build_selected_product_moment_tables(
    *,
    max_L: int,
    r_values: Iterable[int],
    u_grid: FloatArray | None = None,
    u_min: float = -70.0,
    u_max: float = 35.0,
    u_points: int = 16_001,
    laguerre_order: int = 96,
    chunk_size: int = 512,
) -> ProductMomentTables:
    """Precompute product-exponential moments only for selected powers ``r``.

    Large scaling experiments may have ``N = sqrt(d)`` but only a small number
    of distinct count sizes in the sampled profiles.  This constructor avoids
    tabulating every ``r = 0, ..., N + 2`` when only a sparse set of values is
    needed.  For Laplace evaluation of a profile, include ``0, 1, 2`` and
    ``r, r + 1, r + 2`` for every part size ``r`` in the profile.
    """

    if max_L < 1:
        raise ValueError("max_L must be positive")
    try:
        selected_r = tuple(sorted({operator.index(r) for r in r_values}))
    except TypeError as exc:
        raise ValueError("r_values must contain integers") from exc
    if not selected_r:
        raise ValueError("r_values must not be empty")
    if any(r < 0 for r in selected_r):
        raise ValueError("r_values must be non-negative")
    if laguerre_order < 1:
        raise ValueError("laguerre_order must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if u_grid is None:
        if u_points < 2:
            raise ValueError("u_points must be at least 2")
        u_grid = np.linspace(u_min, u_max, u_points, dtype=np.float64)
    else:
        u_grid = np.asarray(u_grid, dtype=np.float64)
        if u_grid.ndim != 1 or u_grid.size < 2:
            raise ValueError("u_grid must be a one-dimensional grid")
    if np.any(np.diff(u_grid) <= 0.0):
        raise ValueError("u_grid must be strictly increasing")

    nodes, weights = laggauss(laguerre_order)
    if np.any(nodes <= 0.0) or np.any(weights <= 0.0):
        raise ValueError("Laguerre quadrature returned non-positive values")

    log_nodes = np.log(nodes)
    log_quadrature_weights = np.log(weights)
    tables: dict[tuple[int, int], FloatArray] = {}

    for r in selected_r:
        current = math.lgamma(r + 1) - (r + 1) * np.logaddexp(0.0, u_grid)
        tables[(1, r)] = current.copy()
        left_log_moment = math.lgamma(r + 1)

        for ell in range(2, max_L + 1):
            next_values = np.empty_like(u_grid)
            quadrature_prefactor = log_quadrature_weights + r * log_nodes

            for start in range(0, u_grid.size, chunk_size):
                stop = min(start + chunk_size, u_grid.size)
                interpolation_points = u_grid[start:stop, None] + log_nodes[None, :]
                interpolated = np.interp(
                    interpolation_points.ravel(),
                    u_grid,
                    current,
                    left=left_log_moment,
                    right=-math.inf,
                ).reshape(interpolation_points.shape)
                log_terms = quadrature_prefactor[None, :] + interpolated
                next_values[start:stop] = _logsumexp(log_terms, axis=1)

            current = next_values
            left_log_moment = ell * math.lgamma(r + 1)
            tables[(ell, r)] = current.copy()

    return ProductMomentTables(
        max_L=max_L,
        max_r=max(selected_r),
        r_values=selected_r,
        u_grid=u_grid,
        log_phi=tables,
    )


def log_q_lambda_closed_l1(*, d: int, partition: tuple[int, ...]) -> QLambdaResult:
    """Exact ``log q_lambda`` for one simplex draw, i.e. ``L=1``."""

    _validate_q_inputs(d=d, L=1, partition=partition)
    N = sum(partition)
    log_q = (
        math.lgamma(d)
        - math.lgamma(d + N)
        + sum(math.lgamma(part + 1) for part in partition)
    )
    return QLambdaResult(
        log_q=float(log_q),
        method="closed_l1",
        d=d,
        L=1,
        N=N,
        partition=tuple(partition),
        message="closed form for L=1",
    )


def log_q_lambda_grid(
    *,
    d: int,
    L: int,
    partition: tuple[int, ...],
    tables: ProductMomentTables,
) -> QLambdaResult:
    """Compute ``log q_lambda`` by trapezoidal integration on the table grid."""

    _validate_q_inputs(d=d, L=L, partition=partition)
    N = sum(partition)
    if N == 0:
        return QLambdaResult(0.0, "grid", d, L, N, tuple(partition))
    if L == 1:
        return log_q_lambda_closed_l1(d=d, partition=partition)

    _validate_tables_for_partition(
        tables=tables,
        L=L,
        partition=partition,
        extra_orders=0,
    )
    s = len(partition)
    multiplicity_pairs = partition_multiplicities(partition)
    u_grid = tables.u_grid
    log_integrand = (
        N * u_grid
        - math.lgamma(N)
        + (d - s) * tables.log_phi[(L, 0)]
    )
    for part, count in multiplicity_pairs:
        log_integrand = log_integrand + count * tables.log_phi[(L, part)]

    log_q = _log_trapezoid_integral(log_integrand, u_grid)
    peak = float(np.max(log_integrand))
    return QLambdaResult(
        log_q=float(log_q),
        method="grid",
        d=d,
        L=L,
        N=N,
        partition=tuple(partition),
        left_gap=peak - float(log_integrand[0]),
        right_gap=peak - float(log_integrand[-1]),
        message="trapezoidal integration on product-moment grid",
    )


def log_q_lambda_scan(
    *,
    d: int,
    L: int,
    partition: tuple[int, ...],
    tables: ProductMomentTables,
    significance_gap: float = 40.0,
) -> QLambdaResult:
    """``log q_lambda`` by global peak scan plus Laplace refinement.

    Extension of :func:`log_q_lambda_laplace` for real-corpus profiles, where
    the outer integrand can be *multimodal* (deep layers with heavy counts
    produce a second, far-left peak) and its dominant peak can be narrower
    than the grid spacing (large ``N``).  The method:

    1. evaluates the full log-integrand on the table grid (vectorized), plus
       the analytic ``t -> 0`` branch left of the grid, where
       ``phi_r -> Gamma(r+1)^L`` exactly, making the left tail a pure
       exponential ``exp(N u)`` whose integral is closed-form;
    2. finds every local maximum within ``significance_gap`` nats of the
       global one;
    3. refines each with a bracketed saddle solve and adds Laplace
       contributions by log-sum-exp.  A peak whose bracket fails is
       integrated by local trapezoid instead.
    """

    _validate_q_inputs(d=d, L=L, partition=partition)
    N = sum(partition)
    if N == 0:
        return QLambdaResult(0.0, "scan", d, L, N, tuple(partition))
    if N == 1:
        return QLambdaResult(
            log_q=-math.log(d),
            method="symmetry",
            d=d,
            L=L,
            N=N,
            partition=tuple(partition),
            message="one-symbol profile by exchangeability",
        )
    if L == 1:
        return log_q_lambda_closed_l1(d=d, partition=partition)
    _validate_tables_for_partition(
        tables=tables, L=L, partition=partition, extra_orders=2
    )
    multiplicity_pairs = partition_multiplicities(partition)
    s = len(partition)
    u_grid = tables.u_grid

    # Vectorized log-integrand (without the -lgamma(N) prefactor).
    psi_grid = N * u_grid + (d - s) * tables.log_phi[(L, 0)]
    left_constant = 0.0  # sum of count * L * lgamma(r + 1): analytic left value
    for part, count in multiplicity_pairs:
        psi_grid = psi_grid + count * tables.log_phi[(L, part)]
        left_constant += count * L * math.lgamma(part + 1)

    return _scan_from_psi(
        d=d, L=L, N=N, s=s, partition=tuple(partition),
        multiplicity_pairs=multiplicity_pairs, tables=tables,
        psi_grid=psi_grid, left_constant=left_constant,
        significance_gap=significance_gap,
    )


def _scan_from_psi(
    *,
    d: int,
    L: int,
    N: int,
    s: int,
    partition: tuple[int, ...],
    multiplicity_pairs,
    tables: ProductMomentTables,
    psi_grid: FloatArray,
    left_constant: float,
    significance_gap: float,
) -> QLambdaResult:
    """The peak-finding / refinement half of log_q_lambda_scan, split
    out so that FAMILIES of profiles sharing a grid integrand (a base
    profile and its one-observation augmentations; complexity notes,
    T3) pay for the O(G k) integrand once and reuse it in O(G) per
    member."""

    u_grid = tables.u_grid

    def derivative(u: float) -> float:
        return N - _weighted_rho_sum(
            d=d,
            s=s,
            L=L,
            multiplicity_pairs=multiplicity_pairs,
            tables=tables,
            u=u,
        )

    contributions: list[float] = []
    peak_locations: list[float] = []
    notes: list[str] = []

    # Analytic left tail: for u < u_grid[0], psi(u) = N u + left_constant up
    # to O(e^u), so the tail integral is exp(N u0 + left_constant) / N.
    u0 = float(u_grid[0])
    left_tail = N * u0 + left_constant - math.log(N)
    contributions.append(left_tail)
    peak_locations.append(u0)

    # If the derivative is still positive at the grid's left edge the
    # dominant saddle may sit inside the analytic region; solve there too.
    if derivative(u0) < 0.0:
        lower = u0 - 10.0
        for _ in range(200):
            if derivative(lower) > 0.0:
                break
            lower -= 25.0
        if derivative(lower) > 0.0:
            saddle = float(brentq(derivative, lower, u0, xtol=1e-11, rtol=1e-11))
            curv = -_weighted_rho_prime_sum(
                d=d, s=s, L=L,
                multiplicity_pairs=multiplicity_pairs, tables=tables, u=saddle,
            )
            if math.isfinite(curv) and curv < 0.0:
                psi_val = _log_q_integrand_without_gamma(
                    d=d, L=L, N=N, s=s,
                    multiplicity_pairs=multiplicity_pairs, tables=tables,
                    u=saddle,
                )
                # replace the flat left-tail estimate with the refined peak
                contributions[-1] = _logaddexp_scalar(
                    psi_val + 0.5 * math.log(2.0 * math.pi / (-curv)),
                    left_tail,
                )
                peak_locations[-1] = saddle
                notes.append(f"left-region saddle at u={saddle:.2f}")

    # Interior local maxima of the grid integrand.
    finite = np.isfinite(psi_grid)
    peak_value = float(np.max(psi_grid[finite])) if np.any(finite) else -math.inf
    interior = np.zeros(u_grid.size, dtype=bool)
    interior[1:-1] = (
        (psi_grid[1:-1] >= psi_grid[:-2])
        & (psi_grid[1:-1] >= psi_grid[2:])
        & np.isfinite(psi_grid[1:-1])
    )
    candidate_indices = [
        i for i in np.nonzero(interior)[0]
        if psi_grid[i] >= peak_value - significance_gap
    ]
    for i in candidate_indices:
        lo, hi = float(u_grid[i - 1]), float(u_grid[i + 1])
        d_lo, d_hi = derivative(lo), derivative(hi)
        if d_lo > 0.0 > d_hi:
            saddle = float(brentq(derivative, lo, hi, xtol=1e-11, rtol=1e-11))
            curv = -_weighted_rho_prime_sum(
                d=d, s=s, L=L,
                multiplicity_pairs=multiplicity_pairs, tables=tables, u=saddle,
            )
            if math.isfinite(curv) and curv < 0.0:
                psi_val = _log_q_integrand_without_gamma(
                    d=d, L=L, N=N, s=s,
                    multiplicity_pairs=multiplicity_pairs, tables=tables,
                    u=saddle,
                )
                contributions.append(
                    psi_val + 0.5 * math.log(2.0 * math.pi / (-curv))
                )
                peak_locations.append(saddle)
                notes.append(f"saddle at u={saddle:.4f}")
                continue
        # fallback: local trapezoid over the surrounding grid patch
        window = slice(max(0, i - 200), min(u_grid.size, i + 201))
        contributions.append(
            _log_trapezoid_integral(psi_grid[window], u_grid[window])
        )
        peak_locations.append(float(u_grid[i]))
        notes.append(f"local trapezoid around u={float(u_grid[i]):.4f}")

    right_gap = peak_value - float(psi_grid[-1])
    converged = bool(right_gap >= 10.0 and len(contributions) > 0)
    log_q = _logsumexp_list(contributions) - math.lgamma(N)
    return QLambdaResult(
        log_q=float(log_q),
        method="scan",
        d=d,
        L=L,
        N=N,
        partition=tuple(partition),
        right_gap=right_gap,
        converged=converged,
        message="; ".join(notes) if notes else "left tail only",
        peaks=tuple(zip(peak_locations, contributions)),
    )


def augmented_partition(base: tuple[int, ...], c: int) -> tuple[int, ...]:
    """The profile after one more observation of a symbol whose current
    count is c (c = 0: a previously unseen symbol)."""

    if c == 0:
        return tuple(sorted(base + (1,)))
    lst = list(base)
    lst[lst.index(c)] = c + 1
    return tuple(sorted(lst))


def log_q_lambda_scan_family(
    *,
    d: int,
    L: int,
    base_partition: tuple[int, ...],
    cs: tuple[int, ...],
    tables: ProductMomentTables,
    significance_gap: float = 40.0,
) -> tuple[QLambdaResult, dict[int, QLambdaResult]]:
    """Scan for a base profile AND each one-observation augmentation,
    sharing the grid integrand (complexity notes, T3).

    Augmenting a symbol of count c multiplies the integrand by
    t phi_{c+1}/phi_c (c = 0: one of the d-s unseen symbols moves to
    count one, giving t phi_1/phi_0), so on the grid each member costs
    one O(G) update on top of the base's O(G k).  Requires L >= 2 and
    a nonempty base with sum(base) >= 2; peak refinement runs per
    member (it is the cheap part).  Returns (base result, {c: result}).
    """

    if L < 2:
        raise ValueError("family scan requires L >= 2 (use closed form)")
    base = tuple(base_partition)
    if not base or sum(base) < 1:
        raise ValueError("family scan requires a nonempty base profile")
    for c in cs:
        if c != 0 and c not in base:
            raise ValueError(f"augmentation count {c} not present in base")
    _validate_tables_for_partition(
        tables=tables, L=L, partition=base, extra_orders=2
    )
    for c in cs:
        _validate_tables_for_partition(
            tables=tables, L=L, partition=augmented_partition(base, c),
            extra_orders=2,
        )

    N = sum(base)
    s = len(base)
    u_grid = tables.u_grid
    psi_base = N * u_grid + (d - s) * tables.log_phi[(L, 0)]
    left_constant = 0.0
    multiplicity_pairs = partition_multiplicities(base)
    for part, count in multiplicity_pairs:
        psi_base = psi_base + count * tables.log_phi[(L, part)]
        left_constant += count * L * math.lgamma(part + 1)

    if N == 1:
        base_result = QLambdaResult(
            log_q=-math.log(d), method="symmetry", d=d, L=L, N=1,
            partition=base, message="one-symbol profile by exchangeability",
        )
    else:
        base_result = _scan_from_psi(
            d=d, L=L, N=N, s=s, partition=base,
            multiplicity_pairs=multiplicity_pairs, tables=tables,
            psi_grid=psi_base, left_constant=left_constant,
            significance_gap=significance_gap,
        )

    aug_results: dict[int, QLambdaResult] = {}
    for c in cs:
        aug = augmented_partition(base, c)
        # multiply by t phi_{c+1}/phi_c: O(G) on the shared integrand
        psi_aug = (
            psi_base + u_grid
            + tables.log_phi[(L, c + 1)] - tables.log_phi[(L, c)]
        )
        left_aug = left_constant + L * (
            math.lgamma(c + 2.0) - math.lgamma(c + 1.0))
        aug_results[c] = _scan_from_psi(
            d=d, L=L, N=N + 1, s=len(aug), partition=aug,
            multiplicity_pairs=partition_multiplicities(aug), tables=tables,
            psi_grid=psi_aug, left_constant=left_aug,
            significance_gap=significance_gap,
        )
    return base_result, aug_results


def _logaddexp_scalar(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def _logsumexp_list(values: list[float]) -> float:
    finite = [v for v in values if v != -math.inf]
    if not finite:
        return -math.inf
    m = max(finite)
    return m + math.log(sum(math.exp(v - m) for v in finite))


def log_q_lambda_laplace(
    *,
    d: int,
    L: int,
    partition: tuple[int, ...],
    tables: ProductMomentTables,
    min_endpoint_gap: float = 10.0,
) -> QLambdaResult:
    """Approximate ``log q_lambda`` by a one-dimensional Laplace formula.

    Only the outer integral over the shared normalization variable is
    approximated.  The finite-``L`` moment functions are read from
    :class:`ProductMomentTables`.
    """

    _validate_q_inputs(d=d, L=L, partition=partition)
    N = sum(partition)
    if N == 0:
        return QLambdaResult(0.0, "laplace", d, L, N, tuple(partition))
    if N == 1:
        return QLambdaResult(
            log_q=-math.log(d),
            method="symmetry",
            d=d,
            L=L,
            N=N,
            partition=tuple(partition),
            message="one-symbol profile by exchangeability",
        )
    if L == 1:
        return log_q_lambda_closed_l1(d=d, partition=partition)

    _validate_tables_for_partition(
        tables=tables,
        L=L,
        partition=partition,
        extra_orders=2,
    )
    multiplicity_pairs = partition_multiplicities(partition)
    s = len(partition)

    def derivative(u: float) -> float:
        return N - _weighted_rho_sum(
            d=d,
            s=s,
            L=L,
            multiplicity_pairs=multiplicity_pairs,
            tables=tables,
            u=u,
        )

    # Heavy-count / deep-layer extension: when L*log(r_max + 1) is large the
    # saddle lies left of the tabulated grid.  Below the grid,
    # ``log_phi_value`` returns the exact t -> 0 asymptote
    # ``L * lgamma(r + 1)`` (the interpolation's analytic left fill), under
    # which rho, its derivative, and the integrand are all exact up to
    # O(e^u) corrections, so the bracket may be extended analytically far
    # below the grid at no table cost.
    max_part = max(part for part, _ in multiplicity_pairs)
    asymptotic_lower = -(L * math.log(max_part + 1.0)) - 40.0
    lower = min(float(tables.u_grid[0]), asymptotic_lower)
    upper = float(tables.u_grid[-1])
    lower_value = derivative(lower)
    for _ in range(50):
        if lower_value > 0.0:
            break
        lower -= 50.0
        lower_value = derivative(lower)
    upper_value = derivative(upper)
    if lower_value <= 0.0 or upper_value >= 0.0:
        message = (
            "saddlepoint is outside the product-moment grid: "
            f"derivative({lower:.3g})={lower_value:.3g}, "
            f"derivative({upper:.3g})={upper_value:.3g}"
        )
        return QLambdaResult(
            log_q=math.nan,
            method="laplace",
            d=d,
            L=L,
            N=N,
            partition=tuple(partition),
            saddle_u=None,
            converged=False,
            message=message,
        )

    saddle_u = float(brentq(derivative, lower, upper, xtol=1e-11, rtol=1e-11))
    psi = _log_q_integrand_without_gamma(
        d=d,
        L=L,
        N=N,
        s=s,
        multiplicity_pairs=multiplicity_pairs,
        tables=tables,
        u=saddle_u,
    )
    curvature = -_weighted_rho_prime_sum(
        d=d,
        s=s,
        L=L,
        multiplicity_pairs=multiplicity_pairs,
        tables=tables,
        u=saddle_u,
    )

    if not math.isfinite(curvature) or curvature >= 0.0:
        return QLambdaResult(
            log_q=math.nan,
            method="laplace",
            d=d,
            L=L,
            N=N,
            partition=tuple(partition),
            saddle_u=saddle_u,
            curvature=curvature,
            converged=False,
            message=f"non-negative saddle curvature {curvature:.6g}",
        )

    log_q = (
        -math.lgamma(N)
        + psi
        + 0.5 * math.log(2.0 * math.pi / (-curvature))
    )
    left_psi = _log_q_integrand_without_gamma(
        d=d,
        L=L,
        N=N,
        s=s,
        multiplicity_pairs=multiplicity_pairs,
        tables=tables,
        u=lower,
    )
    right_psi = _log_q_integrand_without_gamma(
        d=d,
        L=L,
        N=N,
        s=s,
        multiplicity_pairs=multiplicity_pairs,
        tables=tables,
        u=upper,
    )
    left_gap = psi - left_psi
    right_gap = psi - right_psi
    converged = left_gap >= min_endpoint_gap and right_gap >= min_endpoint_gap
    message = "Laplace saddlepoint"
    if not converged:
        message += (
            f"; grid endpoint may be close to peak "
            f"(left_gap={left_gap:.3g}, right_gap={right_gap:.3g})"
        )

    return QLambdaResult(
        log_q=float(log_q),
        method="laplace",
        d=d,
        L=L,
        N=N,
        partition=tuple(partition),
        saddle_u=saddle_u,
        curvature=float(curvature),
        left_gap=float(left_gap),
        right_gap=float(right_gap),
        converged=converged,
        message=message,
    )


def compute_log_q_by_partition(
    *,
    d: int,
    L: int,
    N: int,
    method: QMethod = "auto",
    tables: ProductMomentTables | None = None,
    u_min: float = -70.0,
    u_max: float = 35.0,
    u_points: int = 16_001,
    laguerre_order: int = 96,
    chunk_size: int = 512,
) -> dict[tuple[int, ...], float]:
    """Return ``log q_lambda`` for every partition of ``N``.

    ``method="auto"`` uses the exact closed form for ``L=1`` and the Laplace
    approximation otherwise.  Pass ``method="grid"`` for the older validation
    integral.
    """

    if N < 0:
        raise ValueError("N must be non-negative")
    partitions = _integer_partitions(N)
    if N == 0:
        return {(): 0.0}
    if N == 1:
        return {(1,): -math.log(d)}
    if method == "auto":
        method = "closed_l1" if L == 1 else "laplace"
    if method == "closed_l1" and L != 1:
        raise ValueError("method='closed_l1' is only valid for L=1")
    if method == "closed_l1" or L == 1:
        return {
            partition: log_q_lambda_closed_l1(d=d, partition=partition).log_q
            for partition in partitions
        }

    if tables is None:
        tables = build_product_moment_tables(
            max_L=L,
            max_r=N + (2 if method == "laplace" else 0),
            u_min=u_min,
            u_max=u_max,
            u_points=u_points,
            laguerre_order=laguerre_order,
            chunk_size=chunk_size,
        )

    log_q: dict[tuple[int, ...], float] = {}
    for partition in partitions:
        if method == "grid":
            result = log_q_lambda_grid(
                d=d,
                L=L,
                partition=partition,
                tables=tables,
            )
        elif method == "laplace":
            result = log_q_lambda_laplace(
                d=d,
                L=L,
                partition=partition,
                tables=tables,
            )
        else:
            raise ValueError(f"unknown q method {method!r}")
        if not result.converged:
            raise RuntimeError(result.message)
        log_q[partition] = result.log_q
    return log_q


def _weighted_rho_sum(
    *,
    d: int,
    s: int,
    L: int,
    multiplicity_pairs: tuple[tuple[int, int], ...],
    tables: ProductMomentTables,
    u: float,
) -> float:
    total = (d - s) * _rho(L=L, r=0, tables=tables, u=u)
    for r, count in multiplicity_pairs:
        total += count * _rho(L=L, r=r, tables=tables, u=u)
    return float(total)


def _weighted_rho_prime_sum(
    *,
    d: int,
    s: int,
    L: int,
    multiplicity_pairs: tuple[tuple[int, int], ...],
    tables: ProductMomentTables,
    u: float,
) -> float:
    total = (d - s) * _rho_prime(L=L, r=0, tables=tables, u=u)
    for r, count in multiplicity_pairs:
        total += count * _rho_prime(L=L, r=r, tables=tables, u=u)
    return float(total)


def _rho(*, L: int, r: int, tables: ProductMomentTables, u: float) -> float:
    log_phi_r = tables.log_phi_value(L=L, r=r, u=u)
    log_phi_next = tables.log_phi_value(L=L, r=r + 1, u=u)
    return float(math.exp(u + log_phi_next - log_phi_r))


def _rho_prime(*, L: int, r: int, tables: ProductMomentTables, u: float) -> float:
    log_phi_r = tables.log_phi_value(L=L, r=r, u=u)
    log_phi_next = tables.log_phi_value(L=L, r=r + 1, u=u)
    log_phi_second = tables.log_phi_value(L=L, r=r + 2, u=u)
    rho = math.exp(u + log_phi_next - log_phi_r)
    raw_second = math.exp(2.0 * u + log_phi_second - log_phi_r)
    value = rho + rho * rho - raw_second
    if value < 0.0 and value > -1e-10 * max(1.0, abs(rho)):
        return 0.0
    return float(value)


def _log_q_integrand_without_gamma(
    *,
    d: int,
    L: int,
    N: int,
    s: int,
    multiplicity_pairs: tuple[tuple[int, int], ...],
    tables: ProductMomentTables,
    u: float,
) -> float:
    total = N * u + (d - s) * tables.log_phi_value(
        L=L,
        r=0,
        u=u,
    )
    for part, count in multiplicity_pairs:
        total += count * tables.log_phi_value(L=L, r=part, u=u)
    return float(total)


def _validate_q_inputs(*, d: int, L: int, partition: tuple[int, ...]) -> None:
    if d <= 0:
        raise ValueError("d must be positive")
    if L <= 0:
        raise ValueError("L must be positive")
    partition_multiplicities(partition)
    if len(partition) > d:
        raise ValueError("partition uses more symbols than d")


def _validate_tables_for_partition(
    *,
    tables: ProductMomentTables,
    L: int,
    partition: tuple[int, ...],
    extra_orders: int,
) -> None:
    if tables.max_L < L:
        raise ValueError(f"tables only contain L up to {tables.max_L}")
    needed = {0}
    for offset in range(extra_orders + 1):
        needed.add(offset)
    for part, _ in partition_multiplicities(partition):
        for offset in range(extra_orders + 1):
            needed.add(part + offset)
    missing = [r for r in sorted(needed) if (L, r) not in tables.log_phi]
    if missing:
        raise ValueError(f"moment table is missing r values: {missing}")


def _integer_partitions(n: int, *, max_part: int | None = None) -> list[tuple[int, ...]]:
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return [()]
    if max_part is None or max_part > n:
        max_part = n

    partitions = []
    for first in range(max_part, 0, -1):
        for rest in _integer_partitions(n - first, max_part=min(first, n - first)):
            partitions.append((first, *rest))
    return partitions


def _log_trapezoid_integral(log_values: FloatArray, x_grid: FloatArray) -> float:
    if log_values.size != x_grid.size:
        raise ValueError("log_values and x_grid must have the same length")
    if log_values.size < 2:
        raise ValueError("at least two grid points are required")

    max_value = float(np.max(log_values))
    if not math.isfinite(max_value):
        return -math.inf
    integral = np.trapezoid(np.exp(log_values - max_value), x_grid)
    return max_value + math.log(float(integral))


def _logsumexp(values: FloatArray, axis: int | None = None) -> FloatArray:
    max_values = np.max(values, axis=axis, keepdims=True)
    finite = np.isfinite(max_values)
    shifted_sum = np.sum(np.exp(values - max_values), axis=axis, keepdims=True)
    out = max_values + np.log(shifted_sum)
    out = np.where(finite, out, -np.inf)
    if axis is None:
        return np.asarray(out).reshape(())
    return np.squeeze(out, axis=axis)
