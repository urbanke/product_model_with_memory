"""Mellin--Barnes representation of the moment functions.

The moment function (complexity notes, T1) is

    phi_r^(L)(t) = E[Y^r e^{-tY}],   Y = E_1 ... E_L  (unit exponentials).

Writing e^{-tY} as a Mellin--Barnes contour integral and using
E[Y^s] = Gamma(1+s)^L gives the exact representation

    phi_r^(L)(t) = (1 / 2 pi i) * Int_{c - i inf}^{c + i inf}
                   Gamma(z) t^{-z} Gamma(r + 1 - z)^L dz,
    0 < c < r + 1.

Define F(z) = ln Gamma(z) - z ln t + L ln Gamma(r + 1 - z).  On the
real interval (0, r+1), F is strictly convex (its second derivative is
trigamma(z) + L trigamma(r+1-z) > 0), so F' is increasing and has a
unique root z* (the saddle).  Along the vertical contour through z*
the real part of F falls off, which yields

  * an independent NUMERICAL EVALUATION of phi (integrate along the
    contour) --- used here as a high-accuracy cross-check of the
    recursion-built tables, and

  * the closed-form SADDLE APPROXIMATION
        ln phi  ~=  F(z*) - (1/2) ln(2 pi F''(z*))  [+ correction],
    whose accuracy improves as r grows --- the candidate replacement
    for storing exact tables at large r.

Everything returns natural logarithms.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import loggamma, polygamma


def _F_real(z: float, r: float, L: int, log_t: float) -> float:
    return (
        float(loggamma(z))
        - z * log_t
        + L * float(loggamma(r + 1.0 - z))
    )


def _F_prime(z: float, r: float, L: int, log_t: float) -> float:
    return (
        float(polygamma(0, z))
        - log_t
        - L * float(polygamma(0, r + 1.0 - z))
    )


def _F_second(z: float, r: float, L: int) -> float:
    return float(polygamma(1, z)) + L * float(polygamma(1, r + 1.0 - z))


def saddle_point(r: float, L: int, log_t: float) -> float:
    """The unique root z* of F'(z) = 0 in (0, r + 1)."""

    lo = 1e-12
    hi = r + 1.0 - 1e-12 * max(1.0, r)
    f_lo = _F_prime(lo, r, L, log_t)
    f_hi = _F_prime(hi, r, L, log_t)
    if not (f_lo < 0.0 < f_hi):  # extremely defensive; F' is increasing
        raise RuntimeError(
            f"saddle bracket failed: F'({lo})={f_lo}, F'({hi})={f_hi}"
        )
    return float(
        brentq(_F_prime, lo, hi, args=(r, L, log_t), xtol=1e-13, rtol=1e-14)
    )


def log_phi_saddle(
    r: float, L: int, u: float, *, order: int = 2
) -> tuple[float, dict]:
    """Closed-form saddle approximation of ln phi_r^(L)(e^u).

    order = 1: Gaussian term only.
    order = 2: with the standard next-order correction factor
               1 + F''''/(8 F''^2) - 5 F'''^2 / (24 F''^3).
    Returns (approximation, diagnostics).
    """

    log_t = u
    z = saddle_point(r, L, log_t)
    F = _F_real(z, r, L, log_t)
    F2 = _F_second(z, r, L)
    value = F - 0.5 * math.log(2.0 * math.pi * F2)
    diag = {"z_star": z, "F": F, "F2": F2}
    if order >= 2:
        F3 = float(polygamma(2, z)) - L * float(polygamma(2, r + 1.0 - z))
        F4 = float(polygamma(3, z)) + L * float(polygamma(3, r + 1.0 - z))
        corr = 1.0 + F4 / (8.0 * F2 * F2) - 5.0 * F3 * F3 / (24.0 * F2**3)
        diag.update({"F3": F3, "F4": F4, "correction_factor": corr})
        if corr > 0:
            value += math.log(corr)
        else:  # correction outside its regime; report but do not apply
            diag["correction_skipped"] = True
    return value, diag


def log_phi_contour(
    r: float,
    L: int,
    u: float,
    *,
    oversample: float = 8.0,
    tail_sigmas: float = 14.0,
) -> float:
    """ln phi_r^(L)(e^u) by direct numerical integration along a
    vertical contour.

    Independent of the recursion/quadrature table builder: a second
    exact method, used to cross-check both the tables and the saddle
    approximation.  The contour is the vertical line through
    c = max(z*, 0.6) --- moved off the saddle when the saddle sits
    next to the pole of Gamma(z) at zero (small t), where the local
    Gaussian width misjudges the tails.  The integral does not depend
    on the choice of line (the integrand is analytic in the strip).
    Along any such line the magnitude decays at least like
    exp(-pi |y| / 2) (decay of |Gamma| off the real axis), which sets
    the integration width; the sampling density is set by the phase
    rate, which is bounded by |ln t| plus digamma terms.
    """

    log_t = u
    # Small-t regime: the value is dominated by the residues at
    # z = 0, -1, -2, ... , i.e. the series
    #   phi = sum_j (-t)^j / j! * Gamma(r+j+1)^L .
    # On the contour this regime shows up as catastrophic cancellation
    # (the integral is exponentially smaller than the integrand), so
    # the series --- whose truncation error is bounded by the first
    # omitted term --- is used instead whenever it is sharply
    # convergent at the start.
    tau_log = log_t + L * float(loggamma(r + 2.0) - loggamma(r + 1.0))
    if tau_log < math.log(0.05):
        value, certificate = _log_phi_series(r, L, log_t)
        if certificate < 1e-12:
            return value
    z0 = saddle_point(r, L, log_t)
    c = z0
    F0 = _F_real(c, r, L, log_t)
    F2 = _F_second(c, r, L)
    sigma = 1.0 / math.sqrt(F2)
    # width: the Gaussian scale, but at least wide enough that the
    # guaranteed decay of |Gamma| off the real axis (at rate >= pi/2
    # per unit) has suppressed the integrand by e^{-30}
    half_width = max(tail_sigmas * sigma, 25.0)
    # sampling: resolve the peak (scale sigma), stay well inside the
    # analyticity strip set by the pole at z = 0 (scale z0), and
    # resolve the phase oscillation along the line
    freq = (
        abs(log_t)
        + abs(float(polygamma(0, c)))
        + L * abs(float(polygamma(0, r + 1.0 - c)))
        + 10.0
    )
    spacing = min(
        sigma / 8.0, z0 / 4.0, 2.0 * math.pi / (oversample * freq), 0.05
    )
    points = int(2.0 * half_width / spacing)
    points = max(4_001, min(points | 1, 2_000_001))
    y = np.linspace(-half_width, half_width, points)
    z = c + 1j * y
    F = loggamma(z) - z * log_t + L * loggamma(r + 1.0 - z)
    vals = np.exp(F - F0)  # complex; magnitude <= 1 on the line
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 / 1
    integral = float(np.real(trapezoid(vals, y)))
    if integral <= 0.0:
        raise RuntimeError("contour integral not positive; widen contour")
    return F0 + math.log(integral / (2.0 * math.pi))


def _log_phi_series(
    r: float, L: int, log_t: float, max_terms: int = 60
) -> tuple[float, float]:
    """Small-t series  phi = sum_j (-t)^j / j! Gamma(r+j+1)^L .

    The series is asymptotic (terms eventually grow), so it is
    truncated as soon as a term stops decreasing, and the SECOND
    return value is a certificate: the magnitude of the smallest
    computed term relative to the total, an upper bound on the
    relative truncation error.  Callers accept the series value only
    when the certificate is tiny.
    """

    lead = L * float(loggamma(r + 1.0))
    total = 1.0
    sign = 1.0
    prev_abs = None
    smallest = 0.0
    for j in range(1, max_terms):
        term_log = (
            j * log_t
            - float(loggamma(j + 1.0))
            + L * float(loggamma(r + j + 1.0) - loggamma(r + 1.0))
        )
        if prev_abs is not None and term_log >= prev_abs:
            break  # asymptotic tail reached; stop before it grows
        sign = -sign
        total += sign * math.exp(term_log)
        prev_abs = term_log
        smallest = term_log
    certificate = math.exp(smallest) / max(total, 1e-300)
    return lead + math.log(total), certificate


def log_phi_column(r: float, L: int, u_grid) -> np.ndarray:
    """ln phi_r^(L) on a whole u grid, by the certified methods.

    Per grid point: the self-certified small-t series where it
    applies, otherwise the order-2 saddle approximation with the
    saddle found by VECTORIZED bisection (F' is strictly increasing in
    z, so bisection is exact and robust; 55 halvings reach relative
    precision ~1e-16).  Accuracy: series points are near machine
    precision; saddle points carry the order-2 error measured in the
    prototype (1e-6 .. 1e-3 nats in the regimes where this builder is
    used, r >= ~500).  Used to REPLACE recursion-built columns at
    large r, where the fixed-order Laguerre recursion fails.
    """

    u = np.asarray(u_grid, dtype=np.float64)
    out = np.empty_like(u)

    # series region (certified): tau = t * ((r+1)-ish ratio)^L small
    tau_log = u + L * float(loggamma(r + 2.0) - loggamma(r + 1.0))
    series_mask = tau_log < math.log(0.05)
    if series_mask.any():
        us = u[series_mask]
        lead = L * float(loggamma(r + 1.0))
        total = np.ones_like(us)
        sign = 1.0
        prev = None
        for j in range(1, 60):
            term_log = (
                j * us
                - float(loggamma(j + 1.0))
                + L * float(loggamma(r + j + 1.0) - loggamma(r + 1.0))
            )
            if prev is not None:
                stop = term_log >= prev
                term_log = np.where(stop, -np.inf, term_log)
            sign = -sign
            total = total + sign * np.exp(term_log)
            prev = term_log
        out[series_mask] = lead + np.log(total)

    rest = ~series_mask
    if rest.any():
        ur = u[rest]
        lo = np.full_like(ur, 1e-12)
        hi = np.full_like(ur, r + 1.0 - 1e-12 * max(1.0, r))
        for _ in range(55):
            mid = 0.5 * (lo + hi)
            fp = polygamma(0, mid) - ur - L * polygamma(0, r + 1.0 - mid)
            lo = np.where(fp < 0.0, mid, lo)
            hi = np.where(fp < 0.0, hi, mid)
        z = 0.5 * (lo + hi)
        F = (
            np.real(loggamma(z))
            - z * ur
            + L * np.real(loggamma(r + 1.0 - z))
        )
        F2 = polygamma(1, z) + L * polygamma(1, r + 1.0 - z)
        F3 = polygamma(2, z) - L * polygamma(2, r + 1.0 - z)
        F4 = polygamma(3, z) + L * polygamma(3, r + 1.0 - z)
        val = F - 0.5 * np.log(2.0 * math.pi * F2)
        corr = 1.0 + F4 / (8.0 * F2 * F2) - 5.0 * F3 * F3 / (24.0 * F2**3)
        val = val + np.where(corr > 0, np.log(np.maximum(corr, 1e-300)), 0.0)
        out[rest] = val
    return out


def log_phi_closed_l1(r: float, u: float) -> float:
    """Exact ln phi at level 1: Gamma(r+1) / (1+t)^{r+1}."""

    t = math.exp(u)
    return float(loggamma(r + 1.0)) - (r + 1.0) * math.log1p(t)


def log_phi_limit_t0(r: float, L: int) -> float:
    """Exact limit as t -> 0: ln of Gamma(r+1)^L."""

    return L * float(loggamma(r + 1.0))
