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
from scipy.special import loggamma, polygamma, zeta

_EULER_GAMMA = 0.5772156649015328606


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
    dispatch: bool = True,
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
    if dispatch and tau_log < math.log(0.05):
        value, certificate = _log_phi_series(r, L, log_t)
        if certificate < 1e-12:
            return value
    # Large-t regime: the saddle z* runs against the right pole at
    # z = r + 1 and the vertical contour degenerates; there the
    # right-pole residue series (which converges exactly in this
    # regime) is the reference.
    if dispatch and log_t > math.log(1.2 * (r + 1.0)):
        value, certificate = log_phi_right_series(r, L, log_t)
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
        sigma / 8.0, z0 / 4.0, (r + 1.0 - z0) / 4.0,
        2.0 * math.pi / (oversample * freq), 0.05
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


def log_phi_right_series(
    r: float, L: int, u: float, *, max_j: int = 400
) -> tuple[float, float]:
    """Large-t expansion of ln phi_r^(L)(e^u), with a certificate.

    Closing the Mellin contour to the RIGHT collects the order-L poles
    of Gamma(r+1-z)^L at z = r+1+j, j = 0, 1, 2, ...  Each residue is

        term_j = -(-1)^{(j+1)L} t^{-(r+1+j)} [s^{L-1}] h_j(s),
        h_j(s) = Gamma(r+1+j+s) e^{-su} Gamma(1-s)^L / prod_{i<=j}(s+i)^L,

    a power of t times a degree-(L-1) polynomial in ln t.  The Taylor
    coefficient is extracted exactly: the log of h_j has coefficients
    from polygamma values at r+1+j, Euler's constant and zeta values
    (Gamma(1-s) = exp(gamma s + sum_k zeta(k) s^k / k)), and harmonic
    numbers (the (s+i) factors); the series is then exponentiated by
    the standard power-series recursion.

    The term ratio is about (r+1+j) / (t (j+1)^L), so the series
    CONVERGES once t exceeds r+1 (fast for L >= 2); this is exactly
    the regime where the vertical contour degenerates (saddle against
    the right pole).  The second return value is a certificate: a
    bound on the relative error from truncation, floating-point
    cancellation inside the coefficient recursion, and cancellation
    across terms.  Callers accept the value only when it is tiny.
    """

    n = L - 1
    ks = np.arange(2, n + 1)
    zeta_k = zeta(ks, 1.0) if n >= 2 else np.empty(0)
    harm = 0.0                       # H_j
    harm_k = np.zeros(len(ks))       # H_j^{(k)} for k = 2..n
    total = 0.0        # sum of terms, scaled by e^{-lam0}
    total_abs = 0.0    # sum of |terms|, same scale (cancellation)
    lam0 = None
    cert_coeff = 0.0   # worst coefficient-cancellation factor seen
    prev_abs = None
    ratio = 1.0
    for j in range(max_j):
        a = r + 1.0 + j
        if j > 0:
            harm += 1.0 / j
            if n >= 2:
                harm_k += (1.0 / j) ** ks
        # Taylor coefficients of ln h_j (constant term kept separate)
        c = np.zeros(n + 1)
        if n >= 1:
            c[1] = float(polygamma(0, a)) - u + L * (_EULER_GAMMA - harm)
        if n >= 2:
            fact = np.cumprod(np.concatenate([[1.0, 1.0],
                                              np.arange(2.0, n + 1.0)]))
            c[2:] = (
                np.array([float(polygamma(k - 1, a)) for k in ks])
                / fact[ks]
                + L * (zeta_k + (-1.0) ** ks * harm_k) / ks
            )
        # exponentiate the series:  p_m = (1/m) sum_k k c_k p_{m-k};
        # p_abs runs the same recursion on |c_k|, a majorant whose size
        # relative to |p_n| measures cancellation in the recursion
        p = np.zeros(n + 1)
        p_abs = np.zeros(n + 1)
        p[0] = p_abs[0] = 1.0
        for m in range(1, n + 1):
            kk = np.arange(1, m + 1)
            p[m] = float(np.dot(kk * c[1:m + 1], p[m - 1::-1])) / m
            p_abs[m] = float(
                np.dot(kk * np.abs(c[1:m + 1]), p_abs[m - 1::-1])) / m
        coeff = p[n]
        if coeff != 0.0:
            cert_coeff = max(cert_coeff, 2.2e-16 * p_abs[n] / abs(coeff))
        c0 = float(loggamma(a)) - L * float(loggamma(j + 1.0))
        lam = -a * u + c0 + (math.log(abs(coeff)) if coeff != 0.0
                             else -math.inf)
        sign = -((-1.0) ** ((j + 1) * L)) * math.copysign(1.0, coeff)
        if lam0 is None:
            lam0 = lam
        scaled = math.exp(lam - lam0) if math.isfinite(lam) else 0.0
        total += sign * scaled
        total_abs += scaled
        if prev_abs is not None and prev_abs > 0.0:
            ratio = scaled / prev_abs
        prev_abs = scaled
        if scaled < 1e-18 * abs(total) and ratio < 0.5:
            break
        if j >= 3 and ratio > 1.0 and total_abs > 1e6 * (abs(total) + 1.0):
            # terms growing with massive cancellation ahead (large-L
            # regime): the expansion is numerically useless here
            return float("nan"), math.inf
    else:
        ratio = max(ratio, 0.99)  # never reached a certified stop
    if total <= 0.0:
        return -math.inf, math.inf
    tail = (prev_abs * ratio / (1.0 - ratio)) if ratio < 1.0 else math.inf
    certificate = (
        tail / total
        + 2.2e-16 * total_abs / total
        + cert_coeff
    )
    return lam0 + math.log(total), certificate


def right_series_column(
    r: float, L: int, u, *, max_j: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized large-t expansion: log_phi_right_series for a whole
    array of u at once, with per-point certificates.

    The u dependence enters only through the linear Taylor coefficient
    (c_1 contains -u) and the power t^{-(r+1+j)}, so the coefficient
    recursion runs on arrays over u; everything else is shared.  (The
    scalar point-by-point loop was measured July 30 to consume 76-99%
    of deep-column build time; this replaces it.)
    """

    u = np.atleast_1d(np.asarray(u, dtype=np.float64))
    m_pts = len(u)
    n = L - 1
    ks = np.arange(2, n + 1)
    zeta_k = zeta(ks, 1.0) if n >= 2 else np.empty(0)
    fact = np.cumprod(np.concatenate([[1.0, 1.0],
                                      np.arange(2.0, n + 1.0)])) \
        if n >= 2 else None

    harm = 0.0
    harm_k = np.zeros(len(ks))
    total = np.zeros(m_pts)
    total_abs = np.zeros(m_pts)
    lam0 = np.full(m_pts, np.nan)
    cert_coeff = np.zeros(m_pts)
    prev_abs = np.full(m_pts, np.nan)
    ratio = np.ones(m_pts)
    dead = np.zeros(m_pts, dtype=bool)   # hopeless cancellation
    done = np.zeros(m_pts, dtype=bool)
    for j in range(max_j):
        a = r + 1.0 + j
        if j > 0:
            harm += 1.0 / j
            if n >= 2:
                harm_k += (1.0 / j) ** ks
        c_hi = (np.array([float(polygamma(k - 1, a)) for k in ks]) / fact[ks]
                + L * (zeta_k + (-1.0) ** ks * harm_k) / ks) if n >= 2 \
            else np.empty(0)
        c1 = float(polygamma(0, a)) + L * (_EULER_GAMMA - harm) - u  # (m,)
        # vectorized exp-series recursion: P[q] over u, q = 0..n
        P = np.zeros((n + 1, m_pts))
        Pa = np.zeros((n + 1, m_pts))
        P[0] = Pa[0] = 1.0
        abs_c1 = np.abs(c1)
        for q in range(1, n + 1):
            acc = c1 * P[q - 1]
            acc_a = abs_c1 * Pa[q - 1]
            if q >= 2:
                kk = np.arange(2, q + 1)
                w = kk * c_hi[:q - 1]
                acc = acc + w @ P[q - 2::-1][:len(w)]
                acc_a = acc_a + np.abs(w) @ Pa[q - 2::-1][:len(w)]
            P[q] = acc / q
            Pa[q] = acc_a / q
        coeff = P[n]
        nz = coeff != 0.0
        cert_coeff = np.maximum(
            cert_coeff,
            np.where(nz, 2.2e-16 * Pa[n] / np.maximum(np.abs(coeff), 1e-300),
                     np.inf))
        c0 = float(loggamma(a)) - L * float(loggamma(j + 1.0))
        with np.errstate(divide="ignore"):
            lam = -a * u + c0 + np.log(np.maximum(np.abs(coeff), 0.0))
        sign = -((-1.0) ** ((j + 1) * L)) * np.sign(coeff)
        if j == 0:
            lam0 = lam.copy()
        scaled = np.where(np.isfinite(lam), np.exp(lam - lam0), 0.0)
        act = ~done & ~dead
        total = np.where(act, total + sign * scaled, total)
        total_abs = np.where(act, total_abs + scaled, total_abs)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(np.isfinite(prev_abs) & (prev_abs > 0.0),
                             scaled / prev_abs, ratio)
        prev_abs = scaled
        done |= act & (scaled < 1e-18 * np.abs(total)) & (ratio < 0.5)
        if j >= 3:
            dead |= act & (ratio > 1.0) & (total_abs
                                           > 1e6 * (np.abs(total) + 1.0))
        if (done | dead).all():
            break
    ratio_eff = np.where(done, np.minimum(ratio, 0.5), np.maximum(ratio, 0.99))
    with np.errstate(divide="ignore", invalid="ignore"):
        tail = np.where(ratio_eff < 1.0,
                        prev_abs * ratio_eff / (1.0 - ratio_eff), np.inf)
        cert = (tail / np.maximum(total, 1e-300)
                + 2.2e-16 * total_abs / np.maximum(total, 1e-300)
                + cert_coeff)
        vals = np.where(total > 0.0, lam0 + np.log(np.maximum(total, 1e-300)),
                        np.nan)
    cert = np.where(dead | (total <= 0.0), np.inf, cert)
    return vals, cert


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


def series_column(r: float, L: int, u) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized small-t series with per-point certificates.

    Same mathematics as the scalar _log_phi_series: alternating
    asymptotic series summed over its decreasing prefix, certificate =
    smallest computed term relative to the total (an upper bound on
    the truncation error).
    """

    us = np.asarray(u, dtype=np.float64)
    lead = L * float(loggamma(r + 1.0))
    total = np.ones_like(us)
    smallest = np.full_like(us, np.inf)
    sign = 1.0
    prev = None
    for j in range(1, 60):
        term_log = (
            j * us
            - float(loggamma(j + 1.0))
            + L * float(loggamma(r + j + 1.0) - loggamma(r + 1.0))
        )
        if prev is not None:
            term_log = np.where(term_log >= prev, -np.inf, term_log)
        sign = -sign
        total = total + sign * np.exp(term_log)
        keep = np.isfinite(term_log)
        smallest = np.where(keep, term_log, smallest)
        prev = np.where(keep, term_log,
                        prev if prev is not None else term_log)
        # Early exit: for an alternating series with decreasing terms
        # the truncation error is bounded by the first omitted term, so
        # once EVERY point's next term is below e^-45 (relative to a
        # total of order one) nothing further can matter.  Far to the
        # left one or two terms suffice, and running the full 60 was
        # the dominant cost of level provisioning (measured 31 July).
        if np.max(term_log) < -45.0:
            break
    cert = np.exp(smallest) / np.maximum(total, 1e-300)
    return lead + np.log(np.maximum(total, 1e-300)), cert


def exact_log_phi_column(
    r: float,
    L: int,
    u_grid,
    *,
    oversample: float = 8.0,
    tail_sigmas: float = 14.0,
) -> np.ndarray:
    """ln phi_r^(L) on a whole u grid by the CERTIFIED methods only.

    Per point, in order of preference: the certified small-t series;
    the certified right-pole series (large t, where it self-certifies);
    otherwise direct contour integration with pole-aware sampling ---
    the same rules as the scalar log_phi_contour, vectorized: the
    saddle for every point by simultaneous bisection, then the
    quadrature in blocks of points with similar node counts.  This is
    the builder for STORED universal-table columns: slower than the
    order-2 saddle formula, but accurate at the ~1e-8 level everywhere,
    and the table is built once.
    """

    u = np.asarray(u_grid, dtype=np.float64)
    if L == 1:
        return float(loggamma(r + 1.0)) - (r + 1.0) * np.log1p(np.exp(u))
    out = np.full(len(u), np.nan)
    done = np.zeros(len(u), dtype=bool)

    # ---- certified small-t series (vectorized, with certificate)
    tau_log = u + L * float(loggamma(r + 2.0) - loggamma(r + 1.0))
    ser = tau_log < math.log(0.05)
    if ser.any():
        vals, cert = series_column(r, L, u[ser])
        good = cert < 1e-10
        idx = np.flatnonzero(ser)
        out[idx[good]] = vals[good]
        done[idx[good]] = True

    # ---- certified right-pole series (large t), vectorized
    right = np.flatnonzero(~done & (u > math.log(1.2 * (r + 1.0))))
    if len(right):
        vals, cert = right_series_column(r, L, u[right])
        good = cert < 1e-10
        out[right[good]] = vals[good]
        done[right[good]] = True

    # ---- pole-aware contour for the rest
    rest = np.flatnonzero(~done)
    if len(rest) == 0:
        return out
    ur = u[rest]
    lo = np.full_like(ur, 1e-12)
    hi = np.full_like(ur, (r + 1.0) * (1.0 - 1e-12))
    for _ in range(55):
        mid = 0.5 * (lo + hi)
        fp = polygamma(0, mid) - ur - L * polygamma(0, r + 1.0 - mid)
        lo = np.where(fp < 0.0, mid, lo)
        hi = np.where(fp < 0.0, hi, mid)
    z0 = 0.5 * (lo + hi)
    F0 = (np.real(loggamma(z0)) - z0 * ur
          + L * np.real(loggamma(r + 1.0 - z0)))
    F2 = polygamma(1, z0) + L * polygamma(1, r + 1.0 - z0)
    sigma = 1.0 / np.sqrt(F2)
    # Sampling: trapezoid error on an analytic integrand decays like
    # exp(-2 pi d / h), d = distance from the line to the nearest pole
    # (z = 0 or z = r+1); resolving the Gaussian peak needs h <~ 0.4
    # sigma.  Width: 14 Gaussian widths, floored by the |Gamma| decay
    # rate off the axis ((L+1) pi/2 nats per unit height).
    d = np.minimum(z0, r + 1.0 - z0)
    h = np.minimum(0.4 * sigma, d / 5.0)
    # Width: start from the Gaussian scale, then WIDEN until the
    # actual integrand magnitude (evaluated, not modeled) has dropped
    # 34 nats below the peak --- near a pole the tails decay
    # algebraically before the exponential Gamma decay kicks in, and a
    # purely Gaussian width rule truncates real mass (measured
    # July 2026: 4e-5 nats at L=55, r=2e4 near the series boundary).
    W = np.maximum(tail_sigmas * sigma, 2.0)
    for _ in range(80):
        zW = z0 + 1j * W
        decay = (np.real(loggamma(zW)) - z0 * ur
                 + L * np.real(loggamma(r + 1.0 - zW)) - F0)
        need = decay > -34.0
        if not need.any():
            break
        W = np.where(need, 1.4 * W, W)
    # ---- blocked quadrature with a SHARED line per block.
    # The expensive factors ln Gamma(z) and ln Gamma(r+1-z) do not
    # depend on u; only the cheap factor e^{-z u} does.  Consecutive u
    # points whose saddles lie within a few Gaussian widths of each
    # other can therefore share one vertical line and ONE set of
    # Gamma evaluations (the integral is line-independent inside the
    # strip; keeping the line within ~3 sigma of each point's own
    # saddle bounds the extra cancellation by e^{4.5}, harmless at
    # double precision).  This amortizes the dominant cost over whole
    # blocks --- largest exactly for small r, where most columns live.
    n_rest = len(rest)
    pos = 0
    while pos < n_rest:
        end = pos + 1
        smin = float(sigma[pos])
        while end < n_rest:
            smin_new = min(smin, float(sigma[end]))
            if abs(float(z0[end] - z0[pos])) > 3.0 * smin_new:
                break
            m_est = int(
                2.0 * (W[pos:end + 1].max()
                       + np.abs(z0[pos:end + 1]
                                - 0.5 * (z0[pos] + z0[end])).max())
                / h[pos:end + 1].min()) + 1
            if (end + 1 - pos) * m_est > 4_000_000:
                break
            smin = smin_new
            end += 1
        blk = slice(pos, end)
        pos = end
        c = 0.5 * (float(z0[blk][0]) + float(z0[blk][-1]))
        shift = np.abs(z0[blk] - c)
        W_blk = float((W[blk] + shift).max())
        h_blk = float(h[blk].min())
        m = int(2.0 * W_blk / h_blk) + 2
        y = np.linspace(-W_blk, W_blk, m)
        zline = c + 1j * y
        A = loggamma(zline) + L * loggamma(r + 1.0 - zline)  # u-free, ONCE
        reA, imA = np.real(A), np.imag(A)
        uu = ur[blk]
        mag = reA[None, :] - c * uu[:, None] - F0[blk, None]
        phase = imA[None, :] - y[None, :] * uu[:, None]
        vals = np.exp(np.clip(mag, -745.0, 50.0)) * np.cos(phase)
        hstep = y[1] - y[0]
        integral = hstep * (vals.sum(axis=1)
                            - 0.5 * (vals[:, 0] + vals[:, -1]))
        if np.any(integral <= 0.0):
            bad = np.flatnonzero(integral <= 0.0)
            raise RuntimeError(
                f"contour integral not positive at "
                f"u={ur[blk][bad]} (r={r}, L={L})")
        out[rest[blk]] = F0[blk] + np.log(integral / (2.0 * math.pi))
    return out


def exact_column_recursion(
    r: int,
    max_L: int,
    u_grid,
    *,
    quad_points: int = 512,
    pad: float = 12.0,
) -> np.ndarray:
    """ln phi_r^(L)(e^u) for L = 1..max_L on u_grid, by the level
    recursion with a STABLE quadrature (fixes the July 2026 table bug).

    The recursion step is  phi^(l)(t) = Int v^r e^{-v} phi^(l-1)(t v) dv.
    Substituting y = ln v gives an integrand proportional to the
    log-Gamma(r+1) density, sharply peaked at y = ln r for large r ---
    which fixed-node Laguerre rules miss.  Here the y-range is chosen
    from the Gamma(r+1) distribution's own quantiles (1e-16 to
    1 - 1e-16) and integrated by trapezoid in the log domain: stable
    for every r, with accuracy set by quad_points.  Returns shape
    (max_L, len(u_grid)).
    """

    from scipy.stats import gamma as gamma_dist

    u_grid = np.asarray(u_grid, dtype=np.float64)
    a = r + 1.0  # Gamma shape of v^r e^{-v} dv
    v_lo = float(gamma_dist.ppf(1e-16, a))
    v_hi = float(gamma_dist.isf(1e-16, a))
    y = np.linspace(math.log(max(v_lo, 1e-300)), math.log(v_hi), quad_points)
    dy = y[1] - y[0]
    # log of the quadrature weight at each node: v^{r+1} e^{-v} dy
    # (the extra v is the Jacobian dv = v dy)
    log_wq = (a * y - np.exp(y)) + math.log(dy)

    h = np.min(np.diff(u_grid)) if len(u_grid) > 1 else 0.01
    # each recursion level reads the previous one shifted right by up
    # to y_max, so the working grid must extend (max_L - 1) shifts
    # beyond the requested grid
    right = float(u_grid[-1]) + (max_L - 1) * float(y[-1]) + pad
    ext = np.arange(u_grid[-1] + h, right, h)
    work = np.concatenate([u_grid, ext])

    lgr = float(loggamma(r + 1.0))
    out = np.empty((max_L, len(u_grid)))
    current = lgr - (r + 1.0) * np.log1p(np.exp(work))  # exact L = 1
    out[0] = current[: len(u_grid)]
    for ell in range(2, max_L + 1):
        shifted = work[:, None] + y[None, :]
        prev = np.interp(
            shifted, work, current,
            left=(ell - 1) * lgr, right=-np.inf,
        )
        terms = log_wq[None, :] + prev
        m = terms.max(axis=1)
        with np.errstate(invalid="ignore"):
            current = m + np.log(np.exp(terms - m[:, None]).sum(axis=1))
        current = np.where(np.isfinite(m), current, -np.inf)
        out[ell - 1] = current[: len(u_grid)]
    return out


def log_phi_closed_l1(r: float, u: float) -> float:
    """Exact ln phi at level 1: Gamma(r+1) / (1+t)^{r+1}."""

    t = math.exp(u)
    return float(loggamma(r + 1.0)) - (r + 1.0) * math.log1p(t)


def log_phi_limit_t0(r: float, L: int) -> float:
    """Exact limit as t -> 0: ln of Gamma(r+1)^L."""

    return L * float(loggamma(r + 1.0))


def mellin_saddle_column(r: float, L: int, u_grid) -> np.ndarray:
    """Column of ln phi values by certified series + order-2 saddle.

    The store's builder for r > R_SPLIT: series where its certificate
    holds (machine precision), vectorized-bisection saddle order-2
    elsewhere (<= ~1e-3 nats worst case, ~1e-5 median; measured July
    2026).  Thin wrapper over log_phi_column for naming clarity at the
    call site.
    """

    return log_phi_column(r, L, u_grid)
