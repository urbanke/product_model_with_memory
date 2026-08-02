/* Saddle solve for one peak of the log-integrand, in C.
 *
 * The Python path evaluated the derivative through a closure over
 * numpy slices: 2.7 million calls to `derivative` and 3.3 million to
 * `_phis` for a single first-order run on the enwik8 subword stream
 * (31 July), each doing six numpy operations on a vector of length
 * k+1 with k typically under thirty.  At that size the per-call
 * overhead is the cost, not the arithmetic.  This does the whole
 * bracketed solve --- around forty derivative evaluations, then the
 * curvature and the log-integrand at the root --- in one call.
 *
 * ARITHMETIC.  The interpolation, the products and the sums are
 * written to match the numpy path term for term, and the file must be
 * compiled with -ffp-contract=off: at -O3 the compiler fuses a*b+c
 * into one rounding step and the results move in the last bits.  The
 * sums are accumulated in the order numpy's pairwise reduction uses
 * for short vectors (sequential below eight elements, then the
 * eight-way tree).  What cannot be matched is `exp`: numpy dispatches
 * float64 exp to its own SIMD kernel above length eight, and that
 * kernel differs from libm's by up to one ulp.  Below that length the
 * two agree exactly.  So this returns bit-identical results for
 * profiles with at most seven distinct counts and results within
 * about one ulp above it; `PMM_KERNEL=0` restores the Python path for
 * cross-checking.
 *
 * BRENT.  Ported from scipy's Zeros/brentq.c so that the iterate
 * sequence, and therefore the root, is the same one scipy would
 * produce from the same function values.  The tolerance is
 * load-bearing here: relaxing xtol from 1e-11 to 1e-8 moved log q by
 * 1.4 nats in an earlier experiment, so it is passed in rather than
 * chosen locally.
 */

#include <math.h>
#include <stddef.h>

#define PMM_MAXITER 100

typedef struct {
    const double *phi_r;      /* k x m, row-major */
    const double *phi_n;
    const double *phi_s;
    const double *left_r;     /* k */
    const double *left_n;
    const double *left_s;
    const double *counts;     /* k */
    const double *u;          /* m, ascending, uniform not assumed */
    long k;
    long m;
    double N;
} kernel_ctx;

/* numpy's reduction order for a short contiguous float64 vector:
   sequential below eight, then the eight-way pairwise tree. */
static double pairwise_sum(const double *v, long n)
{
    if (n < 8) {
        double s = 0.0;
        for (long i = 0; i < n; ++i) s += v[i];
        return s;
    }
    double r[8];
    for (int j = 0; j < 8; ++j) r[j] = v[j];
    long i = 8;
    for (; i + 8 <= n; i += 8)
        for (int j = 0; j < 8; ++j) r[j] += v[i + j];
    double s = ((r[0] + r[1]) + (r[2] + r[3]))
             + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) s += v[i];
    return s;
}

/* (ln phi_r, ln phi_{r+1}, ln phi_{r+2}) for every part at u.
   Returns 0 on success, 1 when u lies right of the slice (the caller
   must fall back, exactly as the Python `_phis` returns None). */
static int phis(const kernel_ctx *c, double u,
                double *pr, double *pn, double *ps)
{
    if (u < c->u[0]) {
        for (long i = 0; i < c->k; ++i) {
            pr[i] = c->left_r[i];
            pn[i] = c->left_n[i];
            ps[i] = c->left_s[i];
        }
        return 0;
    }
    if (u > c->u[c->m - 1]) return 1;

    /* np.searchsorted(u_local, u) - 1, clamped: side="left", so the
       first index with u_local[i] >= u.  Getting this wrong (side
       "right") changes the interpolation cell at points that sit
       exactly on a grid node --- which the BRACKET ENDPOINTS always
       do.  At a node both cells agree mathematically but not in the
       last bits, and that was enough to move the whole Brent iterate
       sequence: measured 1.5e-4 bits of drift on heavy byte profiles
       before this was fixed (31 July). */
    long lo = 0, hi = c->m;
    while (lo < hi) {                      /* first index with u_local >= u */
        long mid = (lo + hi) / 2;
        if (c->u[mid] < u) lo = mid + 1; else hi = mid;
    }
    long pos = lo - 1;
    if (pos < 0) pos = 0;
    if (pos > c->m - 2) pos = c->m - 2;

    const double w = (u - c->u[pos]) / (c->u[pos + 1] - c->u[pos]);
    const long m = c->m;
    for (long i = 0; i < c->k; ++i) {
        const double a = c->phi_r[i * m + pos], b = c->phi_r[i * m + pos + 1];
        pr[i] = a + w * (b - a);
        const double a2 = c->phi_n[i * m + pos], b2 = c->phi_n[i * m + pos + 1];
        pn[i] = a2 + w * (b2 - a2);
        const double a3 = c->phi_s[i * m + pos], b3 = c->phi_s[i * m + pos + 1];
        ps[i] = a3 + w * (b3 - a3);
    }
    return 0;
}

/* N - sum(count * rho); `bad` is set when u is right of the slice. */
static double derivative(const kernel_ctx *c, double u,
                         double *pr, double *pn, double *ps,
                         double *tmp, int *bad)
{
    if (phis(c, u, pr, pn, ps)) { *bad = 1; return 0.0; }
    for (long i = 0; i < c->k; ++i)
        tmp[i] = c->counts[i] * exp(u + pn[i] - pr[i]);
    return c->N - pairwise_sum(tmp, c->k);
}

static double curvature(const kernel_ctx *c, double u,
                        double *pr, double *pn, double *ps,
                        double *tmp, int *bad)
{
    if (phis(c, u, pr, pn, ps)) { *bad = 1; return 0.0; }
    for (long i = 0; i < c->k; ++i) {
        const double rho = exp(u + pn[i] - pr[i]);
        const double raw_second = exp(2.0 * u + ps[i] - pr[i]);
        double v = rho + rho * rho - raw_second;
        const double bound = 1e-10 * (1.0 > fabs(rho) ? 1.0 : fabs(rho));
        if (v < 0.0 && v > -bound) v = 0.0;
        tmp[i] = c->counts[i] * v;
    }
    return -pairwise_sum(tmp, c->k);
}

static double psi_at(const kernel_ctx *c, double u,
                     double *pr, double *pn, double *ps,
                     double *tmp, int *bad)
{
    if (phis(c, u, pr, pn, ps)) { *bad = 1; return 0.0; }
    for (long i = 0; i < c->k; ++i) tmp[i] = c->counts[i] * pr[i];
    return c->N * u + pairwise_sum(tmp, c->k);
}

/* scipy's brentq, same iterate sequence. */
static double brentq_c(const kernel_ctx *c, double xa, double xb,
                       double xtol, double rtol,
                       double *pr, double *pn, double *ps, double *tmp,
                       int *bad)
{
    double xpre = xa, xcur = xb, xblk = 0.0, fblk = 0.0;
    double spre = 0.0, scur = 0.0, sbis, delta, stry, dpre, dblk;
    double fpre = derivative(c, xpre, pr, pn, ps, tmp, bad);
    double fcur = derivative(c, xcur, pr, pn, ps, tmp, bad);
    if (*bad) return 0.0;
    if (fpre == 0.0) return xpre;
    if (fcur == 0.0) return xcur;
    if (signbit(fpre) == signbit(fcur)) { *bad = 1; return 0.0; }

    for (int i = 0; i < PMM_MAXITER; ++i) {
        if (fpre != 0.0 && fcur != 0.0 &&
            (signbit(fpre) != signbit(fcur))) {
            xblk = xpre; fblk = fpre; spre = scur = xcur - xpre;
        }
        if (fabs(fblk) < fabs(fcur)) {
            xpre = xcur; xcur = xblk; xblk = xpre;
            fpre = fcur; fcur = fblk; fblk = fpre;
        }
        delta = (xtol + rtol * fabs(xcur)) / 2.0;
        sbis = (xblk - xcur) / 2.0;
        if (fcur == 0.0 || fabs(sbis) < delta) return xcur;

        if (fabs(spre) > delta && fabs(fcur) < fabs(fpre)) {
            if (xpre == xblk) {
                stry = -fcur * (xcur - xpre) / (fcur - fpre);
            } else {
                dpre = (fpre - fcur) / (xpre - xcur);
                dblk = (fblk - fcur) / (xblk - xcur);
                stry = -fcur * (fblk * dblk - fpre * dpre)
                       / (dblk * dpre * (fblk - fpre));
            }
            double lim = 3.0 * fabs(sbis) - delta;
            if (fabs(spre) < lim) lim = fabs(spre);
            if (2.0 * fabs(stry) < lim) { spre = scur; scur = stry; }
            else { spre = sbis; scur = sbis; }
        } else {
            spre = sbis; scur = sbis;
        }
        xpre = xcur; fpre = fcur;
        if (fabs(scur) > delta) xcur += scur;
        else xcur += (sbis > 0.0 ? delta : -delta);
        fcur = derivative(c, xcur, pr, pn, ps, tmp, bad);
        if (*bad) return 0.0;
    }
    return xcur;
}

/* Returns 0 on success (saddle with negative curvature found),
   1 when the bracket does not contain a sign change,
   2 when the curvature test rejects the root,
   3 when a point fell right of the provisioned slice. */
int pmm_solve_peak(const double *phi_r, const double *phi_n,
                   const double *phi_s,
                   const double *left_r, const double *left_n,
                   const double *left_s,
                   const double *counts, long k,
                   const double *u_local, long m,
                   double N, double lo, double hi,
                   double xtol, double rtol,
                   double *scratch,          /* 4*k doubles */
                   double *out_saddle, double *out_curv, double *out_psi)
{
    kernel_ctx c = {phi_r, phi_n, phi_s, left_r, left_n, left_s,
                    counts, u_local, k, m, N};
    double *pr = scratch, *pn = scratch + k, *ps = scratch + 2 * k,
           *tmp = scratch + 3 * k;
    int bad = 0;

    double d_lo = derivative(&c, lo, pr, pn, ps, tmp, &bad);
    if (bad) return 3;
    double d_hi = derivative(&c, hi, pr, pn, ps, tmp, &bad);
    if (bad) return 3;
    if (!(d_lo > 0.0 && 0.0 > d_hi)) return 1;

    double saddle = brentq_c(&c, lo, hi, xtol, rtol, pr, pn, ps, tmp, &bad);
    if (bad) return 3;
    double curv = curvature(&c, saddle, pr, pn, ps, tmp, &bad);
    if (bad) return 3;
    if (!(isfinite(curv) && curv < 0.0)) return 2;
    double psi = psi_at(&c, saddle, pr, pn, ps, tmp, &bad);
    if (bad) return 3;

    *out_saddle = saddle;
    *out_curv = curv;
    *out_psi = psi;
    return 0;
}

/* ------------------------------------------------------------------
 * Degree-7 interpolation of one stored column onto the query grid.
 *
 * This is the other half of a run's cost: `log_phi_matrix` serves every
 * (L, r) column a level needs onto one grid, and the numpy version
 * materialises a (n, 8) temporary per column and reduces it.  Measured
 * at 10.4x here (13.4 ms -> 1.31 ms for a 120-column level) and
 * bit-identical, provided two things hold: the file is compiled with
 * -ffp-contract=off, and the eight products are summed in numpy's
 * pairwise tree rather than sequentially.  Both were checked by
 * comparison against the numpy path, and both were WRONG in the first
 * version, each on its own moving the values by 4.5e-13 (31 July).
 *
 * Points the caller must still handle in Python are flagged in `todo`:
 * those left of the column's own start (the series branch), those
 * right of U_MAX, and those whose stencil would run off either end of
 * the stored values (the weights differ there).
 * ------------------------------------------------------------------ */
void pmm_interp_column(const double *vals, long nvals, long i0,
                       const double *u, long n, double grid0, double u_max,
                       const long *k0, const double *w, const double *wsum,
                       const unsigned char *hit, const long *hitcol,
                       double *row, unsigned char *todo)
{
    const long lim = nvals - 8;
    for (long j = 0; j < n; ++j) {
        const double uj = u[j];
        if (uj < grid0 || uj > u_max) { todo[j] = 1; continue; }
        const long st = i0 + k0[j];
        if (st < 0 || st > lim) { todo[j] = 1; continue; }
        todo[j] = 0;
        if (hit[j]) { row[j] = vals[st + hitcol[j]]; continue; }
        const double *ww = w + j * 8;
        const double *v = vals + st;
        const double p0 = ww[0] * v[0], p1 = ww[1] * v[1];
        const double p2 = ww[2] * v[2], p3 = ww[3] * v[3];
        const double p4 = ww[4] * v[4], p5 = ww[5] * v[5];
        const double p6 = ww[6] * v[6], p7 = ww[7] * v[7];
        row[j] = (((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7)))
                 / wsum[j];
    }
}
