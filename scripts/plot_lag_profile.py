#!/usr/bin/env python3
"""Plot the distance profile v(delta) of one or more lag-family runs,
and fit scaling laws to the decay.

For each input, v(delta) = bits(memoryless) - bits(lag delta) is shown
on log-log axes.  Two laws are fit to the points with delta >= --fit-from
(default 2; lag 1 usually sits above any tail law):

    power law    v = c * delta^(-alpha)      (line in log-log)
    exponential  v = c * rho^delta           (line in log-linear)

Both fits are drawn, the better one (by RMS residual in log v) solid,
the other dotted; fitted parameters are printed and annotated.

Example:

    python scripts/plot_lag_profile.py \
        --inputs output/lag_family_v1024/results.json:V=1024 \
                 output/lag_family_v4096/results.json:V=4096 \
        --fit-from 2 --out paper/lag_profile.pdf
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito colorblind-safe hues, assigned in fixed order.
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


def load_profile(path: str) -> tuple[list[int], list[float]]:
    data = json.loads(Path(path).read_text())
    bits = {int(k): v for k, v in data["member_bits_per_token"].items()}
    if 0 not in bits:
        raise SystemExit(f"{path}: no memoryless member (delta=0) to anchor")
    anchor = bits[0]
    deltas = sorted(d for d in bits if d > 0)
    values = [anchor - bits[d] for d in deltas]
    return deltas, values


def fit_offset_laws(deltas, values, fit_from):
    """3-parameter fits v = c*d^-alpha - L and v = c*rho^d - L (linear LS).

    Returns dict law -> (params, rms) with rms in linear v.  The offset L
    is the learning-cost gap between a V-state member and the memoryless
    anchor; the corrected profile I(d) = v(d) + L estimates the raw
    pairwise information.
    """
    from scipy.optimize import curve_fit

    pts = [(d, v) for d, v in zip(deltas, values) if d >= fit_from]
    if len(pts) < 4:
        return {}
    d = np.array([p[0] for p in pts], float)
    v = np.array([p[1] for p in pts], float)
    L0 = max(0.0, -float(v.min())) + 1e-3
    out = {}
    fp = lambda x, c, a, L: c * x ** (-a) - L
    p, _ = curve_fit(fp, d, v, p0=[max(v) + L0, 2.0, L0],
                     bounds=([0, 0, 0], [np.inf, 10, 5]), maxfev=20000)
    out["power"] = (tuple(p),
                    math.sqrt(float(np.mean((v - fp(d, *p)) ** 2))))
    fe = lambda x, c, r, L: c * r ** x - L
    try:
        p2, _ = curve_fit(fe, d, v, p0=[max(v) + L0, 0.6, L0],
                          bounds=([0, 1e-6, 0], [np.inf, 1, 5]),
                          maxfev=20000)
        out["exp"] = (tuple(p2),
                      math.sqrt(float(np.mean((v - fe(d, *p2)) ** 2))))
    except RuntimeError:
        pass
    return out


def fit_laws(deltas, values, fit_from):
    """Least-squares fits in log v; returns dict of law -> (params, rms)."""
    pts = [(d, v) for d, v in zip(deltas, values) if d >= fit_from and v > 0]
    if len(pts) < 3:
        return {}
    d = np.array([p[0] for p in pts], dtype=float)
    logv = np.log(np.array([p[1] for p in pts], dtype=float))
    out = {}
    # power law: log v = log c - alpha * log d
    A = np.vstack([np.ones_like(d), -np.log(d)]).T
    (logc, alpha), *_ = np.linalg.lstsq(A, logv, rcond=None)
    resid = logv - A @ np.array([logc, alpha])
    out["power"] = ((math.exp(logc), alpha),
                    math.sqrt(float(np.mean(resid**2))))
    # exponential: log v = log c + d * log rho
    B = np.vstack([np.ones_like(d), d]).T
    (logc2, logrho), *_ = np.linalg.lstsq(B, logv, rcond=None)
    resid2 = logv - B @ np.array([logc2, logrho])
    out["exp"] = ((math.exp(logc2), math.exp(logrho)),
                  math.sqrt(float(np.mean(resid2**2))))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="results.json:label pairs, e.g. output/x/results.json:V=1024",
    )
    parser.add_argument("--fit-from", type=int, default=2)
    parser.add_argument(
        "--offset-fit", action="store_true",
        help="fit v = c*d^-alpha - L (learning-cost offset) and plot the "
             "corrected profile I(d) = v(d) + L",
    )
    parser.add_argument(
        "--break-at", type=int, default=None,
        help="broken power law: fit the offset law on d >= BREAK_AT "
             "(tail regime, determines L), then a second power law on the "
             "corrected points with fit_from <= d <= BREAK_AT "
             "(short-range regime); requires --offset-fit",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    all_deltas: set[int] = set()
    for i, spec in enumerate(args.inputs):
        path, _, label = spec.partition(":")
        label = label or Path(path).parent.name
        color = COLORS[i % len(COLORS)]
        deltas, values = load_profile(path)
        all_deltas.update(deltas)
        if args.offset_fit:
            tail_from = args.break_at or args.fit_from
            laws = fit_offset_laws(deltas, values, tail_from)
            if not laws:
                raise SystemExit(f"{path}: too few points for offset fit")
            (c_p, alpha, L), rms_p = laws["power"]
            rms_e = laws.get("exp", (None, float("inf")))[1]
            corrected = [(d, v + L) for d, v in zip(deltas, values)
                         if v + L > 0]
            floor = 2.0 * rms_p  # below this, consistent with zero
            solid = [(d, i) for d, i in corrected if i >= floor]
            faint = [(d, i) for d, i in corrected if i < floor]
            ax.plot([d for d, _ in solid], [i for _, i in solid],
                    "o", ms=5, color=color, zorder=3)
            if faint:
                ax.plot([d for d, _ in faint], [i for _, i in faint],
                        "o", ms=5, mfc="none", color=color, zorder=3)
            xs = np.linspace(tail_from, max(deltas), 300)
            ax.plot(xs, c_p * xs ** (-alpha), "-", color=color, lw=1.6,
                    zorder=2)
            if args.break_at:
                # short-range regime: log-log LS on corrected points
                seg = [(d, i) for d, i in solid
                       if args.fit_from <= d <= args.break_at]
                ds = np.log(np.array([p[0] for p in seg]))
                vs = np.log(np.array([p[1] for p in seg]))
                A = np.vstack([np.ones_like(ds), -ds]).T
                (logc2, a_short), *_ = np.linalg.lstsq(A, vs, rcond=None)
                xs2 = np.linspace(args.fit_from, args.break_at, 100)
                ax.plot(xs2, math.exp(logc2) * xs2 ** (-a_short), "-",
                        color=color, lw=1.6, zorder=2)
                ax.plot([], [], "o-", color=color, ms=5,
                        label=(f"{label}:  $\\alpha_s$={a_short:.2f}, "
                               f"$\\alpha_t$={alpha:.2f}, "
                               f"$\\hat L$={L:.3f}"))
                print(f"{label}: short alpha={a_short:.3f} "
                      f"c={math.exp(logc2):.3f} "
                      f"(log-LS, {args.fit_from}<=d<={args.break_at}) | "
                      f"tail c={c_p:.3f} alpha={alpha:.3f} L={L:.4f} "
                      f"rms={rms_p:.5f} (d>={tail_from}) | "
                      f"exp+offset rms={rms_e:.5f}")
            else:
                ax.plot([], [], "o-", color=color, ms=5,
                        label=(f"{label}:  $\\alpha$={alpha:.2f}, "
                               f"$\\hat L$={L:.3f}"))
                print(f"{label}: power+offset c={c_p:.3f} "
                      f"alpha={alpha:.3f} L={L:.4f} rms={rms_p:.4f} | "
                      f"exp+offset rms={rms_e:.4f} "
                      f"(fit over delta>={tail_from}; plotted I=v+L)")
            continue
        ax.plot(deltas, values, "o", ms=5, color=color, zorder=3)
        laws = fit_laws(deltas, values, args.fit_from)
        if laws:
            (c_p, alpha), rms_p = laws["power"]
            (c_e, rho), rms_e = laws["exp"]
            better = "power" if rms_p <= rms_e else "exp"
            xs = np.linspace(args.fit_from, max(deltas), 200)
            for law, style in (("power", "-"), ("exp", ":")):
                if law == "power":
                    ys = c_p * xs ** (-alpha)
                else:
                    ys = c_e * rho ** xs
                lw = 1.6 if law == better else 1.0
                ax.plot(xs, ys, style, color=color, lw=lw, zorder=2)
            ax.plot([], [], "o-", color=color, ms=5,
                    label=(f"{label}:  "
                           f"$\\alpha$={alpha:.2f}, $\\rho$={rho:.2f}"
                           f" ({better} fits)"))
            print(f"{label}: power alpha={alpha:.3f} c={c_p:.3f} "
                  f"rms={rms_p:.3f} | exp rho={rho:.3f} c={c_e:.3f} "
                  f"rms={rms_e:.3f} -> {better} law fits better "
                  f"(fit over delta>={args.fit_from})")
        else:
            ax.plot([], [], "o", color=color, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = sorted(all_deltas)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel(r"lag $\delta$")
    ax.set_ylabel(
        r"$I(\delta) = v(\delta) + \hat L$  (bits/token)"
        if args.offset_fit else
        r"$v(\delta)$  (bits/token over memoryless)"
    )
    ax.grid(True, which="both", lw=0.3, alpha=0.35)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    if out.suffix != ".png":
        fig.savefig(out.with_suffix(".png"), dpi=180)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
