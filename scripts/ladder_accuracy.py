#!/usr/bin/env python3
"""How far apart can stored columns be in r before interpolation hurts?

The store today is a CACHE: an exact column for every (L, r) any run has
ever asked for, no interpolation across r, and 86 GB and growing.  The
proposed replacement is a designed object --- anchor columns on a fixed
geometric grid in r, everything else interpolated between them.  This
script measures the one number that design needs: the interpolation
error as a function of anchor spacing.

Method.  Anchors sit on a FIXED grid r_k = round((1+f)^k), which is what
production would use --- not a grid centred on whatever is being asked
for.  For a target r that is in the store and is NOT an anchor, take the
`--degree`+1 nearest anchors, interpolate, and compare against the
stored exact column.  Interpolation is barycentric Lagrange in
x = ln(r+1), applied to the residual

    Y(u) = ln phi_r^(L)(e^u) - L * lgamma(r+1)

which removes the t -> 0 limit and leaves a function that is smooth and
O(1)-ish in x, rather than one spanning tens of thousands of nats.

ANCHORS ARE VERIFIED BEFORE USE, and that is not defensive padding.  The
first version of this measurement reported that 10% spacing was
catastrophic at L=43 --- 205 nats --- and the real cause was that two of
its eight anchor columns were corrupt.  A ladder result is only as good
as the columns it stands on, so every anchor is checked against contour
integration (or its checksum) before it is allowed to vote, and targets
whose anchors fail are reported separately rather than silently folded
into the error statistics.

What the output means.  Errors are in nats of ln phi.  Converting to
bits/character needs an amplification factor, and the two we have
disagree by an order of magnitude: a uniform +eps bias measures ~254
bits/token per nat, while the deep-level substitution implies ~30, the
difference being that a smooth error cancels and a constant one does
not.  Both are printed as a bracket.  Do not treat either as the answer:
the honest test is to serve a whole run off a ladder and measure the
codelength directly, exactly as the substitution block does.

    python scripts/ladder_accuracy.py --levels 5,12,20,30,40 \\
        --factors 0.02,0.05,0.10 --targets 40
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import loggamma

from product_model_with_memory.mellin import log_phi_contour
from product_model_with_memory.universal_tables import (
    H,
    SERIES_TAIL_NATS,
    U_MAX,
    UniversalTables,
)

# a nat of ln phi error costs somewhere between these, in bits/token;
# see the module docstring for why the bracket is this wide
AMP_SMOOTH = 30.0      # implied by the deep-level substitution
AMP_BIAS = 254.0       # measured for a uniform +eps field
BITS_PER_TOKEN_PER_CHAR = 4.9723    # bpe_text8


def _right_aligned(tab, L: int, r: int) -> np.ndarray:
    """A column with index 0 at u = U_MAX, running leftwards.

    Every stored column lives on the one master grid and differs only by
    where it starts, so aligning at the RIGHT edge puts all of them on a
    common index.  Aligning at index 0 (the left edge) would compare
    different u.
    """

    _i0, vals = tab.column(L, r)
    return vals[::-1]


def anchor_grid(factor: float, r_max: int) -> list[int]:
    """r_k = round((1+f)^k), deduplicated.  Fixed, not target-centred."""

    out, k = [], 0
    while True:
        r = int(round((1.0 + factor) ** k))
        if r > r_max:
            break
        if not out or r > out[-1]:
            out.append(r)
        k += 1
    return out


_VERIFIED: dict[tuple[int, int], float] = {}


def verify_anchor(tab, L: int, r: int, n_probe: int = 4) -> float:
    """Max |stored - contour| at a few interior points, in nats.

    Memoised: an anchor is shared by every target that brackets it, and
    contour integration is the expensive part of this script.  The
    verdict is a property of the bytes on disk, so checking once is
    enough.
    """

    if (L, r) in _VERIFIED:
        return _VERIFIED[(L, r)]
    v = _right_aligned(tab, L, r)
    u = U_MAX - H * np.arange(len(v))
    keep = np.flatnonzero(u > u.min() + SERIES_TAIL_NATS)
    js = np.linspace(keep.min(), keep.max(), n_probe).astype(int)
    ex = np.array([log_phi_contour(float(r), L, float(u[j])) for j in js])
    err = float(np.abs(v[js] - ex).max())
    _VERIFIED[(L, r)] = err
    return err


def interpolate(tab, L: int, target: int, anchors: list[int]) -> np.ndarray:
    """Barycentric Lagrange across ln(r+1) on the residual."""

    cols = [_right_aligned(tab, L, a) for a in anchors]
    m = min(len(c) for c in cols)
    Y = np.array([c[:m] - L * float(loggamma(a + 1.0))
                  for c, a in zip(cols, anchors)])
    xs = np.log(np.asarray(anchors, dtype=np.float64) + 1.0)
    w = np.ones(len(xs))
    for i in range(len(xs)):
        for k in range(len(xs)):
            if k != i:
                w[i] /= (xs[i] - xs[k])
    d = math.log(target + 1.0) - xs
    if np.any(np.abs(d) < 1e-15):        # target IS an anchor
        return Y[int(np.argmin(np.abs(d)))] + L * float(
            loggamma(target + 1.0))
    c = w / d
    return (c[:, None] * Y).sum(0) / c.sum() + L * float(
        loggamma(target + 1.0))


def decimated_grid(meta: dict, L: int, every: int) -> list[int]:
    """Every `every`-th anchor of the store's OWN grid.

    Recomputing round((1+f')^k) for the nominal coarser f' does NOT
    reproduce this, and assuming it did invalidated a whole sweep:
    every 2nd point of round(1.005^k) lies on 1.005^2 = 1.010025^k, and
    asking for 1.01 instead produced a grid of which only 275 of 1031
    points existed in the store, so the rows measured whatever handful
    of accidental coincidences survived.  Decimate the actual list.
    """

    anchors = [int(r) for r in meta["levels"][str(L)]["anchors"]]
    # decimate only above the dense floor; see build_anchor_store
    floor = int(meta.get("dense_below", 0))
    return ([r for r in anchors if r < floor]
            + [r for r in anchors if r >= floor][::every])


def measure(tab, L: int, factor: float, degree: int, n_targets: int,
            rng: np.random.Generator, verify: bool,
            grid_override: list[int] | None = None) -> dict:
    have = set(tab._load_level(L)["index"])
    idx_r = sorted(have)
    if len(idx_r) < degree + 3:
        return {"L": L, "factor": factor, "skipped": "too few columns"}

    # The IDEAL grid is what the design specifies.  The store is a cache
    # and will not contain all of it, and the difference matters: an
    # earlier version kept only the ideal points that happened to be
    # present, so a run labelled "2%" was really interpolating over an
    # irregular subset with gaps --- a wider and unevenly spaced ladder
    # wearing a 2% label.  Measure the real thing: use a target only
    # when ALL of its surrounding ideal anchors exist, and report the
    # coverage so a thin result cannot be mistaken for a good one.
    ideal = (list(grid_override) if grid_override is not None
             else anchor_grid(factor, max(idx_r)))
    grid = [r for r in ideal if r in have]
    coverage = len(grid) / max(1, len(ideal))
    if len(grid) < degree + 1:
        return {"L": L, "factor": factor,
                "skipped": f"only {len(grid)} of {len(ideal)} anchors "
                           "present in store"}

    ideal_set = set(ideal)
    cand = [r for r in idx_r if r not in ideal_set
            and r > ideal[degree // 2]
            and r < ideal[-(degree // 2 + 1)]]
    if not cand:
        return {"L": L, "factor": factor, "skipped": "no interior targets"}
    targets = [int(x) for x in rng.choice(
        cand, size=min(n_targets, len(cand)), replace=False)]

    errs, bad_anchor, used, gappy = [], 0, 0, 0
    ia = np.asarray(ideal)
    for t in sorted(targets):
        j = int(np.searchsorted(ia, t))
        lo = max(0, min(j - (degree + 1) // 2, len(ideal) - (degree + 1)))
        anch = ideal[lo:lo + degree + 1]
        if len(anch) < degree + 1:
            continue
        if any(a not in have for a in anch):
            gappy += 1        # the true ladder is not measurable here
            continue
        if verify:
            # the whole point: a ladder result is only as good as the
            # columns it stands on, and this measurement was wrong once
            # already because two anchors were corrupt
            try:
                if max(verify_anchor(tab, L, a) for a in anch) > 1e-4:
                    bad_anchor += 1
                    continue
            except RuntimeError:          # checksum refused the column
                bad_anchor += 1
                continue
        got = interpolate(tab, L, t, anch)
        ref = _right_aligned(tab, L, t)
        m = min(len(got), len(ref))
        u = U_MAX - H * np.arange(m)
        keep = u > (U_MAX - H * (m - 1)) + SERIES_TAIL_NATS
        if not keep.any():
            continue
        errs.append(float(np.abs(got[:m][keep] - ref[:m][keep]).max()))
        used += 1

    if not errs:
        return {"L": L, "factor": factor, "coverage": coverage,
                "skipped": f"no target has a complete ladder "
                           f"({gappy} gappy, {bad_anchor} bad anchors; "
                           f"store holds {len(grid)} of {len(ideal)} "
                           f"ideal anchors)"}
    e = np.asarray(errs)
    return {"L": L, "factor": factor, "ideal_anchors": len(ideal),
            "anchors_in_store": len(grid), "coverage": coverage,
            "targets": used, "gappy_targets": gappy,
            "bad_anchor_targets": bad_anchor,
            "median_nats": float(np.median(e)),
            "p90_nats": float(np.percentile(e, 90)),
            "max_nats": float(e.max())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tables", default="tables/universal_v2")
    p.add_argument("--levels", default="5,12,20,30,40")
    p.add_argument("--factors", default="0.02,0.05,0.10")
    p.add_argument("--decimate", default=None,
                   help="comma-separated m: use every m-th anchor of the "
                        "store's OWN grid (needs anchors.json).  This is "
                        "the correct way to test coarser ladders --- "
                        "--factors recomputes round((1+f)^k), which does "
                        "NOT coincide with a decimation and silently "
                        "measures a gappy grid instead")
    p.add_argument("--degrees", default="7",
                   help="comma-separated polynomial degrees; degree+1 "
                        "anchors are used.  Sweeping this matters: at "
                        "tight spacing the error is dominated by the "
                        "conditioning of the interpolation rather than by "
                        "truncation, and a LOWER degree can beat a higher "
                        "one on the same anchors")
    p.add_argument("--targets", type=int, default=40)
    p.add_argument("--no-verify", action="store_true",
                   help="skip anchor verification.  Faster, and the "
                        "reason the first version of this measurement "
                        "was wrong -- do not use it to make a decision")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    tab = UniversalTables(args.tables, read_only=True)
    levels = [int(x) for x in args.levels.split(",")]
    factors = [float(x) for x in args.factors.split(",")]
    rng = np.random.default_rng(args.seed)

    degrees = [int(x) for x in args.degrees.split(",")]
    meta = None
    if args.decimate:
        mf = Path(args.tables) / "anchors.json"
        if not mf.exists():
            raise SystemExit(f"--decimate needs {mf}; that store was not "
                             "built by build_anchor_store.py")
        meta = json.loads(mf.read_text())
        modes = [("every " + str(m), int(m))
                 for m in args.decimate.split(",")]
    else:
        modes = [(f"{f:.3%}", f) for f in factors]

    print(f"store: {args.tables}   "
          f"{'ANCHORS VERIFIED' if not args.no_verify else 'UNVERIFIED'}")
    print(f"{'level':>6} {'ladder':>10} {'eff.':>8} {'deg':>4} "
          f"{'anchors':>13} {'used':>6} {'gap':>5} {'bad':>5} "
          f"{'median':>11} {'max (nats)':>11}")
    rows = []
    for deg in degrees:
        for label, mode in modes:
            for L in levels:
                if meta is not None:
                    g = decimated_grid(meta, L, mode)
                    eff = (1.0 + meta["factor"]) ** mode - 1.0
                    r = measure(tab, L, eff, deg, args.targets, rng,
                                not args.no_verify, grid_override=g)
                else:
                    eff = mode
                    r = measure(tab, L, eff, deg, args.targets, rng,
                                not args.no_verify)
                r["degree"] = deg
                r["ladder"] = label
                rows.append(r)
                if "skipped" in r:
                    print(f"{L:>6} {label:>10} {eff:>7.3%} {deg:>4}   "
                          f"{r['skipped']}")
                    continue
                cov = f"{r['anchors_in_store']}/{r['ideal_anchors']}"
                print(f"{L:>6} {label:>10} {eff:>7.3%} {deg:>4} {cov:>13} "
                      f"{r['targets']:>6} {r['gappy_targets']:>5} "
                      f"{r['bad_anchor_targets']:>5} "
                      f"{r['median_nats']:>11.3e} "
                      f"{r['max_nats']:>11.3e}", flush=True)
            print()

    print("Reading this.  `max` is the worst error over a column, in nats "
          "of ln phi.\nTo cost it in the paper's unit, multiply by "
          f"{AMP_SMOOTH:.0f}..{AMP_BIAS:.0f} bits/token per nat and divide "
          f"by {BITS_PER_TOKEN_PER_CHAR:.2f}:")
    print(f"    1e-6 nats  ->  {1e-6 * AMP_SMOOTH / BITS_PER_TOKEN_PER_CHAR:.1e}"
          f" .. {1e-6 * AMP_BIAS / BITS_PER_TOKEN_PER_CHAR:.1e} bits/char")
    print(f"    1e-4 nats  ->  {1e-4 * AMP_SMOOTH / BITS_PER_TOKEN_PER_CHAR:.1e}"
          f" .. {1e-4 * AMP_BIAS / BITS_PER_TOKEN_PER_CHAR:.1e} bits/char")
    print("The bracket is wide because a smooth error cancels and a "
          "constant one does\nnot.  Use it to CHOOSE a spacing to test, "
          "then confirm end to end by serving\na real run off that ladder "
          "--- the same way the substitution block does.")
    print("\n`anchors` is present/ideal.  `gap` counts targets skipped "
          "because the store\nlacks one of their ideal anchors --- those "
          "are not measurable here, and a\nlarge gap count with few `used` "
          "means the row says little about that spacing.\n`bad` counts "
          "targets dropped because an anchor failed verification; "
          "non-zero\nthere means the store is damaged, not that the "
          "ladder is.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"tables": args.tables, "degrees": degrees,
             "decimate": args.decimate, "rows": rows},
            indent=2))
        print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
