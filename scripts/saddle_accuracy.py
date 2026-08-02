#!/usr/bin/env python3
"""Where is the saddle approximation good enough to replace the store?

The universal store costs 350 GB, and 78% of that sits in columns with
r >= 10,000, because a column runs from u ~ -8 - L ln(r+1) up to
U_MAX = 35 at spacing H = 0.02, so its LENGTH grows like L ln r.  The
large-r columns are both the biggest and the least reusable: they come
from the few very heavy states of one particular run, and splitting a
state manufactures new integers, so they rarely recur.

`log_phi_saddle` is a large-parameter asymptotic expansion, so it
should get BETTER exactly where storage gets expensive.  If that is
true below the store's own certified accuracy above some r*, then above
r* the column need not be stored at all -- it can be evaluated.

This script measures it.  For a spread of (L, r) it reads the stored
column, evaluates the saddle at the same u points, and reports the
error distribution.  The left tail of a column is excluded because the
reader never touches it: below grid0 + PMM_SERIES_TAIL nats the code
already uses the certified small-t series instead.

    python scripts/saddle_accuracy.py
    python scripts/saddle_accuracy.py --levels 2,10,26,43 --tables tables/universal_v2
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from product_model_with_memory.mellin import log_phi_saddle
from product_model_with_memory.universal_tables import (
    SERIES_TAIL_NATS,
    UniversalTables,
    _grid_points,
)

# The store's own certified accuracy, measured against the independent
# contour integrator: median 1.5e-12, max 1.3e-5 nats.  A replacement
# only has to beat the max to be indistinguishable in any reported
# codelength; we flag 1e-6 as the comfortable target.
STORE_MAX_NATS = 1.3e-5
TARGET_NATS = 1e-6


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tables", default=os.environ.get(
        "PMM_UNIVERSAL_TABLES", "tables/universal_v2"))
    p.add_argument("--levels", default="2,4,10,18,26,35,43")
    p.add_argument("--per-column", type=int, default=400,
                   help="u points sampled per column")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    root = Path(args.tables)
    tab = UniversalTables(root, read_only=True)
    levels = [int(x) for x in args.levels.split(",")]

    # CONTROL.  At L = 1 the column is the closed form
    # ln phi = lnGamma(r+1) - (r+1) ln(1+t), so saddle-against-column
    # there tests the whole path --- the u axis, the column layout and
    # the approximation --- against something known.  A previous
    # version of this script mislaid the u axis (the master grid counts
    # DOWN from U_MAX) and reported errors of 1e8 nats; without a
    # control that reads as "the saddle is useless" instead of "the
    # harness is wrong".
    ctrl = []
    for r in (251, 4001, 67465):
        i0, vals = tab.column(1, r)
        grid = _grid_points(i0)
        live = grid >= grid[0] + SERIES_TAIL_NATS
        g, v = grid[live][::37], vals[live][::37]
        e = max(abs(log_phi_saddle(float(r), 1, float(u))[0] - float(x))
                for u, x in zip(g, v))
        ctrl.append((r, e))
    print("control at L = 1 (column is a closed form), max error in nats: "
          + ", ".join(f"r={r}: {e:.1e}" for r, e in ctrl))
    if max(e for _r, e in ctrl) > 1e-2:
        raise SystemExit(
            "control failed: the saddle misses a CLOSED-FORM column by "
            "more than 1e-2 nats, so the harness is reading the store "
            "wrongly.  Fix that before believing anything below.")

    rows = []
    print(f"store: {root}")
    print("saddle (order 2) against stored columns, absolute error in nats")
    print(f"{'L':>4} {'r':>9} {'points':>7} {'median':>11} {'p99':>11} "
          f"{'max':>11} {'verdict':>10}")
    for L in levels:
        try:
            index = json.loads(
                (root / f"level_{L:02d}.index.json").read_text())
        except FileNotFoundError:
            continue
        available = sorted(int(k) for k in index)
        if not available:
            continue
        # a geometric spread of the r values this level actually holds
        wanted = [1, 17, 251, 4001, 67465, 450250, 1000000]
        picks = sorted({min(available, key=lambda a, w=w: abs(a - w))
                        for w in wanted})
        for r in picks:
            i0, vals = tab.column(L, r)
            # grid point i sits at u = U_MAX - i*H, counting DOWN from
            # the right edge; index i0 is the LEFTMOST stored point, so
            # the stored values run in INCREASING u.
            grid = _grid_points(i0)
            assert len(grid) == len(vals), (L, r, len(grid), len(vals))
            # the reader never uses the far left tail: the certified
            # small-t series covers it
            live = grid >= grid[0] + SERIES_TAIL_NATS
            if live.sum() < 8:
                continue
            g, v = grid[live], vals[live]
            step = max(1, len(g) // args.per_column)
            g, v = g[::step], v[::step]
            approx = np.array([log_phi_saddle(float(r), L, float(u))[0]
                               for u in g])
            err = np.abs(approx - v)
            med, p99, mx = (float(np.median(err)),
                            float(np.percentile(err, 99)), float(err.max()))
            verdict = ("evaluate" if mx < TARGET_NATS else
                       "borderline" if mx < STORE_MAX_NATS else "store")
            rows.append({"L": L, "r": int(r), "points": int(len(g)),
                         "median": med, "p99": p99, "max": mx,
                         "verdict": verdict,
                         "column_length": int(len(vals)),
                         "bytes": int(len(vals)) * 8})
            print(f"{L:>4} {r:>9,} {len(g):>7,} {med:>11.2e} {p99:>11.2e} "
                  f"{mx:>11.2e} {verdict:>10}", flush=True)

    if not rows:
        raise SystemExit("no columns read; is --tables right?")

    print(f"\nstore's own certified accuracy: median 1.5e-12, "
          f"max {STORE_MAX_NATS:.1e} nats")
    print(f"'evaluate' means the saddle is below {TARGET_NATS:.0e} nats "
          f"everywhere the reader looks, so the column need not be stored.")

    # smallest r above which every measured column says 'evaluate'
    by_r: dict[int, list] = {}
    for row in rows:
        by_r.setdefault(row["r"], []).append(row)
    ok_from = None
    for r in sorted(by_r, reverse=True):
        if all(x["verdict"] == "evaluate" for x in by_r[r]):
            ok_from = r
        else:
            break
    if ok_from is None:
        print("\nNo r* found: the saddle misses the target at every r "
              "measured.  Keep storing.")
    else:
        print(f"\nr* = {ok_from:,}: at and above this every level measured "
              f"is within {TARGET_NATS:.0e} nats.")
        print(f"    => `python scripts/prune_tables.py --max-r {ok_from}` "
              f"is safe to make PERMANENT (evaluate above it), not just a "
              f"cache eviction.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"tables": str(root), "target_nats": TARGET_NATS,
             "store_max_nats": STORE_MAX_NATS, "r_star": ok_from,
             "rows": rows}, indent=2))
        print(f"written: {args.out}")


if __name__ == "__main__":
    main()
