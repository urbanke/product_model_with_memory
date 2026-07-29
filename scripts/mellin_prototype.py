#!/usr/bin/env python3
"""Measure the accuracy of cheap representations of the moment functions.

Three independent computations of ln phi_r^(L)(t):
  A. the recursion + Gauss-Laguerre table builder (the project's method),
  B. direct Mellin-Barnes contour integration / small-t series (module
     mellin; exact up to quadrature on a smooth integrand),
  C. the closed-form saddle approximation (order 1 and 2).

Outputs, as JSON and a printed summary:
  * |A - B|: cross-validation of the existing tables,
  * |C - B| as a function of r and L: where the saddle formula can
    replace stored tables, and the crossover r0,
  * second differences of (B - C) along a geometric r-ladder: how
    smooth the residual is in ln r, i.e. how coarse a correction grid
    could be.

Example:
    python scripts/mellin_prototype.py --out output/mellin_proto
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from product_model_with_memory.layered import build_selected_product_moment_tables
from product_model_with_memory.mellin import (
    log_phi_contour,
    log_phi_saddle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels", default="2,4,8,16,32",
        help="levels L to test (comma separated)")
    parser.add_argument(
        "--r-ladder", default="1,2,3,5,8,13,21,34,55,89,144,233,377,610,"
        "1000,1600,2500,4000,6300,10000,25000,63000,160000,400000,1000000",
        help="count values r to test")
    parser.add_argument(
        "--u-sample", default="-40,-20,-10,-5,-2,0,2,5,10,20,34",
        help="u = ln t sample points")
    parser.add_argument(
        "--table-r-max", type=int, default=4000,
        help="build recursion tables (method A) for r up to this value")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    t0 = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = [int(x) for x in args.levels.split(",")]
    r_ladder = [int(x) for x in args.r_ladder.split(",")]
    u_sample = [float(x) for x in args.u_sample.split(",")]
    max_L = max(levels)

    # ---- method A: recursion tables for the small-r part of the ladder
    r_for_tables = [r for r in r_ladder if r <= args.table_r_max]
    # use the PRODUCTION grid specification (adaptive spacing, dense
    # where the functions are steep) --- a uniform test grid was shown
    # to corrupt the recursion at large r
    from product_model_with_memory.fast_tables import (
        default_grid_spec,
        grid_from_spec,
    )
    spec = default_grid_spec(
        max_L=max_L, r_max=max(r_for_tables), u_max=35.0
    )
    u_grid = grid_from_spec(spec)
    print(f"production grid: {len(u_grid)} points "
          f"[{u_grid[0]:.1f}, {u_grid[-1]:.1f}]", flush=True)
    tables = build_selected_product_moment_tables(
        max_L=max_L, r_values=r_for_tables, u_grid=u_grid
    )
    print(f"tables built for {len(r_for_tables)} r values "
          f"(L <= {max_L}) ({time.time()-t0:.0f}s)", flush=True)

    records = []
    for L in levels:
        for r in r_ladder:
            for u in u_sample:
                ref = log_phi_contour(r, L, u)  # method B (natural log)
                s1, _ = log_phi_saddle(r, L, u, order=1)
                s2, _ = log_phi_saddle(r, L, u, order=2)
                rec = {
                    "L": L, "r": r, "u": u,
                    "ref": ref,
                    "err_saddle1": s1 - ref,
                    "err_saddle2": s2 - ref,
                }
                if r in r_for_tables:
                    a = tables.log_phi_value(L=L, r=r, u=u)
                    rec["err_tables"] = a - ref
                records.append(rec)
        print(f"L={L} done ({time.time()-t0:.0f}s)", flush=True)

    # ---- summaries
    def _quant(vals):
        v = np.abs(np.array(vals))
        return {
            "median": float(np.median(v)),
            "p90": float(np.quantile(v, 0.9)),
            "max": float(np.max(v)),
            "n": len(vals),
        }

    by_r_saddle2 = {}
    for r in r_ladder:
        errs = [x["err_saddle2"] for x in records if x["r"] == r]
        by_r_saddle2[r] = _quant(errs)
    table_errs = [x["err_tables"] for x in records if "err_tables" in x]

    # residual smoothness along the ladder: second differences of
    # (ref - saddle2) in ln r, per (L, u)
    second_diffs = []
    lr = np.log(np.array(r_ladder, dtype=float))
    for L in levels:
        for u in u_sample:
            res = np.array([
                next(x for x in records if x["L"] == L and x["r"] == r
                     and x["u"] == u)["err_saddle2"] * -1.0
                for r in r_ladder
            ])
            for i in range(1, len(r_ladder) - 1):
                h1, h2 = lr[i] - lr[i - 1], lr[i + 1] - lr[i]
                interp = (h2 * res[i - 1] + h1 * res[i + 1]) / (h1 + h2)
                second_diffs.append(abs(res[i] - interp))

    payload = {
        "levels": levels,
        "r_ladder": r_ladder,
        "u_sample": u_sample,
        "tables_vs_contour_nats": _quant(table_errs),
        "saddle2_abs_err_by_r": by_r_saddle2,
        "residual_interp_err_nats": _quant(second_diffs),
        "records": records,
        "seconds": time.time() - t0,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))

    print("\n=== tables (method A) vs contour (method B), nats ===")
    print(json.dumps(_quant(table_errs), indent=2))
    print("\n=== |saddle order-2 - reference| by r (nats) ===")
    for r in r_ladder:
        q = by_r_saddle2[r]
        print(f"  r={r:8d}: median {q['median']:.2e}  p90 {q['p90']:.2e}"
              f"  max {q['max']:.2e}")
    print("\n=== residual interpolation error on the ladder (nats) ===")
    print(json.dumps(_quant(second_diffs), indent=2))
    print(f"\nwritten: {out_dir/'results.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
