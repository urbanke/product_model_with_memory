#!/usr/bin/env python3
"""Seconds-long check of the complete system, on the real code path.

Five full end-to-end runs were spent discovering, one at a time, that
the ladder was broken in a way no unit test covered: log_phi had the
hook but log_phi_matrix did not; provisioning asked for levels the
expansion serves; decimation deleted the small-r anchors; r=0 fell off
the bottom of the grid and was silently extrapolated.  Each cost a
five-minute run to find and produced exactly one bit of information.

This exercises the same code --- the real codelength evaluator, through
the real store --- on a store small enough to build in seconds, and
checks the invariants that each of those bugs violated:

  * an anchor must come back bit-identical to the exact column;
  * r=0 and r=1 must be served (they are the commonest counts, and the
    grid must cover them, not extrapolate to them);
  * a non-anchor must interpolate to within a stated tolerance;
  * log_phi and log_phi_matrix must agree exactly;
  * a real codelength must come out with log2 q <= 0.

Run it after any change to the ladder, the grid, or the provisioning
path, BEFORE spending a real run.

    python scripts/smoke_ladder.py            # builds its own store
    python scripts/smoke_ladder.py --keep /tmp/smoke   # reuse it
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

FACTOR = 0.05
LEVELS = "2-6"
R_MAX = 4000
DENSE = 64


def build(root: Path, jobs: int) -> None:
    cmd = [sys.executable, "scripts/build_anchor_store.py",
           "--out", str(root), "--factor", str(FACTOR),
           "--levels", LEVELS, "--r-max", str(R_MAX),
           "--dense-below", str(DENSE), "--targets-per-level", "6",
           "--jobs", str(jobs), "--go"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout[-3000:] + p.stderr[-3000:])
        raise SystemExit("smoke: store build failed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", default=None,
                    help="build here and reuse if present")
    ap.add_argument("--tables", default=None,
                    help="check an EXISTING anchor store instead of "
                         "building a small one.  Use with --levels to hit "
                         "the risky end first: a full check walks levels "
                         "in order, so the top ones come last, and that is "
                         "where the failures have been")
    ap.add_argument("--levels", default=None,
                    help="comma-separated levels to check, e.g. 43,44,45")
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--degree", type=int, default=11)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="max nats of interpolation error allowed.  This "
                         "is a BREAKAGE detector, not an accuracy "
                         "measurement: the smoke store is deliberately "
                         "coarse (5%% decimated by 4 = 21.5%% effective, "
                         "five times coarser than production), so a tight "
                         "tolerance here fails on grid coarseness rather "
                         "than on a bug.  The failures this exists to "
                         "catch were 1e2 to 1e3 nats.  Use "
                         "ladder_accuracy.py to measure accuracy.")
    args = ap.parse_args()

    if args.tables:
        root = Path(args.tables)
        if not (root / "anchors.json").exists():
            raise SystemExit(f"{root} is not an anchor store")
    else:
        root = Path(args.keep) if args.keep else Path(
            tempfile.mkdtemp()) / "smoke"
    if not (root / "anchors.json").exists():
        print(f"building a small anchor store at {root} ...", flush=True)
        build(root, args.jobs)
    meta = json.loads((root / "anchors.json").read_text())

    import os
    os.environ["PMM_PHI_LADDER_EVERY"] = str(args.every)
    os.environ["PMM_PHI_LADDER_DEGREE"] = str(args.degree)
    os.environ["PMM_UNIVERSAL_TABLES"] = str(root)
    import importlib
    import product_model_with_memory.universal_tables as UT
    importlib.reload(UT)

    tab = UT.UniversalTables(root, read_only=True)
    u = np.linspace(-8.0, 10.0, 33)
    fails = []

    def check(name, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}"
              f"{'' if ok else '   ' + detail}")
        if not ok:
            fails.append(name)

    want = ([int(x) for x in args.levels.split(",")] if args.levels
            else sorted(int(k) for k in meta["levels"]))
    dense = int(meta.get("dense_below", DENSE))
    for L in want:
        if str(L) not in meta["levels"]:
            check(f"L={L} present in store", False, "no such level")
            continue
        grid = tab._ladder_grid(L)
        entry = meta["levels"][str(L)]

        check(f"L={L} grid starts at r=0",
              int(grid[0]) == 0, f"starts at {int(grid[0])}")
        check(f"L={L} dense floor intact after every-{args.every}",
              all(r in set(int(x) for x in grid) for r in range(0, dense)),
              "decimation removed small-r anchors")

        # the commonest counts in any corpus
        for r in (0, 1, 2, 3):
            try:
                got = tab.log_phi(L, r, u)
                ex = tab.log_phi_exact(L, r, u)
                check(f"L={L} r={r} served exactly",
                      np.array_equal(got, ex),
                      f"max diff {np.abs(got - ex).max():.2e}")
            except RuntimeError as exc:
                check(f"L={L} r={r} served exactly", False, str(exc)[:70])

        tg = list(entry["targets"])[:3] + list(entry["targets"])[-2:]
        for t in tg:
            got = tab.log_phi(L, t, u)
            ex = tab.log_phi_exact(L, t, u)
            e = float(np.abs(got - ex).max())
            check(f"L={L} r={t} interpolates within {args.tol:g}",
                  e < args.tol, f"{e:.2e} nats")

        if tg:
            M = tab.log_phi_matrix(L, tg, u)
            rows = np.array([tab.log_phi(L, t, u) for t in tg])
            check(f"L={L} log_phi_matrix == log_phi",
                  np.array_equal(M, rows),
                  f"max diff {np.abs(M - rows).max():.2e}")

    # and a real codelength through the real evaluator
    try:
        import product_model_with_memory.codelength as cl
        importlib.reload(cl)
        profiles = {"a": (1, 1, 1), "b": (2, 1, 1), "c": (1,),
                    "d": (3, 2, 2, 1, 1), "e": (5, 1)}
        res = cl.depth_averaged_codelength_profiles(
            profiles, d=6, l_max=6, jobs=1)
        worst = max(res[k].log2_q_avg for k in profiles)
        check("real codelength: log2 q <= 0", worst <= 0.0,
              f"worst log2 q = {worst:.4g}")
    except Exception as exc:
        check("real codelength: log2 q <= 0", False, str(exc)[:90])

    # Large counts at realistic alphabet size, checked for stability
    # under decimation.  The 2 Aug failure mode --- curvature computed
    # from cancelling rho terms turning to noise and flipping the
    # integration method --- moved codelengths by WHOLE BITS, only on
    # profiles with large counts and realistic d, and passed every
    # small-count check above.  The same profile through the same store
    # at every=1 and the decimated ladder must agree far under a bit:
    # a method flip or curvature garbage shows up as a 1-6 bit jump.
    # (Both children share the store and the evaluator, so this is a
    # BREAKAGE detector for the evaluator+ladder path at large counts,
    # not an accuracy measurement --- compare_evaluators.py measures.)
    CHILD = r'''
import json, sys
sys.path.insert(0, "src")
from product_model_with_memory.codelength import (
    depth_averaged_codelength_profiles)
res = depth_averaged_codelength_profiles(
    {"big": (3500, 800, 90, 7, 1)}, d=1024, l_max=6, jobs=1)
print("@@%.12f" % res["big"].log2_q_avg)
'''

    def _child_q(every: int) -> float:
        env = dict(os.environ)
        env["PMM_UNIVERSAL_TABLES"] = str(root)
        env["PMM_PHI_LADDER_EVERY"] = str(every)
        env["PMM_PHI_LADDER_DEGREE"] = str(args.degree)
        env["PYTHONPATH"] = "src"
        p = subprocess.run([sys.executable, "-c", CHILD],
                           capture_output=True, text=True, env=env)
        for line in p.stdout.splitlines():
            if line.startswith("@@"):
                return float(line[2:])
        raise RuntimeError((p.stdout + p.stderr)[-300:])

    try:
        q1 = _child_q(1)
        qd = _child_q(args.every)
        d_bits = abs(qd - q1)
        check("large-count codelength stable under decimation",
              d_bits < 1e-3 and q1 <= 0.0,
              f"|every-{args.every} - every-1| = {d_bits:.3e} bits, "
              f"log2 q = {q1:.4g}")
    except Exception as exc:
        check("large-count codelength stable under decimation", False,
              str(exc)[:90])

    print()
    if fails:
        print(f"SMOKE FAILED: {len(fails)} check(s) --- do not spend a "
              "real run yet")
        raise SystemExit(1)
    print("smoke passed; the complete system is worth a real run")


if __name__ == "__main__":
    main()
