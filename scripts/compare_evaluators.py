#!/usr/bin/env python3
"""Where does the ladder+expansion disagree with exact columns?

Two paper runs moved when the evaluator changed --- ctree_fullvocab by
8.5e-4 bits/token and pooled_v1024 by 3.9e-2 --- while nine others
reproduced to 1e-6 or better.  The cache they were compared against is
clean (200 samples at each of L=5,10,20,33: zero bad), so the shift is
the new evaluator's.

smoke_ladder.py did not catch it because it tests profiles like
(1,1,1) at l_max=6: small counts, shallow.  The two runs that disagree
are the two with LARGE counts and DEEP l_max.

This computes the same codelength both ways --- once from exact stored
columns, once through the ladder and expansion --- over a grid of
profiles spanning small to large counts and shallow to deep l_max, and
prints where they part company.  No corpus, no checkpoints, nothing but
the evaluator.

THE d MATTERS.  The first version evaluated every profile at
d = max(len(profile)+1, 4).  Real runs use d = alphabet size (256,
1024, 300k), and the (d-s)*log_phi_0 term is what pulls the outer
integrand down at large u: at d~8 the peak of a large-N profile sits
within 10 nats of the grid's right edge and the scan refuses to
converge.  That is why the original huge rows errored on BOTH
evaluators --- the regime above counts of 1e4 was never compared at
all.  Measured 2 Aug: the same (1e6,...) profile that fails at d=8
evaluates cleanly at d=1024.

THE DOMAIN IS MEASURED, NOT GUESSED.  universal_v2 is a cache, so its
per-level indices enumerate every (L, r) any run ever requested:
r up to 17,005,209 at L<=26, up to 1,063,586 at L=27..54, ~1e4 at
L>=55.  (Upper bounds: probe scripts, including this one, also feed
the cache.)  The graded c* rows below walk that range.

    python scripts/compare_evaluators.py \\
        --exact tables/universal_v2 --ladder tables/anchors_prod
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import json


def _tail(c: int) -> tuple[int, ...]:
    """Top count c with a geometrically decaying tail, ending at 1."""

    out: list[int] = []
    while c >= 1 and len(out) < 7:
        out.append(int(c))
        c //= 4
    if out[-1] != 1:
        out.append(1)
    return tuple(out)


REQUIREMENT_BITS = 1e-4

PROFILES = {
    # name: (profile, l_max, d).  d=None means the legacy default
    # max(len(profile)+1, 4), kept so the handover rows stay comparable.
    # The old huge/one-big-count rows are gone: at legacy d they cannot
    # converge on either evaluator and compare nothing (see docstring).
    "tiny/shallow":      ((1, 1, 1), 6, None),
    "tiny/deep":         ((1, 1, 1), 33, None),
    "small/shallow":     ((5, 3, 2, 1, 1), 6, None),
    "small/deep":        ((5, 3, 2, 1, 1), 33, None),
    "medium/deep":       ((300, 120, 40, 10, 3, 1), 33, None),
    "large/deep":        ((10_000, 4_000, 900, 120, 30, 5, 1), 33, None),
    "one medium count":  ((1_000,), 33, None),
    # graded top count at realistic shape and d: crosses the dense
    # floor (256) and walks log-spaced through the interpolated region
    # up to the largest count the cache says a real run ever asked for
    "c200":     (_tail(200), 33, 1024),
    "c256":     (_tail(256), 33, 1024),
    "c300":     (_tail(300), 33, 1024),
    "c500":     (_tail(500), 33, 1024),
    "c1k":      (_tail(1_000), 33, 1024),
    "c3k":      (_tail(3_162), 33, 1024),
    "c10k":     (_tail(10_000), 33, 1024),
    "c30k":     (_tail(31_623), 33, 1024),
    "c100k":    (_tail(100_000), 33, 1024),
    "c300k":    (_tail(316_228), 33, 1024),
    "c1M":      (_tail(1_000_000), 33, 1024),
    "c1M/l53":  (_tail(1_000_000), 53, 1024),
    "c17M":     (_tail(17_005_209), 33, 1024),
    # full-vocab shape: the deep-level r_max the cache actually holds,
    # at the fullvocab alphabet size
    "fv/1M":    (_tail(1_063_586), 33, 300_000),
    # does d itself move the delta, at fixed profile?
    "d256/10k": (_tail(10_000), 33, 256),
}

CHILD = r'''
import json, os, sys
sys.path.insert(0, "src")
from product_model_with_memory.codelength import (
    depth_averaged_codelength_profiles)
spec = json.loads(sys.argv[1])
out = {}
for name, (prof, lmax, d) in spec.items():
    prof = tuple(prof)
    if d is None:
        d = max(len(prof) + 1, 4)
    try:
        res = depth_averaged_codelength_profiles(
            {name: prof}, d=d, l_max=lmax, jobs=1)
        out[name] = float(res[name].log2_q_avg)
    except Exception as exc:
        # a case that FAILS on one evaluator and not the other is itself
        # the finding; do not let it abort the whole comparison
        out[name] = "ERR: " + str(exc)[:300]
print("@@" + json.dumps(out))
'''


def run(env_extra: dict, spec: dict) -> dict:
    env = dict(os.environ)
    for k in ("PMM_UNIVERSAL_TABLES", "PMM_PHI_LADDER_EVERY",
              "PMM_PHI_LADDER_DEGREE", "PMM_PHI_SADDLE_MIN_L",
              "PMM_PHI_LADDER"):
        env.pop(k, None)
    env.update(env_extra)
    env["PYTHONPATH"] = "src"
    # The exact store is an unsealed cache: a probe that asks for a
    # missing column BUILDS it.  Without this, the build would go
    # through the right-series branch whose certificate is known bad at
    # large r --- quietly writing wrong columns into the clean
    # reference, exactly where the measurement happens.
    env["PMM_BUILD_EXACT"] = "1"
    p = subprocess.run([sys.executable, "-c", CHILD, json.dumps(spec)],
                       capture_output=True, text=True, env=env)
    if p.returncode != 0:
        sys.stderr.write(p.stdout[-2000:] + p.stderr[-2000:])
        raise SystemExit("child failed")
    for line in p.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    raise SystemExit("no result from child")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exact", default="tables/universal_v2",
                    help="store read with NO ladder: exact columns")
    ap.add_argument("--ladder", default="tables/anchors_prod",
                    help="anchor store, read through the ladder")
    ap.add_argument("--every", type=int, default=1,
                    help="read-time decimation of the ladder store's "
                         "own grid (PMM_PHI_LADDER_EVERY)")
    ap.add_argument("--degree", type=int, default=11)
    ap.add_argument("--cutoff", type=int, default=54)
    ap.add_argument("--only", default=None,
                    help="comma-separated case names; default all")
    ap.add_argument("--out", default=None,
                    help="append one JSON line of results here")
    args = ap.parse_args()

    cases = dict(PROFILES)
    if args.only:
        want = [w.strip() for w in args.only.split(",")]
        missing = [w for w in want if w not in cases]
        if missing:
            raise SystemExit(f"unknown case(s): {missing}; "
                             f"known: {list(cases)}")
        cases = {k: cases[k] for k in want}
    spec = {k: [list(v[0]), v[1], v[2]] for k, v in cases.items()}

    print(f"exact : {args.exact} (no ladder, no expansion)")
    print(f"ladder: {args.ladder} (every={args.every} "
          f"degree={args.degree} expansion at L>={args.cutoff})")
    print()
    a = run({"PMM_UNIVERSAL_TABLES": args.exact}, spec)
    b = run({"PMM_UNIVERSAL_TABLES": args.ladder,
             "PMM_PHI_LADDER_EVERY": str(args.every),
             "PMM_PHI_LADDER_DEGREE": str(args.degree),
             "PMM_PHI_SADDLE_MIN_L": str(args.cutoff)}, spec)

    print("%-18s %18s %18s %12s" % ("case", "exact", "ladder",
                                    "delta bits"))
    worst = ("", 0.0)
    compared, failed, errs = 0, [], []
    for name in cases:
        x, y = a.get(name), b.get(name)
        if not isinstance(x, float) or not isinstance(y, float):
            errs.append(name)
            print("%-18s  exact : %s" % (name, x))
            print("%-18s  ladder: %s" % ("", y))
            continue
        d = y - x
        if abs(d) >= REQUIREMENT_BITS:
            flag = "   <-- FAIL"
        elif abs(d) >= 1e-6:
            flag = "   <--"
        else:
            flag = ""
        print("%-18s %18.8f %18.8f %+12.3e%s" % (name, x, y, d, flag))
        compared += 1
        if abs(d) >= REQUIREMENT_BITS:
            failed.append((name, d))
        if abs(d) > abs(worst[1]):
            worst = (name, d)

    print()
    print(f"requirement: |delta| < {REQUIREMENT_BITS:g} bits on every "
          "profile the experiments produce")
    if compared == 0:
        print("NOTHING WAS COMPARED --- every case failed on one side "
              "or the other.")
    elif failed:
        print(f"FAIL: {len(failed)} of {compared} cases at or above "
              f"{REQUIREMENT_BITS:g} bits; worst {worst[0]} at "
              f"{worst[1]:+.3e}.  The pattern across the graded c* rows "
              "says whether it tracks the top count, the depth, or "
              "both.")
    else:
        print(f"PASS: all {compared} compared cases below "
              f"{REQUIREMENT_BITS:g} bits (worst {worst[0]} at "
              f"{worst[1]:+.3e}).")
    if errs:
        print(f"not compared (error on at least one side): {errs}")

    if args.out:
        rec = {"exact": args.exact, "ladder": args.ladder,
               "every": args.every, "degree": args.degree,
               "cutoff": args.cutoff,
               "delta_bits": {n: (b.get(n) - a.get(n))
                              for n in cases
                              if isinstance(a.get(n), float)
                              and isinstance(b.get(n), float)},
               "errors": {n: {"exact": a.get(n), "ladder": b.get(n)}
                          for n in errs}}
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"\nappended: {args.out}")


if __name__ == "__main__":
    main()
