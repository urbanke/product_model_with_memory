#!/usr/bin/env python3
"""Build a DESIGNED universal-table store: anchors on a geometric grid.

The store in tables/universal_v2 is a cache.  It holds an exact column
for every (L, r) some run once asked for, which is the correct thing for
a cache and the wrong thing for a foundation: the set of entries is
whatever history produced, it is not reproducible on another machine,
and --- as the ladder measurement discovered the hard way --- it does
not contain the anchor grid anyone wants to interpolate on.  A 2% ladder
measured against it was really an irregular ladder with gaps, because
only 318 of the 697 ideal anchors were present.

This writes the grid itself:

    r_k = round((1 + f)^k),  k = 0, 1, 2, ...   deduplicated, r <= r_max

deterministic, reproducible, and identical on any machine given the same
(f, levels, r_max).

DECIMATION.  Choose f once, at the finest spacing wanted, and coarser
ladders come out as exact subsets: taking every m-th k gives the grid
for (1+f)^m - 1.  Built at 0.5%, one store answers 0.5%, 1.0%, 2.0%,
4.1% ... with no gaps in any of them, so the spacing experiment runs
against real ladders rather than against whatever the cache happened to
retain.  (In the small-r regime consecutive k collide after rounding and
the grid is just 1, 2, 3, ...; every ladder is exact there because the
target IS an anchor, so decimation only matters where r is large enough
for the anchors to separate, which is where interpolation happens.)

TEST TARGETS.  Interpolation has to be checked against something, so a
sample of NON-anchor r values is built too.  They are recorded as
targets in anchors.json and must never be used as anchors --- a ladder
that quietly interpolates from the point it is being tested at measures
nothing.

Dry run by default; pass --go to actually build.

    python scripts/build_anchor_store.py --factor 0.005 --levels 2-45 \\
        --r-max-from tables/universal_v2 --jobs 12 --out tables/anchors_f005
    python scripts/build_anchor_store.py ... --go
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from product_model_with_memory.universal_tables import (
    UniversalTables,
    column_start_index,
)


def anchor_grid(factor: float, r_max: int, dense_below: int = 256,
                pad: int = 96) -> list[int]:
    """Every integer below dense_below, then r_k = round((1+f)^k).

    THE DENSE FLOOR IS NOT AN OPTIMISATION.  Small r is where the data
    lives --- most counts in a corpus are 1, 2, 3 --- and it is where
    ln phi varies fastest in ln(r+1), so it is the worst place to
    interpolate.  A pure geometric grid happens to contain every small
    integer anyway (round(1.02^k) walks 1, 2, 3, ...), which hides the
    problem until the grid is DECIMATED: every 8th point of that list
    is 1, 5, 9, ..., and r=2 then has to be interpolated across nodes
    spanning ln(2)..ln(10).  Measured on a real run: log2 q came out at
    +1321 bits, i.e. a probability of 2^1321.

    Stating the floor explicitly means decimation can preserve it.

    PAD extends the grid above r_max.  Degree-11 interpolation wants six
    anchors on each side; a target near the top of the grid gets a
    one-sided stencil instead, and accuracy falls by two orders --- from
    ~5e-9 nats in the interior to 2.6e-6 for the target nearest the last
    anchor (measured 2 August).  Since the largest counts in a corpus sit
    exactly there, padding past the largest expected r is not optional,
    and it costs a handful of columns per level.
    """

    # from ZERO: r=0 is a legal query, and a grid starting at 1
    # makes it an extrapolation rather than an interpolation
    out = list(range(0, min(dense_below, r_max) + 1))
    k = 0
    while True:
        r = int(round((1.0 + factor) ** k))
        if r > r_max:
            break
        if r > out[-1]:
            out.append(r)
        k += 1
    # ... then `pad` further steps, so the largest real r still has a
    # two-sided stencil
    extra = 0
    while extra < pad:
        r = int(round((1.0 + factor) ** k))
        if r > out[-1]:
            out.append(r)
            extra += 1
        k += 1
    return out


def decimate(anchors: list[int], every: int, dense_below: int) -> list[int]:
    """Decimate only ABOVE the dense floor; keep every integer below it.

    Decimating the whole list is what removed the small integers and
    produced a probability of 2^1321.
    """

    floor = [r for r in anchors if r < dense_below]
    tail = [r for r in anchors if r >= dense_below]
    return floor + tail[::every]


def parse_levels(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def r_max_per_level(src: str | None, levels: list[int],
                    default: int) -> dict[int, int]:
    """Where each level's counts actually stop.

    Taking one global r_max would build far past the data at deep
    levels, where high counts do not occur, and pay for columns nothing
    will ever read.
    """

    if src is None:
        return {L: default for L in levels}
    root = Path(src)
    out = {}
    for L in levels:
        f = root / f"level_{L:02d}.index.json"
        if not f.exists():
            out[L] = default
            continue
        rs = [int(k) for k in json.loads(f.read_text())]
        out[L] = max(rs) if rs else default
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="new store directory")
    p.add_argument("--factor", type=float, default=0.005,
                   help="anchor spacing; 0.005 = 0.5%%.  Build at the "
                        "finest spacing wanted --- coarser ladders are "
                        "exact subsets by decimation")
    p.add_argument("--levels", default="2-45",
                   help="ranges allowed, e.g. 2-45 or 2-40,45")
    p.add_argument("--r-max", type=int, default=1_000_000)
    p.add_argument("--r-max-from", default=None,
                   help="take each level's r_max from an existing store "
                        "instead of one global value")
    p.add_argument("--decimate-from", default=None,
                   help="build every m-th anchor of an existing store's "
                        "grid (with --every).  This is the ONLY correct "
                        "way to build the production store from an "
                        "experiment store: round((1+f')^k) for the "
                        "nominal coarser f' does not coincide with a "
                        "decimation, so a store built by factor would not "
                        "be the grid that was measured")
    p.add_argument("--every", type=int, default=1,
                   help="decimation step, with --decimate-from.  Applies "
                        "only ABOVE the dense floor")
    p.add_argument("--pad-anchors", type=int, default=96,
                   help="anchors added ABOVE r_max so the largest real r "
                        "still gets a two-sided stencil.  Counted in FINE "
                        "anchors, so it must survive decimation: degree 11 "
                        "wants 6 above the query, and 6 x every-16 = 96.  "
                        "Padding by 8 would leave 1 after every-8 --- "
                        "still one-sided, and still two orders worse")
    p.add_argument("--dense-below", type=int, default=256,
                   help="every integer below this is an anchor and is "
                        "served exactly; decimation never removes them")
    p.add_argument("--targets-per-level", type=int, default=40,
                   help="non-anchor columns built as interpolation test "
                        "points; recorded as targets, never anchors")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--go", action="store_true",
                   help="actually build; without it this only prints "
                        "the plan and its cost")
    args = p.parse_args()

    src_meta = None
    if args.decimate_from:
        mf = Path(args.decimate_from) / "anchors.json"
        if not mf.exists():
            raise SystemExit(f"--decimate-from needs {mf}")
        src_meta = json.loads(mf.read_text())
        levels = sorted(int(k) for k in src_meta["levels"])
        if args.levels != "2-45":       # an explicit --levels narrows it
            levels = [L for L in parse_levels(args.levels) if
                      str(L) in src_meta["levels"]]
        eff = (1.0 + src_meta["factor"]) ** args.every - 1.0
        print(f"decimating {args.decimate_from}: every {args.every} of a "
              f"{src_meta['factor']:.3%} grid -> {eff:.3%}")
        args.factor = eff
    else:
        levels = parse_levels(args.levels)
    rmax = r_max_per_level(args.r_max_from, levels, args.r_max)
    rng = np.random.default_rng(args.seed)

    plan, n_cols, n_vals = {}, 0, 0
    for L in levels:
        if src_meta is not None:
            anchors = decimate(
                [int(r) for r in src_meta["levels"][str(L)]["anchors"]],
                args.every, src_meta.get("dense_below", 0))
            rmax[L] = anchors[-1]
        else:
            anchors = anchor_grid(args.factor, rmax[L],
                                  args.dense_below, args.pad_anchors)
        aset = set(anchors)
        # targets sit BETWEEN anchors, in the geometric regime where
        # consecutive anchors actually differ; below that every integer
        # is an anchor and there is nothing to interpolate
        lo = next((a for a, b in zip(anchors, anchors[1:]) if b > a + 1),
                  anchors[-1])
        # Sample targets by REJECTION, never by materialising the range.
        # `[r for r in range(lo+1, r_max)]` is fine at r_max = 1e6 and is
        # an 8 GB list at 2e8 --- 52 of them, one per level.  The symptom
        # is not an error but a silent hang before the first line of
        # output, while the machine swaps.
        span = rmax[L] - lo - 1
        want = min(args.targets_per_level, max(0, span - len(aset)))
        seen: set[int] = set()
        for _ in range(200 * max(1, want)):
            if len(seen) >= want:
                break
            r = int(rng.integers(lo + 1, rmax[L]))
            if r not in aset:
                seen.add(r)
        tgts = sorted(seen)
        plan[L] = {"anchors": anchors, "targets": tgts}
        for r in anchors + tgts:
            n_cols += 1
            n_vals += column_start_index(L, r) + 1

    # 161k values/sec/core measured on the reference machine
    # 161k values/sec/core is the SERIES builder.  PMM_BUILD_EXACT
    # forces contour integration and runs about 6.5x slower (measured
    # 2 August: 15.6k values/sec at r=1e8, 65k at r=1e3) --- and
    # PMM_BUILD_EXACT is set for every real store, so quoting the fast
    # figure understates the wait by nearly an order of magnitude.
    rate = 25_000.0 if os.environ.get("PMM_BUILD_EXACT", "") not in ("", "0") \
        else 161_000.0
    core_sec = n_vals / rate
    print(f"spacing {args.factor:.3%}   levels {levels[0]}..{levels[-1]}"
          f"   r_max {min(rmax.values()):,}..{max(rmax.values()):,}")
    print(f"  anchors/level  {min(len(v['anchors']) for v in plan.values()):,}"
          f" .. {max(len(v['anchors']) for v in plan.values()):,}")
    print(f"  columns        {n_cols:,}")
    print(f"  size           {n_vals * 8 / 2**30:.1f} GiB")
    print(f"  builder        "
          f"{'EXACT (contour)' if rate < 100_000 else 'series shortcut'}")
    print(f"  build          {core_sec / 3600:.1f} core-hours"
          f"  (~{core_sec / max(1, args.jobs) / 60:.0f} min at "
          f"--jobs {args.jobs})")
    dec = [(m, (1.0 + args.factor) ** m - 1.0) for m in (1, 2, 4, 8, 16)]
    print("  decimations    " + ", ".join(
        f"every {m} -> {s:.2%}" for m, s in dec))
    if not args.go:
        print("\ndry run; nothing written.  Add --go to build.")
        return

    out = Path(args.out)
    # Refuse to build into an existing designed store.  The seal in
    # UniversalTables blocks appends, but that is not enough here: if the
    # store is already complete there is nothing to append, so the build
    # would walk every level, print progress, rewrite anchors.json and
    # report success --- having done nothing.  A no-op that looks like a
    # build is worse than an error, because the result is a store you
    # believe you just rebuilt.
    if (out / "anchors.json").exists():
        raise SystemExit(
            f"{out} is already a sealed anchor store.  Building into it "
            "would silently do\nnothing (its columns exist, so none are "
            "appended) while rewriting anchors.json\nand reporting "
            f"success.  Delete it first:\n\n    rm -rf {out}\n")
    if out.exists() and any(out.glob("level_*.bin")):
        raise SystemExit(
            f"{out} already holds level files but no anchors.json --- an "
            "interrupted build\nor a cache.  Building on top of it would "
            "mix designed anchors with whatever is\nalready there, which "
            f"is exactly what a designed store must not be.\n\n    "
            f"rm -rf {out}\n")
    tab = UniversalTables(out)
    t0 = time.time()
    built = 0
    for L in levels:
        pairs = [(L, r) for r in plan[L]["anchors"] + plan[L]["targets"]]
        n = tab.build_columns(pairs, jobs=args.jobs)
        built += n
        done = time.time() - t0
        print(f"  L={L:>2}  {n:>6,} columns   {done / 60:>5.1f} min elapsed",
              flush=True)
    tab.close()

    # the per-level build locks have no meaning once the store is
    # complete and sealed; leaving them behind (52 of them in
    # anchors_prod) reads as an interrupted build
    for lock in out.glob("level_*.lock"):
        lock.unlink(missing_ok=True)

    (out / "anchors.json").write_text(json.dumps({
        "factor": args.factor,
        "pad_anchors": args.pad_anchors,
        "dense_below": (src_meta.get("dense_below", args.dense_below)
                        if src_meta else args.dense_below),
        "grid": "r_k = round((1+factor)**k), deduplicated",
        "levels": {str(L): {"r_max": rmax[L],
                            "anchors": plan[L]["anchors"],
                            "targets": plan[L]["targets"]}
                   for L in levels},
        "note": "targets are TEST points and must never be used as "
                "interpolation anchors",
    }, indent=2))
    print(f"\nbuilt {built:,} columns in {(time.time() - t0) / 60:.1f} min")
    print(f"grid written to {out / 'anchors.json'}")
    print("\nVerify before trusting it:")
    print(f"    python scripts/check_store.py --tables {out} --per-level 40")


if __name__ == "__main__":
    main()
