#!/usr/bin/env python3
"""Would walking levels outward from the mode save anything?

The level truncation in `codelength._LevelWindow` walks L upward from 1
and stops a profile once its term has fallen far below its own running
maximum.  That fires on profiles whose level curve peaks early and never
on profiles whose curve peaks late, and even for the first kind it has
to climb past the mode before it can stop.  Walking outward from the
mode instead would evaluate only the levels that carry the average.

This measures what that would be worth, on real profiles, before anyone
implements it: it evaluates every level of every distinct first-order
state profile of a stream, then counts

  one-sided   levels the current rule evaluates
  two-sided   levels in the smallest contiguous window around the mode
              holding all but `eps` of the mass

The ratio of the totals is the speedup an ideal outward walk would give
on this workload.  It is an upper bound: a real implementation has to
find the mode, which costs a few evaluations per profile.

    PMM_NO_TRUNCATE=1 python scripts/level_window_probe.py \
        --ids output/streams/bpe_enwik8 --jobs 12
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from product_model_with_memory.codelength import (
    LEVEL_DROP_BITS,
    LEVEL_PATIENCE,
    default_l_max,
    depth_averaged_codelength_profiles,
    profile_of,
)
from product_model_with_memory.streams import load_stream, reduce_ids


def one_sided(q, drop=LEVEL_DROP_BITS, patience=LEVEL_PATIENCE) -> int:
    best, last, falling = -math.inf, math.inf, 0
    for i, v in enumerate(q, start=1):
        falling = falling + 1 if v < last else 0
        last = v
        best = max(best, v)
        if falling >= patience and best - v > drop:
            return i
    return len(q)


def two_sided(q, eps: float = 1e-12) -> tuple[int, int]:
    a = np.asarray(q)
    w = np.exp2(a - a.max())
    total, m = w.sum(), int(np.argmax(a))
    lo = hi = m
    acc = w[m]
    while acc < (1 - eps) * total:
        left = w[lo - 1] if lo > 0 else -1.0
        right = w[hi + 1] if hi < len(w) - 1 else -1.0
        if right >= left:
            hi += 1
            acc += w[hi]
        else:
            lo -= 1
            acc += w[lo]
    return hi - lo + 1, m + 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", required=True)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--max-profiles", type=int, default=4000,
                   help="evaluate at most this many distinct profiles, "
                        "sampled across the size range")
    args = p.parse_args()

    ids, meta = load_stream(args.ids)
    top_k = args.top_k or meta["alphabet"] - 1
    reduced, V, _ = reduce_ids(ids, top_k)
    reduced = np.asarray(reduced, dtype=np.int64)
    l_max = default_l_max(V)

    succ: dict[int, Counter] = {}
    for a, b in zip(reduced[:-1].tolist(), reduced[1:].tolist()):
        succ.setdefault(a, Counter())[b] += 1
    profiles = {profile_of(c) for c in succ.values() if sum(c.values()) >= 2}
    profiles = sorted(profiles, key=sum)
    if len(profiles) > args.max_profiles:
        step = len(profiles) / args.max_profiles
        profiles = [profiles[int(i * step)] for i in range(args.max_profiles)]
    print(f"{meta['representation']}: V={V:,}, l_max={l_max}, "
          f"{len(profiles):,} distinct profiles evaluated", flush=True)

    res = depth_averaged_codelength_profiles(
        {i: pr for i, pr in enumerate(profiles)}, d=V, l_max=l_max,
        jobs=args.jobs)

    tot1 = tot2 = 0
    table = []
    for i, pr in enumerate(profiles):
        q = list(res[i].log2_q_by_depth)
        if not all(math.isfinite(v) for v in q):
            raise SystemExit("run this with PMM_NO_TRUNCATE=1 --- the "
                             "curves must be complete")
        a, (b, mode) = one_sided(q), two_sided(q)
        tot1 += a
        tot2 += b
        table.append((sum(pr), len(set(pr)), a, b, mode))

    print(f"\n{'N':>10} {'ktilde':>7} {'one-sided':>10} {'two-sided':>10} "
          f"{'mode':>5}")
    for row in table[::max(1, len(table) // 15)]:
        print(f"{row[0]:>10,} {row[1]:>7} {row[2]:>10} {row[3]:>10} "
              f"{row[4]:>5}")
    print(f"\nlevel evaluations: one-sided {tot1:,}, two-sided {tot2:,} "
          f"--> {tot1/tot2:.2f}x   (full sweep {len(profiles)*l_max:,})")
    print("This is an UPPER bound: finding the mode costs a few "
          "evaluations per profile.")

    out = {"stream": args.ids, "V": V, "l_max": l_max,
           "profiles": len(profiles),
           "one_sided_level_evaluations": tot1,
           "two_sided_level_evaluations": tot2,
           "speedup_upper_bound": tot1 / tot2,
           "full_sweep": len(profiles) * l_max,
           "rows": [{"N": r[0], "ktilde": r[1], "one_sided": r[2],
                     "two_sided": r[3], "mode": r[4]} for r in table]}
    Path(args.ids, "level_window_probe.json").write_text(
        json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
