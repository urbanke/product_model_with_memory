#!/usr/bin/env python3
"""How much does chunked (checkpointed) evaluation lose vs the exact
sequential code, for schemes whose exact value telescopes?

The exact code updates its tables after every token; its codelength
is one exchangeable evaluation per state at the final counts.  The
chunked code freezes tables at C checkpoints and codes each block
with stale tables.  Both are complete honest codes; the difference
is the price of chunking, which this script measures for memory 0
and memory 1 as a function of C and of the checkpoint SPACING
(equal blocks, or geometric blocks that are small early when the
tables learn fastest).

    python scripts/chunking_cost.py --ids output/streams/bpe_text8 \
        --order 1 --checkpoints 8,32,128 --spacing equal,geo \
        --out output/chunkcost_text8_o1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from product_model_with_memory.streams import load_stream, reduce_ids


def edges_for(n: int, C: int, spacing: str, first: int = 2048):
    """Block boundaries 0 = t_0 < t_1 < ... < t_C = n."""
    if spacing == "equal":
        return [round(n * k / C) for k in range(C + 1)]
    # geometric: block k has length ~ first * r^k with r chosen so the
    # lengths sum to n; small blocks early, where the tables move most
    lo, hi = 1.0, 4.0
    for _ in range(200):
        r = (lo + hi) / 2
        tot = first * (C if abs(r - 1) < 1e-12 else (r**C - 1) / (r - 1))
        if tot < n:
            lo = r
        else:
            hi = r
    r = (lo + hi) / 2
    e = [0]
    acc = 0.0
    for k in range(C):
        acc += first * r**k
        e.append(min(n, round(acc)))
    e[-1] = n
    return e


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--order", type=int, choices=(0, 1), required=True)
    ap.add_argument("--top-k", type=int, default=10**9,
                    help="cap the alphabet (smoke only; default: full)")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--checkpoints", default="8,32,128")
    ap.add_argument("--spacing", default="equal,geo")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    ids, meta = load_stream(args.ids)
    if args.n:
        ids = ids[: args.n]
    if args.top_k < 10**9:
        x, V, _ = reduce_ids(ids, args.top_k)
    else:
        x = ids.astype(np.int64)
        V = int(x.max()) + 1
    n = len(x)
    print(f"V={V} n={n:,} order={args.order}", flush=True)

    from product_model_with_memory.pooled_lags import (
        _LayeredPredictiveBuilder, _augmented_profile, _log2sumexp_arr)
    from product_model_with_memory.codelength import default_l_max
    builder = _LayeredPredictiveBuilder(
        V, default_l_max(V), None, args.jobs, None)

    # state of each position: () for order 0, previous token for 1
    off = args.order
    pos = np.arange(off, n)
    st = (np.zeros(len(pos), dtype=np.int64) if args.order == 0
          else x[pos - 1])
    sym = x[pos]
    key_all = st * V + sym

    # ---- exact reference: one evaluation per state at final counts
    order = np.argsort(st, kind="stable")
    st_s, sym_s = st[order], sym[order]
    from collections import Counter
    mult: Counter = Counter()
    i = 0
    while i < len(st_s):
        j = i
        while j < len(st_s) and st_s[j] == st_s[i]:
            j += 1
        cnt = np.bincount(sym_s[i:j])
        mult[tuple(sorted(int(c) for c in cnt[cnt > 0]))] += 1
        i = j
    need = [p for p in mult if p not in builder.memo]
    print(f"  exact: {len(mult)} profiles, {len(need)} to evaluate",
          flush=True)
    B = 2000
    for k in range(0, len(need), B):
        builder._ensure_families({p: () for p in need[k:k + B]})
    ll = math.log2(builder.l_max)
    bits_exact = -sum(m * (_log2sumexp_arr(builder.memo[p]) - ll)
                      for p, m in mult.items())
    print(f"  exact: {bits_exact / len(pos):.4f} bits/token "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- chunked runs
    ratio_memo: dict = {}
    results = {"ids": args.ids, "order": args.order, "V": V,
               "n_tokens": n, "coded_positions": len(pos),
               "bits_exact": bits_exact,
               "exact_bits_per_token": bits_exact / len(pos),
               "runs": []}
    for C in [int(c) for c in args.checkpoints.split(",")]:
        for spacing in args.spacing.split(","):
            tR = time.time()
            e = edges_for(len(pos), C, spacing)
            bits = 0.0
            # running per-state profiles as dict state -> Counter
            prof: dict = {}
            for b in range(C):
                lo, hi = e[b], e[b + 1]
                if hi <= lo:
                    continue
                blk = slice(lo, hi)
                kb, cb = np.unique(key_all[blk], return_counts=True)
                # PASS 1: group by state (kb is sorted), compute each
                # state's frozen profile once, collect the evaluations
                # this block needs but the memo lacks
                todo = []          # (base, cv, count)
                fams: dict = {}    # base -> set of cvs to evaluate
                i2 = 0
                while i2 < len(kb):
                    s_ = int(kb[i2]) // V
                    j2 = i2
                    while j2 < len(kb) and int(kb[j2]) // V == s_:
                        j2 += 1
                    p_ = prof.get(s_)
                    base = (tuple(sorted(p_.values())) if p_ else ())
                    for t2 in range(i2, j2):
                        y_ = int(kb[t2]) % V
                        cv = p_.get(y_, 0) if p_ else 0
                        todo.append((base, cv, int(cb[t2])))
                        if (base, cv) not in ratio_memo:
                            fams.setdefault(base, set()).add(cv)
                    i2 = j2
                # PASS 2: one batched evaluation for the whole block
                if fams:
                    builder._ensure_families(
                        {b_: tuple(sorted(cs)) for b_, cs in fams.items()})
                    for b_, cs in fams.items():
                        for cv in cs:
                            ratio_memo[(b_, cv)] = builder._log2_ratio(
                                b_, _augmented_profile(b_, cv))
                for base, cv, cc in todo:
                    bits -= cc * ratio_memo[(base, cv)]
                # reveal the block
                for kk, cc in zip(kb, cb):
                    s_, y_ = int(kk) // V, int(kk) % V
                    prof.setdefault(s_, Counter())[y_] += int(cc)
            gap = (bits - bits_exact) / len(pos)
            print(f"  C={C:<4d} {spacing:>5s}: "
                  f"{bits / len(pos):.4f} bits/token  "
                  f"gap {gap:+.4f}  ({time.time()-tR:.0f}s)", flush=True)
            results["runs"].append(
                {"C": C, "spacing": spacing,
                 "bits_per_token": bits / len(pos),
                 "gap_per_token": gap})
            # save after every configuration: a killed run keeps
            # everything already measured
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / "results.json").write_text(
                json.dumps(results, indent=2))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"written: {out/'results.json'} ({time.time()-t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
