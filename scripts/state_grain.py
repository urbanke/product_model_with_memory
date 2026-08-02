#!/usr/bin/env python3
"""How much does the state have to remember?

A first-order model gives every distinct previous symbol its own
distribution.  With $71{,}161$ states and $25.8$ million tokens that is
about $362$ observations each, and the distribution is heavily skewed,
so most states are nearly empty.  Coarsening the state --- keeping the
M most frequent symbols and pooling the rest into one state --- lowers
the number of distributions to estimate at the price of a higher
conditional entropy.  This script measures the price, which is the half
of the trade that needs no model:

    H(X | state_M)  for a grid of M

At M = 0 this is the unconditional entropy and at M = V it is the full
first-order conditional entropy, so the curve interpolates between the
memoryless and first-order rows of the paper's tables.  It is the BEST
POSSIBLE column as a function of M, and it can only rise as M falls.
What it cannot say is where the optimum lies, because that depends on
the learning cost, which only the layered model pays.  Read it as the
bias side of the trade and as a way of choosing a sensible M grid.

Also reported: how the observations are distributed over states, since
the mass sitting in states too rare to estimate is what pooling
recovers cheaply.

    python scripts/state_grain.py --ids output/streams/bpe_enwik8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from product_model_with_memory.streams import load_stream


def _entropy_bits(counts: np.ndarray) -> float:
    c = counts[counts > 0].astype(np.float64)
    if c.size == 0:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", required=True)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--grid", default="0,1,2,4,8,16,32,64,128,256,512,1024,"
                                     "2048,4096,8192,16384,32768,65536")
    args = p.parse_args()

    ids, meta = load_stream(args.ids)
    if args.n:
        ids = ids[: args.n]
    ids = ids.astype(np.int64)
    V = int(ids.max()) + 1
    a, b = ids[:-1], ids[1:]
    N = len(a)
    bpc = meta.get("bytes_per_token", 1.0)

    # (state, successor) counts, and the successor histogram of everything
    uniq, cnt = np.unique(a * V + b, return_counts=True)
    st, su = uniq // V, uniq % V
    freq = np.bincount(a, minlength=V)
    order = np.argsort(-freq, kind="stable")
    order = order[freq[order] > 0]
    n_states = len(order)

    # how the observations are distributed over states
    f = freq[order].astype(np.float64)
    share = np.cumsum(f) / N
    quant = {f"states_holding_{int(q*100)}pct_of_positions":
             int(np.searchsorted(share, q) + 1) for q in (0.5, 0.9, 0.99)}
    rare = {f"mass_in_states_with_at_most_{t}_observations":
            float(f[f <= t].sum() / N) for t in (1, 2, 5, 10, 100)}

    # per-state entropy contribution, and the pooled tail
    rowstart = np.searchsorted(st, np.arange(V + 1))     # st is sorted
    tail = np.bincount(b, minlength=V).astype(np.float64)
    promoted_mass = 0.0
    promoted_sum = 0.0                                    # sum (n_s/N) H_s

    grid = sorted({int(x) for x in args.grid.split(",")} | {n_states})
    grid = [m for m in grid if m <= n_states]
    rows, gi, done = [], 0, 0

    for m in grid:
        while done < m:
            s = order[done]
            lo, hi = rowstart[s], rowstart[s + 1]
            c = cnt[lo:hi].astype(np.float64)
            if c.size:
                promoted_sum += c.sum() / N * _entropy_bits(c)
                promoted_mass += c.sum()
                tail[su[lo:hi]] -= c
            done += 1
        tail_mass = N - promoted_mass
        h = promoted_sum + (tail_mass / N) * _entropy_bits(tail)
        rows.append({
            "M": m,
            "states": m + (1 if tail_mass > 0 else 0),
            "backoff_share_of_positions": tail_mass / N,
            "H_given_state_bits_per_symbol": h,
            "H_given_state_bits_per_char": h / bpc,
        })

    h0 = rows[0]["H_given_state_bits_per_symbol"]
    out = {"representation": meta["representation"],
           "source_file": meta["source_file"],
           "symbols": int(len(ids)), "distinct_states": n_states,
           "observations_per_state_mean": N / n_states,
           **quant, **rare, "curve": rows}
    Path(args.ids, "state_grain.json").write_text(json.dumps(out, indent=2))

    print(f"{meta['representation']} / {meta['source_file']}: "
          f"{n_states:,} states, {N/n_states:,.0f} observations each on "
          f"average")
    for k, v in quant.items():
        print(f"  {k:44s} {v:>10,}")
    for k, v in rare.items():
        print(f"  {k:44s} {100*v:>9.2f}%")
    print(f"\n{'M':>8} {'states':>9} {'backoff':>9} {'H(X|state)':>12} "
          f"{'bits/char':>10} {'vs M=0':>9}")
    for r in rows:
        print(f"{r['M']:>8} {r['states']:>9,} "
              f"{100*r['backoff_share_of_positions']:>8.2f}% "
              f"{r['H_given_state_bits_per_symbol']:>12.4f} "
              f"{r['H_given_state_bits_per_char']:>10.4f} "
              f"{h0 - r['H_given_state_bits_per_symbol']:>9.4f}")
    print("\nThe last column is what the state buys before any learning "
          "cost.\nIt can only grow with M; where the net optimum lies "
          "depends on the\nlearning cost, which only the layered model "
          "pays.")


if __name__ == "__main__":
    main()
