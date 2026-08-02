#!/usr/bin/env python3
"""What would a second-order model cost?  Answered by counting.

Running order two properly means evaluating millions of state profiles
and building moment-table columns for all of them.  Before spending
that, the answer can be estimated from quantities that need no model at
all, using the one thing the first-order runs already measured: how
much a state costs to learn, as a function of how many observations it
has (`scripts/state_redundancy.py`).

For a state map sigma, the codelength decomposes as

    model  =  n * H(X | sigma)  +  learning cost,

the first term pure counting and the second the sum over states of an
excess that the redundancy run measured, bucketed by state size.  So
for any candidate map --- here the product map
sigma_{M1}(x_{t-1}) x sigma_{M2}(x_{t-2}) --- we can count the state
sizes, look the excess up per bucket, and add.

The forecast is an extrapolation, so it is validated rather than
trusted: with M2 = 0 the product map IS the first-order family, every
member of which has been evaluated exactly.  The script prints
predicted against measured for those members first.  If the forecast
does not reproduce them it should not be believed about order two
either.

    python scripts/order2_forecast.py --ids output/streams/bpe_enwik8 \\
        --redundancy output/redundancy_bpe_enwik8/results.json \\
        --family output/family_bpe_enwik8/results.json \\
        --m1 "1024,8192,32768,100277" --m2 "0,16,64,256,1024,100277"
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from product_model_with_memory.streams import (
    load_stream,
    reduce_ids,
    state_order_by_id,
)


def sigma(rank: np.ndarray, m: int) -> np.ndarray:
    """sigma_M as a lookup on reduced ids: the M symbols earliest in
    `rank` keep their identity, everything else goes to state M."""

    return np.minimum(rank, m)


def entropy_and_sizes(state: np.ndarray, succ: np.ndarray, V: int):
    """(sum n_s H_s in bits, state sizes, per-state support)."""

    n_state = int(state.max()) + 1
    key = state.astype(np.int64) * V + succ.astype(np.int64)
    uniq, cnt = np.unique(key, return_counts=True)
    owner = uniq // V
    cuts = np.flatnonzero(np.diff(owner)) + 1
    groups = np.split(cnt, cuts)
    sizes = np.array([g.sum() for g in groups], dtype=np.float64)
    support = np.array([len(g) for g in groups], dtype=np.float64)
    bits = 0.0
    for g, n in zip(groups, sizes):
        q = g / n
        bits += float(n) * float(-(q * np.log2(q)).sum())
    return bits, sizes, support


def excess_lookup(buckets):
    lo = np.array([b["n_from"] for b in buckets], dtype=np.float64)
    per = np.array([b["excess_per_token_in_bucket"] for b in buckets])

    def f(sizes: np.ndarray) -> float:
        idx = np.clip(np.searchsorted(lo, sizes, side="right") - 1,
                      0, len(lo) - 1)
        return float((sizes * per[idx]).sum())

    return f


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", required=True)
    p.add_argument("--redundancy", required=True,
                   help="results.json from scripts/state_redundancy.py "
                        "on the SAME stream")
    p.add_argument("--family", default=None,
                   help="results.json from the order-one family sweep, "
                        "for the validation table")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--m1", default="1024,8192,32768,100277")
    p.add_argument("--m2", default="0,16,64,256,1024,100277")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    red = json.loads(Path(args.redundancy).read_text())
    excess_of = excess_lookup(red["buckets"])

    ids, meta = load_stream(args.ids)
    total_tokens = len(ids)
    if args.n:
        ids = ids[: args.n]
    is_prefix = len(ids) < total_tokens
    top_k = args.top_k or meta["alphabet"] - 1
    reduced, V, capped, keep = reduce_ids(ids, top_k, return_keep=True)
    reduced = np.asarray(reduced, dtype=np.int64)
    order = state_order_by_id(keep)
    rank = np.empty(len(keep), dtype=np.int64)
    rank[order] = np.arange(len(order))
    # symbols that never occur as a state still need a rank; they are
    # never selected because they never appear as x_{t-1}
    if len(rank) < V:
        rank = np.concatenate([rank, np.full(V - len(rank), V, np.int64)])

    x2, x1, y = reduced[:-2], reduced[1:-1], reduced[2:]
    N = len(y)
    # Bits per CHARACTER of the original file is the unit used throughout
    # the paper -- but it is only defined for a COMPLETE stream.  On a
    # prefix the file's byte count and vocabulary charge do not apply, and
    # dividing a prefix's bits by the whole file's bytes gives a number
    # that is neither a rate nor comparable between prefixes.  Prefix runs
    # therefore report bits per token, which is comparable throughout.
    nbytes = None if is_prefix else meta.get("n_bytes")
    fixed = 0.0 if is_prefix else meta.get("fixed_bits", 0.0)
    per_char = bool(nbytes)
    unit = ("bits per character" if per_char else
            "bits per TOKEN (a prefix: bits per character needs the whole "
            "file's byte count and vocabulary charge)")

    def bpc(per_token: float) -> float:
        return (per_token * N + fixed) / nbytes if per_char else per_token

    print(f"{meta['representation']} / {meta['source_file']}: "
          f"V={V:,}, {N:,} positions with two predecessors")
    print(f"excess curve from {args.redundancy} "
          f"({red['states']:,} states, {red['excess_over_plugin_bits']:.4f} "
          f"bits/token excess at order one)\n")

    def forecast(m1: int, m2: int):
        s1 = sigma(rank[x1], m1)
        if m2 == 0:
            state = s1
        else:
            s2 = sigma(rank[x2], m2)
            state = s1 * (m2 + 1) + s2
        # densify
        _, state = np.unique(state, return_inverse=True)
        plug, sizes, support = entropy_and_sizes(state, y, V)
        mm = float((support - 1.0).sum()) / (2.0 * math.log(2.0))
        ex = excess_of(sizes)
        return {
            "M1": m1, "M2": m2, "states": int(len(sizes)),
            "obs_per_state": N / len(sizes),
            "plugin_bits_per_token": plug / N,
            "corrected_bits_per_token": (plug + mm) / N,
            "forecast_bits_per_token": (plug + ex) / N,
        }

    if args.family:
        fam = json.loads(Path(args.family).read_text())
        meas = fam["member_bits_per_token"]
        print(f"VALIDATION --- order one (M2 = 0), predicted against "
              f"measured, {unit}")
        print(f"{'M1':>8} {'states':>9} {'predicted':>10} {'measured':>10} "
              f"{'error':>8}")
        errs = []
        for k in sorted(meas, key=lambda x: int(x)):
            m1 = int(k)
            if m1 == 0:
                continue
            r = forecast(m1, 0)
            e = bpc(r["forecast_bits_per_token"]) - bpc(meas[k])
            errs.append(abs(e))
            print(f"{m1:>8} {r['states']:>9,} "
                  f"{bpc(r['forecast_bits_per_token']):>10.4f} "
                  f"{bpc(meas[k]):>10.4f} {e:>+8.4f}")
        print(f"  worst absolute error {max(errs):.4f}\n")

    print(f"FORECAST --- order two, {unit}")
    print(f"{'M1':>8} {'M2':>8} {'states':>12} {'obs/state':>10} "
          f"{'H(X|s)':>9} {'+MM':>9} {'forecast':>9}")
    rows = []
    for m1 in [int(x) for x in args.m1.split(",")]:
        for m2 in [int(x) for x in args.m2.split(",")]:
            r = forecast(m1, m2)
            for k in ("plugin", "corrected", "forecast"):
                r[k + "_bits_per_character"] = bpc(r[k + "_bits_per_token"])
            rows.append(r)
            print(f"{m1:>8} {m2:>8} {r['states']:>12,} "
                  f"{r['obs_per_state']:>10.1f} "
                  f"{r['plugin_bits_per_character']:>9.4f} "
                  f"{r['corrected_bits_per_character']:>9.4f} "
                  f"{r['forecast_bits_per_character']:>9.4f}", flush=True)

    best = min(rows, key=lambda r: r["forecast_bits_per_token"])
    print(f"\nbest forecast: M1={best['M1']}, M2={best['M2']} at "
          f"{best['forecast_bits_per_token']:.4f} bits per token"
          + (f" ({best['forecast_bits_per_character']:.4f} per character)"
             if per_char else ""))
    order1 = next((r for r in rows if r["M2"] == 0), None)
    if order1 is not None:
        delta = (best["forecast_bits_per_token"]
                 - order1["forecast_bits_per_token"])
        print(f"against order one (M2 = 0): order two "
              f"{'WINS' if delta < 0 else 'loses'} by {abs(delta):.4f} "
              f"bits per token"
              + (f", {abs(delta) * N / nbytes:.4f} per character"
                 if per_char else ""))
    print("H(X|s) is a hard floor for that map (no learning cost at all); "
          "the forecast adds the measured cost of learning.")
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "results.json").write_text(
            json.dumps({"stream": args.ids, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
