#!/usr/bin/env python3
"""Could a state just USE the marginal instead of learning its own?

The redundancy measurement says the first-order model's excess is the
cost of every state discovering, from scratch, which symbols follow it,
when the marginal already answers most of that.  Two constructions
could exploit it, and they are not the same thing:

  REPLACEMENT   each state chooses between its own layered model and
                the marginal, the choice paid for and mixed over.
                Cheap: one bit per state, no new mathematics.

  CENTRING      each state keeps its own layered model, but with the
                prior centred on the marginal instead of on the uniform
                distribution.  Expensive: it breaks exchangeability over
                symbols, so profiles no longer deduplicate by counts.

This script bounds what REPLACEMENT could possibly buy, so the decision
between them rests on a number rather than on an argument.  Per state it
computes what the marginal would charge for that state's successors,

    marginal_s  =  sum_y c_sy log2(1/m_y)  =  n_s (H_s + KL(p_s || m)),

takes the better of that and the model's actual cost, and sums.  That is
generous twice over: it lets every state pick with hindsight, and it
ignores that `m` itself would have to be paid for.  If even this
optimistic bound is small, replacement cannot be the answer and only
centring is left.

    python scripts/marginal_bound.py --ids output/streams/bpe_enwik8 \\
        --jobs 12 --out output/marginal_bound_bpe_enwik8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from product_model_with_memory.codelength import (
    default_l_max,
    depth_averaged_codelength_profiles,
)
from product_model_with_memory.streams import load_stream, reduce_ids

BUCKETS = [1, 2, 5, 10, 30, 100, 300, 1000, 10000, 100000, 10 ** 9]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", required=True)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--l-max", type=int, default=None)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    t0 = time.time()
    ids, meta = load_stream(args.ids)
    if args.n:
        ids = ids[: args.n]
    top_k = args.top_k or meta["alphabet"] - 1
    reduced, V, capped = reduce_ids(ids, top_k)
    reduced = np.asarray(reduced, dtype=np.int64)
    l_max = args.l_max or default_l_max(V)
    a, b = reduced[:-1], reduced[1:]
    N = len(a)

    # the marginal over successors, and its cost per symbol
    freq = np.bincount(b, minlength=V).astype(np.float64)
    m = freq / freq.sum()
    with np.errstate(divide="ignore"):
        cost_of = np.where(m > 0, -np.log2(np.maximum(m, 1e-300)), np.inf)

    uniq, cnt = np.unique(a * V + b, return_counts=True)
    owner, succ = uniq // V, uniq % V
    cuts = np.flatnonzero(np.diff(owner)) + 1
    groups = np.split(cnt, cuts)
    succ_groups = np.split(succ, cuts)
    print(f"{meta['representation']} / {meta['source_file']}: V={V:,}, "
          f"{len(groups):,} states, {N:,} coded positions "
          f"({time.time()-t0:.0f}s)", flush=True)

    profiles, n_s, marginal = [], [], []
    for g, sy in zip(groups, succ_groups):
        profiles.append(tuple(sorted((int(c) for c in g), reverse=True)))
        n_s.append(int(g.sum()))
        marginal.append(float((g * cost_of[sy]).sum()))
    n_s = np.asarray(n_s, dtype=np.float64)
    marginal = np.asarray(marginal)

    distinct = {pr: pr for pr in profiles}
    print(f"  {len(distinct):,} distinct profiles to evaluate", flush=True)

    def progress(event, _unused) -> None:
        kind, k, total = event
        if kind == "depth" and (k % 10 == 0 or k == total):
            print(f"  evaluation: depth {k}/{total} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    ev = depth_averaged_codelength_profiles(
        distinct, d=V, l_max=l_max, jobs=args.jobs, progress=progress)
    model = np.array([-ev[pr].log2_q_avg for pr in profiles])

    prefers = marginal < model
    best = np.minimum(model, marginal)
    charged = best.sum() + len(model)          # one bit per state
    bpt_scale = meta.get("bytes_per_token", 1.0)
    nbytes = meta.get("n_bytes") or N

    print(f"\n{'observations':>16} {'states':>9} {'tokens%':>8} "
          f"{'prefer marg.':>13} {'gain b/char':>12}")
    rows = []
    for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:]):
        sel = (n_s >= lo) & (n_s < hi)
        if not sel.any():
            continue
        gain = float((model[sel] - best[sel]).sum())
        row = {"n_from": lo, "n_to": hi, "states": int(sel.sum()),
               "token_share": float(n_s[sel].sum()) / N,
               "states_preferring_marginal": int(prefers[sel].sum()),
               "gain_bits": gain,
               "gain_bits_per_character": gain / nbytes}
        rows.append(row)
        print(f"{str(lo)+'..'+str(hi-1):>16} {row['states']:>9,} "
              f"{100*row['token_share']:>7.2f}% "
              f"{row['states_preferring_marginal']:>13,} "
              f"{row['gain_bits_per_character']:>12.5f}")

    total_gain = float((model - best).sum())
    out = {
        "representation": meta["representation"],
        "source_file": meta["source_file"],
        "states": len(model), "coded_positions": N,
        "model_bits_per_character":
            (float(model.sum()) + meta.get("fixed_bits", 0.0)) / nbytes,
        "states_preferring_marginal": int(prefers.sum()),
        "token_share_of_those": float(n_s[prefers].sum()) / N,
        "gain_bits_per_character_oracle": total_gain / nbytes,
        "gain_bits_per_character_after_one_bit_per_state":
            (float(model.sum()) - charged) / nbytes,
        "buckets": rows,
        "seconds": time.time() - t0,
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "results.json").write_text(json.dumps(out, indent=2))

    print(f"\nstates that would rather use the marginal: "
          f"{out['states_preferring_marginal']:,} of {len(model):,} "
          f"({100*out['token_share_of_those']:.2f}% of tokens)")
    print(f"best possible gain, choosing per state with hindsight: "
          f"{out['gain_bits_per_character_oracle']:.5f} bits/character")
    print(f"after paying one bit per state:                        "
          f"{out['gain_bits_per_character_after_one_bit_per_state']:.5f}")
    print(f"\nfor scale, the model pays "
          f"{out['model_bits_per_character']:.4f} bits/character and its "
          f"excess over the corrected conditional entropy is what this "
          f"would have to dent.")
    print(f"written: {Path(args.out)/'results.json'} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
