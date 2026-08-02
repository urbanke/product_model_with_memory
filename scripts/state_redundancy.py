#!/usr/bin/env python3
"""Where does the first-order model's excess over the plug-in entropy go?

The family sweep settled one thing and opened another.  Coarsening the
state never pays on the enwik8 subword stream --- the codelength falls
monotonically in M, right up to the full first-order model --- so the
shortfall against `H(next | prev)` is not "too many states".  But the
shortfall is large: 7.9534 bits per token against a plug-in conditional
entropy of 6.385.  This says how that 1.57 is composed.

Per state s, with n_s observations and empirical successor distribution
p_s, it reports three quantities:

    model_s     = -log2 q_avg(profile of s), what the code actually pays
    plugin_s    = n_s * H(p_s), what a decoder that already KNEW p_s
                  would pay
    redundancy  = model_s - plugin_s

summed and also bucketed by n_s, because the question is whether the
excess sits in the many nearly-empty states or in the few heavy ones.
Those two possibilities call for completely different constructions: a
predictor that shares strength between states, or a better estimator
for a single crowded one.

The plug-in is optimistic, and by an amount that grows with the number
of states, so it is reported both raw and with the Miller-Madow
correction (support_s - 1) / (2 n_s) nats per state --- otherwise a
comparison across files with different state counts is not a comparison
at all.

    python scripts/state_redundancy.py --ids output/streams/bpe_enwik8 \
        --jobs 12 --out output/redundancy_bpe_enwik8
"""

from __future__ import annotations

import argparse
import json
import math
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

    # (state, successor) counts, grouped by state
    uniq, cnt = np.unique(a * V + b, return_counts=True)
    states = uniq // V
    cuts = np.flatnonzero(np.diff(states)) + 1
    groups = np.split(cnt, cuts)
    heads = states[np.concatenate(([0], cuts))] if len(states) else states
    print(f"{meta['representation']} / {meta['source_file']}: "
          f"V={V:,}, l_max={l_max}, {len(groups):,} states, "
          f"{N:,} coded positions ({time.time()-t0:.0f}s)", flush=True)

    profiles, n_s, support = [], [], []
    plugin_bits = 0.0
    for g in groups:
        pr = tuple(sorted((int(c) for c in g), reverse=True))
        profiles.append(pr)
        tot = g.sum()
        n_s.append(int(tot))
        support.append(len(g))
        q = g / tot
        plugin_bits += float(tot) * float(-(q * np.log2(q)).sum())
    n_s = np.asarray(n_s, dtype=np.float64)
    support = np.asarray(support, dtype=np.float64)

    distinct = {pr: pr for pr in profiles}
    print(f"  {len(distinct):,} distinct profiles to evaluate", flush=True)

    def progress(event, _unused) -> None:
        kind, k, total = event
        if kind == "depth" and (k % 5 == 0 or k == total):
            print(f"  evaluation: depth {k}/{total} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    ev = depth_averaged_codelength_profiles(
        distinct, d=V, l_max=l_max, jobs=args.jobs, progress=progress)
    model = np.array([-ev[pr].log2_q_avg for pr in profiles])

    plugin = np.array([
        float(n) * float(-(np.asarray(g) / n * np.log2(np.asarray(g) / n)).sum())
        for g, n in zip(groups, n_s)])
    red = model - plugin
    # Miller-Madow: the plug-in entropy is low by (support - 1) / (2 n_s)
    # nats per state, so the honest target is larger by this much
    mm = (support - 1.0) / (2.0 * math.log(2.0))

    # Everything is reported in bits per CHARACTER of the original file,
    # the unit the paper uses throughout, so that figures are comparable
    # across representations.  Per-symbol quantities are labelled where
    # they appear.
    bpt = meta.get("bytes_per_token", 1.0)
    nbytes = meta.get("n_bytes") or N
    fixed = meta.get("fixed_bits", 0.0)
    print(f"\n{'observations':>16} {'states':>8} {'tokens':>12} "
          f"{'model':>9} {'plug-in':>9} {'excess':>9} {'share':>7}"
          f"   (bits per character within the bucket)")
    rows = []
    for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:]):
        sel = (n_s >= lo) & (n_s < hi)
        if not sel.any():
            continue
        tok = float(n_s[sel].sum())
        row = {
            "n_from": lo, "n_to": hi, "states": int(sel.sum()),
            "tokens": tok, "token_share": tok / N,
            "model_bits": float(model[sel].sum()),
            "plugin_bits": float(plugin[sel].sum()),
            "miller_madow_bits": float(mm[sel].sum()),
            "excess_bits": float(red[sel].sum()),
            "excess_per_token_in_bucket": float(red[sel].sum()) / tok,
            "excess_share_of_total": float(red[sel].sum()) / float(red.sum()),
        }
        rows.append(row)
        print(f"{str(lo)+'..'+str(hi-1):>16} {row['states']:>8,} "
              f"{int(tok):>12,} {row['model_bits']/tok/bpt:>9.3f} "
              f"{row['plugin_bits']/tok/bpt:>9.3f} "
              f"{row['excess_per_token_in_bucket']/bpt:>9.3f} "
              f"{100*row['excess_share_of_total']:>6.1f}%")

    out = {
        "representation": meta["representation"],
        "source_file": meta["source_file"],
        "V": V, "l_max": l_max, "states": len(groups), "coded_positions": N,
        "model_bits_per_token": float(model.sum()) / N,
        "plugin_conditional_entropy_bits": float(plugin.sum()) / N,
        "miller_madow_correction_bits": float(mm.sum()) / N,
        "corrected_conditional_entropy_bits":
            float(plugin.sum() + mm.sum()) / N,
        "excess_over_plugin_bits": float(red.sum()) / N,
        "excess_over_corrected_bits": float(red.sum() - mm.sum()) / N,
        "buckets": rows,
        "seconds": time.time() - t0,
    }

    def to_bpc(per_token: float, *, absolute: bool = True) -> float:
        """bits per token -> bits per character.  `absolute` adds the
        vocabulary charge; differences do not carry it, since it is
        common to both terms and cancels."""

        return (per_token * N + (fixed if absolute else 0.0)) / nbytes

    out["model_bits_per_character"] = to_bpc(out["model_bits_per_token"])
    out["corrected_conditional_entropy_bits_per_character"] = to_bpc(
        out["corrected_conditional_entropy_bits"])
    out["excess_over_corrected_bits_per_character"] = to_bpc(
        out["excess_over_corrected_bits"], absolute=False)
    out["miller_madow_correction_bits_per_character"] = to_bpc(
        out["miller_madow_correction_bits"], absolute=False)

    print(f"\n{'':30}{'bits/char':>11}{'bits/token':>12}")
    print(f"model                         "
          f"{out['model_bits_per_character']:>11.4f}"
          f"{out['model_bits_per_token']:>12.4f}")
    print(f"corrected H(next|prev)        "
          f"{out['corrected_conditional_entropy_bits_per_character']:>11.4f}"
          f"{out['corrected_conditional_entropy_bits']:>12.4f}")
    print(f"  of which Miller-Madow       "
          f"{out['miller_madow_correction_bits_per_character']:>11.4f}"
          f"{out['miller_madow_correction_bits']:>12.4f}")
    print(f"excess over corrected         "
          f"{out['excess_over_corrected_bits_per_character']:>11.4f}"
          f"{out['excess_over_corrected_bits']:>12.4f}"
          f"   <-- the cost of learning")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwritten: {Path(args.out)/'results.json'} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
