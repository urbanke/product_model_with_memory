#!/usr/bin/env python3
"""How often must the predictor be recomputed?  Three schemes compared.

For a state with running profile lam the predictor is
P(y) = q_avg(lam + y) / q_avg(lam).  How often lam is refreshed defines
a family of schemes, all of them valid codes:

  FINAL       the reported number, -log2 prod_s q_avg(lam_s), evaluated
              once at the final profiles.  ONE computation.

  SEQUENTIAL  lam refreshed at every position, which is what an encoder
              and decoder actually do.  n computations.  This splits in
              TWO, and the split is the whole point:

                POSTERIOR  P(y) = q_avg(lam+y)/q_avg(lam), which is the
                           depth predictors weighted by their posterior
                           given lam.  Telescopes; equals FINAL.

                UNIFORM    P(y) = (1/L) sum_L q^(L)(lam+y)/q^(L)(lam),
                           the depth predictors weighted equally at
                           every step.  Also sums to one over y, so also
                           a code -- but not a ratio of one function at
                           consecutive profiles, so nothing telescopes
                           and it does NOT equal FINAL.

  CHUNKED(K)  lam frozen at each of K chunk boundaries and reused for
              the whole chunk.  K computations, under either weighting.
              A decoder can run this --- it knows the boundaries and has
              decoded the prefix --- so it is a code, just a lazier one.

Why this comparison is worth running.  FINAL and SEQUENTIAL-POSTERIOR
must agree EXACTLY: within a state the running profiles form a chain,
each step's numerator is the next step's denominator, and the product
collapses to q_avg at the final profile.  Every codelength in the paper
rests on that, and here it is checked against a walk over real data
instead of being argued.  SEQUENTIAL-UNIFORM is a different code and the
gap to it is what uniform weighting costs.  CHUNKED(K) is lazier still,
and the sweep in K prices a practical refresh schedule -- approaching
the corresponding sequential scheme as K grows, degrading as K falls.

Cost is dominated by SEQUENTIAL, which needs q_avg at every running
profile: n + S evaluations, deduplicated heavily because most profiles
are tiny.  Use --n to choose a prefix you can afford.

    python scripts/refresh_schedule.py --ids output/streams/bpe_text8 \\
        --n 200000 --chunks 1,2,8,32,128 --jobs 12 \\
        --out output/refresh_text8
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

# MUST precede the package import.  Level truncation drops depths whose
# contribution to q_avg is negligible and reports -inf for them.  That
# is harmless for the posterior scheme, which only ever uses q_avg, but
# fatal for the uniform scheme, which needs EVERY depth's own predictor:
# the difference of two -inf entries is NaN, and the whole comparison
# silently becomes NaN.  (It did, on the first run of this script.)
os.environ["PMM_NO_TRUNCATE"] = "1"

import numpy as np                                          # noqa: E402

from product_model_with_memory.codelength import (           # noqa: E402
    default_l_max,
    depth_averaged_codelength_profiles,
    profile_of,
)
from product_model_with_memory.streams import (                # noqa: E402
    load_stream,
    reduce_ids,
)


def _augment(profile, c: int):
    """`profile` with one part of size c raised to c+1; c = 0 adds a new
    part of size 1.  By exchangeability the identity of the symbol is
    irrelevant --- only how often it had already been seen."""

    p = list(profile)
    if c > 0:
        p.remove(c)
    return tuple(sorted(p + [c + 1], reverse=True))


def _sequential_pairs(reduced):
    """(lam, lam+y) at every coded position, refreshing every step."""

    counters: dict[int, Counter] = {}
    pairs = []
    for t in range(1, len(reduced)):
        s, y = int(reduced[t - 1]), int(reduced[t])
        c = counters.setdefault(s, Counter())
        before = profile_of(c)
        c[y] += 1
        pairs.append((before, profile_of(c)))
    return pairs, counters


def _chunked_pairs(reduced, K):
    """(lam_frozen, lam_frozen + y) at every coded position, with lam
    refreshed only at the K chunk boundaries."""

    n_coded = len(reduced) - 1
    width = max(1, n_coded // K)
    live: dict[int, Counter] = {}
    snap: dict[int, Counter] = {}
    pairs = []
    for i, t in enumerate(range(1, len(reduced))):
        if i % width == 0:                       # boundary: refreeze
            snap = {s: c.copy() for s, c in live.items()}
        s, y = int(reduced[t - 1]), int(reduced[t])
        fc = snap.get(s)
        if fc is None:
            pairs.append(((), (1,)))             # state unseen at the freeze
        else:
            pairs.append((profile_of(fc), _augment(profile_of(fc), fc.get(y, 0))))
        live.setdefault(s, Counter())[y] += 1
    return pairs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", required=True)
    p.add_argument("--n", type=int, default=200000)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--l-max", type=int, default=None)
    p.add_argument("--chunks", default="",
                   help="comma-separated K values for the frozen-predictor "
                        "schemes; EMPTY (the default) runs only FINAL and "
                        "the two sequential schemes, which is the decisive "
                        "comparison and much the cheaper one.  Chunking is "
                        "only worth pricing if UNIFORM turns out to beat "
                        "POSTERIOR, since it is a cheap route to uniform "
                        "and posterior is already cheap")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    t0 = time.time()
    ids, meta = load_stream(args.ids)
    ids = ids[: args.n]
    top_k = args.top_k or meta["alphabet"] - 1
    reduced, V, _ = reduce_ids(ids, top_k)
    reduced = np.asarray(reduced, dtype=np.int64)
    l_max = args.l_max or default_l_max(V)
    Ks = sorted({int(x) for x in args.chunks.split(",") if x.strip()})
    n_coded = len(reduced) - 1
    print(f"{meta['representation']} / {meta['source_file']}: V={V:,}, "
          f"l_max={l_max}, prefix {len(reduced):,} tokens, "
          f"{n_coded:,} coded positions", flush=True)

    seq_pairs, counters = _sequential_pairs(reduced)
    chunk_pairs = {K: _chunked_pairs(reduced, K) for K in Ks}
    print(f"{len(counters):,} states; pair lists built "
          f"({time.time()-t0:.0f}s)", flush=True)

    need = {pr for a, b in seq_pairs for pr in (a, b)}
    for K in Ks:
        need.update(pr for a, b in chunk_pairs[K] for pr in (a, b))
    need.update(profile_of(c) for c in counters.values())
    need.discard(())
    print(f"{len(need):,} distinct profiles to evaluate "
          f"({time.time()-t0:.0f}s)", flush=True)

    def progress(event, _u):
        kind, k, total = event
        if kind == "depth" and (k % 10 == 0 or k == total):
            print(f"  evaluation: depth {k}/{total} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    ev = depth_averaged_codelength_profiles(
        {pr: pr for pr in need}, d=V, l_max=l_max, jobs=args.jobs,
        progress=progress)
    lq = {pr: ev[pr].log2_q_avg for pr in need}
    lq[()] = 0.0
    # per-depth values, needed for the uniform weighting: it mixes the
    # PREDICTORS q^(L)(b)/q^(L)(a), not the averaged q
    zero = np.zeros(l_max)
    bd = {pr: np.asarray(ev[pr].log2_q_by_depth) for pr in need}
    bd[()] = zero
    bad = [pr for pr in need if not np.all(np.isfinite(bd[pr]))]
    if bad:
        raise SystemExit(
            f"{len(bad):,} profiles have a non-finite per-depth value "
            f"(first: {bad[0]}).  The uniform scheme needs every depth, "
            "so this would produce NaN rather than a wrong number -- but "
            "check that PMM_NO_TRUNCATE=1 took effect before the package "
            "was imported.")
    log_L = float(np.log2(l_max))

    def uniform_bits(pairs):
        tot = 0.0
        for a, b in pairs:
            r = bd[b] - bd[a]                    # log2 of each depth's P
            m = float(np.max(r))
            tot -= m + float(np.log2(np.sum(np.exp2(r - m)))) - log_L
        return tot

    rows = {
        "final": -sum(lq[profile_of(c)] for c in counters.values()),
        "sequential_posterior": -sum(lq[b] - lq[a] for a, b in seq_pairs),
        "sequential_uniform": uniform_bits(seq_pairs),
    }
    for K in Ks:
        rows[f"chunked_{K}_posterior"] = -sum(lq[b] - lq[a]
                                              for a, b in chunk_pairs[K])
        rows[f"chunked_{K}_uniform"] = uniform_bits(chunk_pairs[K])

    bpt = meta.get("bytes_per_token", 1.0)
    base = rows["final"]
    print(f"\n{'scheme':>28} {'recomputations':>15} {'bits/char':>11} "
          f"{'vs FINAL':>12}")
    order = [("final", "1"),
             ("sequential_posterior", f"{n_coded:,}"),
             ("sequential_uniform", f"{n_coded:,}")]
    for K in Ks:
        order.append((f"chunked_{K}_posterior", str(K)))
        order.append((f"chunked_{K}_uniform", str(K)))
    out_rows = {}
    for key, howmany in order:
        v = rows[key]
        out_rows[key] = {"bits": v,
                         "bits_per_character": v / n_coded / bpt,
                         "excess_bits_per_character": (v - base) / n_coded / bpt}
        print(f"{key:>28} {howmany:>15} {v/n_coded/bpt:>11.4f} "
              f"{(v-base)/n_coded/bpt:>+12.6f}")

    gap = abs(rows["final"] - rows["sequential_posterior"])
    unif = rows["sequential_uniform"] - rows["final"]
    better = "POSTERIOR" if unif > 0 else "UNIFORM"
    print(f"\n=> {better} weighting is the better code here.")
    if unif > 0:
        print("   Chunking is therefore moot: it is a cheap route to the")
        print("   uniform scheme, and the posterior scheme is both cheaper")
        print("   (one evaluation at the final profiles) and better.")
    else:
        print("   Chunking is worth pricing: it approximates the uniform")
        print("   scheme at K evaluations instead of n.")
    print(f"\nFINAL against SEQUENTIAL-POSTERIOR: {gap:.3e} bits over the "
          f"prefix ({gap/n_coded/bpt:.2e} per character).")
    print("These must agree exactly -- the telescoping identity every")
    print("reported codelength rests on, checked on real data.")
    print(f"\nSEQUENTIAL-UNIFORM costs {unif/n_coded/bpt:+.6f} bits per "
          f"character more.\nIt is a different code, and it does not "
          f"telescope, so it cannot be\nobtained from the final profiles "
          f"at any price.")

    payload = {"stream": args.ids, "prefix_tokens": int(len(reduced)),
               "V": V, "l_max": l_max, "states": len(counters),
               "coded_positions": n_coded, "chunk_counts": Ks,
               "schemes": out_rows, "final_vs_sequential_bits": gap,
               "seconds": time.time() - t0}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"written: {Path(args.out)/'results.json'} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
