#!/usr/bin/env python3
"""Impact probe: does the large-count table error affect real results?

Takes real heavy profiles from text8 (the memoryless profile and
first-order rows at V = 256), evaluates each with the PRODUCTION scan
algorithm twice --- once on recursion-built tables (side A, the
production path), once on the same tables with all columns r >= R_SWITCH
replaced by certified Mellin columns (side B) --- and reports the
difference in bits, per profile and per level.  Same profiles, same
scan, same grid: the difference isolates the table error's effect.

    python scripts/impact_probe.py --corpus data/text8 --out output/impact_probe
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import loggamma, polygamma

from product_model_with_memory.codelength import default_l_max, needed_r_values
from product_model_with_memory.corpus import load_tokens
from product_model_with_memory.fast_tables import default_grid_spec, grid_from_spec
from product_model_with_memory.layered import (
    ProductMomentTables,
    build_selected_product_moment_tables,
    log_q_lambda_scan,
    log_q_lambda_closed_l1,
)
from product_model_with_memory.pairs import reduce_vocabulary

R_SWITCH = 500


def mellin_columns_batch(r_values, L, u_grid):
    """Certified columns for many r at one level: series where certified,
    order-2 saddle elsewhere (vectorized bisection over the (r, u) grid)."""

    R = np.asarray(r_values, dtype=np.float64)[:, None]
    U = np.asarray(u_grid, dtype=np.float64)[None, :]
    out = np.empty((len(r_values), len(u_grid)))

    tau_log = U + L * (loggamma(R + 2.0) - loggamma(R + 1.0))
    series = tau_log < math.log(0.05)

    # ---- series region
    lead = L * loggamma(R + 1.0)
    total = np.ones_like(out)
    sign = 1.0
    prev = None
    for j in range(1, 60):
        term_log = (
            j * U
            - float(loggamma(j + 1.0))
            + L * (loggamma(R + j + 1.0) - loggamma(R + 1.0))
        )
        if prev is not None:
            term_log = np.where(term_log >= prev, -np.inf, term_log)
        sign = -sign
        total = total + sign * np.exp(np.where(series, term_log, -np.inf))
        prev = term_log
    out = lead + np.log(np.maximum(total, 1e-300))

    # ---- saddle region (vectorized bisection; F' increasing in z)
    lo = np.full_like(out, 1e-12)
    hi = (R + 1.0) * (1.0 - 1e-12) * np.ones_like(out)
    for _ in range(55):
        mid = 0.5 * (lo + hi)
        fp = polygamma(0, mid) - U - L * polygamma(0, R + 1.0 - mid)
        lo = np.where(fp < 0.0, mid, lo)
        hi = np.where(fp < 0.0, hi, mid)
    z = 0.5 * (lo + hi)
    F = np.real(loggamma(z)) - z * U + L * np.real(loggamma(R + 1.0 - z))
    F2 = polygamma(1, z) + L * polygamma(1, R + 1.0 - z)
    F3 = polygamma(2, z) - L * polygamma(2, R + 1.0 - z)
    F4 = polygamma(3, z) + L * polygamma(3, R + 1.0 - z)
    val = F - 0.5 * np.log(2.0 * math.pi * F2)
    corr = 1.0 + F4 / (8.0 * F2 * F2) - 5.0 * F3 * F3 / (24.0 * F2**3)
    val = val + np.where(corr > 0, np.log(np.maximum(corr, 1e-300)), 0.0)
    return np.where(series, out, val)


def q_avg_bits(partition, d, l_max, tables):
    """-log2 q_avg via the production scan, plus the per-level values."""

    per_level = []
    for L in range(1, l_max + 1):
        if L == 1:
            res = log_q_lambda_closed_l1(d=d, partition=partition)
        else:
            res = log_q_lambda_scan(d=d, L=L, partition=partition, tables=tables)
        per_level.append(res.log2_q)
    arr = np.array(per_level)
    m = arr.max()
    avg = m + math.log2(np.exp2(arr - m).sum()) - math.log2(l_max)
    return -avg, per_level


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--top-k", type=int, default=255)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    t0 = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens = load_tokens(args.corpus)
    reduced, vocab = reduce_vocabulary(tokens, args.top_k)
    d = len(vocab)
    l_max = default_l_max(d)
    n = len(reduced)

    # heavy profiles: memoryless; row of the most frequent state; a
    # mid-frequency row
    uni = Counter(reduced)
    p_uni = tuple(sorted(uni.values()))
    by_freq = [w for w, _ in uni.most_common()]
    rows = {w: Counter() for w in (by_freq[0], by_freq[99])}
    for a, b in zip(reduced[:-1], reduced[1:]):
        if a in rows:
            rows[a][b] += 1
    profiles = {
        "memoryless": p_uni,
        f"row:{by_freq[0]}": tuple(sorted(rows[by_freq[0]].values())),
        f"row:{by_freq[99]}": tuple(sorted(rows[by_freq[99]].values())),
    }
    r_all = sorted(set().union(*[needed_r_values(p) for p in profiles.values()]))
    r_big = [r for r in r_all if r >= R_SWITCH]
    print(f"d={d} l_max={l_max} n={n:,}; profiles: "
          f"{ {k: len(v) for k, v in profiles.items()} }; "
          f"|R|={len(r_all)} of which {len(r_big)} >= {R_SWITCH} "
          f"({time.time()-t0:.0f}s)", flush=True)

    spec = default_grid_spec(max_L=l_max, r_max=max(r_all), u_max=35.0)
    u_grid = grid_from_spec(spec)
    print(f"grid: {len(u_grid)} points [{u_grid[0]:.0f}, {u_grid[-1]:.0f}]",
          flush=True)

    tables_a = build_selected_product_moment_tables(
        max_L=l_max, r_values=r_all, u_grid=u_grid
    )
    print(f"side A (recursion) built ({time.time()-t0:.0f}s)", flush=True)

    # evaluate side A for all profiles FIRST, then mutate the columns
    # in place --- keeping only one table set in memory
    side_a = {}
    for name, prof in profiles.items():
        side_a[name] = q_avg_bits(prof, d, l_max, tables_a)
        print(f"A {name}: {side_a[name][0]:.3f} bits "
              f"({time.time()-t0:.0f}s)", flush=True)

    for L in range(2, l_max + 1):
        cols = mellin_columns_batch(r_big, L, u_grid)
        for i, r in enumerate(r_big):
            tables_a.log_phi[(L, r)] = cols[i]
        if L % 5 == 0:
            print(f"side B columns: L={L}/{l_max} ({time.time()-t0:.0f}s)",
                  flush=True)
    tables_b = tables_a  # columns replaced in place
    print(f"side B (mellin >= {R_SWITCH}) ready ({time.time()-t0:.0f}s)",
          flush=True)

    report = {}
    for name, prof in profiles.items():
        bits_a, lv_a = side_a[name]
        bits_b, lv_b = q_avg_bits(prof, d, l_max, tables_b)
        lv_delta = [b - a for a, b in zip(lv_a, lv_b)]
        report[name] = {
            "N": int(sum(prof)),
            "k": len(prof),
            "bits_A_production": bits_a,
            "bits_B_corrected": bits_b,
            "delta_bits": bits_b - bits_a,
            "delta_bits_per_token_if_whole_model": (bits_b - bits_a) / n,
            "per_level_delta_log2q": lv_delta,
        }
        print(f"{name}: A={bits_a:.3f}  B={bits_b:.3f}  "
              f"delta={bits_b - bits_a:+.4f} bits "
              f"({(bits_b-bits_a)/n:+.3e} bits/token) "
              f"({time.time()-t0:.0f}s)", flush=True)

    (out_dir / "results.json").write_text(json.dumps({
        "d": d, "l_max": l_max, "n_tokens": n, "r_switch": R_SWITCH,
        "grid_points": len(u_grid), "profiles": report,
        "seconds": time.time() - t0,
    }, indent=2))
    print(f"written: {out_dir/'results.json'}")


if __name__ == "__main__":
    main()
