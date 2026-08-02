#!/usr/bin/env python3
"""Instrumentation for T2(1): where the time goes, and where the peaks are.

Two measurements over the SAME real profiles, both using the production
evaluation path, nothing re-implemented:

PHASE 1 --- the time split of the atom.  For every (profile, level) it
separates provisioning (getting the level's log-phi matrix), integrand
assembly (the O(G k) sum that builds psi on the whole grid), peak
location (the O(G) local-maximum scan, which is the ONLY reason the
assembly exists), and refinement (the bracketed solves, curvature and
Laplace contributions, all of which are pointwise and need no grid).
This is the gate: if assembly plus location is a small share of the
total, eliminating the grid cannot pay and we should truncate the sum
over levels instead.

PHASE 2 --- the peak atlas.  For every (profile, level) it records every
peak the scan found, with its location and its contribution, and for a
base profile with its one-observation augmentations it records the same
for each family member.  From these it reports what the continuation
design turns on: how far the dominant peak moves from one level to the
next, how far it moves under an augmentation, how often a second peak is
significant and where it sits, whether peaks ever appear or vanish
discontinuously in L, and whether log q_L is log-concave in L (which
decides whether a level-truncation tail bound can be made rigorous).

    python scripts/peak_atlas.py --corpus data/text8 --top-k 255 \
        --rows 0,4,9,24,49,99 --out output/atlas --jobs 12

    python scripts/peak_atlas.py --ids output/streams/bpe_enwik8 \
        --top-k 100276 --rows 0,9,99,999,9999 --out output/atlas_bpe \
        --jobs 12

Use --limit to validate the script on a corpus prefix; the geometry of
interest needs the full file, because the second peak is a heavy-count
phenomenon.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from product_model_with_memory.codelength import (
    _provision_level_rows,
    _provision_tables,
    _resolve_tables_source,
    default_l_max,
    needed_r_values,
)
from product_model_with_memory.corpus import load_tokens
from product_model_with_memory.streams import load_stream, reduce_ids
from product_model_with_memory.layered import (
    ProductMomentTables,
    _parts_cache,
    _scan_from_psi,
    augmented_partition,
    partition_multiplicities,
)
from product_model_with_memory.pairs import reduce_vocabulary

SIGNIFICANCE_GAP = 40.0


# --------------------------------------------------------------- profiles
def build_profiles_ids(ids_dir, top_k, ranks, limit):
    """The same profiles from a saved stream, so the split can be
    measured on the representation the paper actually uses rather than
    on a byte corpus.  `ranks` index the states by frequency, so rank 0
    is the most frequent previous symbol."""

    ids, meta = load_stream(ids_dir)
    if limit:
        ids = ids[:limit]
    reduced, d, capped = reduce_ids(ids, top_k)
    reduced = np.asarray(reduced, dtype=np.int64)
    if capped:
        print(f"  WARNING: {capped:,} positions fall outside the top "
              f"{top_k} and are coded as <unk>", flush=True)
    uni = np.bincount(reduced, minlength=d)
    by_freq = np.argsort(-uni, kind="stable")
    by_freq = by_freq[uni[by_freq] > 0]
    a, b = reduced[:-1], reduced[1:]
    profiles = {"memoryless":
                tuple(sorted((int(c) for c in uni if c > 0), reverse=True))}
    for r in ranks:
        if r >= len(by_freq):
            continue
        succ = b[a == by_freq[r]]
        if succ.size < 2:
            continue
        counts = np.bincount(succ)
        pr = tuple(sorted((int(c) for c in counts if c > 0), reverse=True))
        if sum(pr) >= 2:
            profiles[f"row{r}"] = pr
    return d, len(reduced), profiles


def build_profiles(corpus, top_k, ranks, limit):
    tokens = load_tokens(corpus)
    if limit:
        tokens = tokens[:limit]
    reduced, vocab = reduce_vocabulary(tokens, top_k)
    d = len(vocab)
    uni = Counter(reduced)
    by_freq = [w for w, _ in uni.most_common()]
    wanted = {by_freq[r] for r in ranks if r < len(by_freq)}
    rows = {w: Counter() for w in wanted}
    for a, b in zip(reduced[:-1], reduced[1:]):
        c = rows.get(a)
        if c is not None:
            c[b] += 1
    profiles = {"memoryless": tuple(sorted(uni.values(), reverse=True))}
    for r in ranks:
        if r < len(by_freq):
            p = tuple(sorted(rows[by_freq[r]].values(), reverse=True))
            if sum(p) >= 2:
                profiles[f"row{r}"] = p
    return d, len(reduced), profiles


# ------------------------------------------------------------ peak record
def significant(peaks, gap=SIGNIFICANCE_GAP):
    """Peaks whose contribution is within `gap` nats of the largest.

    peaks[0] is always the left region (analytic tail, or the refined
    left saddle when one was found there)."""

    if not peaks:
        return []
    top = max(c for _, c in peaks)
    return [(float(u), float(c)) for u, c in peaks if c >= top - gap]


def dominant(peaks):
    return max(peaks, key=lambda uc: uc[1])[0] if peaks else float("nan")


# ------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default=None)
    p.add_argument("--ids", default=None,
                   help="a saved stream directory (scripts/make_stream.py); "
                        "give exactly one of --corpus or --ids")
    p.add_argument("--top-k", type=int, default=255)
    p.add_argument("--rows", default="0,4,9,24,49,99")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=1,
                   help="parallelism for COLUMN BUILDING only; the "
                        "measurement itself is serial on purpose")
    p.add_argument("--family-cs", type=int, default=0,
                   help="augmentation counts per profile; 0 = ALL distinct "
                        "counts, which is what a real prediction table (T3) "
                        "needs and the only setting whose timing is "
                        "representative")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ranks = [int(x) for x in args.rows.split(",") if x.strip() != ""]

    if (args.corpus is None) == (args.ids is None):
        raise SystemExit("give exactly one of --corpus or --ids")
    if args.ids:
        d, n_tokens, profiles = build_profiles_ids(
            args.ids, args.top_k, ranks, args.limit)
    else:
        d, n_tokens, profiles = build_profiles(
            args.corpus, args.top_k, ranks, args.limit)
    l_max = default_l_max(d)
    print(f"d={d} l_max={l_max} tokens={n_tokens:,}; "
          f"{len(profiles)} profiles ({time.time()-t0:.0f}s)", flush=True)
    for k, v in profiles.items():
        print(f"  {k:12s} N={sum(v):>10,} k={len(v):>5} "
              f"ktilde={len(set(v)):>4} max={max(v):>9,}")

    # the augmentation counts are fixed here, so that the columns their
    # profiles need are provisioned along with the bases'
    fam_cs = {name: sorted({0} | (set(pr) if args.family_cs == 0
                                  else set(sorted(set(pr))[:args.family_cs])))
              for name, pr in profiles.items()}
    r_need: set[int] = set()
    for name, pr in profiles.items():
        r_need |= set(needed_r_values(pr))
        for c in fam_cs[name]:
            r_need |= set(needed_r_values(augmented_partition(pr, c)))
            r_need |= {c, c + 1, c + 2, c + 3}
    all_r = sorted(r_need)
    prov = _provision_tables(
        _resolve_tables_source(None), l_max, all_r, None, None,
        args.jobs, 96,
        lambda evt, _: print(f"  building columns {evt[1]}/{evt[2]} "
                             f"({time.time()-t0:.0f}s)", flush=True)
        if evt[0] == "tables" and (evt[1] % 500 == 0 or evt[1] == evt[2])
        else None)
    u_grid = np.asarray(prov["u_grid"], dtype=np.float64)
    G = len(u_grid)
    print(f"grid: {G:,} points on [{u_grid[0]:.1f}, {u_grid[-1]:.1f}], "
          f"|R|={len(all_r)} ({time.time()-t0:.0f}s)", flush=True)

    timing = {"provision": 0.0, "assemble": 0.0, "locate": 0.0,
              "family_update": 0.0, "refine_base": 0.0, "refine_family": 0.0}
    counts = {"profile_levels": 0, "family_members": 0}
    atlas: dict[str, dict] = {}
    family: dict[str, dict] = {}

    for L in range(2, l_max + 1):
        t = time.perf_counter()
        M = _provision_level_rows(prov, L, all_r)
        tables = ProductMomentTables.from_matrix(
            max_L=L, L=L, r_values=all_r, u_grid=u_grid, matrix=M)
        timing["provision"] += time.perf_counter() - t

        for name, prof in profiles.items():
            N, s = sum(prof), len(prof)
            mult = partition_multiplicities(prof)

            # ---- assembly: the O(G ktilde) sum
            t = time.perf_counter()
            psi = N * u_grid + (d - s) * tables.log_phi[(L, 0)]
            left_c = 0.0
            for part, count in mult:
                psi = psi + count * tables.log_phi[(L, part)]
                left_c += count * L * math.lgamma(part + 1)
            timing["assemble"] += time.perf_counter() - t

            # ---- location: the O(G) reason the assembly exists
            t = time.perf_counter()
            finite = np.isfinite(psi)
            top = float(np.max(psi[finite])) if np.any(finite) else -math.inf
            interior = np.zeros(G, dtype=bool)
            interior[1:-1] = ((psi[1:-1] >= psi[:-2])
                              & (psi[1:-1] >= psi[2:])
                              & np.isfinite(psi[1:-1]))
            n_cand = int(np.count_nonzero(
                interior & (psi >= top - SIGNIFICANCE_GAP)))
            timing["locate"] += time.perf_counter() - t

            # ---- refinement: everything that is already pointwise
            cache = _parts_cache(L, mult, tables)
            t = time.perf_counter()
            res = _scan_from_psi(
                d=d, L=L, N=N, s=s, partition=prof, multiplicity_pairs=mult,
                tables=tables, psi_grid=psi, left_constant=left_c,
                significance_gap=SIGNIFICANCE_GAP, cache=cache)
            timing["refine_base"] += time.perf_counter() - t
            counts["profile_levels"] += 1

            rec = atlas.setdefault(name, {})
            rec[L] = {
                "log2_q": res.log2_q,
                "peaks": significant(res.peaks),
                "n_peaks_all": len(res.peaks),
                "n_grid_candidates": n_cand,
                "right_gap": res.right_gap,
                "converged": bool(res.converged),
                "left_refined": "left-region saddle" in (res.message or ""),
            }

            # ---- family drift: augmentations of this profile
            cs = fam_cs[name]
            fam = family.setdefault(name, {}).setdefault(L, {})
            base_mult = dict(mult)
            for c in cs:
                if c != 0 and c not in base_mult:
                    continue
                t = time.perf_counter()
                psi_a = (psi + u_grid
                         + tables.log_phi[(L, c + 1)] - tables.log_phi[(L, c)])
                timing["family_update"] += time.perf_counter() - t
                t = time.perf_counter()
                fin = np.isfinite(psi_a)
                tp = float(np.max(psi_a[fin])) if np.any(fin) else -math.inf
                ii = np.zeros(G, dtype=bool)
                ii[1:-1] = ((psi_a[1:-1] >= psi_a[:-2])
                            & (psi_a[1:-1] >= psi_a[2:])
                            & np.isfinite(psi_a[1:-1]))
                _ = int(np.count_nonzero(ii & (psi_a >= tp - SIGNIFICANCE_GAP)))
                timing["locate"] += time.perf_counter() - t
                m = dict(base_mult)
                if c > 0:
                    m[c] -= 1
                    if m[c] == 0:
                        del m[c]
                m[c + 1] = m.get(c + 1, 0) + 1
                am = tuple(sorted(m.items()))
                aug = augmented_partition(prof, c)
                t = time.perf_counter()
                ra = _scan_from_psi(
                    d=d, L=L, N=N + 1, s=len(aug), partition=aug,
                    multiplicity_pairs=am, tables=tables, psi_grid=psi_a,
                    left_constant=left_c + L * (math.lgamma(c + 2.0)
                                                - math.lgamma(c + 1.0)),
                    significance_gap=SIGNIFICANCE_GAP,
                    cache=_parts_cache(L, am, tables))
                timing["refine_family"] += time.perf_counter() - t
                counts["family_members"] += 1
                fam[c] = {"peaks": significant(ra.peaks),
                          "log2_q": ra.log2_q}
        print(f"  L={L}/{l_max} done ({time.time()-t0:.0f}s)", flush=True)

    analysis = analyse(atlas, family, u_grid)
    payload = {
        "corpus": args.corpus, "d": d, "l_max": l_max,
        "tokens": n_tokens, "grid_points": G,
        "grid_spacing": float(u_grid[1] - u_grid[0]),
        "profiles": {k: {"N": sum(v), "k": len(v), "ktilde": len(set(v))}
                     for k, v in profiles.items()},
        "timing_seconds": timing,
        "counts": counts,
        "analysis": analysis,
        "atlas": {k: {str(L): v for L, v in d_.items()}
                  for k, d_ in atlas.items()},
        "seconds": time.time() - t0,
    }
    (out_dir / "atlas.json").write_text(json.dumps(payload, indent=2))
    report(payload)
    print(f"written: {out_dir/'atlas.json'}")


# ----------------------------------------------------------- the analysis
def analyse(atlas, family, u_grid):
    h = float(u_grid[1] - u_grid[0])
    drift_level, drift_family, second_gap = [], [], []
    n_peaks_hist = Counter()
    appear = disappear = 0
    concave_ok = concave_all = 0
    second_diffs: list[float] = []
    effective: dict[str, dict] = {}
    per_profile = {}

    for name, by_L in atlas.items():
        Ls = sorted(by_L)
        doms, counts = [], []
        for L in Ls:
            pk = by_L[L]["peaks"]
            n_peaks_hist[len(pk)] += 1
            counts.append(len(pk))
            doms.append(dominant(pk))
            if len(pk) >= 2:
                srt = sorted(pk, key=lambda uc: -uc[1])
                second_gap.append(abs(srt[0][0] - srt[1][0]))
        d_lvl = [abs(b - a) for a, b in zip(doms, doms[1:])
                 if math.isfinite(a) and math.isfinite(b)]
        drift_level += d_lvl
        for a, b in zip(counts, counts[1:]):
            appear += max(0, b - a)
            disappear += max(0, a - b)

        q = np.array([by_L[L]["log2_q"] for L in Ls])
        # Concavity must be judged against the accuracy of the scan, not
        # against zero: q is ~10^6 bits and the Laplace/scan floor is a
        # few bits, so a +4 bit second difference on -961,430 is noise.
        tol = 1e-5 * float(np.abs(q).max())
        for i in range(1, len(q) - 1):
            concave_all += 1
            dd = float(q[i + 1] - 2 * q[i] + q[i - 1])
            second_diffs.append(dd)
            if dd <= tol:
                concave_ok += 1

        # How many levels actually carry the depth average?  Terms are
        # 2^{q_L}; count the smallest set whose sum is within eps of the
        # whole, and check that set is a contiguous run around the mode
        # (which is what an outward walk from the mode would produce).
        w = np.exp2(q - q.max())
        order = np.argsort(-w)
        csum = np.cumsum(w[order]) / w.sum()
        eff = {}
        for eps in (1e-6, 1e-9, 1e-12):
            m = int(np.searchsorted(csum, 1.0 - eps) + 1)
            idx = sorted(order[:m].tolist())
            eff[f"levels_for_{eps:g}"] = m
            eff[f"contiguous_{eps:g}"] = bool(
                idx == list(range(idx[0], idx[0] + m)))
        eff["levels_available"] = len(q)
        eff["mode_level"] = int(Ls[int(np.argmax(q))])
        effective[name] = eff

        fam_d = []
        for L, by_c in family.get(name, {}).items():
            base = dominant(by_L[L]["peaks"])
            for c, rec in by_c.items():
                u = dominant(rec["peaks"])
                if math.isfinite(base) and math.isfinite(u):
                    fam_d.append(abs(u - base))
        drift_family += fam_d
        per_profile[name] = {
            "max_level_drift": max(d_lvl) if d_lvl else None,
            "median_level_drift": float(np.median(d_lvl)) if d_lvl else None,
            "max_family_drift": max(fam_d) if fam_d else None,
        }

    def stats(v, label):
        if not v:
            return {}
        a = np.asarray(v)
        return {f"{label}_median": float(np.median(a)),
                f"{label}_p99": float(np.quantile(a, 0.99)),
                f"{label}_max": float(a.max()),
                f"{label}_max_in_grid_steps": float(a.max() / h),
                f"{label}_count": int(a.size)}

    out = {"grid_spacing": h,
           "n_significant_peaks_histogram": dict(sorted(n_peaks_hist.items())),
           "peak_appearances_along_L": appear,
           "peak_disappearances_along_L": disappear,
           "log_concave_in_L_fraction":
               (concave_ok / concave_all) if concave_all else None,
           "log_concave_checks": concave_all,
           "second_difference_max": (max(second_diffs)
                                     if second_diffs else None),
           "second_difference_median": (float(np.median(second_diffs))
                                        if second_diffs else None),
           "effective_levels": effective,
           "per_profile": per_profile}
    out.update(stats(drift_level, "level_drift"))
    out.update(stats(drift_family, "family_drift"))
    out.update(stats(second_gap, "second_peak_distance"))
    return out


def report(payload):
    t = payload["timing_seconds"]
    c = payload["counts"]
    a = payload["analysis"]
    tot = sum(t.values())
    print(f"\n=== PHASE 1: where the time goes (serial, {tot:.1f} s total)")
    for k in ("provision", "assemble", "locate", "family_update",
              "refine_base", "refine_family"):
        print(f"    {k:16s} {t[k]:9.2f} s   {100*t[k]/tot:5.1f}%")
    print(f"    {c['profile_levels']:,} (profile, level) pairs, "
          f"{c['family_members']:,} family members")

    # Provisioning is per LEVEL and amortises over every profile at that
    # level; a real run has orders of magnitude more profiles than this
    # probe, so it must not enter the gate.  The gate is the share of
    # the PER-PROFILE work that exists only because of the grid.
    # _scan_from_psi runs the same O(G) local-maximum scan internally
    # for every call, so the refine buckets contain one locate each;
    # `locate` is measured once per call outside, so subtracting it
    # moves that grid work out of the pointwise bucket where it does
    # not belong.
    grid = t["assemble"] + t["locate"] + t["family_update"]
    point = t["refine_base"] + t["refine_family"] - t["locate"]
    share = 100 * grid / (grid + point)
    print(f"\n    per-profile work: grid-dependent {grid:.2f} s "
          f"({share:.1f}%), pointwise {point:.2f} s")
    print(f"    ceiling on the T2(1) speedup: {(grid+point)/point:.2f}x "
          f"(if the grid work went to zero)")
    print(f"    provisioning excluded from the gate: {t['provision']:.2f} s, "
          f"amortised over all profiles at each level")

    print(f"\n=== PHASE 2: the peak atlas "
          f"(grid spacing {a['grid_spacing']:.3f})")
    print(f"    significant peaks per (profile, level): "
          f"{a['n_significant_peaks_histogram']}")
    print(f"    peaks appearing / vanishing along L: "
          f"{a['peak_appearances_along_L']} / "
          f"{a['peak_disappearances_along_L']}")
    for label in ("level_drift", "family_drift", "second_peak_distance"):
        if f"{label}_max" in a:
            print(f"    {label:22s} median {a[label+'_median']:8.4f}  "
                  f"p99 {a[label+'_p99']:8.4f}  max {a[label+'_max']:8.4f}"
                  f"  ({a[label+'_max_in_grid_steps']:.0f} grid steps)")
    print("\n    levels that actually carry the depth average "
          "(of the levels available):")
    for name, e in a["effective_levels"].items():
        print(f"      {name:12s} mode L={e['mode_level']:<3d}  "
              f"1e-6: {e['levels_for_1e-06']:>3d}   "
              f"1e-9: {e['levels_for_1e-09']:>3d}   "
              f"1e-12: {e['levels_for_1e-12']:>3d}   "
              f"of {e['levels_available']:>3d}   "
              f"contiguous={e['contiguous_1e-09']}")
    if a["log_concave_in_L_fraction"] is not None:
        print(f"    log2 q_L concave in L: "
              f"{100*a['log_concave_in_L_fraction']:.1f}% of "
              f"{a['log_concave_checks']} interior levels; "
              f"second difference median "
              f"{a['second_difference_median']:+.4g}, "
              f"max {a['second_difference_max']:+.4g} bits")


if __name__ == "__main__":
    main()
