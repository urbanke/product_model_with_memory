#!/usr/bin/env python3
"""Pairwise order-two members, tables-free, vectorized, threaded.

Same members and same math as the tables-free evaluator, but every
per-token and per-pair quantity is computed by numpy on whole blocks,
and every heavy numpy phase is split into chunks that run on all
--jobs cores (numpy releases Python's lock inside large array
operations).  No plain-Python loop ever walks tokens, triples, or
pairs.  The only Python loops left are over the distinct STATES of a
block (to build count profiles for the layered builder, batched into
one request) and over fixed-size grids.

Key identity that removes the per-pair union walk: for the product
member with exponents (b1, b2), g = 1 - b1 - b2, the normalizer of
the pair (a1, a2) decomposes as

    Z = S(g) * 2^(b1 c1 + b2 c2)
      + 2^(b2 c2) * A1(a1)  +  2^(b1 c1) * A2(a2)  +  X(a1, a2)

where c1, c2 are the states' unseen-symbol constants, S(g) is the
power sum of the unigram row over the whole alphabet, A1/A2 are
per-STATE sums over each state's observed successors, and X is a sum
over the intersection of the two successor lists.  S, A1, A2 are
vectorized outright; X is gathered for all pairs of the block in one
batched join (repeat/searchsorted), no per-pair Python.

    python scripts/pairwise_experiment.py --ids output/streams/bpe_text8 \
        --checkpoints 32 --jobs 8 --out output/pw_text8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from product_model_with_memory.streams import load_stream, reduce_ids

MIX_GRID = [(0.02, 0.05), (0.02, 0.1), (0.02, 0.2), (0.02, 0.3),
            (0.05, 0.05), (0.05, 0.1), (0.05, 0.2), (0.05, 0.3)]
PROD_GRID = [(b1, b2) for b1 in (0.75, 1.0) for b2 in (0.25, 0.5, 0.75, 1.0)]
NEG = -1e30                      # log2 of an impossible event


def geo_edges(lo0: int, n: int, C: int, first: int = 2048):
    m_ = n - lo0
    lo, hi = 1.0, 4.0
    for _ in range(200):
        r = (lo + hi) / 2
        tot = first * (C if abs(r - 1) < 1e-12 else (r**C - 1) / (r - 1))
        if tot < m_:
            lo = r
        else:
            hi = r
    r = (lo + hi) / 2
    e, acc = [lo0], 0.0
    for k in range(C):
        acc += first * r**k
        e.append(min(n, lo0 + round(acc)))
    e[-1] = n
    return e


def merge_counts(keys: np.ndarray, cnts: np.ndarray,
                 newk: np.ndarray, newc: np.ndarray):
    """Merge sorted (key, count) arrays with a new batch (any order)."""
    if newk.size == 0:
        return keys, cnts
    k = np.concatenate([keys, newk])
    c = np.concatenate([cnts, newc])
    uk, inv = np.unique(k, return_inverse=True)
    uc = np.bincount(inv, weights=c).astype(np.int64)
    return uk, uc


class Structure:
    """Sorted packed (state*V + y) -> count, with per-state grouping."""

    def __init__(self, V: int):
        self.V = V
        self.keys = np.empty(0, dtype=np.int64)
        self.cnts = np.empty(0, dtype=np.int64)

    def add_block(self, states: np.ndarray, ys: np.ndarray):
        packed = states * self.V + ys
        uk, uc = np.unique(packed, return_counts=True)
        self.keys, self.cnts = merge_counts(
            self.keys, self.cnts, uk, uc.astype(np.int64))


def chunk_ranges(total: int, pieces: int):
    edges = np.linspace(0, total, pieces + 1).astype(np.int64)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])
            if b > a]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--top-k", type=int, default=10**9)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--checkpoints", type=int, default=32)
    ap.add_argument("--spacing", choices=("equal", "geo"), default="geo")
    ap.add_argument("--smooth", type=float, default=64.0,
                    help="backoff pseudo-count for the markov2 reference")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    ids, meta = load_stream(args.ids)
    if args.n:
        ids = ids[: args.n]
    if args.top_k < 10**9:
        x, V, _ = reduce_ids(ids, args.top_k)
        x = x.astype(np.int64)
    else:
        x = ids.astype(np.int64)
        V = int(x.max()) + 1
    n = len(x)
    print(f"V={V} n={n:,} C={args.checkpoints} spacing={args.spacing} "
          f"jobs={args.jobs}", flush=True)

    from product_model_with_memory.pooled_lags import (
        _LayeredPredictiveBuilder, _augmented_profile)
    from product_model_with_memory.codelength import default_l_max
    builder = _LayeredPredictiveBuilder(
        V, default_l_max(V), None, args.jobs, None)
    ratio_memo: dict = {}
    J = max(1, int(args.jobs))
    ex = ThreadPoolExecutor(max_workers=J)

    if args.spacing == "equal":
        edges = [round(2 + (n - 2) * k / args.checkpoints)
                 for k in range(args.checkpoints + 1)]
    else:
        edges = geo_edges(2, n, args.checkpoints)

    names = (["lag1", "lag2", "markov2"]
             + [f"mix:{a:g},{b:g}" for a, b in MIX_GRID]
             + [f"prod:{a:g},{b:g}" for a, b in PROD_GRID])
    bits = {k: 0.0 for k in names}
    s = float(args.smooth)
    B1 = np.array([p[0] for p in PROD_GRID])
    B2 = np.array([p[1] for p in PROD_GRID])
    G = 1.0 - B1 - B2
    nP = len(PROD_GRID)

    # ---- causal state (all numpy)
    uni = np.zeros(V, dtype=np.int64)
    uni += np.bincount(x[:2], minlength=V)
    S1 = Structure(V)            # lag-1 successors: a1*V + y
    S2 = Structure(V)            # lag-2 successors: a2*V + y
    SP = Structure(V)            # pair totals key a1*V + a2 (y folded out)
    ST = Structure(V)            # triples: (a1*V + a2)*V + y
    coded = 0
    phase = {"prep_py": 0.0, "prep_tab": 0.0, "prep": 0.0,
             "states": 0.0, "X": 0.0, "score": 0.0, "reveal": 0.0}

    def state_prep(struct: Structure, blk_states: np.ndarray,
                   fams: dict, plan: list):
        """Collect (profile, count) needs for the block's states and a
        per-state plan; ONE builder request happens later for all."""
        keys, cnts = struct.keys, struct.cnts
        st_all = keys // V
        lo_i = np.searchsorted(st_all, blk_states, side="left")
        hi_i = np.searchsorted(st_all, blk_states, side="right")
        for a, i0, i1 in zip(blk_states.tolist(),
                             lo_i.tolist(), hi_i.tolist()):
            if i1 > i0:
                cv = cnts[i0:i1]
                base = tuple(sorted(cv.tolist()))
                need = set(cv.tolist())
                if i1 - i0 < V:
                    need.add(0)
            else:
                base, need = (), {0}
            miss = [c_ for c_ in need if (base, c_) not in ratio_memo]
            if miss:
                fams.setdefault(base, set()).update(miss)
            plan.append((a, i0, i1, base))

    def state_arrays(struct: Structure, plan: list, lm_all, nblk: int,
                     Bexp: np.ndarray):
        """Per-state constants c[], per-entry log-ratios aligned with
        struct.keys (only at this block's states), per-state A sums."""
        keys, cnts = struct.keys, struct.cnts
        cconst = np.empty(nblk)
        vals = np.empty(len(keys))          # only block slices filled
        slot_of = {}
        segs = []                            # (i0, i1) per slot
        for slot, (a, i0, i1, base) in enumerate(plan):
            slot_of[a] = slot
            segs.append((i0, i1))
            if i1 > i0:
                cv = cnts[i0:i1]
                ucv = np.unique(cv)
                v_ = np.array([ratio_memo[(base, int(c_))] for c_ in ucv])
                vals[i0:i1] = v_[np.searchsorted(ucv, cv)]
                cconst[slot] = (ratio_memo[(base, 0)]
                                if i1 - i0 < V else NEG)
            else:
                cconst[slot] = ratio_memo[((), 0)]
        # gathered entry indices, concatenated in slot order
        lens = np.array([i1 - i0 for i0, i1 in segs], dtype=np.int64)
        tot = int(lens.sum())
        if tot:
            starts = np.array([i0 for i0, _ in segs], dtype=np.int64)
            off = np.repeat(np.cumsum(lens) - lens, lens)
            gidx = np.repeat(starts, lens) + np.arange(tot) - off
        else:
            gidx = np.empty(0, dtype=np.int64)
        bounds = np.concatenate([[0], np.cumsum(lens)])
        # A sums per member, one thread per member: sum over entries of
        #   2^(g*lm_y) * (2^(b*val) - 2^(b*c))
        lmE = lm_all[keys[gidx] % V]
        vE = vals[gidx]
        cE = np.repeat(cconst, lens)
        A = np.empty((nblk, nP))
        ext = np.empty(tot + 1)

        def a_one(j):
            term = np.exp2(G[j] * lmE) * (
                np.exp2(Bexp[j] * vE) - np.exp2(Bexp[j] * cE))
            e_ = np.concatenate([term, [0.0]])
            A[:, j] = np.add.reduceat(e_, bounds[:-1])

        list(ex.map(a_one, range(nP)))
        A[lens == 0, :] = 0.0
        return cconst, vals, slot_of, lens, gidx, bounds, A

    for c in range(args.checkpoints):
        lo, hi = edges[c], edges[c + 1]
        if hi <= lo:
            continue
        tblk = time.time()

        # ---- frozen unigram row and power sums
        nz = uni[uni > 0]
        gbase = tuple(sorted(nz.tolist()))
        cset = set(nz.tolist())
        seen = len(nz)
        if seen < V:
            cset.add(0)
        cnts_sorted = np.array(sorted(cset), dtype=np.int64)
        miss = [int(c_) for c_ in cnts_sorted
                if (gbase, int(c_)) not in ratio_memo]
        gfams = {gbase: set(miss)} if miss else {}

        # ---- per-state needs (lag1 and lag2), ONE builder request
        t_ = np.arange(lo, hi)
        b_, a1_, a2_ = x[t_], x[t_ - 1], x[t_ - 2]
        u1 = np.unique(a1_)
        u2 = np.unique(a2_)
        plan1: list = []
        plan2: list = []
        fams: dict = {k_: set(v_) for k_, v_ in gfams.items()}
        tpy = time.time()
        state_prep(S1, u1, fams, plan1)
        state_prep(S2, u2, fams, plan2)
        phase["prep_py"] += time.time() - tpy
        ttab = time.time()
        if fams:
            builder._ensure_families(
                {bb: tuple(sorted(cs)) for bb, cs in fams.items()})
            for bb, cs in fams.items():
                for cv in cs:
                    ratio_memo[(bb, cv)] = builder._log2_ratio(
                        bb, _augmented_profile(bb, cv))
        phase["prep_tab"] += time.time() - ttab

        m_val = np.array([ratio_memo[(gbase, int(c_))]
                          for c_ in cnts_sorted])
        lm_all = m_val[np.searchsorted(cnts_sorted, uni)]
        Spow = np.array([np.exp2(g_ * lm_all).sum() for g_ in G])
        phase["prep"] += time.time() - tblk

        # ---- per-state arrays and A sums
        t1s = time.time()
        (c1c, v1, slot1, len1, gidx1, bnd1, A1) = state_arrays(
            S1, plan1, lm_all, len(u1), B1)
        (c2c, v2, slot2, len2, gidx2, bnd2, A2) = state_arrays(
            S2, plan2, lm_all, len(u2), B2)
        phase["states"] += time.time() - t1s

        # ---- distinct triples of the block
        trip = (a1_ * V + a2_) * V + b_
        ut, ct = np.unique(trip, return_counts=True)
        cc = ct.astype(np.float64)
        yy = ut % V
        pk = ut // V
        ta1 = pk // V
        ta2 = pk % V
        # distinct pairs and the triple->pair map
        up, pinv = np.unique(pk, return_inverse=True)
        pa1 = up // V
        pa2 = up % V
        sl1 = np.array([slot1[int(a)] for a in pa1])
        sl2 = np.array([slot2[int(a)] for a in pa2])

        # ---- X: intersection term, batched join (no per-pair loop).
        # For each pair, walk the SHORTER successor list and look the
        # partner up in the other structure; pair chunks run on all
        # cores, each accumulating into its own local X.
        tX = time.time()
        d1p = len1[sl1]
        d2p = len2[sl2]
        Xp = np.zeros((len(up), nP))
        side1 = d1p <= d2p

        def x_chunk(sel_idx, struct_a, gidx_a, bnd_a, sl_a, v_a, c_a,
                    struct_b, v_b, c_b, sl_b, swap):
            ssl = sl_a[sel_idx]
            lens_ = (bnd_a[ssl + 1] - bnd_a[ssl])
            tot = int(lens_.sum())
            if tot == 0:
                return None
            off = np.repeat(np.cumsum(lens_) - lens_, lens_)
            gi = np.repeat(bnd_a[ssl], lens_) + np.arange(tot) - off
            eidx = gidx_a[gi]                    # entry idx in struct a
            yv = struct_a.keys[eidx] % V
            other_state = (pa2 if not swap else pa1)[sel_idx]
            qkey = np.repeat(other_state, lens_) * V + yv
            nb = len(struct_b.keys)
            if nb == 0:
                return None
            pos = np.minimum(np.searchsorted(struct_b.keys, qkey), nb - 1)
            hit = struct_b.keys[pos] == qkey
            if not hit.any():
                return None
            prow = np.repeat(sel_idx, lens_)[hit]
            lmh = lm_all[yv[hit]]
            va = v_a[eidx[hit]]
            vb = v_b[pos[hit]]
            ca = np.repeat(c_a[ssl], lens_)[hit]
            cb = np.repeat(c_b[sl_b[sel_idx]], lens_)[hit]
            Xl = np.zeros((len(up), nP))
            for j in range(nP):
                ba, bb_ = (B1[j], B2[j]) if not swap else (B2[j], B1[j])
                term = np.exp2(G[j] * lmh) * (
                    np.exp2(ba * va) - np.exp2(ba * ca)) * (
                    np.exp2(bb_ * vb) - np.exp2(bb_ * cb))
                np.add.at(Xl[:, j], prow, term)
            return Xl

        tasks = []
        for swap, mask in ((False, side1), (True, ~side1)):
            idxs = np.flatnonzero(mask)
            if len(idxs) == 0:
                continue
            aa = ((S1, gidx1, bnd1, sl1, v1, c1c, S2, v2, c2c, sl2)
                  if not swap else
                  (S2, gidx2, bnd2, sl2, v2, c2c, S1, v1, c1c, sl1))
            for i0, i1 in chunk_ranges(len(idxs), J):
                tasks.append(ex.submit(x_chunk, idxs[i0:i1], *aa, swap))
        for f_ in tasks:
            Xl = f_.result()
            if Xl is not None:
                Xp += Xl
        phase["X"] += time.time() - tX

        # ---- per-pair normalizers (pair chunks on all cores)
        tS = time.time()
        c1p = c1c[sl1]
        c2p = c2c[sl2]
        A1p = A1[sl1, :]
        A2p = A2[sl2, :]
        logZ = np.empty((len(up), nP))

        def z_chunk(i0, i1):
            for j in range(nP):
                e1 = np.exp2(B1[j] * c1p[i0:i1])
                e2 = np.exp2(B2[j] * c2p[i0:i1])
                Z = (Spow[j] * e1 * e2 + e2 * A1p[i0:i1, j]
                     + e1 * A2p[i0:i1, j] + Xp[i0:i1, j])
                logZ[i0:i1, j] = np.log2(np.maximum(Z, 1e-300))

        list(ex.map(lambda ab: z_chunk(*ab), chunk_ranges(len(up), J)))

        # ---- markov2 frozen pair counts (vector lookups)
        nc = np.zeros(len(up))
        if len(SP.keys):
            posP = np.minimum(np.searchsorted(SP.keys, up),
                              len(SP.keys) - 1)
            hitP = SP.keys[posP] == up
            nc[hitP] = SP.cnts[posP[hitP]]

        # ---- per-triple member scores (triple chunks on all cores)
        def t_chunk(i0, i1):
            yy_ = yy[i0:i1]
            key1 = ta1[i0:i1] * V + yy_
            key2 = ta2[i0:i1] * V + yy_
            pv = pinv[i0:i1]
            cc_ = cc[i0:i1]
            lp1 = c1p[pv].copy()
            if len(S1.keys):
                p1_ = np.minimum(np.searchsorted(S1.keys, key1),
                                 len(S1.keys) - 1)
                h1 = S1.keys[p1_] == key1
                lp1[h1] = v1[p1_[h1]]
            lp2 = c2p[pv].copy()
            if len(S2.keys):
                p2_ = np.minimum(np.searchsorted(S2.keys, key2),
                                 len(S2.keys) - 1)
                h2 = S2.keys[p2_] == key2
                lp2[h2] = v2[p2_[h2]]
            lm = lm_all[yy_]
            local = {}
            local["lag1"] = float((cc_ * lp1).sum())
            local["lag2"] = float((cc_ * lp2).sum())
            ncb = np.zeros(i1 - i0)
            if len(ST.keys):
                pT = np.minimum(np.searchsorted(ST.keys, ut[i0:i1]),
                                len(ST.keys) - 1)
                hT = ST.keys[pT] == ut[i0:i1]
                ncb[hT] = ST.cnts[pT[hT]]
            pm2 = (ncb + s * np.exp2(lp1)) / (nc[pv] + s)
            local["markov2"] = float((cc_ * np.log2(pm2)).sum())
            p_m = np.exp2(lm)
            p_1 = np.exp2(lp1)
            p_2 = np.exp2(lp2)
            for w0, w2 in MIX_GRID:
                w1 = 1.0 - w0 - w2
                p = w0 * p_m + w1 * p_1 + w2 * p_2
                local[f"mix:{w0:g},{w2:g}"] = float(
                    (cc_ * np.log2(p)).sum())
            for j, (b1, b2) in enumerate(PROD_GRID):
                ln = G[j] * lm + b1 * lp1 + b2 * lp2 - logZ[pv, j]
                local[f"prod:{b1:g},{b2:g}"] = float((cc_ * ln).sum())
            return local

        parts = list(ex.map(lambda ab: t_chunk(*ab),
                            chunk_ranges(len(ut), 2 * J)))
        for loc in parts:
            for k2, v2_ in loc.items():
                bits[k2] -= v2_
        phase["score"] += time.time() - tS

        # ---- reveal the block (four independent merges, threaded)
        tR = time.time()
        coded += hi - lo
        uni += np.bincount(b_, minlength=V)
        uT, cT = np.unique(trip, return_counts=True)

        def r1():
            S1.add_block(a1_, b_)

        def r2():
            S2.add_block(a2_, b_)

        def r3():
            SP.add_block(a1_, a2_)

        def r4():
            ST.keys, ST.cnts = merge_counts(
                ST.keys, ST.cnts, uT, cT.astype(np.int64))

        list(ex.map(lambda f_: f_(), [r1, r2, r3, r4]))
        phase["reveal"] += time.time() - tR
        if c < 3 or (c + 1) % 8 == 0 or c == args.checkpoints - 1:
            print(f"  checkpoint {c + 1}/{args.checkpoints}: coded "
                  f"{coded:,} ({time.time() - t0:.0f}s, block "
                  f"{time.time() - tblk:.0f}s; "
                  + " ".join(f"{k_}={v_:.0f}s"
                             for k_, v_ in phase.items()) + ")",
                  flush=True)

    per = {k: v / coded for k, v in bits.items()}
    logq = np.array([-bits[k] for k in names])
    mx = logq.max()
    fam = -(mx + math.log2(np.sum(2.0 ** (logq - mx)))) + math.log2(
        len(names))
    out = {"ids": args.ids, "V": V, "n_tokens": n,
           "coded_positions": coded, "checkpoints": args.checkpoints,
           "spacing": args.spacing, "smooth": s,
           "phase_seconds": {k_: round(v_, 1) for k_, v_ in phase.items()},
           "member_bits_per_token": per,
           "family_bits_per_token": fam / coded,
           "best_member": min(per, key=per.get),
           "seconds": time.time() - t0}
    o = Path(args.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "results.json").write_text(json.dumps(out, indent=2))
    for k in sorted(per, key=per.get):
        print(f"  {k:>16s}  {per[k]:.4f}")
    print(f"written: {o/'results.json'} ({time.time()-t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
