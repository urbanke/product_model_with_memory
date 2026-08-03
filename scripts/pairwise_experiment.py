#!/usr/bin/env python3
"""Memory of order two from PAIRWISE statistics: which combiner wins?

Order-two states lose to order one on text8 and enwik8 purely on
learning cost (paper, Table order-two).  This experiment asks how much
of the order-two ceiling can be recovered while learning only pairwise
objects: the three tables (x_t, x_{t-1}), (x_t, x_{t-2}) and
(x_{t-1}, x_{t-2}), of which the third is, by stationarity, the same
object as the first.  Roughly 3 V^2 numbers instead of the V^3 scale of
order-two states.

Combiners compared, all consuming the SAME smoothed tables, so the
comparison isolates the combining rule:

  lag1          p1(b | a1) alone (the order-one reference)
  markov2       order-two states with backoff to lag1 (the ceiling-side
                reference; V^3-scale learning, kept sparse)
  mix:w0,w2     w0 m(b) + w1 p1(b|a1) + w2 p2(b|a2), w1 = 1-w0-w2
  prod:b1,b2    m(b) (p1/m)^b1 (p2/m)^b2, normalized.  b1=b2=1 is the
                star/maxent point: exact iff the two lags are
                conditionally independent GIVEN the target
  calibrated    the Case-B predictor: psi01(b,a1) psi02(b,a2),
                potentials fitted by IPF so the implied triangle model
                reproduces all three pair distributions (paper,
                appendix on the calibrated pairwise predictor)

Sequential construction: none of these predictors telescopes to a
one-shot evaluation, so the code is the checkpointed one: cut the
stream into C blocks, freeze all tables (and the IPF calibration) on
the data strictly before each block, code the block, refresh.
Staleness is priced into every row equally.

Chow-Liu diagnostic: with lags {1,2} the maximum-MI tree simply picks
two of the three edges; because MI(x_t, x_{t-1}) = MI(x_{t-1}, x_{t-2})
identically (same table), the tree keeps (t,t-1),(t-1,t-2) whenever
lag 1 dominates lag 2, i.e. it reduces to lag1.  Printed per
checkpoint, not a scheme.

    python scripts/pairwise_experiment.py --ids output/streams/bpe_text8 \
        --top-k 1023 --checkpoints 32 --out output/pairwise_v1024
    # smoke: --n 300000 --top-k 63 --checkpoints 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from product_model_with_memory.streams import load_stream, reduce_ids


def smoothed_conditional(counts: np.ndarray, m: np.ndarray,
                         s: float) -> np.ndarray:
    """Rows: context a.  Each row smoothed toward the unigram m by s
    pseudo-tokens; rows sum to one even for unseen contexts."""

    return (counts + s * m[None, :]) / (
        counts.sum(axis=1, keepdims=True) + s)


def smoothed_joint(counts: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Strictly positive joint distribution for the IPF margins."""

    return (counts + eps) / (counts.sum() + eps * counts.size)


def ipf_triangle(P01, P02, P12, psi01, psi02, psi12,
                 iters: int, tol: float):
    """Fit psi01(x0,x1) psi02(x0,x2) psi12(x1,x2) to the three pair
    margins by iterative proportional fitting.  The V^3 joint is never
    materialized: each pair margin is one V x V matrix product.
    Warm-startable; returns (psi01, psi02, psi12, sweeps, residual)."""

    for it in range(1, iters + 1):
        M01 = psi01 * (psi02 @ psi12.T)
        psi01 *= P01 / np.maximum(M01 / M01.sum(), 1e-300)
        M02 = psi02 * (psi01 @ psi12)
        psi02 *= P02 / np.maximum(M02 / M02.sum(), 1e-300)
        M12 = psi12 * (psi01.T @ psi02)
        psi12 *= P12 / np.maximum(M12 / M12.sum(), 1e-300)
        M01 = psi01 * (psi02 @ psi12.T)
        resid = np.abs(M01 / M01.sum() - P01).sum()
        if resid < tol:
            return psi01, psi02, psi12, it, float(resid)
    return psi01, psi02, psi12, iters, float(resid)


def mutual_information(J: np.ndarray) -> float:
    pa = J.sum(axis=1, keepdims=True)
    pb = J.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = J * (np.log2(J) - np.log2(pa) - np.log2(pb))
    return float(np.nansum(t))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", required=True,
                    help="stream directory from make_stream.py")
    ap.add_argument("--top-k", type=int, default=1023)
    ap.add_argument("--checkpoints", type=int, default=32)
    ap.add_argument("--smooth", type=float, default=64.0,
                    help="pseudo-tokens toward the unigram, per row")
    ap.add_argument("--n", type=int, default=None,
                    help="use only the first n tokens")
    ap.add_argument("--ipf-iters", type=int, default=300)
    ap.add_argument("--ipf-tol", type=float, default=1e-9)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    ids, meta = load_stream(args.ids)
    if args.n:
        ids = ids[:args.n]
    x, V, capped = reduce_ids(ids, args.top_k)
    x = x.astype(np.int64)
    n = x.size
    s = float(args.smooth)
    C = args.checkpoints
    print(f"V={V} n={n:,} capped={capped:,} checkpoints={C} "
          f"smooth={s:g}", flush=True)

    # members
    mix_grid = [(w0, w2) for w0 in (0.02, 0.05)
                for w2 in (0.05, 0.10, 0.20, 0.30)]
    prod_grid = [(b1, b2) for b1 in (0.75, 1.0)
                 for b2 in (0.25, 0.5, 0.75, 1.0)]
    names = (["lag1", "markov2", "calibrated"]
             + [f"mix:{w0:g},{w2:g}" for w0, w2 in mix_grid]
             + [f"prod:{b1:g},{b2:g}" for b1, b2 in prod_grid])
    bits = {k: 0.0 for k in names}

    # causal state
    N1 = np.zeros((V, V), dtype=np.int64)     # (x_{t-1}, x_t)
    N2 = np.zeros((V, V), dtype=np.int64)     # (x_{t-2}, x_t)
    uni = np.zeros(V, dtype=np.int64)
    tri_c: list[np.ndarray] = []              # context id = a1*V + a2
    tri_b: list[np.ndarray] = []
    Ctri = sparse.csr_matrix((V * V, V), dtype=np.int64)
    psi01 = np.ones((V, V))
    psi02 = np.ones((V, V))
    psi12 = np.ones((V, V))

    edges = np.linspace(2, n, C + 1).astype(np.int64)
    coded = 0
    diagnostics = []
    CH = max(1, (1 << 23) // V)               # scoring chunk length

    for c in range(C):
        lo, hi = int(edges[c]), int(edges[c + 1])
        if hi <= lo:
            continue
        # ---- freeze tables on the prefix strictly before this block
        tot = int(uni.sum())
        m = (uni + 1.0) / (tot + V)
        p1 = smoothed_conditional(N1, m, s)
        p2 = smoothed_conditional(N2, m, s)
        L1, L2, Lm = np.log(p1), np.log(p2), np.log(m)
        J1 = smoothed_joint(N1)
        J2 = smoothed_joint(N2)
        # roles: P01[x0,x1] joint of (target, lag1); P12 is the same
        # table with roles (x_{t-1}, x_{t-2}); P02 from the lag-2 pairs
        P01, P02, P12 = J1.T.copy(), J2.T.copy(), J1.T.copy()
        psi01, psi02, psi12, sweeps, resid = ipf_triangle(
            P01, P02, P12, psi01, psi02, psi12,
            args.ipf_iters, args.ipf_tol)
        K01, K02 = np.log(psi01), np.log(psi02)
        mi01, mi02 = mutual_information(J1), mutual_information(J2)
        diagnostics.append({"checkpoint": c + 1, "ipf_sweeps": sweeps,
                            "ipf_residual_l1": resid,
                            "mi_lag1": mi01, "mi_lag2": mi02,
                            "chow_liu": "lag1-chain" if mi01 >= mi02
                                        else "lag2-edge"})
        if Ctri.nnz or tri_c:
            rowsum = np.asarray(Ctri.sum(axis=1)).ravel()
        else:
            rowsum = np.zeros(V * V)

        # ---- score the block
        for j0 in range(lo, hi, CH):
            j1 = min(j0 + CH, hi)
            t = np.arange(j0, j1)
            b, a1, a2 = x[t], x[t - 1], x[t - 2]
            r1b = p1[a1, b]
            r2b = p2[a2, b]
            mb = m[b]
            bits["lag1"] -= np.log2(r1b).sum()
            # markov2 with backoff to lag1
            cid = a1 * V + a2
            ncb = np.asarray(Ctri[cid, b]).ravel()
            nc = rowsum[cid]
            pm2 = (ncb + s * r1b) / (nc + s)
            bits["markov2"] -= np.log2(pm2).sum()
            for w0, w2 in mix_grid:
                w1 = 1.0 - w0 - w2
                p = w0 * mb + w1 * r1b + w2 * r2b
                bits[f"mix:{w0:g},{w2:g}"] -= np.log2(p).sum()
            # normalization-based members, (chunk, V) logits
            R1 = L1[a1] - Lm[None, :]
            R2 = L2[a2] - Lm[None, :]
            for b1, b2 in prod_grid:
                logits = Lm[None, :] + b1 * R1 + b2 * R2
                mx = logits.max(axis=1, keepdims=True)
                lz = mx[:, 0] + np.log(
                    np.exp(logits - mx).sum(axis=1))
                sel = logits[np.arange(len(t)), b]
                bits[f"prod:{b1:g},{b2:g}"] -= (
                    (sel - lz) / math.log(2)).sum()
            logits = K01[:, a1].T + K02[:, a2].T
            mx = logits.max(axis=1, keepdims=True)
            lz = mx[:, 0] + np.log(np.exp(logits - mx).sum(axis=1))
            sel = logits[np.arange(len(t)), b]
            bits["calibrated"] -= ((sel - lz) / math.log(2)).sum()
        coded += hi - lo

        # ---- reveal the block to the counters
        t = np.arange(lo, hi)
        np.add.at(N1, (x[t - 1], x[t]), 1)
        np.add.at(N2, (x[t - 2], x[t]), 1)
        uni += np.bincount(x[t], minlength=V)
        if c == 0:                       # tokens before the first target
            uni += np.bincount(x[:2], minlength=V)
        tri_c.append(x[t - 1] * V + x[t - 2])
        tri_b.append(x[t])
        new = sparse.coo_matrix(
            (np.ones(hi - lo, dtype=np.int64),
             (tri_c[-1], tri_b[-1])), shape=(V * V, V)).tocsr()
        Ctri = (Ctri + new).tocsr()
        print(f"  checkpoint {c + 1}/{C}: coded {coded:,} "
              f"(ipf {sweeps} sweeps, resid {resid:.1e}) "
              f"({time.time() - t0:.0f}s)", flush=True)

    per_tok = {k: v / coded for k, v in bits.items()}
    logq = np.array([-bits[k] for k in names])      # log2 q per member
    mx = logq.max()
    fam = -(mx + math.log2(np.exp(
        (logq - mx) * math.log(2)).sum()) - math.log2(len(names)))
    post = np.exp((logq - logq.max()) * math.log(2))
    post /= post.sum()

    print(f"\n{'member':>16}  bits/token")
    for k in sorted(names, key=lambda k: per_tok[k]):
        print(f"{k:>16}  {per_tok[k]:.4f}"
              + ("   <-- best" if per_tok[k] == min(per_tok.values())
                 else ""))
    print(f"{'family':>16}  {fam / coded:.4f}")
    print(f"posterior mass on best: {post.max():.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({
        "ids": args.ids, "top_k": args.top_k, "V": V, "n_tokens": n,
        "coded_positions": coded, "checkpoints": C, "smooth": s,
        "member_bits_per_token": per_tok,
        "family_bits_per_token": fam / coded,
        "posterior_max": float(post.max()),
        "best_member": min(per_tok, key=per_tok.get),
        "diagnostics": diagnostics,
        "seconds": time.time() - t0,
    }, indent=2))
    print(f"written: {out / 'results.json'} "
          f"({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
