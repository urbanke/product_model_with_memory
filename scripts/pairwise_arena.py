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

Sequential construction: with one exception, none of these predictors
telescopes to a one-shot evaluation, so the code is the checkpointed
one: cut the stream into C blocks, freeze all tables (and the IPF
calibration) on the data strictly before each block, code the block,
refresh.  Staleness is priced into every row equally.

The exception is the layered order-two reference.  It is a mixture,
so its exact sequential code (tables updated after EVERY token)
telescopes to one evaluation per context pair at the final counts.
--exact add computes this number (markov2-layered-exact) alongside
the checkpointed members; --exact only computes just it and merges it
into an existing results.json.  The gap between the exact and the
checkpointed version measures what the checkpoint schedule costs; if
it is large, C blocks are too few.

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
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
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


def project_margins(J: np.ndarray, m: np.ndarray,
                    iters: int = 500, tol: float = 1e-13) -> np.ndarray:
    """Sinkhorn projection: rescale J so BOTH its margins equal m.

    The three estimated pair tables come from slightly different
    position ranges and smoothing, so their singleton margins disagree
    at the 1e-3 level.  IPF over inconsistent margins does not
    converge (it cycles), which was visible as the sweep cap being hit
    at every checkpoint with a stalled residual.  Projecting each
    table to one common set of margins first makes the triangle
    constraints mutually consistent, and IPF then converges."""

    J = J.copy()
    for _ in range(iters):
        J *= (m / J.sum(axis=1))[:, None]
        J *= (m / J.sum(axis=0))[None, :]
        if np.abs(J.sum(axis=1) - m).sum() < tol:
            break
    return J


def _ipf_anderson(P01, P02, P12, psi01, psi02, psi12,
                  iters, tol, floor):
    """Anderson-accelerated IPF.  Same map, same residual test, same
    unique optimum as plain IPF.  The state advances in the linear
    domain (the map's natural, cheap form); the acceleration history
    lives in the LOG domain in float64 --- reduced precision fails in
    BOTH domains (measured 4 Aug: linear differences underflow;
    floored log entries reach magnitude ~700 where float32 carries
    only ~4e-5 absolute).  Extrapolations are proposals only:
    exponentiation keeps them positive, and the full-precision
    residual test decides."""

    V_ = psi01.shape[0]
    n2 = V_ * V_
    n = 3 * n2
    m = 4 if V_ <= 1500 else 2

    def G_inplace(a01, a02, a12):
        def step(a, P, M):
            sM = M.sum()
            if not np.isfinite(sM) or sM <= 0.0:
                return False
            a *= P / np.maximum(M / sM, floor)
            a /= a.max()
            return True
        ok = (step(a01, P01, a01 * (a02 @ a12.T))
              and step(a02, P02, a02 * (a01 @ a12))
              and step(a12, P12, a12 * (a01.T @ a02)))
        if not ok:
            return None
        M01 = a01 * (a02 @ a12.T)
        sM = M01.sum()
        return (np.abs(M01 / sM - P01).sum()
                if np.isfinite(sM) and sM > 0.0 else float("inf"))

    def logflat(a01, a02, a12):
        out = np.empty(n)
        out[:n2] = np.log(np.maximum(a01, 1e-300)).ravel()
        out[n2:2 * n2] = np.log(np.maximum(a02, 1e-300)).ravel()
        out[2 * n2:] = np.log(np.maximum(a12, 1e-300)).ravel()
        return out

    a01 = np.maximum(psi01, 1e-300)
    a02 = np.maximum(psi02, 1e-300)
    a12 = np.maximum(psi12, 1e-300)
    lx = logflat(a01, a02, a12)
    lx_prev = None
    lf_prev = None
    dX = np.zeros((m, n))
    dF = np.zeros((m, n))
    nh = 0
    head = 0
    sweeps = 0
    while sweeps < iters:
        resid = G_inplace(a01, a02, a12)
        sweeps += 1
        if resid is None:
            a01 = np.ones((V_, V_))
            a02 = np.ones((V_, V_))
            a12 = np.ones((V_, V_))
            lx = logflat(a01, a02, a12)
            nh = 0
            lx_prev = lf_prev = None
            continue
        if resid < tol:
            return a01, a02, a12, sweeps, float(resid)
        lgx = logflat(a01, a02, a12)
        lf = lgx - lx
        if lx_prev is not None:
            dX[head] = lx - lx_prev
            dF[head] = lf - lf_prev
            head = (head + 1) % m
            nh = min(nh + 1, m)
        lx_prev, lf_prev = lx, lf
        did_mix = False
        if nh >= 1:
            D_F = dF[:nh]
            D_X = dX[:nh]
            A = D_F @ D_F.T
            A[np.diag_indices_from(A)] += 1e-8 * max(1.0, A.max())
            try:
                gam = np.linalg.solve(A, D_F @ lf)
                cand_log = lgx - D_X.T @ gam - D_F.T @ gam
                if np.isfinite(cand_log).all():
                    v = np.exp(cand_log)
                    # no renormalization here: the map is scale-
                    # invariant, and lx must stay the exact log of
                    # the state for the history to be consistent
                    a01 = v[:n2].reshape(V_, V_)
                    a02 = v[n2:2 * n2].reshape(V_, V_)
                    a12 = v[2 * n2:].reshape(V_, V_)
                    lx = cand_log
                    did_mix = True
            except np.linalg.LinAlgError:
                nh = 0
                lx_prev = lf_prev = None
        if not did_mix:
            lx = lgx
    return a01, a02, a12, sweeps, float(resid)


def ipf_triangle(P01, P02, P12, psi01, psi02, psi12,
                 iters: int, tol: float, ex=None, jobs: int = 1,
                 solver: str = "ipf"):
    """Fit psi01(x0,x1) psi02(x0,x2) psi12(x1,x2) to the three pair
    margins by iterative proportional fitting.  The V^3 joint is never
    materialized: each pair margin is one V x V matrix product.
    Warm-startable; returns (psi01, psi02, psi12, sweeps, residual)."""

    # Numerics (fix, 4 Aug 2026): each factor is rescaled to max 1
    # right after its update.  The fitted model is invariant to a
    # per-factor scale (every use of the factors normalizes over the
    # target), but WITHOUT the rescaling the scales drift
    # exponentially across warm-started checkpoints and overflow to
    # nan --- observed at checkpoint 6 on enwik8 and enwik9, which
    # cost those runs their calibrated member.  The floor moved from
    # 1e-300 (one update away from the float64 edge) to 1e-150.  If a
    # margin still degenerates, the factors are reset and the sweep
    # restarted, and a degenerate final residual reports inf, never
    # nan.
    floor = 1e-150
    V_ = psi01.shape[0]
    if solver == "anderson":
        return _ipf_anderson(P01, P02, P12, psi01, psi02, psi12,
                             iters, tol, floor)

    if ex is not None and jobs > 1:
        nb = min(jobs * 2, V_)
        edges = np.linspace(0, V_, nb + 1, dtype=int)
        slices = [slice(int(a_), int(b_))
                  for a_, b_ in zip(edges[:-1], edges[1:])
                  if b_ > a_]
    else:
        slices = None

    def _step(psi, P, M):
        # All reductions (sum, max) stay global and serial, so the
        # threaded elementwise updates are bit-identical to serial.
        s = M.sum()
        if not np.isfinite(s) or s <= 0.0:
            return False
        if slices is None:
            psi *= P / np.maximum(M / s, floor)
            psi /= psi.max()
        else:
            def _upd(sl):
                psi[sl] *= P[sl] / np.maximum(M[sl] / s, floor)
            list(ex.map(_upd, slices))
            mx = psi.max()
            def _nrm(sl):
                psi[sl] /= mx
            list(ex.map(_nrm, slices))
        return True

    resid = float("inf")
    for it in range(1, iters + 1):
        ok = (_step(psi01, P01, psi01 * (psi02 @ psi12.T))
              and _step(psi02, P02, psi02 * (psi01 @ psi12))
              and _step(psi12, P12, psi12 * (psi01.T @ psi02)))
        if not ok:
            psi01[:] = 1.0
            psi02[:] = 1.0
            psi12[:] = 1.0
            continue
        M01 = psi01 * (psi02 @ psi12.T)
        s = M01.sum()
        resid = (np.abs(M01 / s - P01).sum()
                 if np.isfinite(s) and s > 0.0 else float("inf"))
        if resid < tol:
            return psi01, psi02, psi12, it, float(resid)
    return psi01, psi02, psi12, iters, float(resid)


def exact_order2_bits(builder, Ctri) -> float:
    """Exact codelength of the order-two layered code: the per-pair
    mixture updated after every token.  The codelength telescopes, so
    it is one evaluation per context pair at the FINAL counts,
    deduplicated over count profiles."""

    from collections import Counter
    from product_model_with_memory.pooled_lags import _log2sumexp_arr

    mult: Counter = Counter()
    indptr, data = Ctri.indptr, Ctri.data
    for u in range(Ctri.shape[0]):
        d0, d1 = indptr[u], indptr[u + 1]
        if d1 > d0:
            mult[tuple(sorted(int(v) for v in data[d0:d1]))] += 1
    profs = [p for p in mult if p not in builder.memo]
    print(f"  exact: {len(mult)} distinct profiles, "
          f"{len(profs)} to evaluate", flush=True)
    B = 2000
    for i in range(0, len(profs), B):
        builder._ensure_families({p: () for p in profs[i:i + B]})
        print(f"  exact: {min(i + B, len(profs))}/{len(profs)}",
              flush=True)
    ll = math.log2(builder.l_max)
    return -sum(
        m * (_log2sumexp_arr(builder.memo[p]) - ll)
        for p, m in mult.items())


def merge_exact(out: Path, bits_x: float, coded: int,
                secs: float) -> None:
    """Write markov2-layered-exact into an existing results.json (or a
    fresh one), recomputing the family code and the best member."""

    res_path = out / "results.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    prev = res.get("coded_positions")
    if prev is not None and prev != coded:
        raise SystemExit(
            f"coded positions differ: results.json has {prev}, this "
            f"stream gives {coded}; same corpus, --n and cap?")
    per = res.setdefault("member_bits_per_token", {})
    per["markov2-layered-exact"] = bits_x / coded
    res["coded_positions"] = coded
    logq = np.array([-v * coded for v in per.values()])
    mx = logq.max()
    fam = -(mx + math.log2(np.exp(
        (logq - mx) * math.log(2)).sum()) - math.log2(len(per)))
    post = np.exp((logq - mx) * math.log(2))
    post /= post.sum()
    res["family_bits_per_token"] = fam / coded
    res["posterior_max"] = float(post.max())
    res["best_member"] = min(per, key=per.get)
    res["exact_seconds"] = secs
    out.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(res, indent=2))
    print(f"markov2-layered-exact: {bits_x / coded:.4f} bits/token")
    print(f"written: {res_path}")


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
                    help="pseudo-tokens toward the unigram, per row "
                         "(counts mode; also the markov2 backoff)")
    ap.add_argument("--tables", choices=("counts", "layered"),
                    default="counts",
                    help="where the probability tables come from: "
                         "simple count blending, or the layered "
                         "estimator evaluated on the counts so far")
    ap.add_argument("--cap", choices=("freq", "id"), default="freq",
                    help="vocabulary cap: most frequent in this file "
                         "(internal comparisons) or lowest vocabulary "
                         "id (decoder-reproducible)")
    ap.add_argument("--order2", choices=("backoff", "layered", "both"),
                    default="backoff",
                    help="order-two reference: count backoff, the "
                         "share-nothing layered estimator per context "
                         "pair, or both")
    ap.add_argument("--exact", choices=("off", "add", "only"),
                    default="off",
                    help="exact order-two layered reference (tables "
                         "updated after every token; telescopes to "
                         "one evaluation per context pair at the "
                         "final counts): 'add' computes it alongside "
                         "the checkpointed members, 'only' computes "
                         "just it and merges it into an existing "
                         "results.json in --out")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--n", type=int, default=None,
                    help="use only the first n tokens")
    ap.add_argument("--ipf-iters", type=int, default=300)
    ap.add_argument("--spacing", choices=("equal", "geo"),
                    default="geo",
                    help="checkpoint placement; geo (default) puts "
                         "small blocks early where tables learn "
                         "fastest --- measured 5x less staleness at "
                         "C=32 than equal")
    ap.add_argument("--ipf-solver", choices=("ipf", "anderson"),
                    default="ipf",
                    help="anderson: accelerated fit, same optimum,"
                         " typically 3-10x fewer sweeps; digits can"
                         " differ only where neither run converges")
    ap.add_argument("--ipf-lag", type=int, default=1,
                    help="warm-start each calibration fit from the fit"
                         " this many checkpoints back; k>1 runs k fit"
                         " chains concurrently (changes calibrated"
                         " digits only where the sweep cap binds)")
    ap.add_argument("--ipf-tol", type=float, default=1e-9)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    ids, meta = load_stream(args.ids)
    if args.n:
        ids = ids[:args.n]
    if args.cap == "id":
        # lowest vocabulary ids keep their identity: fixed before this
        # file exists, so the decoder can reproduce the cap
        x = np.where(ids < args.top_k, ids, args.top_k).astype(np.int64)
        V = args.top_k + 1
        capped = int(np.count_nonzero(x == args.top_k))
    else:
        x, V, capped = reduce_ids(ids, args.top_k)
        x = x.astype(np.int64)
    n = x.size
    s = float(args.smooth)
    C = args.checkpoints
    print(f"V={V} n={n:,} capped={capped:,} checkpoints={C} "
          f"smooth={s:g} tables={args.tables} cap={args.cap} "
          f"order2={args.order2}", flush=True)

    builder = None
    ratio_memo: dict = {}
    if (args.tables == "layered" or args.order2 in ("layered", "both")
            or args.exact != "off"):
        for k_, v_ in (("PMM_UNIVERSAL_TABLES", "tables/anchors_prod"),
                       ("PMM_PHI_LADDER_EVERY", "1"),
                       ("PMM_PHI_LADDER_DEGREE", "11"),
                       ("PMM_PHI_SADDLE_MIN_L", "54")):
            os.environ.setdefault(k_, v_)
        print(f"  store: {os.environ['PMM_UNIVERSAL_TABLES']}",
              flush=True)
        from product_model_with_memory.pooled_lags import (
            _LayeredPredictiveBuilder, _layered_log_tables,
            _augmented_profile)
        from product_model_with_memory.codelength import default_l_max
        builder = _LayeredPredictiveBuilder(
            V, default_l_max(V), None, args.jobs, None)
        globals()["_augmented_profile"] = _augmented_profile

    if args.exact == "only":
        t = np.arange(2, n)
        Cfin = sparse.coo_matrix(
            (np.ones(n - 2, dtype=np.int64),
             (x[t - 1] * V + x[t - 2], x[t])),
            shape=(V * V, V)).tocsr()
        bits_x = exact_order2_bits(builder, Cfin)
        merge_exact(Path(args.out), bits_x, n - 2, time.time() - t0)
        return

    # members
    mix_grid = [(w0, w2) for w0 in (0.02, 0.05)
                for w2 in (0.05, 0.10, 0.20, 0.30)]
    prod_grid = [(b1, b2) for b1 in (0.75, 1.0)
                 for b2 in (0.25, 0.5, 0.75, 1.0)]
    names = ["lag1", "calibrated"]
    if args.order2 in ("backoff", "both"):
        names.append("markov2")
    if args.order2 in ("layered", "both"):
        names.append("markov2-layered")
    names += ([f"mix:{w0:g},{w2:g}" for w0, w2 in mix_grid]
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

    if args.spacing == "equal":
        edges = np.linspace(2, n, C + 1).astype(np.int64)
    else:
        # geometric: small blocks early, where the tables learn
        # fastest.  Measured (chunking_cost.py, 4 Aug 2026): at
        # C=32 on bpe_text8/V=1024 first order, equal spacing costs
        # 0.170 bits/token of staleness, geometric 0.032.
        first = 2048
        m_ = n - 2
        lo_, hi_ = 1.0, 4.0
        for _ in range(200):
            r_ = (lo_ + hi_) / 2
            tot = first * (C if abs(r_ - 1) < 1e-12
                           else (r_**C - 1) / (r_ - 1))
            if tot < m_:
                lo_ = r_
            else:
                hi_ = r_
        r_ = (lo_ + hi_) / 2
        e_ = [2]
        acc = 0.0
        for k_ in range(C):
            acc += first * r_**k_
            e_.append(min(n, 2 + round(acc)))
        e_[-1] = n
        edges = np.asarray(e_, dtype=np.int64)
    coded = 0
    diagnostics = []
    pool = (ThreadPoolExecutor(max_workers=args.jobs)
            if args.jobs > 1 else None)
    scorer = ThreadPoolExecutor(max_workers=1)
    pending = None
    CH = max(1, (1 << 23) // V)               # scoring chunk length
    phase_seconds = {"tables": 0.0, "ipf": 0.0, "score": 0.0,
                     "layered": 0.0, "reveal": 0.0}

    # ---- fit/score pipeline.  --ipf-lag k warm-starts each fit from
    # the fit k checkpoints back, splitting the single fit chain into
    # k independent chains that run concurrently (Ruediger, 4 Aug).
    # lag 1 is the classical single chain.  Scoring always merges in
    # checkpoint order.  The layered member forces the strictly
    # serial path.
    lag = max(1, int(args.ipf_lag))
    if pool is None or "markov2-layered" in bits:
        lag = 1
    sync_mode = (pool is None) or ("markov2-layered" in bits)
    fitpool = ThreadPoolExecutor(max_workers=lag)
    fit_futs = []                 # submission-indexed fit futures
    outq = []                     # (c, coded_after, score_fut, diag)
    WIN = max(2, lag + 1)         # bound on unmerged checkpoints

    def _fit_task(P01_, P02_, warm_fut):
        t_ = time.time()
        if warm_fut is None:
            w01 = np.ones((V, V))
            w02 = np.ones((V, V))
            w12 = np.ones((V, V))
        else:
            w01, w02, w12 = warm_fut.result()[0]
        r01, r02, r12, sweeps_, resid_ = ipf_triangle(
            P01_, P02_, P01_, w01, w02, w12,
            args.ipf_iters, args.ipf_tol, ex=None, jobs=1,
            solver=args.ipf_solver)
        return ((r01, r02, r12), np.log(r01), np.log(r02),
                sweeps_, resid_, time.time() - t_)

    def _merge_head():
        c_, coded_, sfut_, diag_ = outq.pop(0)
        tot_, sweeps_, resid_, ipf_dt = sfut_.result()
        for k2, v2 in tot_.items():
            bits[k2] -= v2
        phase_seconds["ipf"] += ipf_dt
        diag_["ipf_sweeps"] = sweeps_
        diag_["ipf_residual_l1"] = resid_
        diagnostics.append(diag_)
        print(f"  checkpoint {c_ + 1}/{C}: coded {coded_:,} "
              f"(ipf {sweeps_} sweeps, resid {resid_:.1e}) "
              f"({time.time() - t0:.0f}s; "
              + " ".join(f"{k_}={v_:.0f}s"
                         for k_, v_ in phase_seconds.items()) + ")",
              flush=True)

    for c in range(C):
        tblk = time.time()
        lo, hi = int(edges[c]), int(edges[c + 1])
        if hi <= lo:
            continue
        # ---- freeze tables on the prefix strictly before this block
        tot = int(uni.sum())
        if args.tables == "layered":
            lq0, (T1, T2) = _layered_log_tables(
                builder, uni.astype(float),
                [N1.astype(float), N2.astype(float)])
            m = np.exp2(lq0)
            p1 = np.exp2(T1)
            p2 = np.exp2(T2)
        else:
            m = (uni + 1.0) / (tot + V)
            p1 = smoothed_conditional(N1, m, s)
            p2 = smoothed_conditional(N2, m, s)
        L1, L2, Lm = np.log(p1), np.log(p2), np.log(m)
        # joints built FROM the smoothed conditionals, so every
        # scheme consumes the same estimate; the per-cell eps joint
        # was effectively unsmoothed (eps V^2 ~ 4 pseudo-tokens
        # against the conditionals' s V) and the calibrated fit
        # overfitted rare pairs, scoring below lag1 --- which its
        # strong-discount limit forbids
        J1 = m[:, None] * p1
        J2 = m[:, None] * p2
        # roles: P01[x0,x1] joint of (target, lag1); P12 is the same
        # table with roles (x_{t-1}, x_{t-2}); P02 from the lag-2 pairs
        P01 = project_margins(J1.T, m)
        P02 = project_margins(J2.T, m)
        phase_seconds["tables"] += time.time() - tblk

        # ---- dispatch the fit, chained to its ancestor 'lag' back
        warm = (fit_futs[len(fit_futs) - lag]
                if len(fit_futs) >= lag else None)
        ffut = fitpool.submit(_fit_task, P01, P02, warm)
        fit_futs.append(ffut)

        mi01, mi02 = mutual_information(J1), mutual_information(J2)
        diag0 = {"checkpoint": c + 1,
                 "mi_lag1": mi01, "mi_lag2": mi02,
                 "chow_liu": "lag1-chain" if mi01 >= mi02
                             else "lag2-edge"}
        if Ctri.nnz or tri_c:
            rowsum = np.asarray(Ctri.sum(axis=1)).ravel()
        else:
            rowsum = np.zeros(V * V)

        # ---- data-side snapshot for the scorer
        tph = time.time()
        tb_all = np.arange(lo, hi)
        cid_all = x[tb_all - 1] * V + x[tb_all - 2]
        ncb_all = (np.asarray(Ctri[cid_all, x[tb_all]]).ravel()
                   if "markov2" in bits else None)
        snap = dict(p1=p1, p2=p2, m=m, L1=L1, L2=L2, Lm=Lm,
                    rowsum=rowsum, cid_all=cid_all, ncb_all=ncb_all,
                    lo=lo, hi=hi)

        def _score_chunk(sn, j0, j1):
            local = {}
            t = np.arange(j0, j1)
            b, a1, a2 = x[t], x[t - 1], x[t - 2]
            r1b = sn["p1"][a1, b]
            r2b = sn["p2"][a2, b]
            mb = sn["m"][b]
            local["lag1"] = np.log2(r1b).sum()
            cid = sn["cid_all"][j0 - sn["lo"]:j1 - sn["lo"]]
            if "markov2" in bits:
                ncb = sn["ncb_all"][j0 - sn["lo"]:j1 - sn["lo"]]
                nc = sn["rowsum"][cid]
                pm2 = (ncb + s * r1b) / (nc + s)
                local["markov2"] = np.log2(pm2).sum()
            for w0, w2 in mix_grid:
                w1 = 1.0 - w0 - w2
                p = w0 * mb + w1 * r1b + w2 * r2b
                local[f"mix:{w0:g},{w2:g}"] = np.log2(p).sum()
            R1 = sn["L1"][a1] - sn["Lm"][None, :]
            R2 = sn["L2"][a2] - sn["Lm"][None, :]
            for b1, b2 in prod_grid:
                logits = sn["Lm"][None, :] + b1 * R1 + b2 * R2
                mx = logits.max(axis=1, keepdims=True)
                lz = mx[:, 0] + np.log(
                    np.exp(logits - mx).sum(axis=1))
                sel = logits[np.arange(len(t)), b]
                local[f"prod:{b1:g},{b2:g}"] = (
                    (sel - lz) / math.log(2)).sum()
            logits = sn["K01"][:, a1].T + sn["K02"][:, a2].T
            mx = logits.max(axis=1, keepdims=True)
            lz = mx[:, 0] + np.log(np.exp(logits - mx).sum(axis=1))
            sel = logits[np.arange(len(t)), b]
            local["calibrated"] = ((sel - lz) / math.log(2)).sum()
            return local

        def _score_block(sn):
            tb = time.time()
            rng = [(j0, min(j0 + CH, sn["hi"]))
                   for j0 in range(sn["lo"], sn["hi"], CH)]
            if pool is not None:
                parts = list(pool.map(
                    lambda ab: _score_chunk(sn, *ab), rng))
            else:
                parts = [_score_chunk(sn, *ab) for ab in rng]
            tot_ = {}
            for loc in parts:
                for k2, v2 in loc.items():
                    tot_[k2] = tot_.get(k2, 0.0) + v2
            phase_seconds["score"] += time.time() - tb
            return tot_

        def _score_after_fit(sn, ff):
            _, K1_, K2_, sweeps_, resid_, ipf_dt = ff.result()
            sn = dict(sn)
            sn["K01"] = K1_
            sn["K02"] = K2_
            return _score_block(sn), sweeps_, resid_, ipf_dt

        coded += hi - lo
        outq.append((c, coded,
                     scorer.submit(_score_after_fit, snap, ffut),
                     diag0))
        if sync_mode:
            _merge_head()
        else:
            while len(outq) >= WIN:
                _merge_head()

        tph = time.time()
        if "markov2-layered" in bits:
            ranges = [(j0, min(j0 + CH, hi))
                      for j0 in range(lo, hi, CH)]
            for j0, j1 in ranges:
                t = np.arange(j0, j1)
                b, a1, a2 = x[t], x[t - 1], x[t - 2]
                cid = a1 * V + a2
                ncb = np.asarray(Ctri[cid, b]).ravel()
                # share-nothing layered estimator per context pair:
                # p(b|c) = q(profile_c + one more b) / q(profile_c),
                # one evaluation per distinct (profile, count) pair,
                # memoized across blocks and checkpoints
                uc, inv = np.unique(cid, return_inverse=True)
                bases = []
                for u in uc:
                    d0, d1 = Ctri.indptr[u], Ctri.indptr[u + 1]
                    bases.append(tuple(sorted(
                        int(v) for v in Ctri.data[d0:d1])))
                key = inv * (int(ncb.max()) + 2) + ncb
                upair, pinv = np.unique(key, return_inverse=True)
                fams: dict = {}
                need = []
                for kk in upair:
                    bi, cv = int(kk) // (int(ncb.max()) + 2), \
                             int(kk) % (int(ncb.max()) + 2)
                    base = bases[bi]
                    need.append((base, cv))
                    if (base, cv) not in ratio_memo:
                        fams.setdefault(base, set()).add(cv)
                if fams:
                    builder._ensure_families(
                        {bb: tuple(sorted(cs)) for bb, cs in fams.items()})
                    for bb, cs in fams.items():
                        for cv in cs:
                            ratio_memo[(bb, cv)] = builder._log2_ratio(
                                bb, _augmented_profile(bb, cv))
                lut = np.array([ratio_memo[k2] for k2 in need])
                bits["markov2-layered"] -= lut[pinv].sum()
        phase_seconds["layered"] += time.time() - tph

        # ---- reveal the block to the counters (safe alongside the
        # background fits and scorer: they only read their snapshots)
        tph = time.time()
        t = np.arange(lo, hi)
        N1 += np.bincount(x[t - 1].astype(np.int64) * V + x[t],
                          minlength=V * V).reshape(V, V)
        N2 += np.bincount(x[t - 2].astype(np.int64) * V + x[t],
                          minlength=V * V).reshape(V, V)
        uni += np.bincount(x[t], minlength=V)
        if c == 0:                       # tokens before the first target
            uni += np.bincount(x[:2], minlength=V)
        tri_c.append(x[t - 1] * V + x[t - 2])
        tri_b.append(x[t])
        new = sparse.coo_matrix(
            (np.ones(hi - lo, dtype=np.int64),
             (tri_c[-1], tri_b[-1])), shape=(V * V, V)).tocsr()
        Ctri = (Ctri + new).tocsr()
        phase_seconds["reveal"] += time.time() - tph

    while outq:
        _merge_head()


    if args.exact == "add":
        tx = time.time()
        bits["markov2-layered-exact"] = exact_order2_bits(builder, Ctri)
        names.append("markov2-layered-exact")
        print(f"  exact order-2 layered: "
              f"{bits['markov2-layered-exact'] / coded:.4f} bits/token "
              f"({time.time() - tx:.0f}s)", flush=True)

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
        "phase_seconds": {k_: round(v_, 1)
                          for k_, v_ in phase_seconds.items()},
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
