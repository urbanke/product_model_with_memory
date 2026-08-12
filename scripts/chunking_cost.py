#!/usr/bin/env python3
"""How much does chunked (checkpointed) evaluation lose vs the exact
sequential code, for schemes whose exact value telescopes?

The exact code updates its tables after every token; its codelength
is one exchangeable evaluation per state at the final counts.  The
chunked code freezes tables at C checkpoints and codes each block
with stale tables.  Both are complete honest codes; the difference
is the price of chunking, which this script measures for memory 0
and memory 1 as a function of C and of the checkpoint SPACING
(equal blocks, or geometric blocks that are small early when the
tables learn fastest).

    python scripts/chunking_cost.py --ids output/streams/bpe_text8 \
        --order 1 --checkpoints 8,32,128 --spacing equal,geo \
        --out output/chunkcost_text8_o1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from product_model_with_memory.streams import load_stream, reduce_ids
from product_model_with_memory.production_coding import (
    PRODUCTION_SEQUENCE_ESTIMATOR,
)


def edges_for(n: int, C: int, spacing: str, first: int = 2048):
    """Block boundaries 0 = t_0 < t_1 < ... < t_C = n."""
    if spacing == "equal":
        return [round(n * k / C) for k in range(C + 1)]
    if spacing == "equal_suffix":
        if C < 2:
            raise ValueError("equal_suffix needs an initial block and a suffix")
        if not 0 < first < n:
            raise ValueError("the fixed first checkpoint must lie inside the stream")
        # Hold the initial-prefix convention fixed, then divide only the
        # actually scored suffix into equal intervals.  This is the fair
        # comparator for production's geometric schedule: plain `equal`
        # silently changes the first checkpoint as C changes.
        return [0, first] + [
            round(first + (n - first) * k / (C - 1))
            for k in range(1, C)
        ]
    if spacing not in {"geo"}:
        raise ValueError(f"unknown checkpoint spacing {spacing!r}")
    # geometric: block k has length ~ first * r^k with r chosen so the
    # lengths sum to n; small blocks early, where the tables move most
    lo, hi = 1.0, 4.0
    for _ in range(200):
        r = (lo + hi) / 2
        tot = first * (C if abs(r - 1) < 1e-12 else (r**C - 1) / (r - 1))
        if tot < n:
            lo = r
        else:
            hi = r
    r = (lo + hi) / 2
    e = [0]
    acc = 0.0
    for k in range(C):
        acc += first * r**k
        e.append(min(n, round(acc)))
    e[-1] = n
    return e


def raw_predictive_log2_normalizer(
    vocabulary_size: int,
    counts,
    log2_probability_for_count,
) -> float:
    """Return log2 of the total mass of a count-symmetric predictor.

    ``counts`` contains one current count for every observed symbol.  Every
    unseen symbol shares the probability returned for count zero.  Keeping
    this helper independent of the layered evaluator makes the production
    row-normalization accounting directly testable.
    """

    observed = tuple(int(value) for value in counts)
    if vocabulary_size < len(observed):
        raise ValueError("more observed symbols than the vocabulary permits")
    terms = [float(log2_probability_for_count(value)) for value in observed]
    if len(observed) < vocabulary_size:
        terms.append(
            float(log2_probability_for_count(0))
            + math.log2(vocabulary_size - len(observed))
        )
    if not terms:
        raise ValueError("an empty vocabulary has no predictive distribution")
    values = np.asarray(terms, dtype=np.float64)
    maximum = float(values.max())
    return maximum + math.log2(float(np.exp2(values - maximum).sum()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--order", type=int, choices=(0, 1), required=True)
    ap.add_argument("--top-k", type=int, default=10**9,
                    help="cap the alphabet (smoke only; default: full)")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--checkpoints", default="8,32,128")
    ap.add_argument("--spacing", default="equal,geo")
    ap.add_argument(
        "--first-checkpoint", type=int, default=2_050,
        help=(
            "absolute first fitted/checkpoint prefix for geometric spacing; "
            "use 65536 to reproduce the scheduled production runs"
        ),
    )
    ap.add_argument(
        "--edges-manifest",
        help=(
            "checkpoint manifest whose absolute prefix edges replace the "
            "synthetic C/spacing grid; intended for exact replay audits"
        ),
    )
    ap.add_argument(
        "--start-position", type=int,
        help=(
            "first target position represented by the profiles (default: "
            "the Markov order); use 2 to match the two-lag pipeline"
        ),
    )
    ap.add_argument(
        "--exclude-first-block", action="store_true",
        help=(
            "reveal the first block as training data but report a separate "
            "scored total only for subsequent blocks"
        ),
    )
    ap.add_argument(
        "--skip-exact", action="store_true",
        help="skip the telescoping full-stream reference (audit speed-up)",
    )
    ap.add_argument(
        "--score-blocks",
        help=(
            "comma-separated zero-based blocks to evaluate; earlier blocks "
            "are still revealed so every selected block has its exact "
            "historical prefix"
        ),
    )
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # A diagnostic comparison must use the exact table family used by the
    # production encoder.  Otherwise it silently compares two estimators.
    for key, value in (
        ("PMM_UNIVERSAL_TABLES", "tables/anchors_prod"),
        ("PMM_PHI_LADDER_EVERY", "1"),
        ("PMM_PHI_LADDER_DEGREE", "11"),
        ("PMM_PHI_SADDLE_MIN_L", "54"),
    ):
        os.environ.setdefault(key, value)

    t0 = time.time()
    ids, meta = load_stream(args.ids)
    if args.n:
        ids = ids[: args.n]
    if args.top_k < 10**9:
        x, V, _ = reduce_ids(ids, args.top_k)
    else:
        x = ids.astype(np.int64)
        V = int(x.max()) + 1
    n = len(x)
    print(f"V={V} n={n:,} order={args.order}", flush=True)

    from product_model_with_memory.pooled_lags import (
        _LayeredPredictiveBuilder, _augmented_profile, _log2sumexp_arr)
    from product_model_with_memory.codelength import default_l_max
    builder = _LayeredPredictiveBuilder(
        V, default_l_max(V), None, args.jobs, None)

    # state of each position: () for order 0, previous token for 1
    off = args.order if args.start_position is None else args.start_position
    if not args.order <= off < n:
        ap.error("start-position must lie between the Markov order and n-1")
    pos = np.arange(off, n)
    st = (np.zeros(len(pos), dtype=np.int64) if args.order == 0
          else x[pos - 1])
    sym = x[pos]
    key_all = st * V + sym

    from collections import Counter
    # ---- exact reference: one evaluation per state at final counts
    bits_exact = None
    if not args.skip_exact:
        order = np.argsort(st, kind="stable")
        st_s, sym_s = st[order], sym[order]
        mult: Counter = Counter()
        i = 0
        while i < len(st_s):
            j = i
            while j < len(st_s) and st_s[j] == st_s[i]:
                j += 1
            cnt = np.bincount(sym_s[i:j])
            mult[tuple(sorted(int(c) for c in cnt[cnt > 0]))] += 1
            i = j
        need = [p for p in mult if p not in builder.memo]
        print(f"  exact: {len(mult)} profiles, {len(need)} to evaluate",
              flush=True)
        B = 2000
        for k in range(0, len(need), B):
            builder._ensure_families({p: () for p in need[k:k + B]})
        ll = math.log2(builder.l_max)
        bits_exact = -sum(m * (_log2sumexp_arr(builder.memo[p]) - ll)
                          for p, m in mult.items())
        print(f"  exact: {bits_exact / len(pos):.4f} bits/token "
              f"({time.time()-t0:.0f}s)", flush=True)
    else:
        print("  exact: skipped", flush=True)

    # ---- chunked runs
    ratio_memo: dict = {}

    def log2_q_avg(profile: tuple) -> float:
        """Log2 of the depth-averaged joint law for one count profile."""

        if not profile:
            return 0.0
        if profile not in builder.memo:
            builder._ensure_families({profile: ()})
        return _log2sumexp_arr(builder.memo[profile]) - math.log2(builder.l_max)

    def exact_prefix_bits(profiles: dict) -> float:
        """Canonical sequential layered cost represented by row profiles.

        The Markov-1 code is a product of one exchangeable layered law per
        context row.  Consequently this quantity is the exact cost of the
        complete transition prefix and differences at two boundaries are the
        exact sequential cost of the intervening interval.
        """

        multiplicity: Counter = Counter(
            tuple(sorted(int(value) for value in counts.values()))
            for counts in profiles.values()
            if counts
        )
        missing = [profile for profile in multiplicity
                   if profile not in builder.memo]
        if missing:
            builder._ensure_families({profile: () for profile in missing})
        return -float(sum(
            count * log2_q_avg(profile)
            for profile, count in multiplicity.items()
        ))

    def ratio(base: tuple, count: int) -> float:
        """Memoized raw posterior-mixture predictive log2 ratio."""

        key = (base, count)
        value = ratio_memo.get(key)
        if value is None:
            value = builder._log2_ratio(
                base, _augmented_profile(base, count)
            )
            ratio_memo[key] = value
        return value

    def raw_row_log2_normalizer(base: tuple, counts: Counter) -> float:
        """Normalizer of the numerically evaluated raw q_avg ratios.

        It is exactly zero in exact arithmetic.  Production normalizes each
        row explicitly, so charging this term separately distinguishes
        numerical quadrature/table effects from checkpoint staleness.
        All required augmented profiles must already have been ensured.
        """

        return raw_predictive_log2_normalizer(
            V, counts.values(), lambda count: ratio(base, count)
        )

    results = {"ids": args.ids, "order": args.order, "V": V,
               "n_tokens": n, "coded_positions": len(pos),
               "sequence_estimator": PRODUCTION_SEQUENCE_ESTIMATOR,
               "l_max": builder.l_max,
               "first_checkpoint": args.first_checkpoint,
               "bits_exact": bits_exact,
               "exact_bits_per_token": (
                   bits_exact / len(pos) if bits_exact is not None else None
               ),
               "runs": []}
    selected_blocks = None
    if args.score_blocks:
        selected_blocks = {int(value) for value in args.score_blocks.split(",")}
    if args.edges_manifest:
        checkpoint_manifest = json.loads(
            Path(args.edges_manifest).read_text()
        )
        absolute_edges = [int(value) for value in checkpoint_manifest["edges"]]
        e = [0] + [value - off for value in absolute_edges]
        if e[0] != 0 or any(b <= a for a, b in zip(e, e[1:])):
            ap.error("manifest checkpoint edges are not strictly increasing")
        if e[-1] != len(pos):
            ap.error("manifest final edge does not match the selected stream")
        configurations = [(len(e) - 1, "manifest", e)]
    else:
        configurations = [
            (
                C,
                spacing,
                edges_for(
                    len(pos), C, spacing,
                    first=max(1, args.first_checkpoint - off),
                ),
            )
            for C in [int(c) for c in args.checkpoints.split(",")]
            for spacing in args.spacing.split(",")
        ]
    for C, spacing, e in configurations:
        if selected_blocks is not None and (
            min(selected_blocks, default=0) < 0
            or max(selected_blocks, default=0) >= C
        ):
            ap.error("score-blocks contains a block outside the run")
        tR = time.time()
        raw_bits = 0.0
        normalized_bits = 0.0
        block_bits_raw = []
        block_bits = []
        block_normalization_bits = []
        # Cost of the exact sequential code at every block boundary.  Its
        # successive differences are the interval oracle proposed in the
        # audit discussion.
        oracle_prefix_bits = [0.0]
        # running per-state profiles as dict state -> Counter
        prof: dict = {}
        for b in range(C):
            lo, hi = e[b], e[b + 1]
            if hi <= lo:
                continue
            blk = slice(lo, hi)
            kb, cb = np.unique(key_all[blk], return_counts=True)
            evaluate = selected_blocks is None or b in selected_blocks

            # PASS 1: group by state.  Besides the counts actually queried
            # by this interval, ensure every distinct count in the frozen
            # row (plus the unseen case).  That is what is needed to compute
            # the production row normalizer exactly.
            todo = []          # (base, current symbol count, multiplicity)
            state_rows = []    # (base, frozen Counter, block observations)
            fams: dict = {}    # base -> augmented count values to evaluate
            i2 = 0
            while i2 < len(kb):
                s_ = int(kb[i2]) // V
                j2 = i2
                while j2 < len(kb) and int(kb[j2]) // V == s_:
                    j2 += 1
                frozen = prof.get(s_, Counter())
                base = tuple(sorted(int(value) for value in frozen.values()))
                needed = set(base)
                if len(frozen) < V:
                    needed.add(0)
                fams.setdefault(base, set()).update(needed)
                observations = 0
                for t2 in range(i2, j2):
                    y_ = int(kb[t2]) % V
                    cv = int(frozen.get(y_, 0))
                    cc = int(cb[t2])
                    observations += cc
                    if evaluate:
                        todo.append((base, cv, cc))
                    fams.setdefault(base, set()).add(cv)
                if evaluate:
                    state_rows.append((base, frozen, observations))
                i2 = j2

            if fams:
                builder._ensure_families({
                    base: tuple(sorted(counts))
                    for base, counts in fams.items()
                })

            if evaluate:
                raw_block = -float(sum(
                    cc * ratio(base, cv) for base, cv, cc in todo
                ))
                normalization_block = float(sum(
                    observations * raw_row_log2_normalizer(base, frozen)
                    for base, frozen, observations in state_rows
                ))
                normalized_block = raw_block + normalization_block
                raw_bits += raw_block
                normalized_bits += normalized_block
                block_bits_raw.append(raw_block)
                block_normalization_bits.append(normalization_block)
                block_bits.append(normalized_block)
            else:
                block_bits_raw.append(None)
                block_normalization_bits.append(None)
                block_bits.append(None)

            # Reveal the block only after it has been scored with the frozen
            # checkpoint distribution, then evaluate the exact telescoping
            # prefix cost at the new boundary.
            for kk, cc in zip(kb, cb):
                s_, y_ = int(kk) // V, int(kk) % V
                prof.setdefault(s_, Counter())[y_] += int(cc)
            oracle_prefix_bits.append(exact_prefix_bits(prof))
            if C >= 16:
                if evaluate:
                    regret = normalized_block - (
                        oracle_prefix_bits[-1] - oracle_prefix_bits[-2]
                    )
                    print(
                        f"    block {b + 1:>2d}/{C}: "
                        f"{hi - lo:,} tokens, regret {regret / (hi - lo):+.6f} "
                        f"bit/token, normalization "
                        f"{normalization_block:+.3e} bits",
                        flush=True,
                    )
                else:
                    print(
                        f"    block {b + 1:>2d}/{C}: oracle only",
                        flush=True,
                    )

        oracle_block_bits = [
            right - left
            for left, right in zip(
                oracle_prefix_bits[:-1], oracle_prefix_bits[1:]
            )
        ]
        if len(oracle_block_bits) != C:
            raise RuntimeError("oracle boundary count differs from block count")
        oracle_final_difference = (
            oracle_prefix_bits[-1] - bits_exact
            if bits_exact is not None else None
        )
        if (oracle_final_difference is not None
                and abs(oracle_final_difference) > 1e-6):
            raise RuntimeError(
                "prefix oracle does not reproduce the final exact code: "
                f"{oracle_final_difference:+.6g} bits"
            )
        evaluated = [index for index, value in enumerate(block_bits)
                     if value is not None]
        gap = None
        if bits_exact is not None and len(evaluated) == C:
            gap = (normalized_bits - bits_exact) / len(pos)
        scored_from = 1 if args.exclude_first_block else 0
        scored_blocks = [index for index in evaluated if index >= scored_from]
        scored_bits = float(sum(block_bits[index] for index in scored_blocks))
        scored_oracle_bits = float(sum(
            oracle_block_bits[index] for index in scored_blocks
        ))
        scored_positions = int(sum(
            e[index + 1] - e[index] for index in scored_blocks
        ))
        gap_text = f"gap {gap:+.4f}" if gap is not None else "partial audit"
        print(f"  C={C:<4d} {spacing:>8s}: "
              f"{normalized_bits / max(scored_positions, 1):.4f} "
              f"evaluated bits/token  "
              f"{gap_text}  ({time.time()-tR:.0f}s)", flush=True)
        interval_rows = []
        total_interval_regret = float(sum(
            block_bits[index] - oracle_block_bits[index]
            for index in evaluated
        ))
        cumulative_regret = 0.0
        for index in range(C):
            actual = block_bits[index]
            raw = block_bits_raw[index]
            oracle = oracle_block_bits[index]
            width = e[index + 1] - e[index]
            regret = actual - oracle if actual is not None else None
            if regret is not None:
                cumulative_regret += regret
            interval_rows.append({
                "block": index,
                "target_start": off + e[index],
                "target_stop": off + e[index + 1],
                "tokens": width,
                "oracle_sequential_bits": oracle,
                "frozen_raw_ratio_bits": raw,
                "row_normalization_bits": block_normalization_bits[index],
                "frozen_production_bits": actual,
                "staleness_bits_before_normalization": (
                    raw - oracle if raw is not None else None
                ),
                "total_regret_bits": (
                    regret
                ),
                "total_regret_bits_per_token": (
                    regret / width
                    if actual is not None else None
                ),
                "fraction_of_evaluated_regret": (
                    regret / total_interval_regret
                    if regret is not None and total_interval_regret else None
                ),
                "cumulative_regret_bits": (
                    cumulative_regret if regret is not None else None
                ),
            })
        results["runs"].append(
            {"C": C, "spacing": spacing,
             "bits_per_token": (
                 normalized_bits / len(pos) if len(evaluated) == C else None
             ),
             "gap_per_token": gap,
             "block_edges": e,
             "target_position_edges": [off + value for value in e],
             "block_bits": block_bits,
             "block_bits_raw_ratio": block_bits_raw,
             "block_row_normalization_bits": block_normalization_bits,
             "oracle_prefix_bits": oracle_prefix_bits,
             "oracle_block_bits": oracle_block_bits,
             "oracle_final_difference_bits": oracle_final_difference,
             "total_row_normalization_bits": float(sum(
                 value for value in block_normalization_bits
                 if value is not None
             )),
             "total_interval_regret_bits": total_interval_regret,
             "intervals": interval_rows,
             "evaluated_blocks": evaluated,
             "scored_from_block": scored_from,
             "scored_positions": scored_positions,
             "scored_bits": scored_bits,
             "scored_oracle_bits": scored_oracle_bits,
             "scored_regret_bits": scored_bits - scored_oracle_bits,
             "scored_bits_per_token": (
                 scored_bits / scored_positions
                 if scored_positions else float("nan")
             )})
        # save after every configuration: a killed run keeps
        # everything already measured
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(results, indent=2))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"written: {out/'results.json'} ({time.time()-t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
