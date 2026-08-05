#!/usr/bin/env python3
"""Score established product/Markov baselines on persisted calibration blocks."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from product_model_with_memory.graphical_calibration import (
    SparseGroupedProblem,
    SparseGroupedResult,
    sparse_gated_log_probabilities,
)
from product_model_with_memory.streams import load_stream, reduce_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--ids", default="output/streams/bpe_text8")
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--beta1", type=float, default=0.75)
    parser.add_argument("--beta2", type=float, default=0.5)
    parser.add_argument("--smooth", type=float, default=64.0)
    parser.add_argument("--mixture-grid", default="0,.25,.5,.75,1",
                        help="arithmetic weight on calibrated predictor")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run = Path(args.run)
    summary = json.loads((run / "results.json").read_text())
    states = sorted((run / "states").glob("checkpoint_*.npz"))
    ids, _ = load_stream(args.ids)
    x, _, _ = reduce_ids(ids[:summary["n"]], args.top_k)
    x = x.astype(np.int64)
    prefixes = [int(np.load(path)["prefix"]) for path in states]

    triple: dict[tuple[int, int, int], int] = defaultdict(int)
    context_total: dict[tuple[int, int], int] = defaultdict(int)
    revealed = 2
    product_bits = 0.0
    markov_bits = 0.0
    mixture_grid = [float(value) for value in args.mixture_grid.split(",")]
    mixture_bits = {weight: 0.0 for weight in mixture_grid}
    records = 0
    rows = []
    for index in range(len(states) - 1):
        lo, hi = prefixes[index], prefixes[index + 1]
        for t in range(revealed, lo):
            key = (int(x[t - 1]), int(x[t - 2]))
            context_total[key] += 1
            triple[(key[0], key[1], int(x[t]))] += 1
        revealed = lo

        data = np.load(states[index])
        p_ya = data["fallback_ya"]
        p_yb = data["fallback_yb"]
        py = p_ya.sum(axis=1)
        pa = p_ya.sum(axis=0)
        pb = p_yb.sum(axis=0)
        log_py = np.log(py)
        log_p1 = np.log(p_ya) - np.log(pa)[None, :]
        log_p2 = np.log(p_yb) - np.log(pb)[None, :]
        block_product = 0.0
        block_markov = 0.0
        product_logp = np.empty(hi - lo)
        for position, t in enumerate(range(lo, hi)):
            y, a, b = int(x[t]), int(x[t - 1]), int(x[t - 2])
            logits = (
                log_py
                + args.beta1 * (log_p1[:, a] - log_py)
                + args.beta2 * (log_p2[:, b] - log_py)
            )
            product_logp[position] = logits[y] - logsumexp(logits)
            block_product -= float(product_logp[position])
            backoff = p_ya[y, a] / pa[a]
            total = context_total[(a, b)]
            probability = (
                triple[(a, b, y)] + args.smooth * backoff
            ) / (total + args.smooth)
            block_markov -= float(np.log(probability))
        problem = SparseGroupedProblem(
            vocabulary_size=len(data["target_y"]),
            edge_a=data["edge_a"], edge_b=data["edge_b"],
            edge_probability=data["edge_probability"],
            target_y=data["target_y"],
            active_ya_y=data["active_ya_y"],
            active_ya_a=data["active_ya_a"], target_ya=data["target_ya"],
            active_yb_y=data["active_yb_y"],
            active_yb_b=data["active_yb_b"], target_yb=data["target_yb"],
        )
        calibrated = SparseGroupedResult(
            log_base_y=data["log_base_y"],
            correction_ya=data["correction_ya"],
            correction_yb=data["correction_yb"],
            iterations=0, grouped_residual_ya_l1=np.nan,
            grouped_residual_yb_l1=np.nan, residual_y_l1=np.nan,
            converged=True,
        )
        target = x[lo:hi]
        lag1 = x[lo - 1:hi - 1]
        lag2 = x[lo - 2:hi - 2]
        calibrated_logp = sparse_gated_log_probabilities(
            problem, calibrated, target, lag1, lag2, p_ya, p_yb
        )
        for weight in mixture_grid:
            if weight == 0.0:
                mixed = product_logp
            elif weight == 1.0:
                mixed = calibrated_logp
            else:
                mixed = np.logaddexp(
                    np.log1p(-weight) + product_logp,
                    np.log(weight) + calibrated_logp,
                )
            mixture_bits[weight] -= float(mixed.sum()) / np.log(2.0)
        scale = 1.0 / np.log(2.0)
        product_bits += block_product * scale
        markov_bits += block_markov * scale
        records += hi - lo
        rows.append({
            "fit_prefix": lo,
            "scored_records": hi - lo,
            "product_bpc": block_product * scale / (hi - lo),
            "markov2_bpc": block_markov * scale / (hi - lo),
        })
    payload = {
        "source": str(run),
        "scored_records": records,
        "product_beta1": args.beta1,
        "product_beta2": args.beta2,
        "smooth": args.smooth,
        "product_bpc": product_bits / records,
        "markov2_bpc": markov_bits / records,
        "calibrated_product_mixture_bpc": {
            str(weight): bits / records
            for weight, bits in mixture_bits.items()
        },
        "calibrated_product_family_bpc": (
            min(mixture_bits.values())
            - np.log2(sum(
                2.0 ** (-(bits - min(mixture_bits.values())))
                for bits in mixture_bits.values()
            ))
            + np.log2(len(mixture_bits))
        ) / records,
        "rows": rows,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
