#!/usr/bin/env python3
"""Compare eager and bounded-lazy stochastic block preparation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intersection_topology_audit import load_problem
from validate_layered_checkpoint_store import reorder_values
from product_model_with_memory.graphical_calibration import (
    BirthMajorSparseSupport,
    SparseGroupedProblem,
    checkpoint_in_birth_major_support,
    build_ab_major_intersection_graph,
    load_layered_intersection_graph,
    load_ab_major_intersection_graph,
    pair_product_warm_start,
    save_ab_major_intersection_graph,
    stochastic_sparse_dual_approach,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--factors")
    parser.add_argument(
        "--warm", choices=("factors", "pair_product"), default="factors"
    )
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=128)
    parser.add_argument("--cache", type=int, default=8)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--replicas", type=int, default=12)
    parser.add_argument(
        "--fused-ab-batch", action="store_true",
        help="use experimental indexed fusion of the fixed replica batch",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exact-interval", type=int, default=50)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument(
        "--modes", default="eager,lazy,ab_major",
        help="comma-separated subset of eager,lazy,ab_major",
    )
    args = parser.parse_args()

    store = Path(args.store)
    manifest = json.loads((store / "manifest.json").read_text())
    support_dir = store / "support"
    load = lambda name: np.load(  # noqa: E731
        support_dir / f"{name}.npy", mmap_mode="r"
    )
    final = SparseGroupedProblem(
        vocabulary_size=int(manifest["vocabulary_size"]),
        edge_a=load("edge_a"), edge_b=load("edge_b"),
        edge_probability=np.zeros(len(load("edge_a"))),
        target_y=load("target_y"),
        active_ya_y=load("active_ya_y"),
        active_ya_a=load("active_ya_a"),
        target_ya=np.zeros(len(load("active_ya_y"))),
        active_yb_y=load("active_yb_y"),
        active_yb_b=load("active_yb_b"),
        target_yb=np.zeros(len(load("active_yb_y"))),
    )
    support = BirthMajorSparseSupport(
        final, load("birth_ya"), load("birth_yb"), load("birth_ab")
    )
    state_name = f"checkpoint_{args.checkpoint:03d}.npz"
    original = load_problem(Path(args.problems) / "states" / state_name)
    problem = checkpoint_in_birth_major_support(
        original, support, args.checkpoint
    )
    if args.warm == "pair_product":
        log_base, c1, c2 = pair_product_warm_start(problem)
    else:
        if not args.factors:
            parser.error("--factors is required for a factors warm start")
        with np.load(Path(args.factors) / "states" / state_name) as state:
            log_base = state["log_base_y"]
            c1 = reorder_values(
                original.active_ya_y, original.active_ya_a,
                state["correction_ya"], problem.active_ya_y,
                problem.active_ya_a, problem.vocabulary_size,
            )
            c2 = reorder_values(
                original.active_yb_y, original.active_yb_b,
                state["correction_yb"], problem.active_yb_y,
                problem.active_yb_b, problem.vocabulary_size,
            )
    graph = load_layered_intersection_graph(store / "graph")
    ab_path = store / "ab_graph"
    if (ab_path / "manifest.json").exists():
        ab_graph = load_ab_major_intersection_graph(ab_path)
    else:
        ab_graph = build_ab_major_intersection_graph(
            final, support.birth_ya, support.birth_yb, support.birth_ab
        )
        save_ab_major_intersection_graph(ab_graph, ab_path)
        ab_graph = load_ab_major_intersection_graph(ab_path)

    available_modes = {
        name: (layered, direct_ab) for name, layered, direct_ab in (
        ("eager", False, False),
        ("lazy", True, False),
        ("ab_major", True, True),
        )
    }
    requested_modes = [name.strip() for name in args.modes.split(",")]
    unknown_modes = set(requested_modes) - set(available_modes)
    if unknown_modes:
        parser.error(f"unknown --modes value(s): {sorted(unknown_modes)}")
    rows = []
    for name in requested_modes:
        layered, direct_ab = available_modes[name]
        started = time.perf_counter()
        result = stochastic_sparse_dual_approach(
            problem, log_base, c1, c2,
            steps=args.steps, batch_size=1,
            sampling="blocks", edge_blocks=args.blocks,
            replicas=args.replicas, stochastic_workers=args.workers,
            variance_reduction=True, exact_interval=args.exact_interval,
            exact_margin_workers=args.workers,
            certificate_tolerance=args.tolerance,
            optimizer="adam_plateau",
            seed=args.seed,
            exact_layered_graph=graph if layered else None,
            exact_layered_checkpoint=args.checkpoint if layered else None,
            sampled_ab_major_graph=ab_graph if direct_ab else None,
            fused_ab_batch=args.fused_ab_batch,
            lazy_block_cache=args.cache,
        )
        rows.append({
            "mode": name,
            "seconds": time.perf_counter() - started,
            "steps": result.steps,
            "certificate": result.best_exact_certificate,
            "intersection_plan_bytes": result.intersection_plan_bytes,
            "reference_cache_bytes": result.reference_cache_bytes,
            "reference_cache_seconds": result.reference_cache_seconds,
            "sampled_phase_timing": result.sampled_phase_timing,
            "sampled_gradient_seconds": result.sampled_gradient_seconds,
            "optimizer_seconds": result.optimizer_seconds,
            "exact_seconds": result.exact_seconds,
        })
        print(json.dumps(rows[-1]), flush=True)
    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
