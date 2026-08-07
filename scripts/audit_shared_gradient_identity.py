#!/usr/bin/env python3
"""Audit exact and block-gradient identities on one saved checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
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
    load_ab_major_intersection_graph,
    load_layered_intersection_graph,
    sparse_edge_block_from_bounds,
    sparse_factorized_dual_evaluation,
    sparse_factorized_margins_ab_major,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--checkpoint", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=128)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--directions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()

    store = Path(args.store)
    manifest = json.loads((store / "manifest.json").read_text())
    support_dir = store / "support"

    def load(name):
        return np.load(
            support_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False
        )

    final = SparseGroupedProblem(
        int(manifest["vocabulary_size"]), load("edge_a"), load("edge_b"),
        np.zeros(len(load("edge_a"))), load("target_y"),
        load("active_ya_y"), load("active_ya_a"),
        np.zeros(len(load("active_ya_y"))), load("active_yb_y"),
        load("active_yb_b"), np.zeros(len(load("active_yb_y"))),
    )
    support = BirthMajorSparseSupport(
        final, load("birth_ya"), load("birth_yb"), load("birth_ab")
    )
    problem_path = (
        Path(args.problems) / "states"
        / f"checkpoint_{args.checkpoint:03d}.npz"
    )
    original = load_problem(problem_path)
    problem = checkpoint_in_birth_major_support(
        original, support, args.checkpoint
    )
    with np.load(args.state, allow_pickle=False) as state:
        factors = (
            np.asarray(state["log_base_y"]),
            reorder_values(
                original.active_ya_y, original.active_ya_a,
                state["correction_ya"], problem.active_ya_y,
                problem.active_ya_a, problem.vocabulary_size,
            ),
            reorder_values(
                original.active_yb_y, original.active_yb_b,
                state["correction_yb"], problem.active_yb_y,
                problem.active_yb_b, problem.vocabulary_size,
            ),
        )

    exact_graph = load_layered_intersection_graph(store / "graph")
    ab_graph = load_ab_major_intersection_graph(store / "ab_graph")

    def exact(values):
        return sparse_factorized_dual_evaluation(
            problem, *values, compute_certificate=True,
            layered_graph=exact_graph,
            layered_checkpoint=args.checkpoint,
            margin_workers=args.workers,
        )

    exact_evaluation = exact(factors)
    exact_gradient = exact_evaluation.gradient()

    edge_mass_a = np.bincount(
        problem.edge_a, weights=problem.edge_probability,
        minlength=problem.vocabulary_size,
    )
    edge_mass_b = np.bincount(
        problem.edge_b, weights=problem.edge_probability,
        minlength=problem.vocabulary_size,
    )
    target_mass_a = np.bincount(
        problem.active_ya_a, weights=problem.target_ya,
        minlength=problem.vocabulary_size,
    )
    target_mass_b = np.bincount(
        problem.active_yb_b, weights=problem.target_yb,
        minlength=problem.vocabulary_size,
    )
    count_a = np.bincount(
        problem.active_ya_a, minlength=problem.vocabulary_size
    )
    count_b = np.bincount(
        problem.active_yb_b, minlength=problem.vocabulary_size
    )
    saturated_a = count_a == problem.vocabulary_size
    saturated_b = count_b == problem.vocabulary_size
    saturated_error_a = target_mass_a[saturated_a] - edge_mass_a[saturated_a]
    saturated_error_b = target_mass_b[saturated_b] - edge_mass_b[saturated_b]
    inactive_mass_a = edge_mass_a - target_mass_a
    inactive_mass_b = edge_mass_b - target_mass_b

    active = np.r_[0, np.cumsum(
        ab_graph.birth <= args.checkpoint, dtype=np.int64
    )]
    ptr = ab_graph.edge_ptr[:len(problem.edge_probability) + 1]
    work = active[ptr[1:]] - active[ptr[:-1]] + 1
    cumulative = np.r_[0, np.cumsum(work, dtype=np.int64)]
    targets = np.linspace(0, cumulative[-1], args.blocks + 1)
    boundaries = np.unique(np.searchsorted(cumulative, targets))
    boundaries[0] = 0
    boundaries[-1] = len(problem.edge_probability)
    bounds = [
        (int(lo), int(hi)) for lo, hi in zip(boundaries[:-1], boundaries[1:])
        if hi > lo
    ]

    weighted_model = np.zeros_like(exact_gradient)
    masses = []
    for lo, hi in bounds:
        block = sparse_edge_block_from_bounds(
            problem, lo, hi, build_plan=False
        )
        margins = sparse_factorized_margins_ab_major(
            block.problem, ab_graph, args.checkpoint, lo, *factors,
            workers=1,
        )
        mass = block.probability_mass
        masses.append(mass)
        weighted_model[:problem.vocabulary_size] += mass * margins.target_y
        first = problem.vocabulary_size
        second = first + len(problem.target_ya)
        weighted_model[first:second] += mass * margins.active_ya
        weighted_model[second:] += mass * margins.active_yb
    target = np.concatenate([
        problem.target_y, problem.target_ya, problem.target_yb
    ])
    block_gradient = weighted_model - target
    difference = block_gradient - exact_gradient

    rng = np.random.default_rng(args.seed)
    directional = []
    for _ in range(args.directions):
        direction = rng.normal(size=len(exact_gradient))
        direction /= np.linalg.norm(direction)
        vector = np.concatenate(factors)
        first = problem.vocabulary_size
        second = first + len(problem.target_ya)
        unpack = lambda vector: (  # noqa: E731
            vector[:first], vector[first:second], vector[second:]
        )
        analytic = float(exact_gradient @ direction)
        directional.append({
            "analytic": analytic,
            "finite_differences": [
                {
                    "epsilon": epsilon,
                    "value": finite,
                    "absolute_error": abs(analytic - finite),
                }
                for epsilon in (1e-2, 1e-3, 1e-4)
                for finite in [(
                    exact(unpack(vector + epsilon * direction)).objective
                    - exact(unpack(vector - epsilon * direction)).objective
                ) / (2.0 * epsilon)]
            ],
        })

    vector = np.concatenate(factors)
    first = problem.vocabulary_size
    second = first + len(problem.target_ya)
    step_rows = []
    for step_size in (0.1, 1.0, 10.0, 100.0):
        candidate = vector - step_size * exact_gradient
        evaluation = exact((
            candidate[:first], candidate[first:second], candidate[second:]
        ))
        step_rows.append({
            "step_size": step_size,
            "objective": float(evaluation.objective),
            "objective_change": float(
                evaluation.objective - exact_evaluation.objective
            ),
            "certificate": float(evaluation.certificate),
        })

    adam_rows = []
    for learning_rate in (0.03, 0.003, 0.0003):
        # At Adam's first update m_hat=g and v_hat=g^2, so each active
        # coordinate moves by approximately the learning rate, independently
        # of the gradient magnitude.  This makes the Euclidean step grow as
        # sqrt(number of parameters) unless the rate is scaled accordingly.
        update = learning_rate * exact_gradient / (
            np.abs(exact_gradient) + 1e-8
        )
        candidate = vector - update
        candidate[:first] -= np.log(np.exp(candidate[:first]).sum())
        evaluation = exact((
            candidate[:first], candidate[first:second], candidate[second:]
        ))
        adam_rows.append({
            "learning_rate": learning_rate,
            "update_l2": float(np.linalg.norm(update)),
            "update_linf": float(np.max(np.abs(update))),
            "objective": float(evaluation.objective),
            "objective_change": float(
                evaluation.objective - exact_evaluation.objective
            ),
            "certificate": float(evaluation.certificate),
        })

    print(json.dumps({
        "checkpoint": args.checkpoint,
        "parameters": len(exact_gradient),
        "blocks": len(bounds),
        "block_mass_sum": float(sum(masses)),
        "exact_objective": float(exact_evaluation.objective),
        "exact_certificate": float(exact_evaluation.certificate),
        "exact_gradient_l2": float(np.linalg.norm(exact_gradient)),
        "block_gradient_l2": float(np.linalg.norm(block_gradient)),
        "gradient_difference_l1": float(np.sum(np.abs(difference))),
        "gradient_difference_linf": float(np.max(np.abs(difference))),
        "gradient_relative_l2": float(
            np.linalg.norm(difference)
            / max(np.linalg.norm(exact_gradient), np.finfo(float).tiny)
        ),
        "target_consistency": {
            "edge_mass_sum": float(problem.edge_probability.sum()),
            "target_y_sum": float(problem.target_y.sum()),
            "saturated_a_rows": int(saturated_a.sum()),
            "saturated_b_rows": int(saturated_b.sum()),
            "saturated_a_error_l1": float(
                np.abs(saturated_error_a).sum()
            ),
            "saturated_b_error_l1": float(
                np.abs(saturated_error_b).sum()
            ),
            "saturated_a_error_linf": float(
                np.max(np.abs(saturated_error_a), initial=0.0)
            ),
            "saturated_b_error_linf": float(
                np.max(np.abs(saturated_error_b), initial=0.0)
            ),
            "most_negative_inactive_a_mass": float(inactive_mass_a.min()),
            "most_negative_inactive_b_mass": float(inactive_mass_b.min()),
        },
        "directional_derivatives": directional,
        "negative_exact_gradient_steps": step_rows,
        "first_exact_adam_steps": adam_rows,
    }, indent=2))


if __name__ == "__main__":
    main()
