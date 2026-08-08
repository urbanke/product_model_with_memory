#!/usr/bin/env python3
"""Create a phase-oriented or pipelined U/A/B/C/G/F/E schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--checkpoints", type=int, default=3)
    parser.add_argument("--first-checkpoint", type=int, default=2050)
    parser.add_argument("--policy", choices=("phased", "pipeline"), required=True)
    parser.add_argument("--python", default=".venv/bin/python3")
    parser.add_argument("--maximum-workers", type=int, default=8)
    parser.add_argument("--construction-workers", type=int, default=2)
    parser.add_argument("--fitting-workers", type=int, default=4)
    parser.add_argument("--fitting-replicas", type=int, default=12)
    parser.add_argument(
        "--fitting-blocks", type=int, default=128,
        help="number of stochastic graph blocks",
    )
    parser.add_argument(
        "--fitting-block-cache", type=int, default=128,
        help="maximum number of lazy reference blocks retained",
    )
    parser.add_argument(
        "--fitting-exact-interval", type=int, default=50,
        help="updates between exact certificate/snapshot evaluations",
    )
    parser.add_argument("--evaluation-workers", type=int, default=1)
    parser.add_argument(
        "--fitting-steps", type=int, default=1_000,
        help="stochastic safety ceiling; plateau/certificate normally stops first",
    )
    parser.add_argument("--construction-only", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.checkpoints < 2:
        parser.error("at least two checkpoints are required")

    root = Path(args.root)
    problems = root / "problems"
    counts = root / "counts"
    stream = root / "reduced_stream"
    count_deltas = root / "count_deltas"
    pairs = root / "pairs"
    deltas = root / "graph_deltas"
    fitted = root / "fitted"
    scores = root / "scores"
    manifests = root / "job_manifests"
    jobs = []

    def add(
        job_id, kind, checkpoint, command, dependencies, outputs, workers,
        minimum_workers=1,
    ):
        jobs.append({
            "id": job_id,
            "type": kind,
            "checkpoint": checkpoint,
            "command": command,
            "dependencies": dependencies,
            "outputs": [str(path) for path in outputs],
            "completion_manifest": str(manifests / f"{job_id}.json"),
            "workers": workers,
            "minimum_workers": minimum_workers,
            "private_memory_bytes": 0,
        })

    add(
        "S", "S", -1,
        [args.python, "-u", "scripts/prepare_reduced_checkpoint_stream.py",
         "--ids", args.ids, "--top-k", str(args.top_k),
         "--n", str(args.n), "--checkpoints", str(args.checkpoints),
         "--first-checkpoint", str(args.first_checkpoint),
         "--out", str(stream)],
        [], [stream / "manifest.json"], 1,
    )
    for checkpoint in range(args.checkpoints):
        count_dir = counts / f"checkpoint_{checkpoint:03d}"
        delta_dir = count_deltas / f"checkpoint_{checkpoint:03d}"
        unigram_dir = pairs / f"checkpoint_{checkpoint:03d}" / "unigram"
        add(
            f"D{checkpoint}", "D", checkpoint,
            [args.python, "-u", "scripts/prepare_checkpoint_count_delta.py",
             "--stream", str(stream), "--checkpoint", str(checkpoint),
             "--out", str(delta_dir)],
            ["S"], [delta_dir / "manifest.json"], 1,
        )
        u_dependencies = [] if checkpoint == 0 else [f"U{checkpoint - 1}"]
        u_dependencies.append(f"D{checkpoint}")
        u_command = [
            args.python, "-u", "scripts/merge_checkpoint_counts.py",
            "--delta", str(delta_dir), "--out", str(count_dir),
        ]
        if checkpoint:
            u_command.extend([
                "--previous", str(counts / f"checkpoint_{checkpoint - 1:03d}")
            ])
        add(
            f"U{checkpoint}", "U", checkpoint,
            u_command,
            u_dependencies, [count_dir / "manifest.json"], 1,
        )
        add(
            f"M{checkpoint}", "M", checkpoint,
            [args.python, "-u", "scripts/estimate_checkpoint_unigram.py",
             "--counts", str(count_dir),
             "--jobs", str(args.construction_workers),
             "--out", str(unigram_dir)],
            [f"U{checkpoint}"], [unigram_dir / "manifest.json"],
            args.construction_workers,
        )
        for pair, label in (("ya", "A"), ("yb", "B")):
            pair_dir = pairs / f"checkpoint_{checkpoint:03d}" / pair
            add(
                f"{label}{checkpoint}", label, checkpoint,
                [args.python, "-u", "scripts/estimate_checkpoint_pair.py",
                 "--counts", str(count_dir), "--unigram", str(unigram_dir),
                 "--pair", pair,
                 "--jobs", str(args.construction_workers),
                 "--out", str(pair_dir)],
                [f"M{checkpoint}"], [pair_dir / "manifest.json"],
                args.construction_workers,
            )
        problem_path = problems / "states" / f"checkpoint_{checkpoint:03d}.npz"
        add(
            f"C{checkpoint}", "C", checkpoint,
            [args.python, "-u", "scripts/assemble_checkpoint_problem.py",
             "--counts", str(count_dir),
             "--ya", str(pairs / f"checkpoint_{checkpoint:03d}" / "ya"),
             "--yb", str(pairs / f"checkpoint_{checkpoint:03d}" / "yb"),
             "--out", str(problem_path)],
            [f"A{checkpoint}", f"B{checkpoint}"], [problem_path], 1,
        )
        if args.construction_only:
            continue
        g_dependencies = [f"C{checkpoint}"]
        if checkpoint:
            g_dependencies.append(f"G{checkpoint - 1}")
        delta_manifest = (
            deltas / "deltas" / f"checkpoint_{checkpoint:03d}"
            / "manifest.json"
        )
        add(
            f"G{checkpoint}", "G", checkpoint,
            [
                args.python, "-u", "scripts/prepare_checkpoint_graph.py",
                "--problems", str(problems),
                "--delta-store", str(deltas),
                "--checkpoint", str(checkpoint),
            ],
            g_dependencies,
            [delta_manifest], 1,
        )
        # Every checkpoint uses the same data-independent initialization
        # portfolio.  Fitting is therefore not a causal chain: once G_k is
        # available, F_k may run concurrently with every other ready fit.
        fit_dir = fitted / f"checkpoint_{checkpoint:03d}"
        fit_state = fit_dir / "states" / f"checkpoint_{checkpoint:03d}.npz"
        add(
            f"F{checkpoint}", "F", checkpoint,
            [
                args.python, "-u", "scripts/fit_shared_graph_checkpoints.py",
                "--delta-store", str(deltas), "--problems", str(problems),
                "--out", str(fit_dir),
                "--workers", str(args.fitting_workers),
                "--replicas", str(args.fitting_replicas),
                "--max-stochastic-steps", str(args.fitting_steps),
                "--relaxed", "--slack-precision", "1",
                "--stationarity-tolerance", "1e-4",
                "--accept-stochastic-plateau",
                "--tolerance", "1e-2", "--exact-interval",
                str(args.fitting_exact_interval),
                "--blocks", str(args.fitting_blocks),
                "--cache", str(args.fitting_block_cache),
                "--start", str(checkpoint), "--stop", str(checkpoint + 1),
                "--cold-start",
            ],
            [f"G{checkpoint}"],
            [fit_state],
            args.fitting_workers,
            min(2, args.fitting_workers),
        )
        if checkpoint + 1 < args.checkpoints:
            add(
                f"E{checkpoint}", "E", checkpoint,
                [
                    args.python, "-u", "scripts/score_checkpoint_interval.py",
                    "--state", str(fit_state),
                    "--ids", args.ids, "--top-k", str(args.top_k),
                    "--n", str(args.n),
                    "--next-problem", str(
                        problems / "states"
                        / f"checkpoint_{checkpoint + 1:03d}.npz"
                    ),
                    "--checkpoint", str(checkpoint),
                    "--out", str(scores / f"checkpoint_{checkpoint:03d}.json"),
                ],
                [f"F{checkpoint}", f"C{checkpoint + 1}"],
                [scores / f"checkpoint_{checkpoint:03d}.json"],
                args.evaluation_workers,
            )

    if args.policy == "phased":
        construction_waves = (
            [["S"]]
            + [[f"D{k}"] for k in range(args.checkpoints)]
            + [[f"U{k}"] for k in range(args.checkpoints)]
            + [[f"M{k}"] for k in range(args.checkpoints)]
            + [[f"A{k}", f"B{k}"] for k in range(args.checkpoints)]
            + [[f"C{k}"] for k in range(args.checkpoints)]
        )
        waves = construction_waves if args.construction_only else (
            construction_waves
            + [[f"G{k}"] for k in range(args.checkpoints)]
            + [["F0"]]
            + [
                [f"F{k}", f"E{k - 1}"]
                for k in range(1, args.checkpoints)
            ]
        )
    else:
        # A deterministic correctness schedule.  The analytic executor uses
        # the dependency graph above and is free to overlap U(k+1), A(k),
        # B(k), and downstream work subject to live worker capacity.
        waves = [["S"]]
        for checkpoint in range(args.checkpoints):
            waves.append([f"D{checkpoint}"])
            waves.append([f"U{checkpoint}"])
            waves.append([f"M{checkpoint}"])
            waves.append([f"A{checkpoint}", f"B{checkpoint}"])
            waves.append([f"C{checkpoint}"])
            if args.construction_only:
                continue
            waves.append([f"G{checkpoint}"])
            waves.append([f"F{checkpoint}"])
            if checkpoint:
                waves.append([f"E{checkpoint - 1}"])

    payload = {
        "version": 1,
        "policy": args.policy,
        "maximum_workers": args.maximum_workers,
        "maximum_private_memory_bytes": 0,
        "jobs": jobs,
        "waves": waves,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    print(destination)


if __name__ == "__main__":
    main()
