#!/usr/bin/env python3
"""Create a phase-oriented or pipelined hand-authored C/G/F/E schedule."""

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
    parser.add_argument("--evaluation-workers", type=int, default=1)
    parser.add_argument("--fitting-steps", type=int, default=500)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.checkpoints < 2:
        parser.error("at least two checkpoints are required")

    root = Path(args.root)
    problems = root / "problems"
    deltas = root / "graph_deltas"
    fitted = root / "fitted"
    scores = root / "scores"
    manifests = root / "job_manifests"
    jobs = []

    def add(job_id, kind, checkpoint, command, dependencies, outputs, workers):
        jobs.append({
            "id": job_id,
            "type": kind,
            "checkpoint": checkpoint,
            "command": command,
            "dependencies": dependencies,
            "outputs": [str(path) for path in outputs],
            "completion_manifest": str(manifests / f"{job_id}.json"),
            "workers": workers,
            "private_memory_bytes": 0,
        })

    for checkpoint in range(args.checkpoints):
        c_dependencies = [] if checkpoint == 0 else [f"C{checkpoint - 1}"]
        add(
            f"C{checkpoint}", "C", checkpoint,
            [
                args.python, "-u", "scripts/calibration_checkpoint_probe.py",
                "--ids", args.ids, "--top-k", str(args.top_k),
                "--n", str(args.n), "--checkpoints", str(args.checkpoints),
                "--first-checkpoint", str(args.first_checkpoint),
                "--interleave", "1", "--jobs", str(args.construction_workers),
                "--projection-tolerance", "1e-10",
                "--projection-iterations", "100000", "--sparse-upstream",
                "--stream-checkpoints", "--construct-only",
                "--resume-streamed", "--uncompressed-states",
                "--stop-after-checkpoint", str(checkpoint),
                "--out", str(problems),
            ],
            c_dependencies,
            [problems / "states" / f"checkpoint_{checkpoint:03d}.npz"],
            args.construction_workers,
        )
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
        f_dependencies = [f"G{checkpoint}"]
        if checkpoint:
            f_dependencies.append(f"F{checkpoint - 1}")
        add(
            f"F{checkpoint}", "F", checkpoint,
            [
                args.python, "-u", "scripts/fit_shared_graph_checkpoints.py",
                "--delta-store", str(deltas), "--problems", str(problems),
                "--out", str(fitted),
                "--workers", str(args.fitting_workers), "--replicas", "4",
                "--max-stochastic-steps", str(args.fitting_steps),
                "--relaxed", "--slack-precision", "1",
                "--stationarity-tolerance", "1e-4",
                "--tolerance", "1e-2", "--exact-interval", "5",
                "--blocks", "16", "--cache", "16",
                "--start", str(checkpoint), "--stop", str(checkpoint + 1),
            ],
            f_dependencies,
            [fitted / "states" / f"checkpoint_{checkpoint:03d}.npz"],
            args.fitting_workers,
        )
        if checkpoint + 1 < args.checkpoints:
            add(
                f"E{checkpoint}", "E", checkpoint,
                [
                    args.python, "-u", "scripts/score_checkpoint_interval.py",
                    "--state", str(
                        fitted / "states" / f"checkpoint_{checkpoint:03d}.npz"
                    ),
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
        waves = (
            [[f"C{k}"] for k in range(args.checkpoints)]
            + [[f"G{k}"] for k in range(args.checkpoints)]
            + [["F0"]]
            + [
                [f"F{k}", f"E{k - 1}"]
                for k in range(1, args.checkpoints)
            ]
        )
    else:
        waves = [["C0"]]
        if args.checkpoints > 1:
            waves.append(["G0", "C1"])
        for checkpoint in range(args.checkpoints):
            wave = [f"F{checkpoint}"]
            if checkpoint:
                wave.append(f"E{checkpoint - 1}")
            if checkpoint + 1 < args.checkpoints:
                wave.append(f"G{checkpoint + 1}")
            if checkpoint + 2 < args.checkpoints:
                wave.append(f"C{checkpoint + 2}")
            waves.append(wave)

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
