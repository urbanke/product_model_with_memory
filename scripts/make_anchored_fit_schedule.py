#!/usr/bin/env python3
"""Create restartable topology/fit jobs for hard-YA checkpoint problems."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from product_model_with_memory.production_coding import PRODUCTION_SEQUENCE_ESTIMATOR


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[start:start + size] for start in range(0, len(values), size)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--python", default=".venv/bin/python3")
    parser.add_argument("--maximum-workers", type=int, default=8)
    parser.add_argument("--fitting-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--exact-interval", type=int, default=50)
    parser.add_argument("--slack-precision", type=float, default=1.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.maximum_workers < 1 or not 1 <= args.fitting_workers <= args.maximum_workers:
        parser.error("invalid worker allocation")
    problem_paths = sorted((Path(args.problems) / "states").glob("checkpoint_*.npz"))
    if not problem_paths:
        parser.error("no checkpoint problems found")
    root = Path(args.root)
    jobs = []
    manifests = root / "job_manifests"
    topology_ids, fit_ids = [], []
    for checkpoint, problem in enumerate(problem_paths):
        label = f"checkpoint_{checkpoint:03d}"
        topology = root / "topology" / label
        fit = root / "fits" / label
        tid, fid = f"T{checkpoint}", f"F{checkpoint}"
        topology_ids.append(tid)
        fit_ids.append(fid)
        jobs.append({
            "id": tid, "type": "T", "checkpoint": checkpoint,
            "command": [args.python, "-u", "scripts/prepare_anchored_ya_topology.py",
                        "--problem", str(problem), "--out", str(topology)],
            "dependencies": [], "outputs": [str(topology / "manifest.json")],
            "completion_manifest": str(manifests / f"{tid}.json"),
            "workers": 1, "minimum_workers": 1, "private_memory_bytes": 0,
        })
        jobs.append({
            "id": fid, "type": "F", "checkpoint": checkpoint,
            "command": [
                args.python, "-u", "scripts/fit_anchored_ya_checkpoint.py",
                "--problem", str(problem), "--topology", str(topology),
                "--out", str(fit), "--steps", str(args.steps),
                "--batch-size", str(args.batch_size), "--seed", str(args.seed),
                "--workers", str(args.fitting_workers),
                "--learning-rate", str(args.learning_rate),
                "--exact-interval", str(args.exact_interval),
                "--slack-precision", str(args.slack_precision),
            ],
            "dependencies": [tid], "outputs": [str(fit / "manifest.json")],
            "completion_manifest": str(manifests / f"{fid}.json"),
            "workers": args.fitting_workers,
            "minimum_workers": 1, "private_memory_bytes": 0,
        })
    topology_waves = chunks(topology_ids, args.maximum_workers)
    simultaneous_fits = max(1, args.maximum_workers // args.fitting_workers)
    fit_waves = chunks(fit_ids, simultaneous_fits)
    payload = {
        "version": 1,
        "model": "anchored_ya_relaxed_pair_graph_v1",
        "sequence_estimator": PRODUCTION_SEQUENCE_ESTIMATOR,
        "maximum_workers": args.maximum_workers,
        "maximum_private_memory_bytes": 0,
        "jobs": jobs,
        "waves": topology_waves + fit_waves,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    print(destination)


if __name__ == "__main__":
    main()
