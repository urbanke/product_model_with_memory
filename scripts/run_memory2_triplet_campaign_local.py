#!/usr/bin/env python3
"""Resumable, memory-bounded runner for a declared triplet campaign."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from product_model_with_memory.production_coding import (
    PRODUCTION_SEQUENCE_ESTIMATOR,
)


def _complete(path: Path, expected: list[str], triplet_count: int) -> bool:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    accounting = payload.get("honest_accounting", {})
    return (
        payload.get("sequence_estimator") == PRODUCTION_SEQUENCE_ESTIMATOR
        and payload.get("grid") == expected
        and accounting.get("sequence_estimator")
        == PRODUCTION_SEQUENCE_ESTIMATOR
        and accounting.get("triplet_grid_size") == triplet_count
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", default="output/memory2_triplet_campaign_20260809/plan.json"
    )
    parser.add_argument("--ids", default="output/streams/bpe_enwik9")
    parser.add_argument(
        "--root", default="output/memory2_triplet_campaign_20260809"
    )
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--members-per-batch", type=int, default=4)
    parser.add_argument(
        "--cache-65536",
        default=("output/memory2_frontier_20260809/"
                 "enwik8_v65536_baseline_neighbors/cache"),
    )
    args = parser.parse_args()
    if args.jobs < 1 or args.members_per_batch < 1:
        parser.error("jobs and members-per-batch must be positive")

    plan_path = Path(args.plan)
    if not plan_path.exists():
        subprocess.run([
            sys.executable, "scripts/plan_memory2_triplet_campaign.py",
            "--out", str(plan_path),
        ], check=True)
    plan = json.loads(plan_path.read_text())
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = "src" + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    assignments: list[str] = []
    for vocabulary_text, grid in plan["grids_by_vocabulary"].items():
        vocabulary = int(vocabulary_text)
        cache = (
            Path(args.cache_65536) if vocabulary == 65536
            else root / f"cache_v{vocabulary}"
        )
        for batch_index, start in enumerate(
            range(0, len(grid), args.members_per_batch)
        ):
            members = grid[start:start + args.members_per_batch]
            out = root / f"v{vocabulary}" / f"batch_{batch_index:02d}"
            result = out / "results.json"
            assignments.append(f"{vocabulary}={result}")
            if _complete(result, members, int(plan["triplet_count"])):
                print(f"reuse {result}", flush=True)
                continue
            out.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, "-u", "scripts/product_family_experiment.py",
                "--ids", args.ids,
                "--top-k", str(vocabulary - 1),
                "--grid", ",".join(members),
                "--state-order", "frequency",
                "--jobs", str(args.jobs),
                "--alphabet-grid-size", "8",
                "--triplet-grid-size", str(plan["triplet_count"]),
                "--cache-dir", str(cache),
                "--out", str(out),
            ]
            print(
                f"run V={vocabulary} batch={batch_index} members={members}",
                flush=True,
            )
            subprocess.run(command, check=True, env=env)

    summary = root / "accounting.json"
    command = [
        sys.executable, "scripts/summarize_memory2_triplet_campaign.py",
        "--plan", str(plan_path), "--out", str(summary),
    ]
    for assignment in assignments:
        command.extend(["--result", assignment])
    subprocess.run(command, check=True, env=env)
    print(f"complete: {summary}", flush=True)


if __name__ == "__main__":
    main()
