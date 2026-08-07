#!/usr/bin/env python3
"""Plan U/M/A/B/C/G/F/E work without machine timing measurements."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from product_model_with_memory.analytic_schedule import (
    checkpoint_tasks,
    checkpoint_work_profile,
    geometric_prefixes,
    plan_moldable_tasks,
)
from product_model_with_memory.streams import load_stream, reduce_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--checkpoints", type=int, default=32)
    parser.add_argument("--first-checkpoint", type=int, default=2_050)
    parser.add_argument("--maximum-workers", type=int, default=8)
    parser.add_argument("--construction-maximum-workers", type=int, default=4)
    parser.add_argument("--fitting-maximum-workers", type=int, default=4)
    parser.add_argument("--stochastic-steps", type=int, default=5_000)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--exact-interval", type=int, default=5)
    parser.add_argument("--out", required=True)
    parser.add_argument("--construction-only", action="store_true")
    args = parser.parse_args()

    ids, metadata = load_stream(args.ids)
    stop = min(args.n, len(ids))
    reduced, vocabulary_size, capped = reduce_ids(ids[:stop], args.top_k)
    counts = np.bincount(reduced, minlength=vocabulary_size).astype(np.float64)
    probabilities = counts / counts.sum()
    prefixes = geometric_prefixes(
        stop, args.checkpoints, first_prefix=args.first_checkpoint,
    )
    profile = checkpoint_work_profile(
        prefixes, probabilities,
        stochastic_steps=args.stochastic_steps,
        replicas=args.replicas,
        blocks=args.blocks,
        exact_interval=args.exact_interval,
    )
    tasks = checkpoint_tasks(
        profile,
        construction_maximum_workers=args.construction_maximum_workers,
        fitting_maximum_workers=args.fitting_maximum_workers,
    )
    if args.construction_only:
        tasks = tuple(task for task in tasks if task.task_id[0] in "SDUMABC")
    plan = plan_moldable_tasks(tasks, args.maximum_workers)
    payload = {
        "version": 1,
        "model": "portable_analytic_prior",
        "units": "dimensionless estimated primitive visits, not seconds",
        "stream": args.ids,
        "representation": metadata.get("representation"),
        "observations": stop,
        "vocabulary_size": vocabulary_size,
        "capped_positions": capped,
        "maximum_workers": args.maximum_workers,
        "worker_speedup_prior": {"1": 1.0, "2": 1.5, "3": 1.62, "4": 1.68},
        "profile": [asdict(row) for row in profile],
        "tasks": [asdict(task) for task in tasks],
        "plan": [asdict(row) for row in plan],
        "estimated_makespan": max(row.finish for row in plan),
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "written": str(destination),
        "tasks": len(tasks),
        "estimated_makespan": payload["estimated_makespan"],
    }))


if __name__ == "__main__":
    main()
