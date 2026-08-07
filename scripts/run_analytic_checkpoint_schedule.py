#!/usr/bin/env python3
"""Run checkpoint jobs using a portable analytic plan and live capacity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from product_model_with_memory.checkpoint_scheduler import (
    load_fixed_schedule, run_planned_schedule,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True, help="fixed schedule JSON")
    parser.add_argument("--plan", required=True, help="analytic plan JSON")
    parser.add_argument("--working-directory", default=".")
    parser.add_argument("--maximum-workers", type=int)
    args = parser.parse_args()

    schedule = load_fixed_schedule(args.jobs)
    payload = json.loads(Path(args.plan).read_text())
    planned = {row["task_id"]: row for row in payload["plan"]}
    maximum_workers = args.maximum_workers or payload["maximum_workers"]
    records = run_planned_schedule(
        schedule.jobs,
        {task_id: int(row["workers"]) for task_id, row in planned.items()},
        {task_id: float(row["start"]) for task_id, row in planned.items()},
        maximum_workers=maximum_workers,
        working_directory=args.working_directory,
    )
    print(json.dumps({"completed_jobs": len(records)}))


if __name__ == "__main__":
    main()
