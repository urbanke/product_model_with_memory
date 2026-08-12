#!/usr/bin/env python3
"""Run checkpoint jobs using a portable analytic plan and live capacity."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from product_model_with_memory.checkpoint_scheduler import (
    load_fixed_schedule, run_planned_schedule,
)
from product_model_with_memory.production_coding import (
    PRODUCTION_SEQUENCE_ESTIMATOR,
    require_production_sequence_estimator,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True, help="fixed schedule JSON")
    parser.add_argument("--plan", required=True, help="analytic plan JSON")
    parser.add_argument("--working-directory", default=".")
    parser.add_argument("--maximum-workers", type=int)
    parser.add_argument(
        "--event-log",
        help="append scheduler launch/finish events as JSON lines",
    )
    args = parser.parse_args()

    # The analytic executor uses only the dependency DAG and enforces live
    # capacity itself.  Fixed waves are retained as a portable fallback but
    # their aggregate capacity is irrelevant here.
    schedule = load_fixed_schedule(args.jobs, enforce_wave_capacity=False)
    payload = json.loads(Path(args.plan).read_text())
    require_production_sequence_estimator(
        payload.get("sequence_estimator", PRODUCTION_SEQUENCE_ESTIMATOR),
        source=str(args.plan),
    )
    planned = {row["task_id"]: row for row in payload["plan"]}
    tasks = {row["task_id"]: row for row in payload["tasks"]}
    maximum_workers = args.maximum_workers or payload["maximum_workers"]
    event_path = Path(args.event_log) if args.event_log else None
    if event_path:
        event_path.parent.mkdir(parents=True, exist_ok=True)

    def record_event(event: dict) -> None:
        event = {"unix_seconds": time.time(), **event}
        line = json.dumps(event)
        print(line, flush=True)
        if event_path:
            with event_path.open("a") as stream:
                stream.write(line + "\n")

    records = run_planned_schedule(
        schedule.jobs,
        {task_id: int(row["workers"]) for task_id, row in planned.items()},
        {task_id: float(row["start"]) for task_id, row in planned.items()},
        maximum_workers=maximum_workers,
        worker_caps={
            task_id: min(int(row["maximum_workers"]), maximum_workers)
            for task_id, row in tasks.items()
        },
        worker_floors={
            job.job_id: min(job.minimum_workers, maximum_workers)
            for job in schedule.jobs
        },
        working_directory=args.working_directory,
        event_callback=record_event,
    )
    print(json.dumps({"completed_jobs": len(records)}))


if __name__ == "__main__":
    main()
