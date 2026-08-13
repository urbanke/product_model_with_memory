#!/usr/bin/env python3
"""Run a restartable checkpoint schedule in fixed-wave or live-DAG mode."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from product_model_with_memory.checkpoint_scheduler import (
    load_fixed_schedule, run_dependency_schedule,
    run_fixed_schedule,
)
from product_model_with_memory.production_coding import require_production_schedule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schedule")
    parser.add_argument("--working-directory", default=".")
    parser.add_argument(
        "--dependency-driven", action="store_true",
        help="launch every dependency-ready job without fixed-wave barriers",
    )
    parser.add_argument("--maximum-workers", type=int)
    parser.add_argument("--event-log")
    args = parser.parse_args()
    raw_schedule = json.loads(Path(args.schedule).read_text())
    require_production_schedule(raw_schedule, source=args.schedule)
    schedule = load_fixed_schedule(
        args.schedule, enforce_wave_capacity=not args.dependency_driven
    )
    if args.dependency_driven:
        event_path = Path(args.event_log) if args.event_log else None
        if event_path:
            event_path.parent.mkdir(parents=True, exist_ok=True)

        def record(event: dict) -> None:
            row = {"unix_seconds": time.time(), **event}
            print(json.dumps(row), flush=True)
            if event_path:
                with event_path.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")

        records = run_dependency_schedule(
            schedule, working_directory=args.working_directory,
            maximum_workers=args.maximum_workers,
            event_callback=record,
        )
    else:
        if args.maximum_workers is not None:
            parser.error("--maximum-workers requires --dependency-driven")
        records = run_fixed_schedule(
            schedule, working_directory=args.working_directory
        )
    print(json.dumps({"completed_jobs": len(records), "records": records}, indent=2))


if __name__ == "__main__":
    main()
