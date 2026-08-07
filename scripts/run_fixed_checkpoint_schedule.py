#!/usr/bin/env python3
"""Run a hand-authored checkpoint schedule with explicit concurrent waves."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from product_model_with_memory.checkpoint_scheduler import (
    load_fixed_schedule,
    run_fixed_schedule,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schedule")
    parser.add_argument("--working-directory", default=".")
    args = parser.parse_args()
    schedule = load_fixed_schedule(args.schedule)
    records = run_fixed_schedule(
        schedule, working_directory=args.working_directory
    )
    print(json.dumps({"completed_jobs": len(records), "records": records}, indent=2))


if __name__ == "__main__":
    main()
