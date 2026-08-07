#!/usr/bin/env python3
"""Publish G_k and its temporary legacy compatibility store."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--delta-store", required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    commands = (
        (
            sys.executable, str(scripts / "publish_checkpoint_graph_delta.py"),
            "--problems", args.problems,
            "--store", args.delta_store,
            "--checkpoint", str(args.checkpoint),
        ),
        (
            sys.executable,
            str(scripts / "materialize_checkpoint_delta_store.py"),
            "--problems", args.problems,
            "--delta-store", args.delta_store,
            "--checkpoint", str(args.checkpoint),
            "--out", args.out,
        ),
    )
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
