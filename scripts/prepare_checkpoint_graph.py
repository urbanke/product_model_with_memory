#!/usr/bin/env python3
"""Publish the immutable append-only graph delta G_k."""

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
    parser.add_argument("--out", help="deprecated; no store is materialized")
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    command = (
        sys.executable, str(scripts / "publish_checkpoint_graph_delta.py"),
        "--problems", args.problems,
        "--store", args.delta_store,
        "--checkpoint", str(args.checkpoint),
    )
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
