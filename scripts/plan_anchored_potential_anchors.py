#!/usr/bin/env python3
"""Publish the fixed stratified anchor manifest for a potential experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from product_model_with_memory.potential_sampling import (
    POTENTIAL_SAMPLING_DESIGN,
    plan_potential_anchors,
)
from product_model_with_memory.production_coding import PRODUCTION_SEQUENCE_ESTIMATOR


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True, help="reduced stream.npy")
    parser.add_argument("--minimum-prefix", type=int, default=2050)
    parser.add_argument("--windows", default="64,256,1024,4096")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--early-strata", type=int, default=8)
    parser.add_argument("--late-strata", type=int, default=24)
    parser.add_argument("--samples-per-stratum", type=int, default=2)
    parser.add_argument("--early-fraction", type=float, default=0.05)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    stream = Path(args.stream).resolve()
    windows = sorted(set(int(value) for value in args.windows.split(",")))
    if not windows or windows[0] < 1:
        parser.error("windows must be positive integers")
    import numpy as np
    n = len(np.load(stream, mmap_mode="r", allow_pickle=False))
    anchors = plan_potential_anchors(
        n, minimum_prefix=args.minimum_prefix, maximum_window=windows[-1],
        seed=args.seed, early_strata=args.early_strata,
        late_strata=args.late_strata,
        samples_per_stratum=args.samples_per_stratum,
        early_fraction=args.early_fraction,
    )
    eligible = n - windows[-1] + 1 - args.minimum_prefix
    payload = {
        "version": 1,
        "design": POTENTIAL_SAMPLING_DESIGN,
        "sequence_estimator": PRODUCTION_SEQUENCE_ESTIMATOR,
        "stream": str(stream), "stream_sha256": sha256(stream), "n": n,
        "minimum_prefix": args.minimum_prefix, "windows": windows,
        "seed": args.seed, "early_strata": args.early_strata,
        "late_strata": args.late_strata,
        "samples_per_stratum": args.samples_per_stratum,
        "early_fraction": args.early_fraction,
        "eligible_prefix_positions": eligible,
        "anchors": [asdict(anchor) for anchor in anchors],
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(destination)
    print(destination)


if __name__ == "__main__":
    main()
