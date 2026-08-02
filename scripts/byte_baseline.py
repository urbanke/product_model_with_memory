#!/usr/bin/env python3
"""Memoryless (order-0) baseline over RAW BYTES, for any file.

This is the anchor row of the benchmark table: the file is treated as
a sequence over the 256 possible byte values and coded with the
depth-averaged layered mixture, exactly as the word-level memoryless
experiment does for tokens.  Nothing needs to be tokenized, escaped or
converted --- the result is already bits per byte, which for these
corpora IS bits per character, directly comparable to the published
lists.

The alphabet is fixed at d = 256 (not "the number of byte values that
happen to occur"), because a decoder does not know in advance which
values occur; the unseen values cost the mixture almost nothing but
the accounting stays honest.

    python scripts/byte_baseline.py --file data/text8   --out output/byte_text8
    python scripts/byte_baseline.py --file data/enwik8  --out output/byte_enwik8
    python scripts/byte_baseline.py --file data/enwik9  --out output/byte_enwik9 --jobs 12
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from product_model_with_memory.codelength import (
    default_l_max,
    depth_averaged_codelength_profiles,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", required=True, help="path to the raw file")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--l-max", type=int, default=None)
    p.add_argument("--bytes", type=int, default=None,
                   help="use only the first N bytes (for quick checks)")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    raw = np.fromfile(args.file, dtype=np.uint8)
    if args.bytes:
        raw = raw[: args.bytes]
    n = len(raw)
    counts = np.bincount(raw, minlength=256).astype(np.int64)
    occurring = int((counts > 0).sum())
    profile = tuple(sorted((int(c) for c in counts if c > 0), reverse=True))

    d = 256
    l_max = args.l_max or default_l_max(d)
    freq = counts[counts > 0] / n
    h0 = float(-(freq * np.log2(freq)).sum())
    print(f"{args.file}: {n:,} bytes, {occurring} distinct values, "
          f"order-0 entropy {h0:.4f} bits/byte, d={d}, L<={l_max}",
          flush=True)

    def progress(evt, _):
        kind, k, total = evt
        if kind == "tables" and (k % 500 == 0 or k == total):
            print(f"  tables: {k}/{total} ({time.time()-t0:.0f}s)", flush=True)
        elif kind == "depth" and (k % 5 == 0 or k == total):
            print(f"  depth: {k}/{total} ({time.time()-t0:.0f}s)", flush=True)

    res = depth_averaged_codelength_profiles(
        {0: profile}, d=d, l_max=l_max, jobs=args.jobs, progress=progress,
    )[0]

    bits_per_byte = res.bits_per_token          # one "token" = one byte
    payload = {
        "file": args.file,
        "bytes": n,
        "distinct_byte_values": occurring,
        "d": d,
        "l_max": l_max,
        "order0_entropy_bits_per_byte": h0,
        "codelength_bits_per_byte": bits_per_byte,
        "redundancy_bits_per_byte": bits_per_byte - h0,
        "total_bits": -res.log2_q_avg,
        "total_bytes_compressed": -res.log2_q_avg / 8.0,
        "posterior_mode_depth": res.posterior_mode,
        "log2_q_by_depth": list(res.log2_q_by_depth),
        "seconds": time.time() - t0,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))

    print(f"\n  order-0 entropy   : {h0:.4f} bits/byte")
    print(f"  layered memoryless: {bits_per_byte:.4f} bits/byte")
    print(f"  redundancy        : {bits_per_byte - h0:+.6f} bits/byte")
    print(f"  posterior-mode L  : {res.posterior_mode}")
    print(f"  compressed size   : {payload['total_bytes_compressed']:,.0f} bytes")
    print(f"\nwritten: {out_dir/'results.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
