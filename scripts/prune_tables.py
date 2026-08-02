#!/usr/bin/env python3
"""Drop large-r columns from the universal store to reclaim disk.

The store is append-only: it grows when an experiment asks for a
moment order it has never seen and it never evicts.  After the
order-two runs it reached 350 GB, and the distribution is extremely
lopsided --- columns get physically longer as r grows, so counts above
10,000 hold about 78% of the bytes while counts below 1,000, which
recur in every single experiment, hold under 2%.

Large r values come from the handful of very heavy states in a run and
are specific to that run: splitting a state produces new integers, so
they rarely recur.  Dropping them is therefore cheap in expectation,
and safe in any case, because `UniversalTables.column` rebuilds a
missing column on demand.

The rewrite is per level.  Kept columns are streamed into a new data
file with a fresh index, then both are swapped in.  A crash between the
two swaps leaves a stale index pointing past the end of a shorter file,
which `_load_level` already detects and refuses loudly rather than
serving wrong numbers --- and the level would then simply rebuild.

    python scripts/prune_tables.py --max-r 10000 --dry-run
    python scripts/prune_tables.py --max-r 10000
"""

from __future__ import annotations

import argparse
import json
import os
import zlib
from pathlib import Path

import numpy as np


def level_files(root: Path):
    for idx in sorted(root.glob("level_*.index.json")):
        yield idx.with_suffix("").with_suffix(".bin"), idx


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tables", default=None,
                   help="store directory; defaults to PMM_UNIVERSAL_TABLES "
                        "or ./tables/universal_v2")
    p.add_argument("--max-r", type=int, required=True,
                   help="drop every column with r strictly greater than this")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verify", type=int, default=3,
                   help="re-read this many surviving columns per level "
                        "after the rewrite")
    args = p.parse_args()

    root = Path(args.tables or os.environ.get(
        "PMM_UNIVERSAL_TABLES", "tables/universal_v2"))
    if not (root / "manifest.json").exists():
        raise SystemExit(f"no store at {root}")

    total_before = total_after = 0
    dropped = kept = 0
    print(f"{'level':>8} {'columns':>10} {'dropped':>9} {'GB before':>10} "
          f"{'GB after':>9}")
    for data_f, idx_f in level_files(root):
        index = {int(k): tuple(v) for k, v in
                 json.loads(idx_f.read_text()).items()}
        if not index:
            continue
        keep_rs = sorted(r for r in index if r <= args.max_r)
        drop_rs = [r for r in index if r > args.max_r]
        # entries are (off, n) before checksums existed and (off, n, crc)
        # after; both forms must be handled or pruning silently mangles a
        # checksummed store
        before = sum(e[1] for e in index.values()) * 8
        after = sum(index[r][1] for r in keep_rs) * 8
        total_before += before
        total_after += after
        kept += len(keep_rs)
        dropped += len(drop_rs)
        print(f"{data_f.stem:>8} {len(index):>10,} {len(drop_rs):>9,} "
              f"{before/2**30:>10.2f} {after/2**30:>9.2f}")
        if args.dry_run or not drop_rs:
            continue

        tmp_bin = data_f.with_suffix(".bin.new")
        new_index: dict[int, tuple] = {}
        pos = 0
        with open(data_f, "rb") as src, open(tmp_bin, "wb") as dst:
            for r in keep_rs:
                entry = index[r]
                off, n = entry[0], entry[1]
                crc = entry[2] if len(entry) > 2 else None
                src.seek(off * 8)
                raw = src.read(n * 8)
                if len(raw) != n * 8:
                    tmp_bin.unlink(missing_ok=True)
                    raise SystemExit(
                        f"{data_f.name}: column r={r} is short; the store "
                        "was already damaged, refusing to rewrite it")
                # pruning MOVES a column, so this is the one place that
                # must re-check the checksum before rewriting: copying a
                # corrupted column to a new offset would launder it
                if crc is not None and (zlib.crc32(raw) & 0xFFFFFFFF) != crc:
                    tmp_bin.unlink(missing_ok=True)
                    raise SystemExit(
                        f"{data_f.name}: column r={r} fails its checksum; "
                        "the store is already corrupted, refusing to "
                        "rewrite it.  Delete this level and let it rebuild.")
                dst.write(raw)
                new_index[r] = (pos, n, crc) if crc is not None else (pos, n)
                pos += n
        tmp_idx = idx_f.with_suffix(".json.new")
        tmp_idx.write_text(json.dumps(
            {str(k): list(v) for k, v in new_index.items()}))
        os.replace(tmp_bin, data_f)
        os.replace(tmp_idx, idx_f)

        # read a few survivors back and check they are finite
        for r in keep_rs[:: max(1, len(keep_rs) // max(1, args.verify))][
                :args.verify]:
            off, n = new_index[r][0], new_index[r][1]
            with open(data_f, "rb") as f:
                f.seek(off * 8)
                vals = np.frombuffer(f.read(n * 8), dtype=np.float64)
            if len(vals) != n or not np.all(np.isfinite(vals)):
                raise SystemExit(
                    f"{data_f.name}: verification failed at r={r}; delete "
                    f"level_{data_f.stem.split('_')[1]}.* and let it rebuild")

    print(f"\n{'':>8} {kept:>10,} {dropped:>9,} "
          f"{total_before/2**30:>10.2f} {total_after/2**30:>9.2f}")
    freed = (total_before - total_after) / 2 ** 30
    if args.dry_run:
        print(f"\nDRY RUN: would drop {dropped:,} columns and free "
              f"{freed:.1f} GB.  Re-run without --dry-run to do it.")
    else:
        print(f"\ndropped {dropped:,} columns, freed {freed:.1f} GB.")
    print("Anything dropped rebuilds automatically the next time an "
          "experiment asks for it.")


if __name__ == "__main__":
    main()
