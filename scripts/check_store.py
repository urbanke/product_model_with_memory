#!/usr/bin/env python3
"""Verify stored columns against an independent exact method.

The store is append-only and is written during long runs.  If a run dies
between appending a column's data and flushing the level index --- a
crash, an OOM kill, a full disk --- the index can end up pointing at the
wrong region of the level file.  A column read through a bad offset is
still SMOOTH: it is real data, just the wrong column's.  Nothing about
it looks broken from the inside, no interpolation check fires, and every
downstream number is quietly wrong.

Found on 1 August in a local copy: 285 of 594 columns at L=43 disagreed
with contour integration by up to 47,000 nats, while L=35 was clean and
neighbouring columns at the same level were exact to 1e-12.

The check is two-stage, because the honest reference is slow.  A
saddlepoint screen runs first: cheap, and at L >= SADDLE_OK_L it agrees
with contour integration to about 1e-6 nats, so anything above the
screen threshold there is a real defect rather than an approximation
gap.  Every flagged column is then CONFIRMED by contour integration
before it is reported, so a screen failure alone never condemns a
column.  Below SADDLE_OK_L the expansion is too coarse to screen with
(~5e-3 nats at L=2), so those levels are sampled with contour directly.

    python scripts/check_store.py                     # sample every level
    python scripts/check_store.py --levels 40,43,54 --per-level 200
    python scripts/check_store.py --all --jobs 8      # every column

Repair: a bad column cannot be fixed in place, because its bytes belong
to a different column.  Delete the level and let it rebuild --

    rm tables/universal_v2/level_43.*

-- which is safe: `UniversalTables.column` rebuilds on demand by the
certified route.  Rebuilding is slow, so check first and delete only the
levels that fail.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

from product_model_with_memory.mellin import log_phi_contour, log_phi_saddle
from product_model_with_memory.universal_tables import (
    H,
    U_MAX,
    UniversalTables,
)

SADDLE_OK_L = 20      # depth above which the expansion is a fair screen
SCREEN_NATS = 1e-2    # ~1e4 times the screen's own error at those depths
CONFIRM_NATS = 1e-4   # a confirmed defect: far beyond any method's error


def _probe_u(n_vals: int, k: int) -> np.ndarray:
    """A few u points inside the column, away from both edges."""

    u = U_MAX - H * np.arange(n_vals)[::-1]
    lo = int(0.55 * len(u))          # right of the far-left tail
    return u[np.linspace(lo, len(u) - 4, k).astype(int)]


def check_column(tab, L: int, r: int) -> tuple[float, float]:
    """(screen error, confirmed error).  The second is computed only if
    the first fires, and only it condemns a column."""

    _i0, vals = tab.column(L, r)
    u = _probe_u(len(vals), 5)
    idx = np.searchsorted(U_MAX - H * np.arange(len(vals))[::-1], u)
    idx = np.clip(idx, 0, len(vals) - 1)
    stored = vals[::-1][len(vals) - 1 - idx] if False else np.array(
        [vals[int(len(vals) - 1 - round((U_MAX - x) / H))] for x in u])

    if L >= SADDLE_OK_L:
        approx = np.array([log_phi_saddle(float(r), L, float(x))[0]
                           for x in u])
        screen = float(np.abs(approx - stored).max())
        if screen < SCREEN_NATS:
            return screen, 0.0
    else:
        screen = float("inf")        # no usable screen; go straight to it

    exact = np.array([log_phi_contour(float(r), L, float(x)) for x in u])
    return screen, float(np.abs(exact - stored).max())


def self_test() -> int:
    """Forge the damage, then confirm this script detects it.

    A clean report and a broken detector look identical from the
    outside, so the checker needs a positive control.  This builds a
    throwaway store, repoints one column's index entry at another
    column's bytes --- precisely what concurrent writers produced --- and
    asserts that check_column reports it.  If this fails, no clean
    report from this script means anything.
    """

    import json as _json
    import tempfile

    from product_model_with_memory.universal_tables import UniversalTables

    ok = True
    # Two paths need proving, and they are NOT the same path.  A store
    # written since checksums exist is caught by the CRC on read.  A
    # store written before them --- which is every store that predates
    # 1 August, and therefore the one anybody is actually checking ---
    # has no CRC to fail, so detection rests entirely on the saddlepoint
    # screen and the contour confirmation.  Proving only the first would
    # be a control for the case that needs no control.
    #
    # The level must also be at or above SADDLE_OK_L, or the screen is
    # skipped and the test says nothing about how real levels behave.
    L = max(SADDLE_OK_L, 24)
    for legacy in (False, True):
        label = "legacy (no checksum)" if legacy else "checksummed"
        root = Path(tempfile.mkdtemp()) / "selftest"
        tab = UniversalTables(root)
        good, victim = 40, 90
        tab.column(L, good)
        tab.column(L, victim)
        tab.close()

        idx_f = root / f"level_{L:02d}.index.json"
        idx = _json.loads(idx_f.read_text())
        if legacy:                      # strip CRCs: an old-format store
            idx = {k: v[:2] for k, v in idx.items()}
        off_g, n_g = idx[str(good)][0], idx[str(good)][1]
        entry = idx[str(victim)]
        # the victim now points at the other column's bytes, keeping its
        # own length: finite, smooth, correctly sized, and wrong
        idx[str(victim)] = [off_g, min(n_g, entry[1])] + list(entry[2:])
        idx_f.write_text(_json.dumps(idx))

        tab = UniversalTables(root, read_only=True)
        healthy = check_column(tab, L, good)[1]
        if healthy > CONFIRM_NATS:
            print(f"  {label}: FAIL -- healthy column reported corrupt "
                  f"({healthy:.2e})")
            ok = False
        try:
            screen, confirmed = check_column(tab, L, victim)
            how = (f"screen {screen:.2e} -> contour {confirmed:.2e}"
                   if confirmed > CONFIRM_NATS else "NOT DETECTED")
            print(f"  {label:22s}: {how}")
            if confirmed <= CONFIRM_NATS:
                print(f"      FAIL: forged corruption survived the "
                      f"{'numeric' if legacy else 'checksum'} path")
                ok = False
        except RuntimeError as exc:
            if legacy:                  # nothing should refuse it here
                print(f"  {label:22s}: FAIL -- unexpected refusal: "
                      f"{str(exc)[:60]}")
                ok = False
            else:
                print(f"  {label:22s}: refused on read (checksum)")
        tab.close()

    print("\nself-test PASSED: both the checksum path and the numeric "
          "path detect\nforged corruption at L=%d.  A clean report from "
          "this script means something." % L if ok else
          "\nself-test FAILED: do not trust a clean report from this "
          "script.")
    return 0 if ok else 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true",
                   help="forge known damage and confirm it is detected; "
                        "run this before trusting a clean report")
    p.add_argument("--tables", default=os.environ.get(
        "PMM_UNIVERSAL_TABLES", "tables/universal_v2"))
    p.add_argument("--levels", default=None,
                   help="comma-separated; default every level present")
    p.add_argument("--per-level", type=int, default=120,
                   help="columns sampled per level (ignored with --all)")
    p.add_argument("--all", action="store_true",
                   help="check every column; slow, but conclusive")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.self_test:
        raise SystemExit(self_test())

    root = Path(args.tables)
    if not (root / "manifest.json").exists():
        raise SystemExit(f"no store at {root}")
    tab = UniversalTables(root, read_only=True)

    if args.levels:
        levels = [int(x) for x in args.levels.split(",")]
    else:
        levels = sorted(int(f.stem.split("_")[1].split(".")[0])
                        for f in root.glob("level_*.index.json"))

    rng = random.Random(args.seed)
    print(f"store: {root}")
    print(f"{'level':>6} {'columns':>9} {'checked':>8} {'flagged':>8} "
          f"{'CONFIRMED BAD':>14} {'worst (nats)':>13}")
    report, total_bad = [], 0
    for L in levels:
        idx_f = root / f"level_{L:02d}.index.json"
        if not idx_f.exists():
            continue
        rs = sorted(int(k) for k in json.loads(idx_f.read_text()))
        if not rs:
            continue
        # index consistency first: an offset past the end of the file is
        # unambiguous damage and needs no numerics at all
        size = (root / f"level_{L:02d}.bin").stat().st_size // 8
        index = json.loads(idx_f.read_text())
        # index entries are (off, n) or (off, n, crc) -- sum() over the
        # whole entry would fold the checksum into the arithmetic
        past_eof = [r for r in rs
                    if index[str(r)][0] + index[str(r)][1] > size]

        pick = rs if args.all else rng.sample(rs, min(args.per_level, len(rs)))
        flagged, bad, worst = 0, [], 0.0
        for r in sorted(pick):
            try:
                screen, confirmed = check_column(tab, L, r)
            except Exception as exc:                     # unreadable column
                bad.append((r, float("inf")))
                print(f"    L={L} r={r}: unreadable ({exc})", file=sys.stderr)
                continue
            if screen >= SCREEN_NATS:
                flagged += 1
            if confirmed > CONFIRM_NATS:
                bad.append((r, confirmed))
                worst = max(worst, confirmed)
        total_bad += len(bad)
        report.append({"L": L, "columns": len(rs), "checked": len(pick),
                       "flagged": flagged, "confirmed_bad": len(bad),
                       "worst_nats": worst, "index_past_eof": past_eof,
                       "bad_r": [r for r, _ in bad[:50]]})
        mark = "  <-- REBUILD" if bad or past_eof else ""
        print(f"{L:>6} {len(rs):>9,} {len(pick):>8,} {flagged:>8,} "
              f"{len(bad):>14,} {worst:>13.3e}{mark}", flush=True)
        if past_eof:
            print(f"       {len(past_eof)} column(s) indexed past the end "
                  f"of the file", flush=True)

    print()
    if total_bad == 0:
        print("No corrupted columns found.  Note this is a SAMPLE unless "
              "--all was given:\nabsence of evidence at 120 columns per "
              "level is not proof for a level of 40,000.")
    else:
        levs = [r["L"] for r in report if r["confirmed_bad"] or
                r["index_past_eof"]]
        print(f"{total_bad} corrupted column(s) confirmed against contour "
              f"integration.")
        print("A bad column's bytes belong to a different column, so it "
              "cannot be repaired\nin place.  Delete the affected levels "
              "and let them rebuild:\n")
        print("    rm " + " ".join(
            f"{root}/level_{L:02d}.*" for L in levs))
        print("\nAny result computed from these levels should be recomputed.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"tables": str(root), "sampled": not args.all,
             "levels": report}, indent=2))
        print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
