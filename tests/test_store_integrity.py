"""The store must not be able to corrupt itself silently.

On 1 August, 285 of 594 columns at L=43 were wrong by up to 47,000 nats.
The cause was `_append_column` recording each column's offset from
`level["size"]`, an in-memory counter, while appending in "ab" mode.
Two processes writing one level each kept their own counter, so both
wrote real bytes and both recorded plausible offsets, but each one's
offsets were short by whatever the other had written; `_flush_index`
then overwrote the index with a single process's view.

What made it expensive was not the corruption but the SILENCE.  A
column read through a wrong offset is real data from elsewhere in the
file: finite, smooth, right length, right dtype.  Every existing guard
passed it --- the finite check, the length check, the interpolation
roughness check --- and it took contour integration to see anything was
wrong at all.  The store had been certified once at build time on 600
samples and never re-verified on read.

So these tests are about detection as much as prevention:

  * two concurrent writers must not produce a wrong column, and
  * if one ever does, the checksum must catch it on the next read.

test_a_wrong_offset_is_caught_on_read is the important one.  It forges
exactly the damage the bug produced --- a valid column body under
another column's index entry --- and asserts we now refuse it.  Without
the checksum that read succeeds and returns smooth, plausible nonsense.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import zlib
from pathlib import Path

import numpy as np
import pytest

from product_model_with_memory.universal_tables import UniversalTables


def _writer(path: str, L: int, rs: list[int]) -> None:
    tab = UniversalTables(path)
    for r in rs:
        tab.column(L, r)
    tab.close()


@pytest.fixture
def store(tmp_path: Path) -> Path:
    UniversalTables(tmp_path / "s").close()
    return tmp_path / "s"


def test_columns_carry_a_checksum(store: Path):
    tab = UniversalTables(store)
    tab.column(3, 40)
    tab.close()
    idx = json.loads((store / "level_03.index.json").read_text())
    assert idx, "nothing was written"
    for entry in idx.values():
        assert len(entry) == 3, "column written without a checksum"


def test_a_wrong_offset_is_caught_on_read(store: Path):
    """The exact damage the bug produced, forged deliberately.

    Two good columns, then one's index entry is repointed at the other's
    bytes.  The result is finite, smooth, and the right length --- so
    every guard that existed before this change passes it.
    """

    tab = UniversalTables(store)
    a, b = 40, 90
    tab.column(3, a)
    tab.column(3, b)
    tab.close()

    idx_f = store / "level_03.index.json"
    idx = json.loads(idx_f.read_text())
    off_b, n_b, _crc_b = idx[str(b)]
    off_a, n_a, crc_a = idx[str(a)]
    # point a at b's bytes, keeping a's length: a valid column body
    idx[str(a)] = [off_b, min(n_a, n_b), crc_a]
    idx[str(b)] = [off_b, min(n_a, n_b), _crc_b]
    idx_f.write_text(json.dumps(idx))

    tab = UniversalTables(store, read_only=True)
    with pytest.raises(RuntimeError, match="checksum"):
        tab.column(3, a)
    tab.close()


def test_the_forged_column_would_have_passed_the_old_guards(store: Path):
    """Why the checksum is necessary and not belt-and-braces.

    Re-forge the same damage and confirm the bytes are finite, smooth,
    and correctly sized.  Nothing short of comparing against an
    independent evaluation could have flagged them.
    """

    tab = UniversalTables(store)
    tab.column(3, 40)
    tab.column(3, 90)
    tab.close()
    idx = json.loads((store / "level_03.index.json").read_text())
    off_b, n_b, _ = idx["90"]
    raw = (store / "level_03.bin").read_bytes()[off_b * 8:(off_b + n_b) * 8]
    vals = np.frombuffer(raw, dtype=np.float64)
    assert np.all(np.isfinite(vals))                    # finite check: passes
    assert len(vals) == n_b                             # length check: passes
    d2 = np.abs(np.diff(vals, n=2))
    assert d2.max() < 1e-2                              # smoothness: passes


def test_concurrent_writers_do_not_corrupt(store: Path):
    """The regression test for the bug itself.

    Two processes build disjoint columns of one level at the same time.
    Every column either reads back correctly or is absent; none may read
    back as someone else's data.
    """

    L = 3
    left = [30, 31, 32, 33, 34, 35, 36, 37]
    right = [80, 81, 82, 83, 84, 85, 86, 87]
    ctx = mp.get_context("fork")
    ps = [ctx.Process(target=_writer, args=(str(store), L, rs))
          for rs in (left, right)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(300)
    assert all(p.exitcode == 0 for p in ps), "a writer died"

    idx = json.loads((store / f"level_{L:02d}.index.json").read_text())
    # neither writer's entries may have been clobbered by the other
    missing = [r for r in left + right if str(r) not in idx]
    assert not missing, f"index lost columns: {missing}"

    tab = UniversalTables(store, read_only=True)
    for r in left + right:
        _i0, vals = tab.column(L, r)          # raises if the CRC fails
        assert np.all(np.isfinite(vals))
    tab.close()

    rep = tab.verify_level(L) if hasattr(tab, "verify_level") else None
    if rep is not None:
        assert not rep["bad"], f"corrupted after concurrent write: {rep}"


def test_verify_level_reports_damage(store: Path):
    tab = UniversalTables(store)
    for r in (40, 90, 140):
        tab.column(3, r)
    tab.close()

    idx_f = store / "level_03.index.json"
    idx = json.loads(idx_f.read_text())
    idx["90"][2] = (idx["90"][2] + 1) & 0xFFFFFFFF       # wrong checksum
    idx_f.write_text(json.dumps(idx))

    rep = UniversalTables(store, read_only=True).verify_level(3)
    assert rep["bad"] == [90], rep
    assert rep["unchecked"] == []


def test_a_checksummed_store_opens_at_all(store: Path):
    """Widening the index entry from (off, n) to (off, n, crc) breaks
    every site that unpacked it as a pair.  The first attempt missed one
    in _load_level, which made a checksummed store unopenable --- caught
    only because the existing suite happened to reopen a store.

    This pins the contract directly: write, drop all caches, reopen from
    disk, read back.  Any new two-value unpack on the load path fails
    here rather than in whatever unrelated test reopens a store next.
    """

    tab = UniversalTables(store)
    tab.column(3, 40)
    tab.column(3, 90)
    tab.close()

    reopened = UniversalTables(store)          # fresh object, reads JSON
    _i0, vals = reopened.column(3, 40)
    assert np.all(np.isfinite(vals))
    reopened.close()

    ro = UniversalTables(store, read_only=True)
    _i0, vals = ro.column(3, 90)
    assert np.all(np.isfinite(vals))
    ro.close()


def test_both_index_forms_load(store: Path):
    """Old (off, n) and new (off, n, crc) entries must coexist in one
    level: a store part-written before the change and appended to after
    is the normal case, not an exotic one."""

    tab = UniversalTables(store)
    for r in (40, 90, 140):
        tab.column(3, r)
    tab.close()

    idx_f = store / "level_03.index.json"
    idx = json.loads(idx_f.read_text())
    idx["90"] = idx["90"][:2]                  # demote one to the old form
    idx_f.write_text(json.dumps(idx))

    tab = UniversalTables(store, read_only=True)
    for r in (40, 90, 140):
        _i0, vals = tab.column(3, r)
        assert np.all(np.isfinite(vals))
    rep = tab.verify_level(3)
    assert rep["bad"] == []
    assert rep["unchecked"] == [90]
    tab.close()


def test_legacy_columns_report_as_unchecked_not_healthy(store: Path):
    """A store written before checksums existed must not be reported
    clean.  Absence of a checksum is absence of evidence."""

    tab = UniversalTables(store)
    tab.column(3, 40)
    tab.close()
    idx_f = store / "level_03.index.json"
    idx = json.loads(idx_f.read_text())
    idx["40"] = idx["40"][:2]                            # old two-field form
    idx_f.write_text(json.dumps(idx))

    rep = UniversalTables(store, read_only=True).verify_level(3)
    assert rep["bad"] == []
    assert rep["unchecked"] == [40]


def test_a_designed_store_is_sealed(tmp_path: Path):
    """An anchor store must never grow.

    Its value is that its contents are a stated function of its grid, so
    one on-demand append makes it unreproducible while still looking
    like a designed object.  That is not hypothetical: pointing an
    ordinary run at an anchor store started rebuilding the whole cache
    inside it, and the only symptom was an apparent hang.

    anchors.json is the seal, and build_anchor_store.py writes it last,
    so a store is mutable exactly while it is being built.
    """

    root = tmp_path / "anch"
    tab = UniversalTables(root)
    tab.column(3, 40)
    tab.column(3, 90)
    tab.close()
    assert not UniversalTables(root).sealed        # not sealed while built

    (root / "anchors.json").write_text(json.dumps(
        {"factor": 0.5, "levels": {"3": {"anchors": [40, 90],
                                         "targets": []}}}))

    tab = UniversalTables(root)
    assert tab.sealed
    _i0, vals = tab.column(3, 40)                  # existing column: fine
    assert np.all(np.isfinite(vals))
    with pytest.raises(RuntimeError, match="SEALED|sealed"):
        tab.column(3, 41)                          # missing: must refuse
    with pytest.raises(RuntimeError, match="SEALED|sealed"):
        tab.ensure_columns(3, [41])
    tab.close()

    # and nothing was written on the way out
    idx = json.loads((root / "level_03.index.json").read_text())
    assert sorted(int(k) for k in idx) == [40, 90]
