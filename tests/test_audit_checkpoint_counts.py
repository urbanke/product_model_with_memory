import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_checkpoint_counts.py"
SPEC = importlib.util.spec_from_file_location("audit_checkpoint_counts", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_direct_counts_respects_interval_births():
    stream = np.asarray([4, 1, 3, 2, 1, 0], dtype=np.uint16)
    whole = MODULE.direct_counts(stream, 5, 0, 6)
    first = MODULE.direct_counts(stream, 5, 0, 3)
    second = MODULE.direct_counts(stream, 5, 3, 6)
    assert np.array_equal(whole["unigram"], first["unigram"] + second["unigram"])
    for label in ("ya", "yb", "ab"):
        combined = {}
        for keys, counts in (
            (first[f"keys_{label}"], first[f"counts_{label}"]),
            (second[f"keys_{label}"], second[f"counts_{label}"]),
        ):
            for key, count in zip(keys, counts):
                combined[int(key)] = combined.get(int(key), 0) + int(count)
        assert np.array_equal(whole[f"keys_{label}"], np.asarray(sorted(combined)))
        assert np.array_equal(
            whole[f"counts_{label}"],
            np.asarray([combined[key] for key in sorted(combined)]),
        )


def test_comparison_reports_mismatch():
    expected = {name: np.asarray([1, 2]) for name in MODULE.ARRAY_NAMES}
    actual = {name: value.copy() for name, value in expected.items()}
    actual["counts_ab"][1] = 3
    result = MODULE._comparison(expected, actual)
    assert not result["equal"]
    assert result["arrays"]["counts_ab"]["first_mismatch"] == 1
    assert result["arrays"]["counts_ab"]["mismatch_count"] == 1
