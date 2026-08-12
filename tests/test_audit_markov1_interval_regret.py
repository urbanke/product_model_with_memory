"""Unit tests for the production-artifact Markov-1 regret audit."""

from collections import Counter

import numpy as np
import pytest

from scripts.audit_markov1_interval_regret import profile_multiplicities
from scripts.calibrate_markov1_sampling import context_profiles_at_stops


def test_profile_multiplicities_groups_rows_and_identical_profiles():
    # V=4: row 0 has counts (3,1), row 2 has the same profile in the
    # opposite symbol order, and row 3 has profile (2,).
    keys = np.asarray([0, 2, 9, 11, 14], dtype=np.int64)
    counts = np.asarray([3, 1, 1, 3, 2], dtype=np.int64)
    assert profile_multiplicities(keys, counts, 4) == Counter({
        (1, 3): 2,
        (2,): 1,
    })


def test_profile_multiplicities_rejects_unsorted_keys():
    with pytest.raises(ValueError, match="unique, sorted"):
        profile_multiplicities(
            np.asarray([4, 1]), np.asarray([1, 1]), 4
        )


def test_sparse_profiles_at_stops_match_direct_transition_counts():
    stream = np.asarray([0, 1, 0, 2, 0, 1], dtype=np.int64)
    profiles = context_profiles_at_stops(stream, {3, 6}, 3)
    assert profiles[3] == Counter({(1,): 2})
    assert profiles[6] == Counter({(1,): 2, (2, 1): 1})
