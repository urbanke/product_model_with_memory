import math

import numpy as np
import pytest

from product_model_with_memory.anchored_state_maps import (
    map_reduced_context,
    nested_state_subset_bits,
    state_alphabet_size,
    state_map_manifest,
    validate_anchored_alphabets,
)
from product_model_with_memory.memory2_frontier import enumerative_subset_bits


def test_equal_endpoint_is_identity_and_free():
    x = np.array([0, 3, 7], dtype=np.int32)
    assert state_alphabet_size(8, 8) == 8
    assert np.array_equal(map_reduced_context(x, 8, 8), x)
    assert nested_state_subset_bits(8, 8, 8) == 0.0


def test_nested_maps_use_one_shared_backoff_state():
    x = np.array([0, 1, 2, 3, 7], dtype=np.int32)
    assert state_alphabet_size(8, 3) == 4
    assert np.array_equal(map_reduced_context(x, 8, 3), [0, 1, 2, 3, 3])


def test_asymmetric_nested_subset_charge():
    expected = enumerative_subset_bits(16, 16) + enumerative_subset_bits(16, 8)
    assert math.isclose(nested_state_subset_bits(16, 16, 8), expected)
    manifest = state_map_manifest(16, 16, 8)
    assert manifest["first_lag_alphabet_size"] == 16
    assert manifest["second_lag_alphabet_size"] == 9
    assert manifest["state_subset_description_bits"] == expected


@pytest.mark.parametrize("point", [(1, 1, 1), (8, 0, 0), (8, 4, 5), (8, 9, 4)])
def test_invalid_alphabets_are_rejected(point):
    with pytest.raises(ValueError):
        validate_anchored_alphabets(*point)
