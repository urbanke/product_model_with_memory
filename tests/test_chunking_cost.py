"""Checkpoint-grid invariants for the Markov staleness audit."""

import numpy as np

from scripts.chunking_cost import edges_for, raw_predictive_log2_normalizer


def test_equal_suffix_holds_first_checkpoint_fixed():
    edges = edges_for(10_000, 16, "equal_suffix", first=536)
    assert len(edges) == 17
    assert edges[:2] == [0, 536]
    assert edges[-1] == 10_000
    suffix_widths = np.diff(edges[1:])
    assert int(suffix_widths.max() - suffix_widths.min()) <= 1


def test_geometric_grid_holds_first_checkpoint_fixed():
    edges = edges_for(10_000, 16, "geo", first=536)
    assert len(edges) == 17
    assert edges[:2] == [0, 536]
    assert edges[-1] == 10_000
    assert all(right > left for left, right in zip(edges, edges[1:]))


def test_count_symmetric_row_normalizer_counts_unseen_multiplicity():
    # Two observed symbols have probabilities 1/2 and 1/4.  The other two
    # symbols are unseen and each has probability 1/8.
    probabilities_by_count = {3: 0.5, 1: 0.25, 0: 0.125}
    value = raw_predictive_log2_normalizer(
        4,
        (3, 1),
        lambda count: np.log2(probabilities_by_count[count]),
    )
    assert abs(value) < 1e-14


def test_row_normalizer_exposes_numerical_mass_error():
    # Multiplying every symbol probability by the same factor should report
    # exactly that factor; this is the per-token correction production adds.
    scale = 1.01
    probabilities_by_count = {
        3: 0.5 * scale,
        1: 0.25 * scale,
        0: 0.125 * scale,
    }
    value = raw_predictive_log2_normalizer(
        4,
        (3, 1),
        lambda count: np.log2(probabilities_by_count[count]),
    )
    assert abs(value - np.log2(scale)) < 1e-14
