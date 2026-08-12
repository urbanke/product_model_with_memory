"""Nested lag-state maps for unequal-alphabet anchored production runs.

The emission alphabet has ``V`` symbols.  A lag parameter ``M < V`` keeps
the first ``M`` transmitted reduced symbols distinct and maps every other
symbol to one shared backoff state, so its operational alphabet has ``M+1``
states.  ``M == V`` is the identity map and has exactly ``V`` states.  The
kept sets are nested, and their metadata is charged by the same enumerative
description used by the honest memory-two experiments.

This module contains no estimator.  It only defines deterministic metadata
and symbol maps; every resulting data-bearing sequence must still be coded by
``layered_depth_averaged_product_simplex_v1`` at its operational alphabet.
"""

from __future__ import annotations

import numpy as np

from product_model_with_memory.memory2_frontier import enumerative_subset_bits


def validate_anchored_alphabets(v: int, m1: int, m2: int) -> None:
    """Validate the unequal anchored grid, including the equal endpoint."""

    if v < 2 or not (1 <= m2 <= m1 <= v):
        raise ValueError("anchored alphabets require 1 <= M2 <= M1 <= V")


def state_alphabet_size(v: int, m: int) -> int:
    """Operational number of states for the declared top-``M`` map."""

    if not 1 <= m <= v:
        raise ValueError("state parameter must satisfy 1 <= M <= V")
    return v if m == v else m + 1


def map_reduced_context(values: np.ndarray, v: int, m: int) -> np.ndarray:
    """Apply the nested retained-symbol/backoff map to reduced IDs."""

    size = state_alphabet_size(v, m)
    source = np.asarray(values)
    if source.size and (int(source.min()) < 0 or int(source.max()) >= v):
        raise ValueError("context symbol lies outside the emission alphabet")
    if m == v:
        return source.astype(np.int64, copy=False)
    # Reduced IDs 0..M-1 name the transmitted retained subset; M is the
    # single shared backoff state.  The decoder reconstructs this map from
    # the nested subset descriptions charged below.
    mapped = np.minimum(source, m).astype(np.int64, copy=False)
    if mapped.size and int(mapped.max()) >= size:
        raise AssertionError("mapped context exceeds its declared alphabet")
    return mapped


def nested_state_subset_bits(v: int, m1: int, m2: int) -> float:
    """Metadata bits for the nested first- and second-lag retained sets.

    Unlike :func:`nested_frequency_subset_bits`, this accepts the equal cases
    needed by the anchored grid.  A full set costs zero; when ``M2 == M1``
    the second set is known once the first has been transmitted.
    """

    validate_anchored_alphabets(v, m1, m2)
    bits = enumerative_subset_bits(v, m1)
    if m2 < m1:
        bits += enumerative_subset_bits(m1, m2)
    return bits


def state_map_manifest(v: int, m1: int, m2: int) -> dict:
    """Immutable provenance recorded by every unequal campaign artifact."""

    validate_anchored_alphabets(v, m1, m2)
    return {
        "emission_vocabulary_size": int(v),
        "first_lag_parameter": int(m1),
        "second_lag_parameter": int(m2),
        "first_lag_alphabet_size": state_alphabet_size(v, m1),
        "second_lag_alphabet_size": state_alphabet_size(v, m2),
        "state_map": "nested_transmitted_subset_plus_shared_backoff_v1",
        "state_subset_description_bits": nested_state_subset_bits(v, m1, m2),
    }
