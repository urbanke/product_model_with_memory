"""Averaging over a nested family of state maps (memory as a depth-like axis).

The family: fix the emission vocabulary (top-K tokens + <unk>, V = K + 1
symbols) and define, for each M in a grid, the state map

    sigma_M : previous token  ->  itself if among the top M tokens,
                                  else one shared backoff state,

so member M has at most M + 1 states.  M = 0 collapses to a single state:
the memoryless (unigram) predictor.  The family is nested and spans "no
memory" to "full first-order memory over V".

The predictor per member is the share-nothing construction of the companion
paper's Section 5.4: conditioned on the state, the successor sub-sequence is
exchangeable, so

    Q^(sigma_M)(x^n)  =  prod_states  q_avg(successor profile of the state),

each factor the per-state depth-averaged layered mixture, evaluated exactly
from counts.  This is an honest sequential code.

The family average is the uniform mixture Q = (1/|grid|) sum_M Q^(sigma_M);
its codelength is a log-sum-exp of member codelengths and the posterior over
M is reported.  Predictions to check: the family tracks the best member to
within log2 |grid| / n, and the member codelengths show an interior optimum
in M (the data-determined effective memory) where the posterior concentrates.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from product_model_with_memory.codelength import (
    DepthAveragedCodelength,
    default_l_max,
    depth_averaged_codelength_profiles,
    profile_of,
)

BACKOFF = "<other>"


def member_state_profiles(
    reduced: Sequence[str], state_vocab: Sequence[str], m: int
) -> dict[str, tuple[int, ...]]:
    """Successor count profiles per state for the member with top-``m`` states.

    ``state_vocab`` lists candidate state tokens in frequency order (the
    frequency order of the *first* coordinates of the sliding pairs).
    """

    keep = set(state_vocab[:m]) if m > 0 else set()
    successors: dict[str, Counter] = defaultdict(Counter)
    for a, b in zip(reduced[:-1], reduced[1:]):
        state = a if a in keep else BACKOFF
        successors[state][b] += 1
    return {
        state: profile_of(counts) for state, counts in successors.items()
    }


def state_family_codelengths(
    reduced: Sequence[str],
    *,
    vocabulary_size: int,
    m_grid: Sequence[int],
    l_max: int | None = None,
    cache_dir: str | Path,
    jobs: int = 1,
    progress=None,
) -> dict:
    """Codelengths of every family member, their mixture, and the posterior."""

    if l_max is None:
        l_max = default_l_max(vocabulary_size)
    n_coded = len(reduced) - 1  # tokens x_2 .. x_n are coded

    # frequency order of states (first coordinates of pairs)
    first_counts = Counter(reduced[:-1])
    state_vocab = [w for w, _ in first_counts.most_common()]

    # all state profiles across all members, evaluated in one pass
    profiles: dict[tuple[int, str], tuple[int, ...]] = {}
    for m in m_grid:
        for state, prof in member_state_profiles(
            reduced, state_vocab, m
        ).items():
            profiles[(m, state)] = prof

    results: Mapping[tuple[int, str], DepthAveragedCodelength] = (
        depth_averaged_codelength_profiles(
            profiles,
            d=vocabulary_size,
            l_max=l_max,
            cache_dir=cache_dir,
            jobs=jobs,
            progress=progress,
        )
    )

    member_log2_q: dict[int, float] = {}
    member_states: dict[int, int] = {}
    for (m, _state), res in results.items():
        member_log2_q[m] = member_log2_q.get(m, 0.0) + res.log2_q_avg
        member_states[m] = member_states.get(m, 0) + 1

    grid = sorted(m_grid)
    logs = np.array([member_log2_q[m] for m in grid])
    prior = -math.log2(len(grid))
    family_log2_q = _log2sumexp(logs + prior)
    posterior = np.exp((logs - logs.max()) * math.log(2.0))
    posterior /= posterior.sum()

    return {
        "n_coded": n_coded,
        "l_max": l_max,
        "m_grid": grid,
        "member_bits_per_token": {
            m: -member_log2_q[m] / n_coded for m in grid
        },
        "member_states_observed": member_states,
        "family_bits_per_token": -family_log2_q / n_coded,
        "posterior_over_m": {
            m: float(p) for m, p in zip(grid, posterior)
        },
        "best_member": min(grid, key=lambda m: -member_log2_q[m]),
    }


def _log2sumexp(values: np.ndarray) -> float:
    m = float(np.max(values))
    return m + math.log2(float(np.sum(np.exp((values - m) * math.log(2.0)))))
