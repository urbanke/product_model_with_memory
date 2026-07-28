"""Order-two state maps with asymmetric resolution (product states).

Extension of :mod:`product_model_with_memory.state_family` to states built
from the previous TWO tokens, each at its own resolution:

    sigma_{M1,M2} :  s_t = ( b_{M1}(x_{t-1}),  b_{M2}(x_{t-2}) ),

where ``b_M`` maps a token to itself if among the top ``M`` tokens (by
frequency in the reduced stream) and to a shared backoff symbol otherwise;
``b_0`` maps everything to the backoff symbol.  Members:

  * (0, 0)   -- memoryless,
  * (M, 0)   -- the first-order family of ``state_family``,
  * (M1, M2) -- genuine order-two memory, with the resolution split across
                the two distances into the past chosen by the data.

The emission vocabulary V is fixed for all members, so every member is a
probability assignment on the same sequence space; each is scored by the
honest share-nothing construction (product over states of per-state
depth-averaged layered mixtures, exact from counts), and the family is a
uniform mixture over the supplied grid of (M1, M2) pairs with the posterior
reported.  All members code tokens x_3 .. x_n (the first two tokens carry
no complete order-two context), so codelengths are comparable across
members.

Computation: successor profiles are deduplicated globally --- rare contexts
overwhelmingly share identical tiny profiles such as (1,), (1,1), (2,) ---
and each unique profile is evaluated once; a member's codelength is the
multiplicity-weighted sum.  This collapses hundreds of thousands of states
into a few thousand evaluations.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from product_model_with_memory.codelength import (
    default_l_max,
    depth_averaged_codelength_profiles,
    profile_of,
)

BACKOFF = "<other>"


def _bucket(token: str, keep: set) -> str:
    return token if token in keep else BACKOFF


def member_profile_multiset(
    reduced: Sequence[str],
    order_vocab: Sequence[str],
    m1: int,
    m2: int,
) -> Counter:
    """Multiset of successor-count profiles over states of member (m1, m2).

    Returns ``Counter{profile_tuple: number_of_states_with_this_profile}``.
    """

    keep1 = set(order_vocab[:m1]) if m1 > 0 else set()
    keep2 = set(order_vocab[:m2]) if m2 > 0 else set()
    successors: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for t in range(2, len(reduced)):
        state = (_bucket(reduced[t - 1], keep1), _bucket(reduced[t - 2], keep2))
        successors[state][reduced[t]] += 1
    multiset: Counter = Counter()
    for counts in successors.values():
        multiset[profile_of(counts)] += 1
    return multiset


def product_family_codelengths(
    reduced: Sequence[str],
    *,
    vocabulary_size: int,
    grid: Sequence[tuple[int, int]],
    l_max: int | None = None,
    cache_dir: str | Path,
    jobs: int = 1,
    progress=None,
) -> dict:
    """Codelengths of every (M1, M2) member, their mixture, and the posterior."""

    if l_max is None:
        l_max = default_l_max(vocabulary_size)
    n_coded = len(reduced) - 2  # tokens x_3 .. x_n are coded

    frequency = Counter(reduced)
    order_vocab = [w for w, _ in frequency.most_common()]

    members: dict[tuple[int, int], Counter] = {}
    unique_profiles: set[tuple[int, ...]] = set()
    for m1, m2 in grid:
        multiset = member_profile_multiset(reduced, order_vocab, m1, m2)
        members[(m1, m2)] = multiset
        unique_profiles.update(multiset)
    if progress is not None:
        progress(("profiles", len(unique_profiles), len(unique_profiles)), None)

    results: Mapping[tuple[int, ...], object] = (
        depth_averaged_codelength_profiles(
            {p: p for p in unique_profiles},
            d=vocabulary_size,
            l_max=l_max,
            cache_dir=cache_dir,
            jobs=jobs,
            progress=progress,
        )
    )
    log2_q_of = {p: res.log2_q_avg for p, res in results.items()}

    member_log2_q: dict[tuple[int, int], float] = {}
    member_states: dict[tuple[int, int], int] = {}
    for key, multiset in members.items():
        member_log2_q[key] = sum(
            mult * log2_q_of[p] for p, mult in multiset.items()
        )
        member_states[key] = sum(multiset.values())

    keys = list(members)
    logs = np.array([member_log2_q[k] for k in keys])
    prior = -math.log2(len(keys))
    family_log2_q = _log2sumexp(logs + prior)
    posterior = np.exp((logs - logs.max()) * math.log(2.0))
    posterior /= posterior.sum()

    return {
        "n_coded": n_coded,
        "l_max": l_max,
        "grid": keys,
        "member_bits_per_token": {
            k: -member_log2_q[k] / n_coded for k in keys
        },
        "member_states_observed": member_states,
        "unique_profiles": len(unique_profiles),
        "family_bits_per_token": -family_log2_q / n_coded,
        "posterior": {k: float(p) for k, p in zip(keys, posterior)},
        "best_member": min(keys, key=lambda k: -member_log2_q[k]),
    }


def _log2sumexp(values: np.ndarray) -> float:
    m = float(np.max(values))
    return m + math.log2(float(np.sum(np.exp((values - m) * math.log(2.0)))))
