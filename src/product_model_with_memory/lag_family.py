"""Lag family: single-lag state maps sigma_delta(s_t) = x_{t-delta}.

Level 0 of the pooled-lag-experts scheme (paper, outlook scheme 5).
Each member conditions the next token on the token exactly delta steps
back, at full vocabulary resolution; the member is scored by the honest
share-nothing construction (per-state depth-averaged layered mixtures,
evaluated exactly from successor count profiles).  All members code
x_{delta_max+1} .. x_n --- the tokens for which every member's state is
defined --- so they are mixable, and the uniform mixture's posterior
over delta is the DISTANCE PROFILE of memory value: how predictive
information decays with the gap.  A memoryless member (delta = 0 by
convention: a single state) anchors the family.

Profiles are deduplicated globally across members, as in the product
family: distant lags produce near-unigram rows that overwhelmingly
coincide.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from product_model_with_memory.codelength import (
    default_l_max,
    depth_averaged_codelength_profiles,
    profile_of,
)


def member_profile_multiset(
    reduced: Sequence[str], delta: int, delta_max: int
) -> Counter:
    """Multiset {successor profile: multiplicity} for the lag-delta member.

    Codes tokens x_t for t >= delta_max (0-indexed), conditioning on
    x_{t-delta}; delta = 0 means a single unconditional state.
    """

    counts: dict = defaultdict(Counter)
    n = len(reduced)
    if delta == 0:
        counts[()] = Counter(reduced[delta_max:])
    else:
        for t in range(delta_max, n):
            counts[reduced[t - delta]][reduced[t]] += 1
    multiset: Counter = Counter()
    for c in counts.values():
        multiset[profile_of(c)] += 1
    return multiset


def lag_family_codelengths(
    reduced: Sequence[str],
    *,
    vocabulary_size: int,
    deltas: Sequence[int],
    l_max: int | None = None,
    cache_dir: str | Path,
    jobs: int = 1,
    progress=None,
) -> dict:
    """Codelengths of all lag members, the uniform mixture, and posterior."""

    if l_max is None:
        l_max = default_l_max(vocabulary_size)
    delta_max = max(deltas)
    n_coded = len(reduced) - delta_max

    multisets = {
        d: member_profile_multiset(reduced, d, delta_max) for d in deltas
    }
    unique = set()
    for m in multisets.values():
        unique.update(m.keys())
    if progress is not None:
        total_states = sum(len(m) for m in multisets.values())
        progress(("profiles", len(unique), total_states), None)

    results = depth_averaged_codelength_profiles(
        {p: p for p in unique},
        d=vocabulary_size,
        l_max=l_max,
        cache_dir=cache_dir,
        jobs=jobs,
        progress=progress,
    )

    member_bits = {}
    states_observed = {}
    for d, multiset in multisets.items():
        total = sum(
            results[p].log2_q_avg * mult for p, mult in multiset.items()
        )
        member_bits[d] = -total / n_coded
        states_observed[d] = sum(multiset.values())

    # uniform mixture over members and posterior over delta
    log_terms = {
        d: -bits * n_coded - math.log2(len(deltas))
        for d, bits in member_bits.items()
    }
    m = max(log_terms.values())
    log_mix = m + math.log2(
        sum(2.0 ** (v - m) for v in log_terms.values())
    )
    family_bits = -log_mix / n_coded
    posterior = {
        d: 2.0 ** (v - log_mix) for d, v in log_terms.items()
    }

    return {
        "n_coded": n_coded,
        "l_max": l_max,
        "deltas": list(deltas),
        "unique_profiles": len(unique),
        "states_observed": states_observed,
        "member_bits_per_token": member_bits,
        "family_bits_per_token": family_bits,
        "posterior": posterior,
    }
