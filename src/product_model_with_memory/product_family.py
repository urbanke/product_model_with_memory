"""Order-two state maps with asymmetric resolution (product states).

Extension of :mod:`product_model_with_memory.state_family` to states built
from the previous TWO tokens, each at its own resolution:

    sigma_{M1,M2} :  s_t = ( b_{M1}(x_{t-1}),  b_{M2}(x_{t-2}) ),

where ``b_M`` maps a token to itself if among the top ``M`` tokens (by
frequency in the reduced stream) and to a shared backoff symbol otherwise;
``b_0`` maps everything to the backoff symbol.  Members:

  * (0, 0)   -- memoryless,
  * (M, 0)   -- the first-order family of ``state_family`` exactly,
  * (M1, M2) -- genuine order-two memory, with the resolution split across
                the two distances into the past chosen by the data.

The missing second-lag context before ``x_2`` is deterministically assigned
to the second-lag backoff bucket.  Consequently every member codes
``x_2 .. x_n`` and the complete ``(M, 0)`` slice is bit-for-bit identical to
the corresponding first-order member; it is not merely identical after
discarding ``x_2``.

The emission vocabulary V is fixed for all members, so every member is a
probability assignment on the same sequence space; each is scored by the
honest share-nothing construction (product over states of per-state
depth-averaged layered mixtures, exact from counts), and the family is a
uniform mixture over the supplied grid of (M1, M2) pairs with the posterior
reported.  All members code tokens x_2 .. x_n.  The unavailable second-lag
context for x_2 uses the declared backoff bucket, so codelengths are
comparable across members and the M2=0 identity is exact.

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
    for t in range(1, len(reduced)):
        # Before x_2 there is no second-lag symbol.  Sending that missing
        # context to the already-declared backoff bucket makes M2=0 exactly
        # the first-order model on the full stream, including x_2.
        lag2 = BACKOFF if t == 1 else _bucket(reduced[t - 2], keep2)
        state = (_bucket(reduced[t - 1], keep1), lag2)
        successors[state][reduced[t]] += 1
    multiset: Counter = Counter()
    for counts in successors.values():
        multiset[profile_of(counts)] += 1
    return multiset


def member_profile_multiset_ids(
    ids: np.ndarray, rank: np.ndarray, m1: int, m2: int
) -> Counter:
    """Multiset of successor profiles over the states of member
    (m1, m2), for an INTEGER symbol stream.

    ``rank[j]`` is the position of symbol ``j`` in the promotion order,
    so ``b_M(j)`` keeps ``j`` when ``rank[j] < M`` and sends it to the
    shared backoff state otherwise; ``min(rank, M)`` computes both at
    once with M as the backoff index.

    Same result as :func:`member_profile_multiset`, computed by sorting
    packed (state, successor) keys.  The dictionary path costs one
    Python iteration per position, which is fine for a few million
    tokens and impossible for the 2.7e8 of enwik9.
    """

    x1, y = ids[:-1], ids[1:]
    # `rank` is int64, so this is the one full-length int64 work array.  Pack
    # both lag buckets and the successor into it in place.  In particular, do
    # not materialize b1, b2, state, and key as four corpus-length arrays: on
    # enwik9 each such array costs about 2.2 GB although they encode the same
    # intermediate integer at successive stages.
    state = np.minimum(rank[x1], m1)
    if m2 > 0:
        state *= m2 + 1
        state[0] += m2  # the missing second lag uses the backoff bucket
        # Bound the only second-lag temporary independently of corpus length.
        chunk_size = 8_000_000
        for start in range(1, len(y), chunk_size):
            stop = min(start + chunk_size, len(y))
            state[start:stop] += np.minimum(
                rank[ids[start - 1:stop - 1]], m2
            )
    V = int(ids.max()) + 1
    # `state` need not be dense: multiplying its existing integer label by V
    # already gives a collision-free (state, successor) key.  The former
    # np.unique(..., return_inverse=True) allocated another full-length int64
    # array and then immediately discarded the dense-label interpretation.
    # Avoiding it is exact and materially lowers the enwik9 counting peak.
    state *= V
    state += y
    uniq, counts = np.unique(state, return_counts=True)
    owner = uniq // V
    cuts = np.flatnonzero(np.diff(owner)) + 1
    multiset: Counter = Counter()
    for g in np.split(counts, cuts):
        multiset[tuple(sorted((int(c) for c in g), reverse=True))] += 1
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
    state_order=None,
) -> dict:
    """Codelengths of every (M1, M2) member, their mixture, and the posterior.

    ``state_order`` lists symbols in the order they are promoted out of
    the backoff state, at BOTH lags.  Pass `streams.state_order_by_id`
    for the admissible family: that order is fixed by the vocabulary,
    so the decoder knows it before the file is seen.  Left as None the
    order is this file's own frequency ranking, which is not admissible
    without transmitting the ranking, and is kept only to reproduce the
    earlier text8 runs.
    """

    if l_max is None:
        l_max = default_l_max(vocabulary_size)
    n_coded = len(reduced) - 1  # tokens x_2 .. x_n are coded

    fast = isinstance(reduced, np.ndarray) and reduced.dtype.kind in "iu"
    if fast:
        # Keep the compact integer dtype produced by stream reduction.  Rank
        # lookup promotes the packed state keys to int64 where that width is
        # actually needed; converting every token here doubled the persistent
        # enwik9 stream allocation for no mathematical reason.
        ids = reduced
        n_sym = int(ids.max()) + 1
        if state_order is None:
            first = np.bincount(ids[1:-1], minlength=n_sym)
            state_order = np.argsort(-first, kind="stable")
        state_order = np.asarray(state_order, dtype=np.int64)
        rank = np.full(n_sym, n_sym, dtype=np.int64)
        inside = state_order[state_order < n_sym]
        rank[inside] = np.arange(len(inside))
    else:
        frequency = Counter(reduced)
        order_vocab = [w for w, _ in frequency.most_common()]

    members: dict[tuple[int, int], Counter] = {}
    unique_profiles: set[tuple[int, ...]] = set()
    for m1, m2 in grid:
        multiset = (member_profile_multiset_ids(ids, rank, m1, m2) if fast
                    else member_profile_multiset(reduced, order_vocab, m1, m2))
        members[(m1, m2)] = multiset
        unique_profiles.update(multiset)
        if progress is not None:
            progress(("member", len(members), len(grid)), None)
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
