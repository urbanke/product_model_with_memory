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


def member_state_profiles_ids(
    ids: np.ndarray, state_order: np.ndarray, m: int
) -> dict:
    """Successor profiles per state, for an INTEGER symbol stream.

    Same result as :func:`member_state_profiles`, computed by sorting
    packed (state, successor) keys instead of filling a dictionary of
    counters.  The dictionary path costs one Python iteration per
    adjacent pair and one dict entry per distinct pair, which is fine
    for a few million tokens and impossible for the 4.3e8 tokens and
    1.8e6 states of enwik9 under our tokenizer.
    """

    a, b = ids[:-1], ids[1:]
    V = int(ids.max()) + 1
    if m > 0:
        lut = np.full(V, -1, dtype=np.int64)
        keep = state_order[:m]
        lut[keep] = keep
        st = lut[a]
    else:
        st = np.full(len(a), -1, dtype=np.int64)
    # np.unique sorts internally, so sorting first would only add a
    # full extra copy of the key array --- 2.2 GB at enwik9 scale
    uniq, counts = np.unique((st + 1) * V + b.astype(np.int64),
                             return_counts=True)
    states = uniq // V - 1
    cuts = np.flatnonzero(np.diff(states)) + 1
    groups = np.split(counts, cuts)
    heads = states[np.concatenate(([0], cuts))] if len(states) else states
    return {
        (BACKOFF if int(sid) < 0 else int(sid)):
            tuple(sorted((int(c) for c in g), reverse=True))
        for sid, g in zip(heads, groups)
    }


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
    state_order=None,
) -> dict:
    """Codelengths of every family member, their mixture, and the posterior.

    ``state_order`` lists symbols in the order they are promoted to
    states, so member M gives its own state to ``state_order[:M]`` and
    pools the rest.  Pass `streams.state_order_by_id` for the ADMISSIBLE
    family: that order is fixed by the vocabulary and so is known to the
    decoder before the file is seen.  Left as None the order is this
    file's own frequency ranking, which is not admissible without
    transmitting the ranking (about 1.5 million bits at V = 100,277);
    it is kept for comparison with the earlier text8 runs.
    """

    if l_max is None:
        l_max = default_l_max(vocabulary_size)
    n_coded = len(reduced) - 1  # tokens x_2 .. x_n are coded

    fast = isinstance(reduced, np.ndarray) and reduced.dtype.kind in "iu"
    if fast:
        ids = reduced.astype(np.int64, copy=False)
        first = np.bincount(ids[:-1], minlength=int(ids.max()) + 1)
        if state_order is None:
            state_order = np.argsort(-first, kind="stable")
        else:
            state_order = np.asarray(state_order, dtype=np.int64)
        # symbols that never occur as a state would contribute an empty
        # profile and shift every M by one for nothing
        state_order = state_order[first[state_order] > 0]
    else:
        # frequency order of states (first coordinates of pairs)
        first_counts = Counter(reduced[:-1])
        state_vocab = [w for w, _ in first_counts.most_common()]

    # all state profiles across all members, evaluated in one pass
    profiles: dict[tuple[int, str], tuple[int, ...]] = {}
    for m in m_grid:
        per_state = (member_state_profiles_ids(ids, state_order, m) if fast
                     else member_state_profiles(reduced, state_vocab, m))
        for state, prof in per_state.items():
            profiles[(m, state)] = prof

    # Distinct profiles only.  A state's codelength depends on its
    # profile and nothing else, and across a grid of M values most
    # states share a profile --- (1) and (1, 1) alone account for the
    # bulk of a heavy-tailed alphabet --- so evaluating per state would
    # repeat the same integral thousands of times.
    distinct = {prof: prof for prof in profiles.values()}
    evaluated: Mapping = depth_averaged_codelength_profiles(
        distinct,
        d=vocabulary_size,
        l_max=l_max,
        cache_dir=cache_dir,
        jobs=jobs,
        progress=progress,
    )

    member_log2_q: dict[int, float] = {}
    member_states: dict[int, int] = {}
    for (m, _state), prof in profiles.items():
        member_log2_q[m] = member_log2_q.get(m, 0.0) + evaluated[prof].log2_q_avg
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
