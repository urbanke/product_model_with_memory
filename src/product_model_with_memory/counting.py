"""Memory-frugal counting (complexity notes, T4).

Replaces the hash-table counting (dicts of string tuples -> Counter,
measured ~200 GB at full vocabulary, depth 2) by SORTING: each
position's (context, next symbol) window is packed into fixed-width
integers, the packed keys are sorted, and every context's row is then
one contiguous run --- the same information the suffix-array idea
reads off suffix intervals, obtained with a few flat arrays of size n.
Memory: a small constant number of 8-byte-per-position arrays,
independent of how many distinct contexts exist.

Output is organized per depth d = 0..max_depth:
  * n_contexts[d]      --- number of distinct contexts at depth d
  * profile_id[d]      --- for each context (in sorted key order),
                            an index into `profiles`
  * parent[d]          --- for each context at depth d, the index of
                            its parent context at depth d-1 (drop the
                            oldest symbol)
  * group_of_position  --- (internal) which context each position
                            belongs to, used to build `parent`
plus the global deduplicated list `profiles`.

Only positions t >= max_depth are counted (full history), exactly as
the previous implementation did, so all downstream numbers match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ContextProfileTables:
    """Per-depth context structure + deduplicated profiles."""

    max_depth: int
    n_coded: int
    profiles: list[tuple[int, ...]]       # deduplicated
    profile_id: list[np.ndarray]          # depth -> (n_contexts,) int64
    parent: list[np.ndarray | None]       # depth -> (n_contexts,) or None
    n_contexts: list[int]

    def profile_of_context(self, d: int, i: int) -> tuple[int, ...]:
        return self.profiles[int(self.profile_id[d][i])]


def _digits_per_word(V: int) -> int:
    """How many base-V digits fit into a signed 64-bit word."""

    return max(1, int(62 // max(1.0, math.log2(V))))


def _pack_keys(ids: np.ndarray, V: int, d: int, start: int) -> list[np.ndarray]:
    """Packed (context, token) keys for depth d, one entry per position
    t = start..n-1, as a list of words, MOST significant word first.

    Digit order within the key (most to least significant):
    x_{t-d}, ..., x_{t-1}, x_t --- so that sorting groups by context
    (all digits but the last) and, inside a context, by next symbol.
    """

    n = len(ids)
    m = n - start
    digits = [ids[start - d + j:n - d + j] for j in range(d)]  # oldest first
    digits.append(ids[start:n])                                # the token
    per_word = _digits_per_word(V)
    words: list[np.ndarray] = []
    for w_start in range(0, len(digits), per_word):
        chunk = digits[w_start:w_start + per_word]
        word = np.zeros(m, dtype=np.int64)
        for dig in chunk:
            word *= V
            word += dig
        words.append(word)
    return words


def _pack_context_output_keys(
    context_ids: np.ndarray,
    output_ids: np.ndarray,
    context_V: int,
    d: int,
    start: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Pack context digits and a separately alphabetized output digit.

    Context words use base ``context_V``.  The output is deliberately a
    separate least-significant word, so its alphabet need not equal the
    context alphabet and no mixed-radix product can overflow.
    """

    n = len(output_ids)
    m = n - start
    digits = [
        context_ids[start - d + j:n - d + j] for j in range(d)
    ]
    context_words: list[np.ndarray] = []
    per_word = _digits_per_word(context_V)
    for w_start in range(0, len(digits), per_word):
        word = np.zeros(m, dtype=np.int64)
        for dig in digits[w_start:w_start + per_word]:
            word *= context_V
            word += dig
        context_words.append(word)
    if not context_words:
        context_words = [np.zeros(m, dtype=np.int64)]
    output_word = np.asarray(output_ids[start:n], dtype=np.int64)
    return context_words + [output_word], context_words


def _run_starts(words: list[np.ndarray], order: np.ndarray) -> np.ndarray:
    """Boolean array: True where a new distinct key starts in sorted
    order (first element always True)."""

    m = len(order)
    new = np.zeros(m, dtype=bool)
    new[0] = True
    for w in words:
        s = w[order]
        new[1:] |= s[1:] != s[:-1]
    return new


def context_profile_tables(
    ids, V: int, max_depth: int, *, context_ids=None,
    context_alphabet_size: int | None = None,
) -> ContextProfileTables:
    """Contexts, rows-as-profiles, and the tree structure for all
    depths 0..max_depth, by sorting packed keys."""

    ids = np.ascontiguousarray(ids, dtype=np.int64)
    n = len(ids)
    if n <= max_depth:
        raise ValueError("sequence shorter than max_depth")
    if np.any(ids < 0) or np.any(ids >= V):
        raise ValueError("ids outside [0, V)")
    separate_context = context_ids is not None
    if separate_context:
        context_ids = np.ascontiguousarray(context_ids, dtype=np.int64)
        if len(context_ids) != n:
            raise ValueError("context_ids and output ids must have equal length")
        if context_alphabet_size is None or context_alphabet_size < 1:
            raise ValueError("a positive context_alphabet_size is required")
        if (np.any(context_ids < 0)
                or np.any(context_ids >= context_alphabet_size)):
            raise ValueError("context_ids outside context alphabet")
    start = max_depth
    m = n - start

    profiles: list[tuple[int, ...]] = []
    profile_index: dict[bytes, int] = {}
    profile_id: list[np.ndarray] = []
    parent: list[np.ndarray | None] = []
    n_contexts: list[int] = []
    prev_group_of_pos: np.ndarray | None = None

    for d in range(0, max_depth + 1):
        if separate_context:
            words, ctx_words = _pack_context_output_keys(
                context_ids, ids, context_alphabet_size, d, start
            )
        else:
            words = _pack_keys(ids, V, d, start)
            # context part = all digits except the last output digit.
            ctx_words = [w.copy() for w in words]
            ctx_words[-1] //= V
        order = (np.argsort(words[0], kind="stable") if len(words) == 1
                 else np.lexsort(words[::-1]))
        new_key = _run_starts(words, order)
        new_ctx = _run_starts(ctx_words, order)
        del ctx_words

        run_id = np.cumsum(new_key) - 1          # per sorted position
        run_counts = np.bincount(run_id)          # count per (ctx, token)
        run_ctx = np.cumsum(new_ctx) - 1          # context of each sorted pos
        n_ctx = int(run_ctx[-1]) + 1
        # context id per RUN (first sorted position of each run)
        first_of_run = np.flatnonzero(new_key)
        ctx_of_run = run_ctx[first_of_run]

        # group id per position (original order), for the parent join
        group_of_pos = np.empty(m, dtype=np.int64)
        group_of_pos[order] = run_ctx

        # profiles: per context, the sorted (descending) run counts
        pid = np.empty(n_ctx, dtype=np.int64)
        ctx_run_starts = np.flatnonzero(
            np.concatenate([[True], ctx_of_run[1:] != ctx_of_run[:-1]]))
        ctx_run_ends = np.concatenate([ctx_run_starts[1:], [len(run_counts)]])
        for g in range(n_ctx):
            row = run_counts[ctx_run_starts[g]:ctx_run_ends[g]]
            key = np.sort(row)[::-1].astype(np.int64).tobytes()
            idx = profile_index.get(key)
            if idx is None:
                idx = len(profiles)
                profile_index[key] = idx
                profiles.append(tuple(
                    int(x) for x in np.frombuffer(key, dtype=np.int64)))
            pid[g] = idx
        profile_id.append(pid)
        n_contexts.append(n_ctx)

        # parent linkage: the parent of a depth-d context is the
        # depth-(d-1) context that any of its positions belongs to
        if d == 0:
            parent.append(None)
        else:
            rep_pos = np.empty(n_ctx, dtype=np.int64)
            rep_pos[run_ctx] = order  # last write per group suffices
            parent.append(prev_group_of_pos[rep_pos])
        prev_group_of_pos = group_of_pos

    return ContextProfileTables(
        max_depth=max_depth,
        n_coded=m,
        profiles=profiles,
        profile_id=profile_id,
        parent=parent,
        n_contexts=n_contexts,
    )
