"""Corpus loading, counts, and empirical entropies for token sequences."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Sequence


def load_tokens(path: str | Path) -> list[str]:
    """Load a whitespace-tokenized corpus (e.g. text8) as a list of tokens."""

    return Path(path).read_text().split()


def empirical_entropy_bits(counts: Counter[str] | dict[str, int]) -> float:
    """Empirical unigram entropy (bits/token) of the sequence behind ``counts``."""

    n = sum(counts.values())
    if n == 0:
        return 0.0
    return -sum(v / n * math.log2(v / n) for v in counts.values() if v > 0)


def empirical_conditional_entropy_bits(
    tokens: Sequence[str],
) -> tuple[float, float, int, int]:
    """Empirical H(next | prev) for a token sequence, in bits/token.

    Returns ``(h_conditional, h_pair, distinct_bigrams, n_pairs)`` where
    ``h_pair`` is the entropy of the sliding-window pair distribution and
    ``h_conditional = h_pair - h_unigram_of_first_coordinate``.
    """

    if len(tokens) < 2:
        return 0.0, 0.0, 0, 0
    pair_counts = Counter(zip(tokens[:-1], tokens[1:]))
    first_counts = Counter(tokens[:-1])
    n = len(tokens) - 1
    h_pair = -sum(v / n * math.log2(v / n) for v in pair_counts.values())
    h_first = -sum(v / n * math.log2(v / n) for v in first_counts.values())
    return h_pair - h_first, h_pair, len(pair_counts), n


def prefix_counts(tokens: Sequence[str], n: int) -> Counter[str]:
    """Counts of the first ``n`` tokens."""

    if n > len(tokens):
        raise ValueError(f"prefix {n} exceeds corpus length {len(tokens)}")
    return Counter(tokens[:n])
