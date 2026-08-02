"""Token streams: one interface for every representation.

The memory experiments (pairs, state-map families, context trees, lag
pools) were written against the \\texttt{text8} word tokenization and
could see nothing else.  This module lets the same machinery run on any
representation --- raw bytes, our tokenizer's token stream, a pretrained
BPE stream --- by reducing all of them to the same object: a sequence of
integer symbol ids plus the metadata needed to turn bits per token back
into bits per character of the ORIGINAL file.

That metadata is the part that is easy to get wrong, so it is explicit:

  n_bytes      length of the original file, the denominator of every
               bits-per-character figure
  alphabet     the full alphabet the representation defines, BEFORE any
               top-K capping done by an experiment
  fixed_bits   the cost of everything the memory model does NOT touch.
               For our tokenizer that is the spelling, case and mask
               streams: a first-order model over the token stream leaves
               them unchanged, so they are added back unmodified.  For
               bytes it is zero.  For BPE it is zero when the vocabulary
               is not charged, and the zipped vocabulary size when it is.

A capped experiment (top-K + <unk>) does not define a decodable code on
its own --- an <unk> does not say which symbol occurred --- exactly as
in the existing text8 experiments, where the gap is closed by the
full-fidelity accounting.  `reduce_ids` therefore reports how many
positions were capped, so the omission is visible rather than implicit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_stream(path, ids, *, representation: str, source_file: str,
                n_bytes: int, alphabet: int, fixed_bits: float = 0.0,
                notes: str = "") -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    ids = np.ascontiguousarray(ids, dtype=np.int32)
    np.save(path / "ids.npy", ids)
    (path / "stream.json").write_text(json.dumps({
        "representation": representation,
        "source_file": source_file,
        "n_bytes": int(n_bytes),
        "n_tokens": int(ids.size),
        "alphabet": int(alphabet),
        "distinct_used": int(np.unique(ids).size),
        "bytes_per_token": n_bytes / max(int(ids.size), 1),
        "fixed_bits": float(fixed_bits),
        "notes": notes,
    }, indent=2))


def load_stream(path) -> tuple[np.ndarray, dict]:
    path = Path(path)
    meta = json.loads((path / "stream.json").read_text())
    return np.load(path / "ids.npy"), meta


def reduce_ids(ids: np.ndarray, top_k: int, *, return_keep: bool = False):
    """Keep the ``top_k`` most frequent ids, map the rest to one <unk>.

    Returns (reduced token list, V = top_k + 1, positions capped), and
    with ``return_keep`` also ``keep``: the ORIGINAL vocabulary id of
    each reduced id, so ``keep[j]`` is the id that reduced id ``j``
    stands for.  Callers need it to build a state map that does not
    depend on this file's frequencies --- see `state_order_by_id`.

    The reduced ids run 0..top_k-1 in frequency order with top_k as
    <unk>, so the alphabet is dense and V is exactly what the estimator
    needs.  That relabelling is itself frequency-derived, but it cannot
    change a codelength: the model is exchangeable over symbols, so
    permuting the alphabet permutes the count profile and leaves q
    alone.  It matters only where something SELECTS symbols by their
    reduced id, which is precisely what a state map does.
    """

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    counts = np.bincount(ids.astype(np.int64))
    order = np.argsort(-counts, kind="stable")
    keep = order[:top_k][counts[order[:top_k]] > 0]
    lut = np.full(counts.size, top_k, dtype=np.int32)
    lut[keep] = np.arange(len(keep), dtype=np.int32)
    reduced = lut[ids.astype(np.int64)]
    capped = int(np.count_nonzero(reduced == top_k))
    if return_keep:
        return reduced, top_k + 1, capped, keep
    return reduced, top_k + 1, capped


def state_order_by_id(keep: np.ndarray) -> np.ndarray:
    """Reduced ids ordered by ASCENDING vocabulary id.

    A family of state maps has to be fixed before the file is seen, or
    the decoder cannot reproduce it and the code is not admissible.
    Ranking the previous symbol by its frequency IN THIS FILE fails
    that test: the ranking of 100,277 symbols would have to be
    transmitted, about 1.5 million bits, 0.19 bits per character on
    enwik8 --- more than the entire first-order gain.

    Vocabulary id is free.  `cl100k_base` assigns ids in merge order, so
    a low id is a subword that was frequent in the tokenizer's training
    corpus, and that ordering is part of the vocabulary already charged
    for in `fixed_bits`.  "Give a state to the symbols whose id is below
    M" is therefore a map both encoder and decoder know in advance, and
    a reasonable proxy for frequency without being derived from the
    data.
    """

    return np.argsort(np.asarray(keep), kind="stable")


def bits_per_character(total_bits: float, meta: dict) -> float:
    """Bits per character of the ORIGINAL file, including the streams the
    memory model does not touch."""

    return (total_bits + meta.get("fixed_bits", 0.0)) / meta["n_bytes"]
