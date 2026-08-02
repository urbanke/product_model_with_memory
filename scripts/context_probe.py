#!/usr/bin/env python3
"""Is first-order memory worth anything on this representation?

A first-order model conditions each symbol on the one before it.  That
is a very different amount of information depending on what a symbol
is: on raw bytes the predecessor is a genuine local context, but on our
tokenizer $63\\%$ of symbols are single punctuation bytes, so a word's
predecessor is usually a space or a bracket and carries almost nothing.

This script settles that with empirical entropies, which cost one pass
and no model.  For each stream it reports

  H(X)                     the memoryless target
  H(X | previous symbol)   what first order can reach
  H(X | previous CONTENT)  what first order could reach if delimiters
                           were skipped when forming the state

and the same three restricted to positions where the symbol being
predicted is itself content, since that is the part we care about.  The
difference between the second and the third is exactly the cost of
spending our context on delimiters --- if it is small the objection is
unfounded, and if it is large the state map has to skip them.

    python scripts/context_probe.py --ids output/streams/ours_enwik8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from product_model_with_memory.streams import load_stream


def _entropy(counts: np.ndarray) -> float:
    c = counts[counts > 0].astype(np.float64)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def _conditional_entropy(states: np.ndarray, symbols: np.ndarray) -> float:
    """H(symbol | state), empirical, in bits."""

    n_sym = int(symbols.max()) + 1
    key = states.astype(np.int64) * n_sym + symbols.astype(np.int64)
    # counts of the OBSERVED pairs only: bincount would allocate one
    # cell per (state, symbol) combination, which is 2.4e10 cells for a
    # 155k vocabulary
    joint = np.unique(key, return_counts=True)[1].astype(np.float64)
    n = joint.sum()
    h_joint = -(joint / n * np.log2(joint / n)).sum()
    st = np.unique(states, return_counts=True)[1].astype(np.float64)
    h_state = -(st / n * np.log2(st / n)).sum()
    return float(h_joint - h_state)


def content_mask(ids: np.ndarray, representation: str) -> np.ndarray:
    """Which symbols carry content rather than acting as separators."""

    if representation == "ours":
        # ids 0..255 are single "other" bytes; 256/257 are the escapes
        # that introduce a word or a number, 258 is EOF, 259+ are
        # vocabulary entries
        return (ids >= 256) & (ids != 258)
    if representation == "bytes":
        # whitespace and the common markup punctuation of XML
        sep = np.zeros(256, dtype=bool)
        for b in b" \t\n\r<>/=\"'&;:,.|[]{}()*#-":
            sep[b] = True
        return ~sep[np.clip(ids, 0, 255)]
    # a BPE vocabulary glues leading spaces onto its subwords, so it has
    # no separate delimiter symbols at all --- which is itself the point
    return np.ones(len(ids), dtype=bool)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", required=True)
    p.add_argument("--n", type=int, default=None)
    args = p.parse_args()

    ids, meta = load_stream(args.ids)
    if args.n:
        ids = ids[: args.n]
    rep = meta["representation"]
    ids = ids.astype(np.int64)
    is_content = content_mask(ids, rep)

    # dense relabelling so bincount stays small
    uniq, dense = np.unique(ids, return_inverse=True)
    dense = dense.astype(np.int64)

    # state 1: the previous symbol
    prev = dense[:-1]
    nxt = dense[1:]

    # state 2: the most recent CONTENT symbol strictly before each
    # position (delimiters skipped); positions before the first content
    # symbol are dropped
    idx = np.where(is_content, np.arange(len(ids)), -1)
    last_content = np.maximum.accumulate(idx)
    # the most recent content symbol STRICTLY BEFORE position k+1 is
    # last_content[k]; using last_content[k+1] would include the target
    # itself whenever the target is content, making the state a copy of
    # the symbol and the conditional entropy exactly zero
    state_idx = last_content[:-1]
    valid = state_idx >= 0
    prev_content = dense[state_idx[valid]]
    nxt_c = dense[1:][valid]

    h0 = _entropy(np.bincount(dense))
    h1 = _conditional_entropy(prev, nxt)
    h2 = _conditional_entropy(prev_content, nxt_c)

    # restricted to predicting a content symbol
    sel = is_content[1:]
    h0r = _entropy(np.bincount(dense[1:][sel]))
    h1r = _conditional_entropy(prev[sel], nxt[sel])
    selv = is_content[1:][valid]
    h2r = _conditional_entropy(prev_content[selv], nxt_c[selv])

    frac_delim_state = float(np.mean(~is_content[:-1]))
    out = {
        "representation": rep,
        "n_symbols": int(len(ids)),
        "fraction_of_states_that_are_delimiters": frac_delim_state,
        "all_symbols": {
            "H": h0, "H_given_prev": h1, "H_given_prev_content": h2,
            "first_order_gain": h0 - h1,
            "gain_if_delimiters_skipped": h0 - h2,
        },
        "content_symbols_only": {
            "H": h0r, "H_given_prev": h1r, "H_given_prev_content": h2r,
            "first_order_gain": h0r - h1r,
            "gain_if_delimiters_skipped": h0r - h2r,
        },
    }
    print(json.dumps(out, indent=2))
    Path(args.ids, "context_probe.json").write_text(json.dumps(out, indent=2))

    b = meta.get("bytes_per_token", 1.0)
    print(f"\n  {rep}: {100*frac_delim_state:.1f}% of states are delimiters")
    print(f"  predicting everything : first order buys "
          f"{h0-h1:.4f} bits/symbol ({(h0-h1)/b:.4f} bits/char), "
          f"{h0-h2:.4f} if delimiters are skipped")
    print(f"  predicting content    : first order buys "
          f"{h0r-h1r:.4f} bits/symbol, "
          f"{h0r-h2r:.4f} if delimiters are skipped")
    print("\n  CAUTION: these are PLUG-IN entropies.  H(X|state) is badly")
    print("  biased downward when the state space is large and many pairs")
    print("  occur once, so the 'delimiters skipped' figures are optimistic")
    print("  and must not be quoted as achievable.  What is NOT biased is")
    print("  the comparison of the two state maps at equal state-space")
    print("  size, and the first column, where the state space is the same")
    print("  as the alphabet.  The honest numbers come from the layered")
    print("  model, which pays the learning cost these do not.")


if __name__ == "__main__":
    main()
