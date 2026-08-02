"""Exactly invertible tokenizer for raw byte files (TOKENIZER.md v3).

Segments a file into maximal ASCII-letter runs, maximal digit runs and
single other bytes, and emits the streams that our estimators model.
Every byte is accounted for: `decode(encode(raw)) == raw` for arbitrary
input, which the tests enforce on all switch combinations.

Two switches, both measured rather than assumed (see TOKENIZER.md):

  numbers = "intern"        digit runs are tokens in the vocabulary
          = "compositional" digits are ordinary single-byte segments

  case    = "conditioned"   token identity is the lowercased form; the
                            case class travels in its own stream
          = "folded"        case is part of the token identity

Alphabet layout of the token stream:

    0..255   the byte tokens (one per possible byte value)
    256      ESC_WORD   a letter run whose form is new
    257      ESC_NUM    a digit run whose form is new (intern only)
    258      EOF        end of the token stream; no header is needed
    259..    vocabulary entries, indexed by order of first occurrence

The decoder assigns indices by the same rule, so no vocabulary is ever
transmitted and nothing is added to a decompressor archive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BYTE_BASE = 0
ESC_WORD = 256
ESC_NUM = 257
EOF = 258
VOCAB_BASE = 259

CASE_LOWER, CASE_CAP, CASE_UPPER, CASE_MIXED = 0, 1, 2, 3
CASE_NAMES = ("lower", "Cap", "UPPER", "mixed")

KIND_OTHER, KIND_LETTER, KIND_DIGIT = 0, 1, 2


@dataclass
class Tokenized:
    """The streams plus what is needed to audit them."""

    tokens: np.ndarray                 # int32, ends with EOF
    word_spellings: list[bytes]        # one per ESC_WORD, in order
    num_spellings: list[bytes]         # one per ESC_NUM, in order
    case_classes: np.ndarray           # int8, one per letter run
    masks: list[np.ndarray]            # one per mixed run
    vocabulary: list[bytes]            # index i <-> token VOCAB_BASE + i
    numbers: str
    case: str
    n_bytes: int
    stats: dict = field(default_factory=dict)

    @property
    def alphabet_size(self) -> int:
        return VOCAB_BASE + len(self.vocabulary)


def _classify(raw: np.ndarray, numbers: str) -> np.ndarray:
    cls = np.zeros(len(raw), dtype=np.int8)
    is_letter = ((raw >= 65) & (raw <= 90)) | ((raw >= 97) & (raw <= 122))
    cls[is_letter] = KIND_LETTER
    if numbers == "intern":
        cls[(raw >= 48) & (raw <= 57)] = KIND_DIGIT
    return cls


def segment(raw: np.ndarray, numbers: str = "intern"):
    """Segment starts, ends and kinds.

    A new segment begins wherever the class changes, and additionally
    at every 'other' byte, since those are segments of length one.
    """

    n = len(raw)
    if n == 0:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64),
                np.zeros(0, np.int8))
    cls = _classify(raw, numbers)
    new = np.empty(n, dtype=bool)
    new[0] = True
    new[1:] = (cls[1:] != cls[:-1]) | (cls[1:] == KIND_OTHER)
    starts = np.flatnonzero(new)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = n
    return starts, ends, cls[starts]


def _case_classes(starts, ends, is_upper):
    """Case class per letter run.  The four classes partition: a single
    uppercase letter is Cap (UPPER requires length >= 2)."""

    lengths = ends - starts
    n_upper = np.add.reduceat(is_upper.astype(np.int64), starts)
    first_upper = is_upper[starts]
    out = np.full(len(starts), CASE_MIXED, dtype=np.int8)
    out[n_upper == 0] = CASE_LOWER
    out[first_upper & (n_upper == 1)] = CASE_CAP
    out[(n_upper == lengths) & (lengths >= 2)] = CASE_UPPER
    return out


def encode(raw, numbers: str = "intern",
           case: str = "conditioned") -> Tokenized:
    """Encode a byte array into the streams of TOKENIZER.md v3."""

    if numbers not in ("intern", "compositional"):
        raise ValueError(f"numbers={numbers!r}")
    if case not in ("conditioned", "folded"):
        raise ValueError(f"case={case!r}")
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    n = len(raw)
    starts, ends, kinds = segment(raw, numbers)
    n_seg = len(starts)

    is_upper = (raw >= 65) & (raw <= 90)
    low = raw.copy()
    low[is_upper] |= 0x20                      # ASCII lowercase

    letter_mask = kinds == KIND_LETTER
    l_starts, l_ends = starts[letter_mask], ends[letter_mask]
    if case == "conditioned" and len(l_starts):
        case_classes = _case_classes(l_starts, l_ends, is_upper)
    else:
        case_classes = np.zeros(0, dtype=np.int8)

    key_buf = (low if case == "conditioned" else raw).tobytes()

    tokens = np.empty(n_seg + 1, dtype=np.int32)
    # byte segments are the majority; fill them without a Python loop
    other_mask = kinds == KIND_OTHER
    if other_mask.any():
        tokens[:-1][other_mask] = raw[starts[other_mask]].astype(np.int32)

    vocab: list[bytes] = []
    index: dict[bytes, int] = {}
    word_spell: list[bytes] = []
    num_spell: list[bytes] = []
    masks: list[np.ndarray] = []

    # case classes are indexed by letter-run order; keep a running
    # counter over letter runs only
    letter_ordinal = np.cumsum(letter_mask) - 1
    want_mask = (case == "conditioned")

    for i in np.flatnonzero(~other_mask):
        i = int(i)
        s, e = int(starts[i]), int(ends[i])
        key = key_buf[s:e]
        idx = index.get(key)
        is_letter = kinds[i] == KIND_LETTER
        if idx is None:
            idx = len(vocab)
            index[key] = idx
            vocab.append(key)
            tokens[i] = ESC_WORD if is_letter else ESC_NUM
            (word_spell if is_letter else num_spell).append(key)
        else:
            tokens[i] = VOCAB_BASE + idx
        if is_letter and want_mask:
            li = int(letter_ordinal[i])
            if case_classes[li] == CASE_MIXED:
                masks.append(is_upper[s:e].astype(np.uint8).copy())
    tokens[-1] = EOF

    n_letter = int(letter_mask.sum())
    n_digit = int((kinds == KIND_DIGIT).sum())
    stats = {
        "segments": n_seg,
        "letter_runs": n_letter,
        "digit_runs": n_digit,
        "byte_segments": n_seg - n_letter - n_digit,
        "vocabulary": len(vocab),
        "new_words": len(word_spell),
        "new_numbers": len(num_spell),
        "mixed_runs": len(masks),
        "bytes_per_segment": n / max(n_seg, 1),
    }
    if case == "conditioned" and n_letter:
        hist = np.bincount(case_classes, minlength=4)
        stats["case_histogram"] = {CASE_NAMES[j]: int(hist[j])
                                   for j in range(4)}
    return Tokenized(
        tokens=tokens, word_spellings=word_spell, num_spellings=num_spell,
        case_classes=case_classes, masks=masks, vocabulary=vocab,
        numbers=numbers, case=case, n_bytes=n, stats=stats,
    )


def decode(tok: Tokenized) -> bytes:
    """Reconstruct the original bytes from the streams alone.

    Uses only what the framing provides: the token stream up to EOF,
    then the spellings, case classes and masks in order.  No header and
    no transmitted vocabulary; the decoder rebuilds the vocabulary by
    the same first-occurrence rule as the encoder, and remembers for
    each entry whether it came from a letter run (which consumes a case
    symbol) or a digit run (which does not).
    """

    out = bytearray()
    vocab: list[bytes] = []
    vocab_is_letter: list[bool] = []
    wi = ni = li = mi = 0
    conditioned = tok.case == "conditioned"

    for t in tok.tokens.tolist():
        if t == EOF:
            break
        if t < 256:
            out.append(t)
            continue
        if t == ESC_WORD:
            form = tok.word_spellings[wi]; wi += 1
            vocab.append(form); vocab_is_letter.append(True)
            is_letter = True
        elif t == ESC_NUM:
            form = tok.num_spellings[ni]; ni += 1
            vocab.append(form); vocab_is_letter.append(False)
            is_letter = False
        else:
            j = t - VOCAB_BASE
            form = vocab[j]
            is_letter = vocab_is_letter[j]
        if not (conditioned and is_letter):
            out += form
            continue
        cls = int(tok.case_classes[li]); li += 1
        if cls == CASE_LOWER:
            out += form
        elif cls == CASE_CAP:
            out += bytes([form[0] & 0xDF]) + form[1:]
        elif cls == CASE_UPPER:
            out += bytes(b & 0xDF for b in form)
        else:
            mask = tok.masks[mi]; mi += 1
            out += bytes((b & 0xDF) if m else b
                         for b, m in zip(form, mask))
    return bytes(out)
