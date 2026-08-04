"""Acceptance tests for the tokenizer (TOKENIZER.md v3).

Test 1 of the specification (round trip) and test 2 (framing
self-sufficiency: the decoder may use only the streams, never the
encoder's vocabulary), plus the claim that the four case classes
partition.
"""

import itertools

import numpy as np

from product_model_with_memory.tokenizer import (
    CASE_CAP,
    CASE_LOWER,
    CASE_MIXED,
    CASE_UPPER,
    Tokenized,
    decode,
    encode,
)

COMBOS = list(itertools.product(["intern", "compositional"],
                                ["conditioned", "folded"]))

CASES = {
    "empty": b"",
    "letters": b"hello world",
    "mixed case": b"The Quick BROWN fox McDonald's iPhone eBay A I",
    "digits": b"2004-03-11T14:23:56Z id=12345 2004 2004 007",
    "markup": b'<page>\n  <title>Anarchism</title>\n  <id>12</id>\n</page>',
    "utf8": "café naïve 日本語".encode(),
    "single letters": b"A a Z z 0 9 ~",
    "all byte values": bytes(range(256)),
    "repeats": b"the The THE tHe " * 200,
}


def test_round_trip_all_combinations():
    for data in CASES.values():
        raw = np.frombuffer(data, dtype=np.uint8)
        for numbers, case in COMBOS:
            tok = encode(raw, numbers, case)
            assert decode(tok) == data, (numbers, case, data[:40])


def test_round_trip_random_bytes():
    rng = np.random.default_rng(7)
    for _ in range(5):
        raw = rng.integers(0, 256, size=50_000, dtype=np.uint8)
        for numbers, case in COMBOS:
            assert decode(encode(raw, numbers, case)) == raw.tobytes()


def test_framing_self_sufficiency():
    """The decoder must not need the encoder's vocabulary: it rebuilds
    it from the spellings by the same first-occurrence rule."""

    raw = np.frombuffer(CASES["markup"] + CASES["digits"], dtype=np.uint8)
    for numbers, case in COMBOS:
        tok = encode(raw, numbers, case)
        stripped = Tokenized(
            tokens=tok.tokens, word_spellings=tok.word_spellings,
            num_spellings=tok.num_spellings, case_classes=tok.case_classes,
            masks=tok.masks, vocabulary=[],          # <- withheld
            numbers=tok.numbers, case=tok.case, n_bytes=tok.n_bytes,
        )
        assert decode(stripped) == raw.tobytes()


def test_case_classes_partition():
    raw = np.frombuffer(
        b"lower Cap UPPER mIxEd A a AB ab aB Ab", dtype=np.uint8)
    tok = encode(raw, "intern", "conditioned")
    cls = list(tok.case_classes)
    assert set(cls) <= {CASE_LOWER, CASE_CAP, CASE_UPPER, CASE_MIXED}
    assert len(cls) == tok.stats["letter_runs"]
    # a single uppercase letter is Cap, never UPPER
    words = b"lower Cap UPPER mIxEd A a AB ab aB Ab".split()
    for w, c in zip(words, cls):
        if len(w) == 1 and w.isupper():
            assert c == CASE_CAP


def test_vocabulary_is_first_occurrence_ordered():
    raw = np.frombuffer(b"beta alpha beta gamma alpha", dtype=np.uint8)
    tok = encode(raw, "intern", "conditioned")
    assert tok.vocabulary[:3] == [b"beta", b"alpha", b"gamma"]


def test_numbers_switch_changes_segmentation_not_content():
    raw = np.frombuffer(b"year 2004 and 12345", dtype=np.uint8)
    a = encode(raw, "intern", "conditioned")
    b = encode(raw, "compositional", "conditioned")
    assert decode(a) == decode(b) == raw.tobytes()
    assert a.stats["digit_runs"] == 2
    assert b.stats["digit_runs"] == 0
    assert b.stats["segments"] > a.stats["segments"]
