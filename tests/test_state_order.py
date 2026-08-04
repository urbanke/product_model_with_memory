"""The family of state maps has to be fixed before the file is seen.

Member M of the family gives its own state to M symbols and pools the
rest.  Which M symbols is part of the code, so the decoder has to know
it in advance.  Ranking by frequency IN THE FILE fails that: the
decoder has not seen the file, and transmitting a ranking of 100,277
symbols costs about 1.5 million bits, 0.19 bits per character on
enwik8 --- more than the whole first-order gain.

Ranking by vocabulary id passes: `cl100k_base` fixes the ids, and the
vocabulary is already charged for in `fixed_bits`.

These tests pin the distinction down operationally.  A data-independent
map must select the SAME symbols when the data changes but the
vocabulary does not; a frequency map must not.  The first test is the
one that matters --- it is the property the admissibility argument
rests on --- and the second exists so that a change making both orders
identical cannot pass silently.
"""

import numpy as np

from product_model_with_memory.state_family import member_state_profiles_ids
from product_model_with_memory.streams import reduce_ids, state_order_by_id


def _stream(pairs):
    """A stream over vocabulary ids from (id, count) pairs, laid out so
    that vocabulary id order and frequency order DISAGREE."""

    out = []
    for token, count in pairs:
        out.extend([token] * count)
    return np.array(out, dtype=np.int64)


# id 7 is the rarest, id 900 the most frequent: the two orders are
# reversed, so a test cannot pass by accident
PAIRS = [(7, 5), (11, 20), (300, 60), (900, 200)]


def _promoted(ids, top_k, how, m):
    """The vocabulary ids member M gives their own state."""

    reduced, _V, _capped, keep = reduce_ids(ids, top_k, return_keep=True)
    if how == "id":
        order = state_order_by_id(keep)
    else:
        first = np.bincount(reduced[:-1], minlength=len(keep) + 1)
        order = np.argsort(-first[: len(keep)], kind="stable")
    return {int(keep[j]) for j in order[:m]}


def test_id_order_is_independent_of_the_data():
    """Change the counts, keep the vocabulary: the map must not move."""

    a = _stream(PAIRS)
    b = _stream([(7, 400), (11, 3), (300, 9), (900, 2)])   # order reversed
    for m in (1, 2, 3):
        assert _promoted(a, 4, "id", m) == _promoted(b, 4, "id", m)


def test_id_order_takes_the_smallest_ids():
    ids = _stream(PAIRS)
    assert _promoted(ids, 4, "id", 1) == {7}
    assert _promoted(ids, 4, "id", 2) == {7, 11}
    assert _promoted(ids, 4, "id", 3) == {7, 11, 300}


def test_frequency_order_does_move_with_the_data():
    """The contrast: the inadmissible map depends on the file, which is
    exactly why it cannot be used without paying for the ranking."""

    a = _stream(PAIRS)
    b = _stream([(7, 400), (11, 3), (300, 9), (900, 2)])
    assert _promoted(a, 4, "frequency", 1) != _promoted(b, 4, "frequency", 1)


def test_m_zero_and_m_full_do_not_depend_on_the_order():
    """The published rows must be untouched.  M = 0 is the memoryless
    model and M = full is plain first order; in both the state map is
    the same function whichever order names the symbols."""

    ids = _stream(PAIRS)
    reduced, _V, _capped, keep = reduce_ids(ids, 4, return_keep=True)
    by_id = state_order_by_id(keep)
    first = np.bincount(reduced[:-1], minlength=len(keep) + 1)
    by_freq = np.argsort(-first[: len(keep)], kind="stable")
    for m in (0, len(keep)):
        pid = member_state_profiles_ids(reduced, by_id, m)
        pfq = member_state_profiles_ids(reduced, by_freq, m)
        assert sorted(pid.values()) == sorted(pfq.values()), m
