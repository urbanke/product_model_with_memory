import math

from product_model_with_memory.memory2_frontier import (
    MemoryTwoFrontier,
    MemoryTwoPoint,
    declared_triplet_grid,
    nested_frequency_subset_bits,
)


def _search():
    return MemoryTwoFrontier(
        vocabulary_grid=[64, 128, 256],
        state_grid=[0, 16, 32, 64, 128, 256],
        baseline_point=MemoryTwoPoint(128, 128, 0),
        baseline_bpc=1.0,
    )


def test_identity_gate_is_first_and_hard():
    search = _search()
    assert search.next_points() == (MemoryTwoPoint(128, 128, 0),)
    try:
        search.record(MemoryTwoPoint(128, 128, 0), 1.01)
    except RuntimeError as exc:
        assert "identity gate failed" in str(exc)
    else:
        raise AssertionError("a broken M2=0 identity was accepted")


def test_seed_ray_starts_at_16_and_stops_at_first_reversal():
    search = _search()
    search.record(MemoryTwoPoint(128, 128, 0), 1.0)
    assert search.next_points() == (MemoryTwoPoint(128, 128, 16),)
    search.record(MemoryTwoPoint(128, 128, 16), 0.90)
    assert search.next_points() == (MemoryTwoPoint(128, 128, 32),)
    search.record(MemoryTwoPoint(128, 128, 32), 0.85)
    assert search.next_points() == (MemoryTwoPoint(128, 128, 64),)
    search.record(MemoryTwoPoint(128, 128, 64), 0.87)

    # The worsening 64 point still beats the ORIGINAL 1.0 baseline, hence
    # remains a winner even though it closes the special upward ray.  Its
    # ordinary radius-two neighborhood therefore still contains M2=128.
    assert MemoryTwoPoint(128, 128, 64) in search.winners
    assert MemoryTwoPoint(128, 128, 128) in search.next_points()


def test_every_point_below_original_baseline_expands_and_no_repeat_occurs():
    search = _search()
    arm = [
        (MemoryTwoPoint(128, 128, 0), 1.0),
        (MemoryTwoPoint(128, 128, 16), 0.90),
        (MemoryTwoPoint(128, 128, 32), 0.95),
    ]
    for point, bpc in arm:
        assert search.next_points() == (point,)
        search.record(point, bpc)

    # 0.95 is worse than the current best 0.90 but is still a winner because
    # the fixed reference is 1.0.
    assert set(search.winners) == {
        MemoryTwoPoint(128, 128, 16),
        MemoryTwoPoint(128, 128, 32),
    }
    pending = search.next_points()
    assert pending
    assert not (set(pending) & set(search.results))
    assert len(pending) == len(set(pending))

    # Exhaust the finite frontier with losing results.  Memoization makes the
    # stopping condition explicit and leaves no point scheduled twice.
    seen = set(search.results)
    while not search.complete:
        batch = search.next_points()
        assert not (seen & set(batch))
        for point in batch:
            search.record(point, 1.1)
            seen.add(point)
    audit = search.audit()
    assert audit["complete"] is True
    assert audit["pending"] == []


def test_original_center_allows_two_distinct_coordinates_not_factor_four():
    search = _search()
    search.record(MemoryTwoPoint(128, 128, 0), 1.0)
    search.record(MemoryTwoPoint(128, 128, 16), 1.1)
    pending = set(search.next_points())

    assert search.winners == ()
    assert search.expansion_centers == (MemoryTwoPoint(128, 128, 0),)
    # A two-step diagonal from the original point is required.
    assert MemoryTwoPoint(128, 64, 16) in pending
    # Repeating a move on M2 would produce 32, but that is not a permitted
    # order-two neighbor.  It is reached only by an improving upward arm.
    assert MemoryTwoPoint(128, 128, 32) not in pending
    # A move on V and a move on M1 is a genuine order-two neighbor.
    assert MemoryTwoPoint(64, 64, 0) in pending


def test_declared_triplet_grid_is_deduplicated_and_enforces_strict_lag_order():
    points = declared_triplet_grid(
        vocabulary_grid=[64, 64, 100],
        first_lag_grid=[16, 32, 64, 100],
        second_lag_grid=[8, 16, 32, 64],
        minimum_second_lag=16,
    )
    assert len(points) == len(set(points))
    assert all(16 <= p.second_lag_states < p.first_lag_states
               <= p.vocabulary_size for p in points)
    assert MemoryTwoPoint(64, 32, 16) in points
    assert MemoryTwoPoint(64, 16, 16) not in points
    assert MemoryTwoPoint(64, 100, 16) not in points


def test_nested_subset_accounting_uses_second_set_inside_first_set():
    point = MemoryTwoPoint(64, 32, 16)
    expected = (
        math.log2(math.comb(64, 32))
        + math.log2(math.comb(32, 16))
    )
    assert abs(nested_frequency_subset_bits(point) - expected) < 1e-10
