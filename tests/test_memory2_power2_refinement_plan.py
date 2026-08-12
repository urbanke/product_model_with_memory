from scripts.plan_memory2_power2_refinement import build_plan


def test_power2_refinement_is_frozen_complete_and_power_of_two():
    plan = build_plan()
    points = [tuple(values) for values in plan["triplets"]]

    assert plan["triplet_count"] == 20
    assert len(points) == len(set(points)) == 20
    assert {point[0] for point in points} == {32768, 65536}
    assert {point[1] for point in points} == {16384, 32768, 65536}
    assert {point[2] for point in points} == {128, 256, 512, 1024}
    assert all(m2 < m1 <= vocabulary for vocabulary, m1, m2 in points)
    assert all(
        value > 0 and value & (value - 1) == 0
        for point in points
        for value in point
    )
    assert {key: len(value) for key, value in
            plan["grids_by_vocabulary"].items()} == {
                "32768": 8,
                "65536": 12,
            }
