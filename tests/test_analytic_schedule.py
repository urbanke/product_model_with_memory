import numpy as np

from product_model_with_memory.analytic_schedule import (
    MoldableTask, checkpoint_tasks, checkpoint_work_profile,
    expected_compatible_triangles,
    expected_pair_support, geometric_prefixes, worker_modes, worker_speedup,
    plan_moldable_tasks,
)


def test_support_estimates_are_monotone_and_bounded():
    p = np.asarray([0.5, 0.3, 0.2])
    pair = [expected_pair_support(n, p) for n in (1, 10, 10000)]
    triangle = [
        expected_compatible_triangles(n, p) for n in (1, 10, 10000)
    ]
    assert pair[0] < pair[1] < pair[2] <= 9
    assert triangle[0] < triangle[1] < triangle[2] <= 27


def test_work_profile_uses_incremental_C_G_and_interval_E():
    rows = checkpoint_work_profile(
        np.asarray([10, 30, 100]), np.asarray([0.6, 0.4]),
        stochastic_steps=20, replicas=4, blocks=8, exact_interval=5,
    )
    assert [row.prefix for row in rows] == [10, 30, 100]
    assert all(row.construction > 0 and row.graph >= 0 for row in rows)
    assert [row.evaluation for row in rows] == [20.0, 70.0, None]
    assert all(row.fitting > row.expected_triangles for row in rows)


def test_initial_worker_prior_is_concave_and_saturates_after_four():
    speed = [worker_speedup(p) for p in range(1, 9)]
    assert speed[:4] == [1.0, 1.9, 2.25, 2.4]
    assert speed[4:] == [2.4] * 4
    assert worker_modes(12) == (1, 2, 3, 4)


def test_geometric_prefixes_end_at_stream_length():
    prefixes = geometric_prefixes(10_000, 8, first_prefix=200)
    assert len(prefixes) == 8
    assert np.all(prefixes[1:] > prefixes[:-1])
    assert prefixes[-1] == 10_000


def test_moldable_plan_respects_chains_and_capacity():
    tasks = (
        MoldableTask("C0", 20),
        MoldableTask("C1", 30, ("C0",)),
        MoldableTask("F0", 40, ("C0",)),
        MoldableTask("F1", 50, ("C1", "F0")),
        MoldableTask("E0", 10, ("C1", "F0"), maximum_workers=1),
    )
    plan = plan_moldable_tasks(tasks, 4)
    by_id = {row.task_id: row for row in plan}
    assert by_id["C1"].start >= by_id["C0"].finish
    assert by_id["F1"].start >= max(
        by_id["C1"].finish, by_id["F0"].finish
    )
    events = sorted({row.start for row in plan} | {row.finish for row in plan})
    for time in events:
        assert sum(
            row.workers for row in plan if row.start <= time < row.finish
        ) <= 4


def test_checkpoint_tasks_encode_the_actual_dependency_graph():
    profile = checkpoint_work_profile(
        np.asarray([10, 30, 100]), np.asarray([0.6, 0.4]),
        stochastic_steps=20, replicas=4, blocks=8, exact_interval=5,
    )
    tasks = {task.task_id: task for task in checkpoint_tasks(profile)}
    assert tasks["C1"].dependencies == ("C0",)
    assert tasks["G1"].dependencies == ("C1", "G0")
    assert tasks["F1"].dependencies == ("G1", "F0")
    assert tasks["E1"].dependencies == ("F1", "C2")
    assert "E2" not in tasks
    assert tasks["G1"].maximum_workers == 1
    assert tasks["E1"].maximum_workers == 1
