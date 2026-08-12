import numpy as np

from product_model_with_memory.anchored_pair_graph import (
    ANCHORED_PAIR_GRAPH_INITIALIZER,
    ANCHORED_PAIR_GRAPH_MODEL,
    anchored_joint,
    anchored_intersection_dual,
    anchored_intersection_sampled_gradient,
    anchored_intersection_sgd,
    anchored_problem_from_intersection,
    anchored_sparse_dual,
    anchored_sparse_sgd,
    cold_pair_midpoint,
    cold_pair_midpoint_from_grouped,
    dense_full_pair_dual,
    dense_layered_pair,
    full_implicit_pair_problem,
    full_implicit_pair_sampled_gradient,
    full_implicit_pair_sgd,
    full_implicit_log_probabilities,
    full_implicit_validation_objective,
    implicit_ya_intersection_dual,
    implicit_ya_log_probabilities,
    implicit_ya_problem_from_grouped,
    implicit_ya_sampled_gradient,
    sparse_problem_from_dense,
)
from product_model_with_memory.graphical_calibration import (
    SparseGroupedProblem,
    SparseProjectedPair,
    build_sparse_intersection_plan,
)


def test_anchored_joint_preserves_ya_for_arbitrary_soft_factors():
    p_ya = np.array([[0.18, 0.07], [0.22, 0.13], [0.16, 0.24]])
    p_ab = np.array([[0.21, 0.04, 0.09], [0.08, 0.31, 0.27]])
    u = np.array([[0.3, -0.4, 0.1], [-0.2, 0.7, -0.1], [0.5, 0.2, -0.8]])
    v = np.array([[0.1, -0.3, 0.8], [-0.5, 0.4, 0.2]])

    joint = anchored_joint(p_ya, p_ab, u, v)

    np.testing.assert_allclose(joint.sum(axis=2), p_ya / p_ya.sum(), atol=1e-14)
    np.testing.assert_allclose(joint.sum(), 1.0, atol=1e-14)


def test_zero_corrections_reduce_to_markov_one_times_b_given_a():
    p_ya = np.array([[0.3, 0.1], [0.2, 0.4]])
    p_ab = np.array([[0.1, 0.2, 0.1], [0.3, 0.1, 0.2]])
    joint = anchored_joint(p_ya, p_ab, np.zeros((2, 3)))
    conditional_y = joint / joint.sum(axis=0, keepdims=True)
    markov_one = p_ya / p_ya.sum(axis=0, keepdims=True)
    np.testing.assert_allclose(
        conditional_y,
        np.broadcast_to(markov_one[:, :, None], conditional_y.shape),
        atol=1e-14,
    )


def test_cold_midpoint_is_deterministic_and_has_neutral_ab_factor():
    p_ya = np.array([[0.3, 0.1], [0.2, 0.4]])
    p_yb = np.array([[0.2, 0.15, 0.05], [0.1, 0.2, 0.3]])
    p_ab = np.array([[0.1, 0.2, 0.1], [0.3, 0.1, 0.2]])
    first = cold_pair_midpoint(p_ya, p_yb, p_ab)
    second = cold_pair_midpoint(p_ya, p_yb, p_ab)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[1], np.zeros_like(p_ab))
    assert ANCHORED_PAIR_GRAPH_MODEL == "anchored_ya_relaxed_pair_graph_v1"
    assert ANCHORED_PAIR_GRAPH_INITIALIZER == "cold_pair_midpoint_v1"


def test_sparse_dual_gradient_matches_finite_differences():
    p_ya = np.array([[0.18, 0.07], [0.22, 0.13], [0.16, 0.24]])
    p_yb = np.array([[0.11, 0.09, 0.05], [0.08, 0.14, 0.13], [0.12, 0.10, 0.18]])
    p_ab = np.array([[0.21, 0.04, 0.09], [0.08, 0.31, 0.27]])
    problem = sparse_problem_from_dense(p_ya, p_yb, p_ab)
    rng = np.random.default_rng(17)
    u = rng.normal(scale=0.2, size=len(problem.target_yb))
    v = rng.normal(scale=0.2, size=len(problem.target_ab))
    value, gu, gv = anchored_sparse_dual(
        problem, u, v, slack_precision=3.0,
    )
    assert np.isfinite(value)
    vector = np.r_[u, v]
    analytic = np.r_[gu, gv]
    epsilon = 1e-6
    numeric = np.empty_like(vector)
    for i in range(len(vector)):
        plus = vector.copy()
        minus = vector.copy()
        plus[i] += epsilon
        minus[i] -= epsilon
        vp = anchored_sparse_dual(
            problem, plus[:len(u)], plus[len(u):], slack_precision=3.0,
        )[0]
        vm = anchored_sparse_dual(
            problem, minus[:len(u)], minus[len(u):], slack_precision=3.0,
        )[0]
        numeric[i] = (vp - vm) / (2.0 * epsilon)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-7, atol=2e-8)


def test_sparse_model_margins_match_dense_reference():
    p_ya = np.array([[0.3, 0.1], [0.2, 0.4]])
    p_yb = np.array([[0.2, 0.15, 0.05], [0.1, 0.2, 0.3]])
    p_ab = np.array([[0.1, 0.2, 0.1], [0.3, 0.1, 0.2]])
    problem = sparse_problem_from_dense(p_ya, p_yb, p_ab)
    u = np.linspace(-0.3, 0.4, len(problem.target_yb))
    v = np.linspace(0.2, -0.25, len(problem.target_ab))
    _, gu, gv = anchored_sparse_dual(problem, u, v)
    dense_u = np.zeros_like(p_yb, dtype=float)
    dense_u[problem.yb_y, problem.yb_b] = u
    dense_v = np.zeros_like(p_ab, dtype=float)
    dense_v[problem.ab_a, problem.ab_b] = v
    joint = anchored_joint(p_ya, p_ab, dense_u, dense_v)
    model_yb = joint.sum(axis=1)
    model_ab = joint.sum(axis=0)
    np.testing.assert_allclose(
        gu + problem.target_yb,
        model_yb[problem.yb_y, problem.yb_b], atol=1e-14,
    )
    np.testing.assert_allclose(
        gv + problem.target_ab,
        model_ab[problem.ab_a, problem.ab_b], atol=1e-14,
    )


def test_cold_start_sgd_is_reproducible_and_improves_exact_objective():
    p_ya = np.array([[0.18, 0.07], [0.22, 0.13], [0.16, 0.24]])
    p_yb = np.array([[0.11, 0.09, 0.05], [0.08, 0.14, 0.13], [0.12, 0.10, 0.18]])
    p_ab = np.array([[0.21, 0.04, 0.09], [0.08, 0.31, 0.27]])
    problem = sparse_problem_from_dense(p_ya, p_yb, p_ab)
    dense_u, _ = cold_pair_midpoint(p_ya, p_yb, p_ab)
    initial_u = dense_u[problem.yb_y, problem.yb_b]
    initial_v = np.zeros_like(problem.target_ab)
    kwargs = dict(
        steps=80, batch_size=64, seed=1234, learning_rate=0.02,
        slack_precision=3.0,
    )
    first = anchored_sparse_sgd(problem, initial_u, initial_v, **kwargs)
    second = anchored_sparse_sgd(problem, initial_u, initial_v, **kwargs)
    np.testing.assert_array_equal(first.correction_yb, second.correction_yb)
    np.testing.assert_array_equal(first.correction_ab, second.correction_ab)
    assert first.best_objective < first.initial_objective
    assert first.batch_size == 64
    assert first.seed == 1234


def test_existing_intersection_graph_adapter_matches_sparse_oracle():
    p_ya = np.array([[0.18, 0.07], [0.22, 0.13], [0.16, 0.24]])
    p_yb = np.array([[0.11, 0.09, 0.05], [0.08, 0.14, 0.13], [0.12, 0.10, 0.18]])
    p_ab = np.array([[0.21, 0.04, 0.09], [0.08, 0.31, 0.27]])
    oracle = sparse_problem_from_dense(p_ya, p_yb, p_ab)
    ya_y, ya_a = np.nonzero(p_ya)
    yb_y, yb_b = np.nonzero(p_yb)
    ab_a, ab_b = np.nonzero(p_ab)
    grouped = SparseGroupedProblem(
        vocabulary_size=3,
        edge_a=ab_a, edge_b=ab_b,
        edge_probability=p_ab[ab_a, ab_b] / p_ab.sum(),
        target_y=p_ya.sum(axis=1) / p_ya.sum(),
        active_ya_y=ya_y, active_ya_a=ya_a,
        target_ya=p_ya[ya_y, ya_a] / p_ya.sum(),
        active_yb_y=yb_y, active_yb_b=yb_b,
        target_yb=p_yb[yb_y, yb_b] / p_yb.sum(),
    )
    plan = build_sparse_intersection_plan(grouped)
    adapted = anchored_problem_from_intersection(grouped, plan)
    rng = np.random.default_rng(919)
    u = rng.normal(scale=0.2, size=len(grouped.target_yb))
    v = rng.normal(scale=0.2, size=len(grouped.edge_probability))
    expected = anchored_sparse_dual(
        oracle, u, v, slack_precision=2.5,
    )
    actual = anchored_intersection_dual(
        adapted, u, v, slack_precision=2.5,
    )
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-14)
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-14)
    np.testing.assert_allclose(actual[2], expected[2], atol=1e-14)
    sparse_start = cold_pair_midpoint_from_grouped(grouped)
    dense_start = cold_pair_midpoint(p_ya, p_yb, p_ab)
    np.testing.assert_allclose(
        sparse_start[0], dense_start[0][yb_y, yb_b], atol=1e-14,
    )
    np.testing.assert_array_equal(sparse_start[1], np.zeros(len(ab_a)))


def test_intersection_baseline_handles_inactive_yb_cells():
    p_ya = np.array([[0.3, 0.1], [0.2, 0.4]])
    p_yb = np.array([[0.25, 0.0, 0.15], [0.0, 0.25, 0.35]])
    p_ab = np.array([[0.1, 0.2, 0.1], [0.3, 0.1, 0.2]])
    ya_y, ya_a = np.nonzero(p_ya)
    yb_y, yb_b = np.nonzero(p_yb)
    ab_a, ab_b = np.nonzero(p_ab)
    grouped = SparseGroupedProblem(
        vocabulary_size=3,
        edge_a=ab_a, edge_b=ab_b, edge_probability=p_ab[ab_a, ab_b],
        target_y=p_ya.sum(axis=1),
        active_ya_y=ya_y, active_ya_a=ya_a, target_ya=p_ya[ya_y, ya_a],
        active_yb_y=yb_y, active_yb_b=yb_b, target_yb=p_yb[yb_y, yb_b],
    )
    adapted = anchored_problem_from_intersection(
        grouped, build_sparse_intersection_plan(grouped),
    )
    u = np.array([0.2, -0.3, 0.4, -0.1])
    v = np.linspace(-0.2, 0.25, len(ab_a))
    _, gu, gv = anchored_intersection_dual(adapted, u, v)
    dense_u = np.zeros_like(p_yb)
    dense_u[yb_y, yb_b] = u
    dense_v = np.zeros_like(p_ab)
    dense_v[ab_a, ab_b] = v
    joint = anchored_joint(p_ya, p_ab, dense_u, dense_v)
    np.testing.assert_allclose(
        gu + grouped.target_yb,
        joint.sum(axis=1)[yb_y, yb_b], atol=1e-14,
    )
    np.testing.assert_allclose(
        gv + grouped.edge_probability,
        joint.sum(axis=0)[ab_a, ab_b], atol=1e-14,
    )


def test_sampled_intersection_gradient_is_worker_invariant():
    p_ya = np.array([[0.3, 0.1], [0.2, 0.4]])
    p_yb = np.array([[0.25, 0.0, 0.15], [0.0, 0.25, 0.35]])
    p_ab = np.array([[0.1, 0.2, 0.1], [0.3, 0.1, 0.2]])
    ya_y, ya_a = np.nonzero(p_ya)
    yb_y, yb_b = np.nonzero(p_yb)
    ab_a, ab_b = np.nonzero(p_ab)
    grouped = SparseGroupedProblem(
        vocabulary_size=3,
        edge_a=ab_a, edge_b=ab_b, edge_probability=p_ab[ab_a, ab_b],
        target_y=p_ya.sum(axis=1),
        active_ya_y=ya_y, active_ya_a=ya_a, target_ya=p_ya[ya_y, ya_a],
        active_yb_y=yb_y, active_yb_b=yb_b, target_yb=p_yb[yb_y, yb_b],
    )
    problem = anchored_problem_from_intersection(
        grouped, build_sparse_intersection_plan(grouped),
    )
    u = np.array([0.2, -0.3, 0.4, -0.1])
    v = np.linspace(-0.2, 0.25, len(ab_a))
    one = anchored_intersection_sampled_gradient(
        problem, u, v, batch_size=257, seed=81, step=9, workers=1,
        slack_precision=2.0,
    )
    four = anchored_intersection_sampled_gradient(
        problem, u, v, batch_size=257, seed=81, step=9, workers=4,
        slack_precision=2.0,
    )
    np.testing.assert_array_equal(one[0], four[0])
    np.testing.assert_array_equal(one[1], four[1])
def test_sampled_intersection_gradient_monte_carlo_matches_exact():
    p_ya = np.array([[0.3, 0.1], [0.2, 0.4]])
    p_yb = np.array([[0.25, 0.0, 0.15], [0.0, 0.25, 0.35]])
    p_ab = np.array([[0.1, 0.2, 0.1], [0.3, 0.1, 0.2]])
    ya_y, ya_a = np.nonzero(p_ya)
    yb_y, yb_b = np.nonzero(p_yb)
    ab_a, ab_b = np.nonzero(p_ab)
    grouped = SparseGroupedProblem(
        vocabulary_size=3,
        edge_a=ab_a, edge_b=ab_b, edge_probability=p_ab[ab_a, ab_b],
        target_y=p_ya.sum(axis=1),
        active_ya_y=ya_y, active_ya_a=ya_a, target_ya=p_ya[ya_y, ya_a],
        active_yb_y=yb_y, active_yb_b=yb_b, target_yb=p_yb[yb_y, yb_b],
    )
    problem = anchored_problem_from_intersection(
        grouped, build_sparse_intersection_plan(grouped),
    )
    u = np.array([0.2, -0.3, 0.4, -0.1])
    v = np.linspace(-0.2, 0.25, len(ab_a))
    _, exact_u, exact_v = anchored_intersection_dual(problem, u, v)
    estimates = [anchored_intersection_sampled_gradient(
        problem, u, v, batch_size=512, seed=710, step=step,
    ) for step in range(160)]
    mean_u = np.mean([item[0] for item in estimates], axis=0)
    mean_v = np.mean([item[1] for item in estimates], axis=0)
    np.testing.assert_allclose(mean_u, exact_u, atol=6e-3)
    np.testing.assert_allclose(mean_v, exact_v, atol=6e-3)


def test_intersection_sgd_is_worker_invariant_and_exactly_selected():
    p_ya = np.array([[0.3, 0.1], [0.2, 0.4]])
    p_yb = np.array([[0.25, 0.0, 0.15], [0.0, 0.25, 0.35]])
    p_ab = np.array([[0.1, 0.2, 0.1], [0.3, 0.1, 0.2]])
    ya_y, ya_a = np.nonzero(p_ya)
    yb_y, yb_b = np.nonzero(p_yb)
    ab_a, ab_b = np.nonzero(p_ab)
    grouped = SparseGroupedProblem(
        vocabulary_size=3,
        edge_a=ab_a, edge_b=ab_b, edge_probability=p_ab[ab_a, ab_b],
        target_y=p_ya.sum(axis=1),
        active_ya_y=ya_y, active_ya_a=ya_a, target_ya=p_ya[ya_y, ya_a],
        active_yb_y=yb_y, active_yb_b=yb_b, target_yb=p_yb[yb_y, yb_b],
    )
    problem = anchored_problem_from_intersection(
        grouped, build_sparse_intersection_plan(grouped),
    )
    initial = cold_pair_midpoint_from_grouped(grouped)
    kwargs = dict(
        steps=100, batch_size=128, seed=44, learning_rate=0.02,
        exact_interval=10, slack_precision=3.0,
    )
    one = anchored_intersection_sgd(problem, *initial, workers=1, **kwargs)
    four = anchored_intersection_sgd(problem, *initial, workers=4, **kwargs)
    np.testing.assert_array_equal(one.correction_yb, four.correction_yb)
    np.testing.assert_array_equal(one.correction_ab, four.correction_ab)
    assert one.best_objective < one.initial_objective
    selected, _, _ = anchored_intersection_dual(
        problem, one.correction_yb, one.correction_ab, slack_precision=3.0,
    )
    assert selected == one.best_objective


def test_implicit_ya_background_matches_dense_full_marginal():
    # Most YA cells live in an implicit context-dependent background; only
    # three corrections are explicit, as in a production layered pair law.
    background = np.array([0.015, 0.025, 0.010])
    active_y = np.array([0, 1, 2])
    active_a = np.array([0, 1, 1])
    delta = np.array([0.18, 0.20, 0.17])
    dense_ya = np.broadcast_to(background, (3, 3)).copy()
    dense_ya[active_y, active_a] += delta
    dense_ya /= dense_ya.sum()
    # Scale the sparse representation identically.
    scale = 1.0 / (3 * background.sum() + delta.sum())
    p_ya = SparseProjectedPair(
        3, np.ones(3), np.ones(3), background * scale,
        active_y, active_a, delta * scale,
    )
    p_yb = np.array([
        [0.16, 0.00, 0.04],
        [0.00, 0.19, 0.08],
        [0.10, 0.00, 0.13],
    ])
    yb_y, yb_b = np.nonzero(p_yb)
    p_ab = np.array([
        [0.08, 0.12, 0.06],
        [0.18, 0.12, 0.14],
        [0.09, 0.11, 0.10],
    ])
    ab_a, ab_b = np.nonzero(p_ab)
    grouped = SparseGroupedProblem(
        vocabulary_size=3,
        edge_a=ab_a, edge_b=ab_b, edge_probability=p_ab[ab_a, ab_b],
        target_y=dense_ya.sum(axis=1),
        active_ya_y=active_y, active_ya_a=active_a,
        target_ya=dense_ya[active_y, active_a],
        active_yb_y=yb_y, active_yb_b=yb_b,
        target_yb=p_yb[yb_y, yb_b],
    )
    implicit = implicit_ya_problem_from_grouped(grouped, p_ya)
    u = np.linspace(-0.25, 0.35, len(yb_y))
    v = np.linspace(0.2, -0.15, len(ab_a))
    _, gu, gv = implicit_ya_intersection_dual(implicit, u, v)
    dense_u = np.zeros_like(p_yb)
    dense_u[yb_y, yb_b] = u
    dense_v = np.zeros_like(p_ab)
    dense_v[ab_a, ab_b] = v
    joint = anchored_joint(dense_ya, p_ab, dense_u, dense_v)
    np.testing.assert_allclose(
        gu + grouped.target_yb,
        joint.sum(axis=1)[yb_y, yb_b], atol=2e-14,
    )
    np.testing.assert_allclose(
        gv + grouped.edge_probability,
        joint.sum(axis=0)[ab_a, ab_b], atol=2e-14,
    )
    # The explicit active cells do not exhaust YA; the omitted mass is still
    # exactly present in the sparse objective and margins.
    assert grouped.target_ya.sum() < 1.0
    np.testing.assert_allclose(implicit.ya_context_probability, dense_ya.sum(axis=0))
    one = implicit_ya_sampled_gradient(
        implicit, u, v, batch_size=257, seed=93, step=4, workers=1,
    )
    four = implicit_ya_sampled_gradient(
        implicit, u, v, batch_size=257, seed=93, step=4, workers=4,
    )
    np.testing.assert_array_equal(one[0], four[0])
    np.testing.assert_array_equal(one[1], four[1])
    estimates = [implicit_ya_sampled_gradient(
        implicit, u, v, batch_size=512, seed=1201, step=step,
    ) for step in range(180)]
    np.testing.assert_allclose(
        np.mean([value[0] for value in estimates], axis=0), gu, atol=6e-3,
    )
    np.testing.assert_allclose(
        np.mean([value[1] for value in estimates], axis=0), gv, atol=6e-3,
    )
    targets, lag1, lag2 = np.indices((3, 3, 3)).reshape(3, -1)
    scored, markov, covered = implicit_ya_log_probabilities(
        implicit, u, v, targets, lag1, lag2,
    )
    dense_conditional = joint / joint.sum(axis=0, keepdims=True)
    np.testing.assert_allclose(
        scored, np.log(dense_conditional[targets, lag1, lag2]), atol=3e-14,
    )
    np.testing.assert_allclose(
        markov,
        np.log(dense_ya[targets, lag1] / dense_ya.sum(axis=0)[lag1]),
        atol=3e-14,
    )
    assert np.all(covered)


def test_full_layered_ab_background_keeps_missing_contexts_normalized():
    background = np.array([0.01, 0.02, 0.015])
    active_y = np.array([0, 1, 2, 0])
    active_a = np.array([0, 0, 1, 2])
    delta = np.array([0.20, 0.18, 0.22, 0.265])
    scale = 1.0 / (3 * background.sum() + delta.sum())
    layered = SparseProjectedPair(
        3, np.ones(3), np.ones(3), background * scale,
        active_y, active_a, delta * scale,
    )
    p_ya = dense_layered_pair(layered)
    # Stationarity orientation: AB(a,b) is the same table with Y renamed A.
    p_ab = p_ya.copy()
    yb_y = np.array([0, 1, 2])
    yb_b = np.array([2, 1, 0])
    target_yb = np.array([0.08, 0.11, 0.09])
    # Deliberately omit every active AB factor for A=2. Its implicit AB
    # background nevertheless gives that hard-YA context a valid normalizer.
    ab_a = np.array([0, 0, 1])
    ab_b = np.array([0, 1, 1])
    target_ab = p_ab[ab_a, ab_b]
    value, gu, gv = dense_full_pair_dual(
        p_ya, p_ab, yb_y, yb_b, target_yb,
        ab_a, ab_b, target_ab,
        np.array([0.2, -0.1, 0.3]), np.array([0.1, -0.2, 0.15]),
    )
    assert np.isfinite(value)
    assert np.all(np.isfinite(gu))
    assert np.all(np.isfinite(gv))
    assert p_ya[:, 2].sum() > 0.0
    p_yb_dense = np.full((3, 3), 0.01)
    p_yb_dense[yb_y, yb_b] = target_yb
    p_yb_dense /= p_yb_dense.sum()
    p_yb_y, p_yb_b = np.nonzero(p_yb_dense)
    p_yb_pair = SparseProjectedPair(
        3, np.ones(3), np.ones(3), np.zeros(3),
        p_yb_y, p_yb_b, p_yb_dense[p_yb_y, p_yb_b],
    )
    grouped = SparseGroupedProblem(
        vocabulary_size=3,
        edge_a=ab_a, edge_b=ab_b,
        edge_probability=target_ab / target_ab.sum(),
        target_y=p_ya.sum(axis=1),
        active_ya_y=active_y, active_ya_a=active_a,
        target_ya=p_ya[active_y, active_a],
        active_yb_y=yb_y, active_yb_b=yb_b, target_yb=target_yb,
    )
    sparse = full_implicit_pair_problem(grouped, layered, p_yb_pair)
    u = np.array([0.2, -0.1, 0.3])
    v = np.array([0.1, -0.2, 0.15])
    _, exact_u, exact_v = dense_full_pair_dual(
        p_ya, p_ab, yb_y, yb_b, target_yb,
        ab_a, ab_b, target_ab, u, v,
    )
    estimates = [full_implicit_pair_sampled_gradient(
        sparse, u, v, batch_size=512, seed=404, step=step,
    ) for step in range(220)]
    np.testing.assert_allclose(
        np.mean([item[0] for item in estimates], axis=0), exact_u, atol=6e-3,
    )
    np.testing.assert_allclose(
        np.mean([item[1] for item in estimates], axis=0), exact_v, atol=6e-3,
    )
    one = full_implicit_pair_sampled_gradient(
        sparse, u, v, batch_size=127, seed=5, step=8, workers=1,
    )
    four = full_implicit_pair_sampled_gradient(
        sparse, u, v, batch_size=127, seed=5, step=8, workers=4,
    )
    np.testing.assert_array_equal(one[0], four[0])
    np.testing.assert_array_equal(one[1], four[1])
    validation = full_implicit_validation_objective(
        sparse, u, v, validation_seed=71, validation_samples=100_000,
    )
    dense_value = dense_full_pair_dual(
        p_ya, p_ab, yb_y, yb_b, target_yb,
        ab_a, ab_b, target_ab, u, v,
    )[0]
    np.testing.assert_allclose(validation, dense_value, atol=3e-3)
    kwargs = dict(
        steps=80, batch_size=128, training_seed=9,
        validation_seed=71, validation_samples=4096,
        learning_rate=0.02, validation_interval=10, slack_precision=3.0,
    )
    fitted1 = full_implicit_pair_sgd(sparse, u, v, workers=1, **kwargs)
    fitted4 = full_implicit_pair_sgd(sparse, u, v, workers=4, **kwargs)
    np.testing.assert_array_equal(fitted1.correction_yb, fitted4.correction_yb)
    np.testing.assert_array_equal(fitted1.correction_ab, fitted4.correction_ab)
    assert fitted1.best_validation_objective <= fitted1.initial_validation_objective
    dense_u = np.zeros_like(p_ya)
    dense_v = np.zeros_like(p_ab)
    dense_u[yb_y, yb_b] = u
    dense_v[ab_a, ab_b] = v
    joint = anchored_joint(p_ya, p_ab, dense_u, dense_v)
    conditional = joint / joint.sum(axis=0, keepdims=True)
    targets, contexts, old = np.indices((3, 3, 3)).reshape(3, -1)
    scored, markov = full_implicit_log_probabilities(
        sparse, u, v, targets, contexts, old,
    )
    np.testing.assert_allclose(
        scored, np.log(conditional[targets, contexts, old]), atol=3e-14,
    )
    np.testing.assert_allclose(
        markov,
        np.log(p_ya[targets, contexts] / p_ya.sum(axis=0)[contexts]),
        atol=3e-14,
    )
