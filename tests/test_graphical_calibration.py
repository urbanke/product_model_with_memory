"""Checks for the three-pair maximum-entropy calibration."""

import numpy as np

from product_model_with_memory.graphical_calibration import (
    GroupedCheckpoint,
    SparseGroupedProblem,
    SparseGroupedResult,
    conditional_ipf,
    first_pair_warm_start,
    fit_grouped_checkpoints,
    grouped_conditional_ipf,
    pair_midpoint_warm_start,
    pair_product_warm_start,
    project_sparse_layered_pair,
    restrict_margins_to_observed_contexts,
    restrict_sparse_margins_to_observed_contexts,
    second_pair_warm_start,
    sparse_gated_log_probabilities,
    sparse_grouped_ipf,
    sparse_problem_from_dense,
    sparse_problem_from_projected,
    sparse_star_log_probabilities,
    star_log_probabilities,
    transfer_sparse_warm_start,
)
from scripts.pairwise_arena import ipf_triangle


def _pair_margins(q):
    return q.sum(axis=2), q.sum(axis=1), q.sum(axis=0)


def _joint_ipf_reference(p_ya, p_yb, p_ab, *, tol=1e-12):
    """Literal V^3 IPF, used only as an independent tiny reference."""

    q = np.ones((p_ya.shape[0], p_ya.shape[1], p_yb.shape[1]))
    q /= q.sum()
    tiny = np.finfo(np.float64).tiny
    for _ in range(20_000):
        q *= (p_ya / np.maximum(q.sum(axis=2), tiny))[:, :, None]
        q *= (p_yb / np.maximum(q.sum(axis=1), tiny))[:, None, :]
        q *= (p_ab / np.maximum(q.sum(axis=0), tiny))[None, :, :]
        m_ya, m_yb, m_ab = _pair_margins(q)
        if max(
            np.abs(m_ya - p_ya).sum(),
            np.abs(m_yb - p_yb).sum(),
            np.abs(m_ab - p_ab).sum(),
        ) < tol:
            return q
    raise AssertionError("reference IPF did not converge")


def test_first_pair_warm_start_exactly_matches_ya_margin():
    rng = np.random.default_rng(20260806)
    raw = rng.gamma(shape=1.1, scale=1.0, size=(5, 4, 3))
    source = raw / raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(source)
    problem = sparse_problem_from_dense(
        p_ya, p_yb, p_ab,
        np.ones_like(p_ya, dtype=bool),
        np.ones_like(p_yb, dtype=bool),
    )
    log_base, correction_ya, correction_yb = first_pair_warm_start(problem)
    result = sparse_grouped_ipf(
        problem,
        max_iterations=0,
        tolerance=1e-14,
        log_base_y=log_base,
        correction_ya=correction_ya,
        correction_yb=correction_yb,
    )
    assert result.grouped_residual_ya_l1 < 2e-14
    assert result.residual_y_l1 < 2e-14
    assert np.array_equal(correction_yb, np.zeros_like(correction_yb))


def test_pair_product_warm_start_uses_both_pair_factors():
    rng = np.random.default_rng(20260807)
    raw = rng.gamma(shape=1.1, scale=1.0, size=(5, 4, 3))
    source = raw / raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(source)
    problem = sparse_problem_from_dense(
        p_ya, p_yb, p_ab,
        np.ones_like(p_ya, dtype=bool),
        np.ones_like(p_yb, dtype=bool),
    )
    log_base, correction_ya, correction_yb = pair_product_warm_start(problem)
    py = p_ya.sum(axis=1)
    pa = p_ab.sum(axis=1)
    pb = p_ab.sum(axis=0)
    expected_first = np.log(p_ya / (py[:, None] * pa[None, :]))
    expected_second = np.log(p_yb / (py[:, None] * pb[None, :]))
    assert np.allclose(log_base, np.log(py), atol=2e-14)
    assert np.allclose(correction_ya, expected_first.ravel(), atol=2e-14)
    assert np.allclose(correction_yb, expected_second.ravel(), atol=2e-14)

    _, midpoint_first, midpoint_second = pair_midpoint_warm_start(problem)
    assert np.allclose(midpoint_first, 0.5 * correction_ya)
    assert np.allclose(midpoint_second, 0.5 * correction_yb)

    _, second_first, second_second = second_pair_warm_start(problem)
    assert np.array_equal(second_first, np.zeros_like(second_first))
    assert np.array_equal(second_second, correction_yb)


def test_conditional_form_matches_literal_three_factor_ipf():
    rng = np.random.default_rng(20260805)
    raw = rng.gamma(shape=0.7, scale=1.0, size=(4, 3, 5))
    source = raw / raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(source)

    got = conditional_ipf(p_ya, p_yb, p_ab, tolerance=1e-12)
    assert got.converged
    reference = _joint_ipf_reference(p_ya, p_yb, p_ab)
    fitted = got.joint(p_ab)

    assert np.max(np.abs(fitted - reference)) < 2e-11
    assert got.residual_ya_l1 < 1e-12
    assert got.residual_yb_l1 < 1e-12
    assert got.residual_ab_l1 < 1e-14


def test_conditional_form_matches_existing_triangle_fitter():
    rng = np.random.default_rng(41)
    raw = rng.gamma(shape=1.2, scale=1.0, size=(5, 4, 3))
    source = raw / raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(source)

    got = conditional_ipf(p_ya, p_yb, p_ab, tolerance=1e-12)
    psi_ya, psi_yb, psi_ab, _, residual = ipf_triangle(
        p_ya,
        p_yb,
        p_ab,
        np.ones_like(p_ya),
        np.ones_like(p_yb),
        np.ones_like(p_ab),
        iters=10_000,
        tol=1e-12,
    )
    reference = (
        psi_ya[:, :, None]
        * psi_yb[:, None, :]
        * psi_ab[None, :, :]
    )
    reference /= reference.sum()

    assert got.converged
    assert residual < 1e-12
    assert np.max(np.abs(got.joint(p_ab) - reference)) < 2e-11


def test_duplicate_lag_is_not_counted_twice():
    # A and B are exact copies.  Both target-lag margins are therefore
    # identical.  The calibrated conditional must equal the one-lag
    # conditional rather than squaring its evidence.
    p_a = np.array([0.2, 0.3, 0.5])
    p_y_given_a = np.array([
        [0.80, 0.15, 0.05],
        [0.10, 0.75, 0.15],
        [0.05, 0.20, 0.75],
    ]).T  # rows Y, columns A
    p_ya = p_y_given_a * p_a[None, :]
    p_yb = p_ya.copy()
    p_ab = np.diag(p_a)

    got = conditional_ipf(p_ya, p_yb, p_ab, tolerance=1e-12)
    assert got.converged
    cond = got.conditional()
    for a in range(3):
        assert np.max(np.abs(cond[:, a, a] - p_y_given_a[:, a])) < 1e-11


def test_inconsistent_pair_margins_are_rejected():
    p_ya = np.full((2, 2), 0.25)
    p_yb = np.full((2, 2), 0.25)
    p_ab = np.array([[0.4, 0.1], [0.1, 0.4]])
    p_yb[0, 0] += 0.05
    p_yb[1, 0] -= 0.05

    try:
        conditional_ipf(p_ya, p_yb, p_ab)
    except ValueError as exc:
        assert "inconsistent" in str(exc)
    else:
        raise AssertionError("inconsistent margins were accepted")


def test_grouped_solver_matches_active_cells_and_inactive_mass():
    rng = np.random.default_rng(92)
    raw = rng.gamma(shape=0.8, scale=1.0, size=(5, 4, 3))
    source = raw / raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(source)
    active_ya = rng.random(p_ya.shape) < 0.45
    active_yb = rng.random(p_yb.shape) < 0.45

    got = grouped_conditional_ipf(
        p_ya, p_yb, p_ab, active_ya, active_yb, tolerance=1e-12
    )
    q = got.joint(p_ab)
    m_ya, m_yb, m_ab = _pair_margins(q)

    assert got.converged
    assert np.max(np.abs(m_ya[active_ya] - p_ya[active_ya])) < 1e-11
    assert np.max(np.abs(m_yb[active_yb] - p_yb[active_yb])) < 1e-11
    for state in range(p_ya.shape[1]):
        inactive = ~active_ya[:, state]
        assert abs(m_ya[inactive, state].sum()
                   - p_ya[inactive, state].sum()) < 1e-11
    for state in range(p_yb.shape[1]):
        inactive = ~active_yb[:, state]
        assert abs(m_yb[inactive, state].sum()
                   - p_yb[inactive, state].sum()) < 1e-11
    assert np.max(np.abs(m_ab - p_ab)) < 1e-14
    assert np.max(np.abs(m_ya.sum(axis=1) - p_ya.sum(axis=1))) < 1e-11


def test_grouped_solver_warm_start_reaches_same_solution_faster():
    rng = np.random.default_rng(117)
    first = rng.gamma(shape=1.0, scale=1.0, size=(5, 4, 4))
    first /= first.sum()
    # A later geometric checkpoint is the old accumulated distribution plus
    # a smaller new block, not an unrelated distribution.
    later = first + 0.002 * rng.gamma(
        shape=1.0, scale=1.0, size=first.shape
    )
    later /= later.sum()
    p1_ya, p1_yb, p1_ab = _pair_margins(first)
    p2_ya, p2_yb, p2_ab = _pair_margins(later)
    active_ya = rng.random(p1_ya.shape) < 0.55
    active_yb = rng.random(p1_yb.shape) < 0.55

    previous = grouped_conditional_ipf(
        p1_ya, p1_yb, p1_ab, active_ya, active_yb, tolerance=1e-11
    )
    cold = grouped_conditional_ipf(
        p2_ya, p2_yb, p2_ab, active_ya, active_yb, tolerance=1e-11
    )
    warm = grouped_conditional_ipf(
        p2_ya,
        p2_yb,
        p2_ab,
        active_ya,
        active_yb,
        tolerance=1e-11,
        log_f_ya=previous.log_f_ya,
        log_f_yb=previous.log_f_yb,
    )

    assert previous.converged and cold.converged and warm.converged
    assert warm.iterations < cold.iterations
    assert np.max(np.abs(warm.joint(p2_ab) - cold.joint(p2_ab))) < 2e-10


def test_interleaved_checkpoint_chains_match_sequential_fits():
    rng = np.random.default_rng(203)
    counts = rng.gamma(shape=1.0, scale=1.0, size=(4, 3, 3))
    active_ya = rng.random((4, 3)) < 0.6
    active_yb = rng.random((4, 3)) < 0.6
    checkpoints = []
    for _ in range(6):
        counts += 0.03 * rng.gamma(shape=1.0, scale=1.0, size=counts.shape)
        q = counts / counts.sum()
        p_ya, p_yb, p_ab = _pair_margins(q)
        checkpoints.append(GroupedCheckpoint(
            p_ya, p_yb, p_ab, active_ya, active_yb
        ))

    sequential = fit_grouped_checkpoints(
        checkpoints, interleave=1, tolerance=1e-11
    )
    interleaved = fit_grouped_checkpoints(
        checkpoints, interleave=3, tolerance=1e-11
    )

    assert all(result.converged for result in sequential + interleaved)
    for point, one, three in zip(checkpoints, sequential, interleaved):
        assert np.max(np.abs(
            one.joint(point.p_ab) - three.joint(point.p_ab)
        )) < 2e-10


def test_observed_context_restriction_produces_consistent_margins():
    rng = np.random.default_rng(771)
    q = rng.gamma(shape=0.9, scale=1.0, size=(5, 4, 4))
    q /= q.sum()
    p_ya, p_yb, p_ab = _pair_margins(q)
    observed = rng.random(p_ab.shape) < 0.7
    # Ensure every A and B value remains represented.
    np.fill_diagonal(observed, True)

    got = restrict_margins_to_observed_contexts(
        p_ya, p_yb, p_ab, observed
    )

    assert np.all(got.p_ab[~observed] == 0.0)
    assert abs(got.p_ab.sum() - 1.0) < 1e-12
    assert abs(got.retained_ab_mass - p_ab[observed].sum()) < 1e-15
    assert np.max(np.abs(got.p_ya.sum(axis=1)
                         - got.p_yb.sum(axis=1))) < 1e-12
    assert np.max(np.abs(got.p_ya.sum(axis=0)
                         - got.p_ab.sum(axis=1))) < 1e-12
    assert np.max(np.abs(got.p_yb.sum(axis=0)
                         - got.p_ab.sum(axis=0))) < 1e-12


def test_observed_context_restriction_allows_unseen_context_states():
    rng = np.random.default_rng(772)
    q = rng.gamma(shape=0.9, scale=1.0, size=(4, 3, 3))
    q /= q.sum()
    p_ya, p_yb, p_ab = _pair_margins(q)
    observed = np.zeros_like(p_ab, dtype=bool)
    observed[:2, :2] = True

    got = restrict_margins_to_observed_contexts(
        p_ya, p_yb, p_ab, observed
    )

    assert np.all(got.p_ya[:, 2] == 0.0)
    assert np.all(got.p_yb[:, 2] == 0.0)
    assert np.max(np.abs(got.p_ya.sum(axis=1)
                         - got.p_yb.sum(axis=1))) < 1e-12
    assert np.max(np.abs(got.p_ya.sum(axis=0)
                         - got.p_ab.sum(axis=1))) < 1e-12
    assert np.max(np.abs(got.p_yb.sum(axis=0)
                         - got.p_ab.sum(axis=0))) < 1e-12


def test_matrix_free_grouped_solver_matches_dense_reference():
    rng = np.random.default_rng(818)
    raw = rng.gamma(shape=0.9, scale=1.0, size=(5, 4, 4))
    raw[:, 0, 3] = 0.0
    raw[:, 2, 1] = 0.0
    q = raw / raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(q)
    active_ya = rng.random(p_ya.shape) < 0.55
    active_yb = rng.random(p_yb.shape) < 0.55
    dense = grouped_conditional_ipf(
        p_ya, p_yb, p_ab, active_ya, active_yb, tolerance=1e-10
    )
    problem = sparse_problem_from_dense(
        p_ya, p_yb, p_ab, active_ya, active_yb
    )
    sparse = sparse_grouped_ipf(problem, tolerance=1e-10)

    assert dense.converged and sparse.converged
    # Compare the sufficient-statistic margins, which uniquely determine
    # this grouped maximum-entropy solution.
    assert sparse.grouped_residual_ya_l1 < 1e-10
    assert sparse.grouped_residual_yb_l1 < 1e-10
    assert sparse.residual_y_l1 < 1e-10
    dense_conditional = dense.conditional()
    map_ya = {(int(y), int(a)): i for i, (y, a) in enumerate(zip(
        problem.active_ya_y, problem.active_ya_a
    ))}
    map_yb = {(int(y), int(b)): i for i, (y, b) in enumerate(zip(
        problem.active_yb_y, problem.active_yb_b
    ))}
    for a, b in zip(problem.edge_a, problem.edge_b):
        score = sparse.log_base_y.copy()
        for y in range(problem.vocabulary_size):
            i = map_ya.get((y, int(a)))
            j = map_yb.get((y, int(b)))
            if i is not None:
                score[y] += sparse.correction_ya[i]
            if j is not None:
                score[y] += sparse.correction_yb[j]
        score -= score.max()
        probability = np.exp(score)
        probability /= probability.sum()
        assert np.max(np.abs(
            probability - dense_conditional[:, a, b]
        )) < 2e-9


def test_anderson_sparse_solver_matches_plain_ipf():
    rng = np.random.default_rng(991)
    raw = rng.gamma(shape=1.1, scale=1.0, size=(6, 5, 5))
    raw[:, 1, 4] = 0.0
    raw[:, 3, 0] = 0.0
    q = raw / raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(q)
    active_ya = rng.random(p_ya.shape) < 0.5
    active_yb = rng.random(p_yb.shape) < 0.5
    problem = sparse_problem_from_dense(
        p_ya, p_yb, p_ab, active_ya, active_yb
    )

    plain = sparse_grouped_ipf(problem, tolerance=1e-10, solver="ipf")
    accelerated = sparse_grouped_ipf(
        problem, tolerance=1e-10, solver="anderson"
    )

    assert plain.converged and accelerated.converged
    assert accelerated.iterations < plain.iterations
    assert accelerated.grouped_residual_ya_l1 < 1e-10
    assert accelerated.grouped_residual_yb_l1 < 1e-10
    assert accelerated.residual_y_l1 < 1e-10


def test_lbfgs_sparse_solver_matches_plain_ipf():
    rng = np.random.default_rng(1234)
    raw = rng.gamma(shape=1.0, scale=1.0, size=(5, 4, 4))
    raw[:, 0, 2] = 0.0
    q = raw / raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(q)
    active_ya = rng.random(p_ya.shape) < 0.6
    active_yb = rng.random(p_yb.shape) < 0.6
    problem = sparse_problem_from_dense(
        p_ya, p_yb, p_ab, active_ya, active_yb
    )

    plain = sparse_grouped_ipf(problem, tolerance=1e-9, solver="ipf")
    quasi_newton = sparse_grouped_ipf(
        problem, tolerance=1e-9, solver="lbfgs", max_iterations=2_000
    )

    assert plain.converged and quasi_newton.converged
    assert quasi_newton.iterations < plain.iterations
    assert quasi_newton.grouped_residual_ya_l1 < 1e-9
    assert quasi_newton.grouped_residual_yb_l1 < 1e-9
    assert quasi_newton.residual_y_l1 < 1e-9


def test_sparse_warm_start_transfers_growing_active_support():
    old = SparseGroupedProblem(
        vocabulary_size=4,
        edge_a=np.array([0]), edge_b=np.array([1]),
        edge_probability=np.array([1.0]), target_y=np.full(4, 0.25),
        active_ya_y=np.array([1]), active_ya_a=np.array([0]),
        target_ya=np.array([0.2]),
        active_yb_y=np.array([2]), active_yb_b=np.array([1]),
        target_yb=np.array([0.3]),
    )
    result = SparseGroupedResult(
        log_base_y=np.arange(4.0), correction_ya=np.array([0.7]),
        correction_yb=np.array([-0.4]), iterations=3,
        grouped_residual_ya_l1=0.0, grouped_residual_yb_l1=0.0,
        residual_y_l1=0.0, converged=True,
    )
    new = SparseGroupedProblem(
        vocabulary_size=4,
        edge_a=np.array([0]), edge_b=np.array([1]),
        edge_probability=np.array([1.0]), target_y=np.full(4, 0.25),
        active_ya_y=np.array([0, 1]), active_ya_a=np.array([0, 0]),
        target_ya=np.array([0.1, 0.2]),
        active_yb_y=np.array([2, 3]), active_yb_b=np.array([1, 1]),
        target_yb=np.array([0.3, 0.1]),
    )

    base, first, second = transfer_sparse_warm_start(old, result, new)

    assert np.array_equal(base, result.log_base_y)
    assert np.array_equal(first, np.array([0.0, 0.7]))
    assert np.array_equal(second, np.array([-0.4, 0.0]))


def test_sparse_gated_scoring_matches_direct_conditionals():
    rng = np.random.default_rng(929)
    raw = rng.gamma(shape=1.2, scale=1.0, size=(4, 4, 4))
    raw /= raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(raw)
    observed = np.ones_like(p_ab, dtype=bool)
    observed[2, 1] = False
    restricted = restrict_margins_to_observed_contexts(
        p_ya, p_yb, p_ab, observed
    )
    active_ya = np.ones_like(p_ya, dtype=bool)
    active_yb = np.ones_like(p_yb, dtype=bool)
    problem = sparse_problem_from_dense(
        restricted.p_ya, restricted.p_yb, restricted.p_ab,
        active_ya, active_yb,
    )
    result = sparse_grouped_ipf(
        problem, solver="lbfgs", tolerance=1e-10, max_iterations=2_000
    )
    targets = np.array([0, 2, 1, 3])
    aa = np.array([0, 1, 2, 2])
    bb = np.array([0, 2, 1, 1])
    got = sparse_gated_log_probabilities(
        problem, result, targets, aa, bb, p_ya, p_yb
    )

    expected = []
    for y, a, b in zip(targets, aa, bb):
        if observed[a, b]:
            score = result.log_base_y.copy()
            score[problem.active_ya_y[problem.active_ya_a == a]] += (
                result.correction_ya[problem.active_ya_a == a]
            )
            score[problem.active_yb_y[problem.active_yb_b == b]] += (
                result.correction_yb[problem.active_yb_b == b]
            )
            expected.append(score[y] - np.logaddexp.reduce(score))
        else:
            star = p_ya[:, a] * p_yb[:, b] / p_ya.sum(axis=1)
            expected.append(np.log(star[y] / star.sum()))
    assert np.max(np.abs(got - expected)) < 1e-11


def test_sparse_layered_pair_projection_matches_dense_sinkhorn():
    rng = np.random.default_rng(1041)
    v = 7
    context_margin = rng.dirichlet(np.ones(v))
    target_margin = rng.dirichlet(np.ones(v))
    active = rng.random((v, v)) < 0.35  # rows context, columns target
    raw = np.ones((v, v))
    raw[active] = rng.lognormal(mean=0.3, sigma=0.7, size=active.sum())
    conditional = raw / raw.sum(axis=1, keepdims=True)
    context, target = np.nonzero(active)
    sparse = project_sparse_layered_pair(
        context_margin,
        1.0 / raw.sum(axis=1),
        target,
        context,
        conditional[context, target],
        target_margin,
        tolerance=1e-13,
    )

    dense = context_margin[:, None] * conditional
    for _ in range(20_000):
        dense *= (context_margin / dense.sum(axis=1))[:, None]
        dense *= (target_margin / dense.sum(axis=0))[None, :]
        if max(
            np.abs(dense.sum(axis=1) - context_margin).sum(),
            np.abs(dense.sum(axis=0) - target_margin).sum(),
        ) < 1e-13:
            break

    assert np.max(np.abs(sparse.dense().T - dense)) < 2e-12
    chosen_y = np.array([0, 3, 6, 2])
    chosen_context = np.array([1, 4, 0, 5])
    assert np.max(np.abs(
        sparse.values(chosen_y, chosen_context)
        - dense[chosen_context, chosen_y]
    )) < 2e-12


def test_sparse_observed_restriction_matches_dense_path():
    rng = np.random.default_rng(1042)
    v = 6
    marginal = rng.dirichlet(np.ones(v))

    def make_pair():
        active = rng.random((v, v)) < 0.4
        np.fill_diagonal(active, True)
        raw = np.ones((v, v))
        raw[active] = rng.lognormal(0.2, 0.6, active.sum())
        conditional = raw / raw.sum(axis=1, keepdims=True)
        context, target = np.nonzero(active)
        pair = project_sparse_layered_pair(
            marginal, 1.0 / raw.sum(axis=1), target, context,
            conditional[context, target], marginal, tolerance=1e-13,
        )
        return pair

    p_ya = make_pair()
    p_yb = make_pair()
    observed = rng.random((v, v)) < 0.45
    np.fill_diagonal(observed, True)
    dense = restrict_margins_to_observed_contexts(
        p_ya.dense(), p_yb.dense(), p_ya.dense(), observed,
        tolerance=1e-12,
    )
    edge_a, edge_b = np.nonzero(observed)
    sparse = restrict_sparse_margins_to_observed_contexts(
        p_ya, p_yb, edge_a, edge_b, tolerance=1e-12,
    )
    problem = sparse_problem_from_projected(sparse)

    assert abs(sparse.retained_ab_mass - dense.retained_ab_mass) < 1e-12
    assert np.max(np.abs(sparse.p_ya.dense() - dense.p_ya)) < 2e-11
    assert np.max(np.abs(sparse.p_yb.dense() - dense.p_yb)) < 2e-11
    expected_edges = dense.p_ab[edge_a, edge_b]
    assert np.max(np.abs(problem.edge_probability - expected_edges)) < 2e-12
    targets = rng.integers(v, size=40)
    context_a = rng.integers(v, size=40)
    context_b = rng.integers(v, size=40)
    sparse_logp = sparse_star_log_probabilities(
        p_ya, p_yb, targets, context_a, context_b
    )
    dense_logp = star_log_probabilities(
        targets, context_a, context_b, p_ya.dense(), p_yb.dense()
    )
    assert np.max(np.abs(sparse_logp - dense_logp)) < 2e-11
    fit = sparse_grouped_ipf(
        problem, solver="lbfgs", tolerance=1e-9, max_iterations=2_000
    )
    gated_sparse = sparse_gated_log_probabilities(
        problem, fit, targets, context_a, context_b, p_ya, p_yb
    )
    gated_dense = sparse_gated_log_probabilities(
        problem, fit, targets, context_a, context_b,
        p_ya.dense(), p_yb.dense(),
    )
    assert np.max(np.abs(gated_sparse - gated_dense)) < 2e-11
