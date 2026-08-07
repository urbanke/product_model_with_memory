"""Checks for the three-pair maximum-entropy calibration."""

import numpy as np
import pytest

from product_model_with_memory.graphical_calibration import (
    GroupedCheckpoint,
    SparseGroupedProblem,
    SparseGroupedResult,
    SparseIntersectionPlan,
    SparseRestrictedMargins,
    build_sparse_edge_blocks,
    build_ab_major_intersection_graph,
    build_sparse_intersection_plan,
    build_layered_intersection_graph,
    birth_major_sparse_support,
    check_grouped_feasibility_lp,
    conditional_ipf,
    checkpoint_in_birth_major_support,
    exact_sparse_dual_wolfe,
    empirical_pair_slack_variances,
    first_pair_warm_start,
    fit_grouped_checkpoints,
    grouped_conditional_ipf,
    intersection_plan_from_layered_graph,
    layered_intersection_graph_from_plan,
    load_ab_major_intersection_graph,
    load_layered_intersection_graph,
    pair_midpoint_warm_start,
    pair_product_warm_start,
    project_sparse_layered_pair,
    projected_pair_warm_start,
    restrict_margins_to_observed_contexts,
    restrict_sparse_margins_to_observed_contexts,
    sample_sparse_grouped_edges,
    save_layered_intersection_graph,
    save_ab_major_intersection_graph,
    second_pair_warm_start,
    sparse_edge_minibatch,
    sparse_edge_block_from_bounds,
    sparse_factorized_dual_evaluation,
    sparse_factorized_dual_hessian_product,
    sparse_factorized_margins,
    sparse_factorized_margins_ab_major,
    sparse_factorized_margins_layered,
    sparse_factorized_margins_layered_reference,
    sparse_factorized_margins_reference,
    sparse_gated_log_probabilities,
    sparse_grouped_newton_cg,
    sparse_grouped_ipf,
    sparse_problem_from_dense,
    sparse_problem_from_projected,
    sparse_problem_with_edge_distribution,
    sparse_star_log_probabilities,
    star_log_probabilities,
    stochastic_sparse_dual_approach,
    stratified_sparse_edge_minibatch,
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


def test_projected_first_pair_start_is_exact_with_inactive_cells():
    v = 4
    marginal = np.full(v, 1.0 / v)
    contexts = np.arange(v)
    first = project_sparse_layered_pair(
        marginal,
        np.full(v, 0.1),
        np.arange(v),
        contexts,
        np.full(v, 0.7),
        marginal,
    )
    second = project_sparse_layered_pair(
        marginal,
        np.full(v, 0.1),
        (np.arange(v) + 1) % v,
        contexts,
        np.full(v, 0.7),
        marginal,
    )
    edge_a = np.repeat(np.arange(v), v)
    edge_b = np.tile(np.arange(v), v)
    margins = SparseRestrictedMargins(
        first, second, edge_a, edge_b,
        np.full(v * v, 1.0 / (v * v)), 1.0,
    )
    problem = sparse_problem_from_projected(margins)
    warm = projected_pair_warm_start(
        first, second, problem.target_y, "first_pair"
    )
    result = sparse_grouped_ipf(
        problem,
        max_iterations=0,
        log_base_y=warm[0],
        correction_ya=warm[1],
        correction_yb=warm[2],
    )
    assert result.grouped_residual_ya_l1 < 2e-14
    assert result.residual_y_l1 < 2e-14


def test_grouped_feasibility_lp_detects_incompatible_pair_margins():
    diagonal = np.array([[0.5, 0.0], [0.0, 0.5]])
    anti_diagonal = np.array([[0.0, 0.5], [0.5, 0.0]])
    active = np.ones((2, 2), dtype=bool)

    feasible_problem = sparse_problem_from_dense(
        diagonal, diagonal, diagonal, active, active
    )
    feasible = check_grouped_feasibility_lp(feasible_problem)
    assert feasible.feasible
    assert feasible.max_equality_residual < 1e-12

    # YA and YB demand Y=A=B, while AB demands A!=B.  Every one-variable
    # margin is nevertheless the same uniform distribution.
    impossible_problem = sparse_problem_from_dense(
        diagonal, diagonal, anti_diagonal, active, active
    )
    impossible = check_grouped_feasibility_lp(impossible_problem)
    assert not impossible.feasible

    # Exact matching has no finite solution, but the quadratic slack model
    # has a finite, certified stationary point while retaining both pairs.
    relaxed = exact_sparse_dual_wolfe(
        impossible_problem,
        np.log(impossible_problem.target_y),
        np.zeros(len(impossible_problem.target_ya)),
        np.zeros(len(impossible_problem.target_yb)),
        pair_slack_precision=100.0,
        max_iterations=2_000,
        tolerance=1e-5,
    )
    assert relaxed.converged
    assert relaxed.stationarity <= 1e-5
    assert relaxed.certificate > 1e-4
    assert np.isfinite(relaxed.correction_ya).all()
    assert np.isfinite(relaxed.correction_yb).all()

    variance_ya, variance_yb = empirical_pair_slack_variances(
        impossible_problem, 100,
    )
    assert np.all(variance_ya >= 1e-4)
    assert np.all(variance_yb >= 1e-4)


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
    trace = []
    sparse = sparse_grouped_ipf(
        problem, tolerance=1e-10, trace=trace, trace_interval=5
    )

    assert dense.converged and sparse.converged
    assert trace
    assert trace[-1]["certificate"] < 1e-10
    assert trace[-1]["limiting_margin"] in {"y", "ya", "yb"}
    assert np.isfinite(trace[-1]["objective"])
    assert np.isfinite(trace[-1]["factor_abs_p99"])
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
    preconditioned = sparse_grouped_ipf(
        problem, tolerance=1e-9, solver="lbfgs", max_iterations=2_000,
        lbfgs_precondition=True,
    )
    unreduced = sparse_grouped_ipf(
        problem, tolerance=1e-9, solver="lbfgs", max_iterations=2_000,
        reduce_gauge=False,
    )

    assert (
        plain.converged and quasi_newton.converged
        and preconditioned.converged and unreduced.converged
    )
    assert quasi_newton.iterations < plain.iterations
    assert quasi_newton.margin_evaluations > 0
    assert quasi_newton.grouped_residual_ya_l1 < 1e-9
    assert quasi_newton.grouped_residual_yb_l1 < 1e-9
    assert quasi_newton.residual_y_l1 < 1e-9
    assert max(
        preconditioned.grouped_residual_ya_l1,
        preconditioned.grouped_residual_yb_l1,
        preconditioned.residual_y_l1,
    ) < 1e-9
    assert max(
        unreduced.grouped_residual_ya_l1,
        unreduced.grouped_residual_yb_l1,
        unreduced.residual_y_l1,
    ) < 1e-9


def test_sharded_edge_normalization_matches_serial_solver():
    rng = np.random.default_rng(20260808)
    raw = rng.gamma(shape=1.0, scale=1.0, size=(7, 6, 6))
    q = raw / raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(q)
    active_ya = rng.random(p_ya.shape) < 0.4
    active_yb = rng.random(p_yb.shape) < 0.4
    problem = sparse_problem_from_dense(
        p_ya, p_yb, p_ab, active_ya, active_yb
    )
    serial = sparse_grouped_ipf(
        problem, max_iterations=20, tolerance=1e-30, margin_workers=1
    )
    sharded = sparse_grouped_ipf(
        problem, max_iterations=20, tolerance=1e-30, margin_workers=3
    )
    factorized = sparse_grouped_ipf(
        problem, max_iterations=20, tolerance=1e-30,
        evaluator="factorized", margin_workers=3,
    )
    assert np.max(np.abs(serial.log_base_y - sharded.log_base_y)) < 1e-12
    assert np.max(
        np.abs(serial.correction_ya - sharded.correction_ya)
    ) < 1e-12
    assert np.max(
        np.abs(serial.correction_yb - sharded.correction_yb)
    ) < 1e-12
    assert abs(
        serial.grouped_residual_ya_l1 - sharded.grouped_residual_ya_l1
    ) < 1e-13
    assert abs(
        serial.grouped_residual_yb_l1 - sharded.grouped_residual_yb_l1
    ) < 1e-13
    assert np.max(
        np.abs(serial.log_base_y - factorized.log_base_y)
    ) < 1e-12
    assert np.max(
        np.abs(serial.correction_ya - factorized.correction_ya)
    ) < 1e-12
    assert np.max(
        np.abs(serial.correction_yb - factorized.correction_yb)
    ) < 1e-12


def test_intersection_factorized_margins_match_dense_evaluation():
    rng = np.random.default_rng(20260809)
    v = 6
    raw = rng.gamma(shape=1.0, scale=1.0, size=(v, v, v))
    source = raw / raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(source)
    active_ya = rng.random((v, v)) < 0.45
    active_yb = rng.random((v, v)) < 0.45
    problem = sparse_problem_from_dense(
        p_ya, p_yb, p_ab, active_ya, active_yb
    )
    log_base = rng.normal(size=v)
    c1 = rng.normal(scale=0.4, size=len(problem.target_ya))
    c2 = rng.normal(scale=0.4, size=len(problem.target_yb))
    got = sparse_factorized_margins_reference(problem, log_base, c1, c2)
    plan = build_sparse_intersection_plan(problem, edge_chunk_size=7)
    assert isinstance(plan, SparseIntersectionPlan)
    vectorized = sparse_factorized_margins(
        problem, plan, log_base, c1, c2
    )

    normalized_base = log_base - np.log(np.exp(log_base).sum())
    score = np.broadcast_to(normalized_base[:, None, None], (v, v, v)).copy()
    score[problem.active_ya_y, problem.active_ya_a, :] += c1[:, None]
    score[problem.active_yb_y, :, problem.active_yb_b] += c2[:, None]
    log_z = np.log(np.exp(score).sum(axis=0))
    conditional = np.exp(score - log_z[None, :, :])
    joint = conditional * p_ab[None, :, :]
    dense_ya, dense_yb, _ = _pair_margins(joint)

    assert np.max(np.abs(got.target_y - joint.sum(axis=(1, 2)))) < 2e-14
    assert np.max(np.abs(
        got.active_ya - dense_ya[problem.active_ya_y, problem.active_ya_a]
    )) < 2e-14
    assert np.max(np.abs(
        got.active_yb - dense_yb[problem.active_yb_y, problem.active_yb_b]
    )) < 2e-14
    assert np.max(np.abs(
        got.log_normalizer - log_z[problem.edge_a, problem.edge_b]
    )) < 2e-14
    assert np.max(np.abs(vectorized.target_y - got.target_y)) < 2e-14
    assert np.max(np.abs(vectorized.active_ya - got.active_ya)) < 2e-14
    assert np.max(np.abs(vectorized.active_yb - got.active_yb)) < 2e-14
    assert np.max(np.abs(
        vectorized.log_normalizer - got.log_normalizer
    )) < 2e-14


def test_native_intersection_plan_matches_scipy_reference(monkeypatch):
    import product_model_with_memory.graphical_calibration as calibration

    if not (
        calibration._graphical_margin_c is not None
        and hasattr(calibration._graphical_margin_c, "intersection_plan")
    ):
        pytest.skip("native intersection builder is not available")
    rng = np.random.default_rng(20260810)
    v = 11
    raw = rng.gamma(shape=0.8, scale=1.0, size=(v, v, v))
    raw /= raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(raw)
    problem = sparse_problem_from_dense(
        p_ya, p_yb, p_ab,
        rng.random((v, v)) < 0.55,
        rng.random((v, v)) < 0.45,
    )
    direct = calibration.build_sparse_intersection_plan(problem)
    monkeypatch.setattr(calibration, "_graphical_margin_c", None)
    reference = calibration.build_sparse_intersection_plan(
        problem, edge_chunk_size=7
    )
    np.testing.assert_array_equal(direct.edge, reference.edge)
    np.testing.assert_array_equal(direct.target_y, reference.target_y)
    np.testing.assert_array_equal(
        direct.correction_ya, reference.correction_ya
    )
    np.testing.assert_array_equal(
        direct.correction_yb, reference.correction_yb
    )


def test_intersection_plan_respects_memory_limit():
    rng = np.random.default_rng(20260811)
    raw = rng.gamma(shape=1.0, scale=1.0, size=(5, 5, 5))
    raw /= raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(raw)
    problem = sparse_problem_from_dense(
        p_ya, p_yb, p_ab,
        np.ones((5, 5), dtype=bool),
        np.ones((5, 5), dtype=bool),
    )
    with pytest.raises(MemoryError):
        build_sparse_intersection_plan(problem, max_intersections=1)


def test_layered_intersection_graph_reconstructs_active_plans_and_margins(
    tmp_path,
):
    rng = np.random.default_rng(20260812)
    v = 7
    raw = rng.gamma(shape=0.9, scale=1.0, size=(v, v, v))
    raw /= raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(raw)
    problem = sparse_problem_from_dense(
        p_ya, p_yb, p_ab,
        rng.random((v, v)) < 0.65,
        rng.random((v, v)) < 0.55,
    )
    plan = build_sparse_intersection_plan(problem)
    layers = 5
    birth_ya = rng.integers(layers, size=len(problem.target_ya))
    birth_yb = rng.integers(layers, size=len(problem.target_yb))
    birth_ab = rng.integers(layers, size=len(problem.edge_probability))
    triangle_birth = np.maximum.reduce([
        birth_ya[plan.correction_ya],
        birth_yb[plan.correction_yb],
        birth_ab[plan.edge],
    ])
    graph = layered_intersection_graph_from_plan(
        problem, plan, triangle_birth, layers=layers
    )
    direct_graph = build_layered_intersection_graph(
        problem, birth_ya, birth_yb, birth_ab, layers=layers
    )
    ab_graph = build_ab_major_intersection_graph(
        problem, birth_ya, birth_yb, birth_ab
    )
    save_ab_major_intersection_graph(ab_graph, tmp_path / "ab_graph")
    mapped_ab = load_ab_major_intersection_graph(tmp_path / "ab_graph")
    assert mapped_ab.edges == ab_graph.edges
    assert all(isinstance(array, np.memmap) for array in (
        mapped_ab.edge_ptr, mapped_ab.correction_ya,
        mapped_ab.correction_yb, mapped_ab.birth,
    ))
    np.testing.assert_array_equal(
        np.diff(ab_graph.edge_ptr),
        np.bincount(plan.edge, minlength=len(problem.edge_probability)),
    )
    np.testing.assert_array_equal(ab_graph.correction_ya, plan.correction_ya)
    np.testing.assert_array_equal(ab_graph.correction_yb, plan.correction_yb)
    np.testing.assert_array_equal(ab_graph.birth, triangle_birth)
    assert ab_graph.nbytes < sum(array.nbytes for array in (
        plan.edge, plan.target_y, plan.correction_ya, plan.correction_yb
    ))
    for expected, actual in zip(graph.row_ptr, direct_graph.row_ptr):
        np.testing.assert_array_equal(actual, expected)
    for expected, actual in zip(
        graph.correction_yb, direct_graph.correction_yb
    ):
        np.testing.assert_array_equal(actual, expected)
    for expected, actual in zip(graph.edge_ab, direct_graph.edge_ab):
        np.testing.assert_array_equal(actual, expected)
    assert graph.layers == layers
    assert graph.edges == len(plan.edge)
    assert graph.nbytes < plan.edge.nbytes * 4 + sum(
        pointer.nbytes for pointer in graph.row_ptr
    )

    log_base = rng.normal(scale=0.2, size=v)
    full_c1 = rng.normal(scale=0.1, size=len(problem.target_ya))
    full_c2 = rng.normal(scale=0.1, size=len(problem.target_yb))
    for checkpoint in range(layers):
        reconstructed = intersection_plan_from_layered_graph(
            problem, graph, checkpoint
        )
        selected = np.flatnonzero(triangle_birth <= checkpoint)
        target = plan.target_y[selected]
        edge = plan.edge[selected]
        order = np.lexsort((target, edge))
        np.testing.assert_array_equal(reconstructed.edge, edge[order])
        np.testing.assert_array_equal(reconstructed.target_y, target[order])
        np.testing.assert_array_equal(
            reconstructed.correction_ya, plan.correction_ya[selected][order]
        )
        np.testing.assert_array_equal(
            reconstructed.correction_yb, plan.correction_yb[selected][order]
        )


    reconstructed = intersection_plan_from_layered_graph(
        problem, graph, layers - 1
    )
    explicit = sparse_factorized_margins(
        problem, plan, log_base, full_c1, full_c2
    )
    layered = sparse_factorized_margins(
        problem, reconstructed, log_base, full_c1, full_c2
    )
    direct_layered = sparse_factorized_margins_layered_reference(
        problem, graph, layers - 1, log_base, full_c1, full_c2
    )
    native_layered = sparse_factorized_margins_layered(
        problem, graph, layers - 1, log_base, full_c1, full_c2
    )
    parallel_layered = sparse_factorized_margins_layered(
        problem, graph, layers - 1, log_base, full_c1, full_c2,
        workers=4,
    )
    ab_major = sparse_factorized_margins_ab_major(
        problem, ab_graph, layers - 1, 0,
        log_base, full_c1, full_c2,
    )
    parallel_ab_major = sparse_factorized_margins_ab_major(
        problem, ab_graph, layers - 1, 0,
        log_base, full_c1, full_c2, workers=4,
    )
    np.testing.assert_allclose(layered.target_y, explicit.target_y,
                               rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(layered.active_ya, explicit.active_ya,
                               rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(layered.active_yb, explicit.active_yb,
                               rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(
        layered.log_normalizer, explicit.log_normalizer,
        rtol=0.0, atol=2e-15,
    )
    np.testing.assert_allclose(direct_layered.target_y, explicit.target_y,
                               rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(direct_layered.active_ya, explicit.active_ya,
                               rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(direct_layered.active_yb, explicit.active_yb,
                               rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(
        direct_layered.log_normalizer, explicit.log_normalizer,
        rtol=0.0, atol=2e-15,
    )
    np.testing.assert_allclose(native_layered.target_y, explicit.target_y,
                               rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(native_layered.active_ya, explicit.active_ya,
                               rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(native_layered.active_yb, explicit.active_yb,
                               rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(
        native_layered.log_normalizer, explicit.log_normalizer,
        rtol=0.0, atol=2e-15,
    )
    np.testing.assert_allclose(parallel_layered.target_y, native_layered.target_y,
                               rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(parallel_layered.active_ya, native_layered.active_ya,
                               rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(parallel_layered.active_yb, native_layered.active_yb,
                               rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(
        parallel_layered.log_normalizer, native_layered.log_normalizer,
        rtol=0.0, atol=2e-15,
    )
    np.testing.assert_allclose(ab_major.target_y, explicit.target_y,
                               rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(ab_major.active_ya, explicit.active_ya,
                               rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(ab_major.active_yb, explicit.active_yb,
                               rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(ab_major.log_normalizer,
                               explicit.log_normalizer,
                               rtol=0.0, atol=2e-15)
    for actual, expected in zip(
        (parallel_ab_major.target_y, parallel_ab_major.active_ya,
         parallel_ab_major.active_yb, parallel_ab_major.log_normalizer),
        (ab_major.target_y, ab_major.active_ya, ab_major.active_yb,
         ab_major.log_normalizer),
    ):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-15)
    edge_lo, edge_hi = 5, 19
    block = sparse_edge_block_from_bounds(problem, edge_lo, edge_hi)
    explicit_block = sparse_factorized_margins(
        block.problem, block.intersection_plan,
        log_base, full_c1, full_c2,
    )
    ab_block = sparse_factorized_margins_ab_major(
        block.problem, ab_graph, layers - 1, edge_lo,
        log_base, full_c1, full_c2,
    )
    for actual, expected in zip(
        (ab_block.target_y, ab_block.active_ya, ab_block.active_yb,
         ab_block.log_normalizer),
        (explicit_block.target_y, explicit_block.active_ya,
         explicit_block.active_yb, explicit_block.log_normalizer),
    ):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-15)
    selected_edges = np.asarray([2, 5, 7, 7, 15], dtype=np.int32)
    selected_probability = np.asarray([1, 2, 3, 4, 5], dtype=float)
    indexed_problem = sparse_problem_with_edge_distribution(
        problem, selected_edges, selected_probability
    )
    indexed_plan = build_sparse_intersection_plan(indexed_problem)
    explicit_indexed = sparse_factorized_margins(
        indexed_problem, indexed_plan, log_base, full_c1, full_c2
    )
    ab_indexed = sparse_factorized_margins_ab_major(
        indexed_problem, ab_graph, layers - 1, 0,
        log_base, full_c1, full_c2, workers=4,
        edge_indices=selected_edges,
    )
    for actual, expected in zip(
        (ab_indexed.target_y, ab_indexed.active_ya,
         ab_indexed.active_yb, ab_indexed.log_normalizer),
        (explicit_indexed.target_y, explicit_indexed.active_ya,
         explicit_indexed.active_yb, explicit_indexed.log_normalizer),
    ):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-15)
    dual_explicit = sparse_factorized_dual_evaluation(
        problem, log_base, full_c1, full_c2,
        intersection_plan=plan, compute_certificate=True,
    )
    dual_layered = sparse_factorized_dual_evaluation(
        problem, log_base, full_c1, full_c2,
        layered_graph=graph, layered_checkpoint=layers - 1,
        margin_workers=4, compute_certificate=True,
    )
    np.testing.assert_allclose(
        dual_layered.gradient(), dual_explicit.gradient(),
        rtol=0.0, atol=3e-15,
    )
    assert abs(dual_layered.objective - dual_explicit.objective) < 3e-15
    assert abs(dual_layered.certificate - dual_explicit.certificate) < 3e-15
    stochastic_layered = stochastic_sparse_dual_approach(
        problem, log_base, full_c1, full_c2,
        steps=0, batch_size=1, exact_margin_workers=4,
        exact_layered_graph=graph,
        exact_layered_checkpoint=layers - 1,
    )
    assert abs(
        stochastic_layered.best_exact_certificate
        - dual_layered.certificate
    ) < 3e-15
    stochastic_lazy = stochastic_sparse_dual_approach(
        problem, log_base, full_c1, full_c2,
        steps=2, batch_size=1, sampling="blocks", edge_blocks=4,
        replicas=2, stochastic_workers=2, variance_reduction=True,
        exact_interval=1, exact_margin_workers=2,
        exact_layered_graph=graph,
        exact_layered_checkpoint=layers - 1,
        lazy_block_cache=1,
    )
    assert stochastic_lazy.steps == 2
    assert np.isfinite(stochastic_lazy.best_exact_certificate)
    stochastic_ab = stochastic_sparse_dual_approach(
        problem, log_base, full_c1, full_c2,
        steps=2, batch_size=1, sampling="blocks", edge_blocks=4,
        replicas=2, stochastic_workers=2, variance_reduction=True,
        exact_interval=1, exact_margin_workers=2,
        exact_layered_graph=graph,
        exact_layered_checkpoint=layers - 1,
        sampled_ab_major_graph=ab_graph,
        lazy_block_cache=1,
        fused_ab_batch=True,
    )
    assert stochastic_ab.steps == 2
    assert stochastic_ab.intersection_plan_bytes == 0
    assert np.isfinite(stochastic_ab.best_exact_certificate)
    stochastic_ab_legacy = stochastic_sparse_dual_approach(
        problem, log_base, full_c1, full_c2,
        steps=2, batch_size=1, sampling="blocks", edge_blocks=4,
        replicas=2, stochastic_workers=2, variance_reduction=True,
        exact_interval=1, exact_margin_workers=2,
        exact_layered_graph=graph,
        exact_layered_checkpoint=layers - 1,
        sampled_ab_major_graph=ab_graph,
        lazy_block_cache=4, fused_ab_batch=False,
    )
    for actual, expected in zip(
        (stochastic_ab.log_base_y, stochastic_ab.correction_ya,
         stochastic_ab.correction_yb),
        (stochastic_ab_legacy.log_base_y,
         stochastic_ab_legacy.correction_ya,
         stochastic_ab_legacy.correction_yb),
    ):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-13)
    fitted_explicit = sparse_grouped_ipf(
        problem, solver="lbfgs", evaluator="factorized",
        tolerance=1e-9, max_iterations=2_000,
    )
    fitted_layered = sparse_grouped_ipf(
        problem, solver="lbfgs", evaluator="layered",
        tolerance=1e-9, max_iterations=2_000,
        _layered_graph=graph, _layered_checkpoint=layers - 1,
    )
    assert fitted_layered.converged
    np.testing.assert_allclose(
        fitted_layered.grouped_residual_ya_l1,
        fitted_explicit.grouped_residual_ya_l1,
        rtol=0.0, atol=2e-10,
    )
    np.testing.assert_allclose(
        fitted_layered.grouped_residual_yb_l1,
        fitted_explicit.grouped_residual_yb_l1,
        rtol=0.0, atol=2e-10,
    )
    recovered_extreme = sparse_grouped_ipf(
        problem, solver="lbfgs", evaluator="layered",
        tolerance=1e-8, max_iterations=2_000,
        log_base_y=log_base,
        correction_ya=np.full(len(problem.target_ya), 800.0),
        correction_yb=np.full(len(problem.target_yb), 800.0),
        _layered_graph=graph, _layered_checkpoint=layers - 1,
    )
    assert recovered_extreme.converged
    assert max(
        recovered_extreme.residual_y_l1,
        recovered_extreme.grouped_residual_ya_l1,
        recovered_extreme.grouped_residual_yb_l1,
    ) < 1e-8


def test_birth_major_support_makes_every_checkpoint_a_graph_prefix(tmp_path):
    rng = np.random.default_rng(20260813)
    v = 8
    raw = rng.gamma(shape=0.9, scale=1.0, size=(v, v, v))
    raw /= raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(raw)
    final = sparse_problem_from_dense(
        p_ya, p_yb, p_ab,
        rng.random((v, v)) < 0.7,
        rng.random((v, v)) < 0.6,
    )
    layers = 4
    births = (
        rng.integers(layers, size=len(final.target_ya), dtype=np.uint8),
        rng.integers(layers, size=len(final.target_yb), dtype=np.uint8),
        rng.integers(
            layers, size=len(final.edge_probability), dtype=np.uint8
        ),
    )
    support = birth_major_sparse_support(final, *births)
    graph = build_layered_intersection_graph(
        support.problem,
        support.birth_ya, support.birth_yb, support.birth_ab,
        layers=layers,
    )
    save_layered_intersection_graph(graph, tmp_path / "graph")
    mapped = load_layered_intersection_graph(tmp_path / "graph")
    assert mapped.edges == graph.edges
    assert all(isinstance(array, np.memmap) for family in (
        mapped.row_ptr, mapped.correction_yb, mapped.edge_ab
    ) for array in family)
    graph = mapped
    for checkpoint in range(layers):
        chosen1 = np.flatnonzero(births[0] <= checkpoint)
        chosen2 = np.flatnonzero(births[1] <= checkpoint)
        chosen_ab = np.flatnonzero(births[2] <= checkpoint)
        rng.shuffle(chosen1)
        rng.shuffle(chosen2)
        rng.shuffle(chosen_ab)
        edge_probability = final.edge_probability[chosen_ab]
        edge_probability /= edge_probability.sum()
        unordered = SparseGroupedProblem(
            vocabulary_size=v,
            edge_a=final.edge_a[chosen_ab],
            edge_b=final.edge_b[chosen_ab],
            edge_probability=edge_probability,
            target_y=final.target_y,
            active_ya_y=final.active_ya_y[chosen1],
            active_ya_a=final.active_ya_a[chosen1],
            target_ya=final.target_ya[chosen1],
            active_yb_y=final.active_yb_y[chosen2],
            active_yb_b=final.active_yb_b[chosen2],
            target_yb=final.target_yb[chosen2],
        )
        aligned = checkpoint_in_birth_major_support(
            unordered, support, checkpoint
        )
        assert np.all(support.birth_ya[:len(aligned.target_ya)] <= checkpoint)
        assert np.all(support.birth_yb[:len(aligned.target_yb)] <= checkpoint)
        assert np.all(support.birth_ab[:len(aligned.edge_probability)] <= checkpoint)
        explicit_plan = build_sparse_intersection_plan(aligned)
        layered_plan = intersection_plan_from_layered_graph(
            aligned, graph, checkpoint
        )
        np.testing.assert_array_equal(layered_plan.edge, explicit_plan.edge)
        np.testing.assert_array_equal(
            layered_plan.target_y, explicit_plan.target_y
        )
        np.testing.assert_array_equal(
            layered_plan.correction_ya, explicit_plan.correction_ya
        )
        np.testing.assert_array_equal(
            layered_plan.correction_yb, explicit_plan.correction_yb
        )

        log_base = rng.normal(scale=0.2, size=v)
        c1 = rng.normal(scale=0.1, size=len(aligned.target_ya))
        c2 = rng.normal(scale=0.1, size=len(aligned.target_yb))
        explicit = sparse_factorized_margins(
            aligned, explicit_plan, log_base, c1, c2
        )
        layered = sparse_factorized_margins_layered(
            aligned, graph, checkpoint, log_base, c1, c2, workers=3
        )
        np.testing.assert_allclose(layered.target_y, explicit.target_y,
                                   rtol=0.0, atol=3e-15)
        np.testing.assert_allclose(layered.active_ya, explicit.active_ya,
                                   rtol=0.0, atol=3e-15)
        np.testing.assert_allclose(layered.active_yb, explicit.active_yb,
                                   rtol=0.0, atol=3e-15)
        np.testing.assert_allclose(
            layered.log_normalizer, explicit.log_normalizer,
            rtol=0.0, atol=3e-15,
        )


def test_sampled_sparse_dual_gradient_is_unbiased_edge_by_edge():
    rng = np.random.default_rng(1701)
    v = 5
    raw = rng.gamma(shape=1.3, scale=1.0, size=(v, v, v))
    raw /= raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(raw)
    active_ya = rng.random((v, v)) < 0.55
    active_yb = rng.random((v, v)) < 0.55
    problem = sparse_problem_from_dense(
        p_ya, p_yb, p_ab, active_ya, active_yb
    )
    lb = rng.normal(scale=0.3, size=v)
    c1 = rng.normal(scale=0.2, size=len(problem.target_ya))
    c2 = rng.normal(scale=0.2, size=len(problem.target_yb))
    exact = sparse_factorized_dual_evaluation(problem, lb, c1, c2)

    mean_objective = 0.0
    mean_gradient = np.zeros_like(exact.gradient())
    for edge, probability in enumerate(problem.edge_probability):
        one_edge = sparse_problem_with_edge_distribution(
            problem, np.array([edge]), np.array([1.0])
        )
        sampled = sparse_factorized_dual_evaluation(one_edge, lb, c1, c2)
        mean_objective += probability * sampled.objective
        mean_gradient += probability * sampled.gradient()

    assert abs(mean_objective - exact.objective) < 2e-13
    assert np.max(np.abs(mean_gradient - exact.gradient())) < 2e-13


def test_sampled_sparse_dual_empirical_distribution_and_validation():
    rng = np.random.default_rng(1702)
    raw = rng.gamma(shape=1.1, scale=1.0, size=(4, 4, 4))
    raw /= raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(raw)
    problem = sparse_problem_from_dense(
        p_ya, p_yb, p_ab,
        np.ones_like(p_ya, dtype=bool),
        np.ones_like(p_yb, dtype=bool),
    )
    sampled = sample_sparse_grouped_edges(problem, 20_000, rng)
    empirical = np.zeros(len(problem.edge_probability))
    lookup = {
        (int(a), int(b)): index
        for index, (a, b) in enumerate(zip(problem.edge_a, problem.edge_b))
    }
    for a, b, probability in zip(
        sampled.edge_a, sampled.edge_b, sampled.edge_probability
    ):
        empirical[lookup[int(a), int(b)]] = probability
    assert np.max(np.abs(empirical - problem.edge_probability)) < 0.012
    assert np.array_equal(sampled.target_y, problem.target_y)
    assert np.array_equal(sampled.target_ya, problem.target_ya)
    assert np.array_equal(sampled.target_yb, problem.target_yb)


def test_sparse_edge_minibatch_reuses_exact_intersection_entries():
    rng = np.random.default_rng(1703)
    raw = rng.gamma(shape=1.2, scale=1.0, size=(6, 6, 6))
    raw /= raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(raw)
    problem = sparse_problem_from_dense(
        p_ya, p_yb, p_ab,
        rng.random((6, 6)) < 0.5,
        rng.random((6, 6)) < 0.5,
    )
    full_plan = build_sparse_intersection_plan(problem)
    sampled, sliced_plan = sparse_edge_minibatch(
        problem, full_plan, 40, rng
    )
    fresh_plan = build_sparse_intersection_plan(sampled)
    assert np.array_equal(sliced_plan.edge, fresh_plan.edge)
    assert np.array_equal(sliced_plan.target_y, fresh_plan.target_y)
    assert np.array_equal(sliced_plan.correction_ya, fresh_plan.correction_ya)
    assert np.array_equal(sliced_plan.correction_yb, fresh_plan.correction_yb)

    lb = rng.normal(scale=0.2, size=6)
    c1 = rng.normal(scale=0.1, size=len(problem.target_ya))
    c2 = rng.normal(scale=0.1, size=len(problem.target_yb))
    sliced = sparse_factorized_dual_evaluation(
        sampled, lb, c1, c2, intersection_plan=sliced_plan
    )
    fresh = sparse_factorized_dual_evaluation(
        sampled, lb, c1, c2, intersection_plan=fresh_plan
    )
    assert abs(sliced.objective - fresh.objective) < 1e-14
    assert np.max(np.abs(sliced.gradient() - fresh.gradient())) < 1e-14


def test_stratified_sparse_edge_sampling_is_empirically_unbiased():
    rng = np.random.default_rng(1705)
    raw = rng.gamma(shape=0.5, scale=1.0, size=(4, 4, 4))
    raw /= raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(raw)
    problem = sparse_problem_from_dense(
        p_ya, p_yb, p_ab,
        np.ones_like(p_ya, dtype=bool),
        np.ones_like(p_yb, dtype=bool),
    )
    plan = build_sparse_intersection_plan(problem)
    lookup = {
        (int(a), int(b)): edge
        for edge, (a, b) in enumerate(zip(problem.edge_a, problem.edge_b))
    }
    mean = np.zeros(len(problem.edge_probability))
    repetitions = 2_000
    for _ in range(repetitions):
        sampled, _ = stratified_sparse_edge_minibatch(
            problem, plan, 8, rng, strata=4
        )
        for a, b, probability in zip(
            sampled.edge_a, sampled.edge_b, sampled.edge_probability
        ):
            mean[lookup[int(a), int(b)]] += probability / repetitions
    assert np.max(np.abs(mean - problem.edge_probability)) < 0.006


def test_probability_weighted_edge_blocks_reproduce_exact_dual():
    rng = np.random.default_rng(1706)
    raw = rng.gamma(shape=0.9, scale=1.0, size=(6, 6, 6))
    raw /= raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(raw)
    problem = sparse_problem_from_dense(
        p_ya, p_yb, p_ab,
        rng.random((6, 6)) < 0.6,
        rng.random((6, 6)) < 0.6,
    )
    plan = build_sparse_intersection_plan(problem)
    blocks = build_sparse_edge_blocks(problem, plan, 7)
    lb = rng.normal(scale=0.2, size=6)
    c1 = rng.normal(scale=0.1, size=len(problem.target_ya))
    c2 = rng.normal(scale=0.1, size=len(problem.target_yb))
    exact = sparse_factorized_dual_evaluation(
        problem, lb, c1, c2, intersection_plan=plan
    )
    objective = 0.0
    gradient = np.zeros_like(exact.gradient())
    for block in blocks:
        evaluation = sparse_factorized_dual_evaluation(
            block.problem, lb, c1, c2,
            intersection_plan=block.intersection_plan,
        )
        objective += block.probability_mass * evaluation.objective
        gradient += block.probability_mass * evaluation.gradient()
    assert abs(sum(block.probability_mass for block in blocks) - 1.0) < 1e-14
    assert abs(objective - exact.objective) < 2e-13
    assert np.max(np.abs(gradient - exact.gradient())) < 2e-13


def test_factorized_hessian_product_matches_gradient_difference():
    rng = np.random.default_rng(1711)
    raw = rng.gamma(shape=0.8, scale=1.0, size=(6, 6, 6))
    raw /= raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(raw)
    problem = sparse_problem_from_dense(
        p_ya, p_yb, p_ab,
        rng.random(p_ya.shape) < 0.65,
        rng.random(p_yb.shape) < 0.65,
    )
    plan = build_sparse_intersection_plan(problem)
    lb = rng.normal(scale=0.2, size=6)
    c1 = rng.normal(scale=0.15, size=len(problem.target_ya))
    c2 = rng.normal(scale=0.15, size=len(problem.target_yb))
    direction = rng.normal(size=6 + len(c1) + len(c2))
    product = sparse_factorized_dual_hessian_product(
        problem, lb, c1, c2, direction, intersection_plan=plan
    )
    epsilon = 1e-5

    def gradient_at(offset):
        vector = np.concatenate([lb, c1, c2]) + offset * direction
        return sparse_factorized_dual_evaluation(
            problem,
            vector[:6], vector[6:6 + len(c1)], vector[6 + len(c1):],
            intersection_plan=plan,
        ).gradient()

    difference = (gradient_at(epsilon) - gradient_at(-epsilon)) / (2 * epsilon)
    assert np.max(np.abs(product - difference)) < 2e-9
    assert float(direction @ product) >= -1e-12
    gauge = np.zeros_like(direction)
    gauge[:6] = 1.0
    gauge_product = sparse_factorized_dual_hessian_product(
        problem, lb, c1, c2, gauge, intersection_plan=plan
    )
    assert np.max(np.abs(gauge_product)) < 2e-14


def test_factorized_hessian_product_matches_literal_covariance():
    rng = np.random.default_rng(1712)
    raw = rng.gamma(shape=0.8, scale=1.0, size=(4, 4, 4))
    raw /= raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(raw)
    mask1 = rng.random(p_ya.shape) < 0.7
    mask2 = rng.random(p_yb.shape) < 0.7
    problem = sparse_problem_from_dense(p_ya, p_yb, p_ab, mask1, mask2)
    plan = build_sparse_intersection_plan(problem)
    v = problem.vocabulary_size
    n1 = len(problem.target_ya)
    n2 = len(problem.target_yb)
    lb = rng.normal(scale=0.2, size=v)
    c1 = rng.normal(scale=0.15, size=n1)
    c2 = rng.normal(scale=0.15, size=n2)
    direction = rng.normal(size=v + n1 + n2)

    literal = np.zeros((len(direction), len(direction)))
    normalized = lb - np.logaddexp.reduce(lb)
    for a, b, edge_mass in zip(
        problem.edge_a, problem.edge_b, problem.edge_probability
    ):
        features = np.zeros((v, len(direction)))
        features[np.arange(v), np.arange(v)] = 1.0
        first = np.flatnonzero(problem.active_ya_a == a)
        second = np.flatnonzero(problem.active_yb_b == b)
        features[problem.active_ya_y[first], v + first] = 1.0
        features[problem.active_yb_y[second], v + n1 + second] = 1.0
        scores = normalized + features[:, v:] @ np.concatenate([c1, c2])
        probability = np.exp(scores - np.logaddexp.reduce(scores))
        mean = probability @ features
        centered = features - mean
        literal += edge_mass * (centered.T * probability) @ centered

    product = sparse_factorized_dual_hessian_product(
        problem, lb, c1, c2, direction, intersection_plan=plan
    )
    assert np.max(np.abs(product - literal @ direction)) < 2e-13


def test_sparse_newton_cg_converges_and_counts_products():
    rng = np.random.default_rng(1713)
    raw = rng.gamma(shape=1.2, scale=1.0, size=(4, 4, 4))
    raw /= raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(raw)
    problem = sparse_problem_from_dense(
        p_ya, p_yb, p_ab,
        np.ones_like(p_ya, dtype=bool),
        np.ones_like(p_yb, dtype=bool),
    )
    result = sparse_grouped_newton_cg(
        problem,
        log_base_y=np.log(problem.target_y),
        correction_ya=np.zeros(len(problem.target_ya)),
        correction_yb=np.zeros(len(problem.target_yb)),
        max_iterations=100,
        tolerance=1e-8,
    )
    assert result.converged
    assert result.margin_evaluations > 0
    assert result.hessian_products > 0
    assert max(
        result.residual_y_l1,
        result.grouped_residual_ya_l1,
        result.grouped_residual_yb_l1,
    ) <= 1e-8

    budgeted = sparse_grouped_newton_cg(
        problem,
        log_base_y=np.log(problem.target_y),
        correction_ya=np.zeros(len(problem.target_ya)),
        correction_yb=np.zeros(len(problem.target_yb)),
        max_iterations=100,
        tolerance=1e-15,
        max_hessian_products=1,
    )
    assert budgeted.hessian_products == 1
    assert budgeted.margin_evaluations >= 1


def test_factorized_margins_stably_recompute_cancelled_edge():
    raw = np.arange(1, 28, dtype=np.float64).reshape(3, 3, 3)
    raw /= raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(raw)
    mask1 = np.zeros_like(p_ya, dtype=bool)
    mask2 = np.zeros_like(p_yb, dtype=bool)
    mask1[0, 0] = True
    mask2[0, 0] = True
    problem = sparse_problem_from_dense(p_ya, p_yb, p_ab, mask1, mask2)
    plan = build_sparse_intersection_plan(problem)
    lb = np.log(problem.target_y)
    c1 = np.array([40.0])
    c2 = np.array([-40.0])
    margins = sparse_factorized_margins(problem, plan, lb, c1, c2)
    assert np.all(np.isfinite(margins.target_y))
    assert np.all(np.isfinite(margins.active_ya))
    assert np.all(np.isfinite(margins.active_yb))
    assert np.all(np.isfinite(margins.log_normalizer))
    selected = np.flatnonzero(
        (problem.edge_a == 0) & (problem.edge_b == 0)
    )
    assert len(selected) == 1
    # The two corrections cancel exactly for y=0 on this edge, so its
    # conditional distribution is the normalized baseline and Z=1.
    assert margins.log_normalizer[selected[0]] == pytest.approx(
        0.0, abs=2e-14
    )


def test_stochastic_sparse_approach_improves_exact_dual():
    rng = np.random.default_rng(1704)
    raw = rng.gamma(shape=0.8, scale=1.0, size=(5, 5, 5))
    raw /= raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(raw)
    problem = sparse_problem_from_dense(
        p_ya, p_yb, p_ab,
        np.ones_like(p_ya, dtype=bool),
        np.ones_like(p_yb, dtype=bool),
    )
    lb = np.log(problem.target_y)
    c1 = np.zeros(len(problem.target_ya))
    c2 = np.zeros(len(problem.target_yb))
    initial = sparse_factorized_dual_evaluation(problem, lb, c1, c2)
    records = []
    result = stochastic_sparse_dual_approach(
        problem, lb, c1, c2,
        steps=250, batch_size=250, learning_rate=0.02,
        exact_interval=25, seed=9, trust_radius=4.0,
        exact_record_callback=lambda step, certificate, *_: records.append(
            (step, certificate)
        ),
    )
    final = sparse_factorized_dual_evaluation(
        problem, result.log_base_y,
        result.correction_ya, result.correction_yb,
    )
    assert final.objective < initial.objective - 1e-3
    assert result.exact_evaluations == 11
    assert result.sampled_edges == 62_500
    assert len(records) == result.exact_evaluations
    assert records[0][0] == 0 and records[-1][0] == 250


def test_exact_armijo_selects_scale_and_reduces_certificate():
    rng = np.random.default_rng(17041)
    raw = rng.gamma(shape=0.8, scale=1.0, size=(5, 5, 5))
    raw /= raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(raw)
    problem = sparse_problem_from_dense(
        p_ya, p_yb, p_ab,
        np.ones_like(p_ya, dtype=bool),
        np.ones_like(p_yb, dtype=bool),
    )
    lb = np.log(problem.target_y)
    c1 = np.zeros(len(problem.target_ya))
    c2 = np.zeros(len(problem.target_yb))
    initial = sparse_factorized_dual_evaluation(
        problem, lb, c1, c2, compute_certificate=True
    )
    result = exact_sparse_dual_wolfe(
        problem, lb, c1, c2, max_iterations=100, tolerance=1e-3
    )
    assert result.objective < initial.objective
    assert result.certificate < initial.certificate
    assert result.evaluations > result.iterations
    assert all(
        later["objective"] <= earlier["objective"] + 1e-12
        for earlier, later in zip(result.trace, result.trace[1:])
    )
    assert any("accepted_step" in row for row in result.trace)


def test_block_svrg_approach_improves_exact_dual():
    rng = np.random.default_rng(1707)
    raw = rng.gamma(shape=0.8, scale=1.0, size=(5, 5, 5))
    raw /= raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(raw)
    problem = sparse_problem_from_dense(
        p_ya, p_yb, p_ab,
        np.ones_like(p_ya, dtype=bool),
        np.ones_like(p_yb, dtype=bool),
    )
    lb = np.log(problem.target_y)
    c1 = np.zeros(len(problem.target_ya))
    c2 = np.zeros(len(problem.target_yb))
    initial = sparse_factorized_dual_evaluation(problem, lb, c1, c2)
    result = stochastic_sparse_dual_approach(
        problem, lb, c1, c2,
        steps=100, batch_size=1, learning_rate=0.02,
        exact_interval=25, seed=10, trust_radius=4.0,
        sampling="blocks", edge_blocks=8, replicas=2,
        variance_reduction=True,
    )
    final = sparse_factorized_dual_evaluation(
        problem, result.log_base_y,
        result.correction_ya, result.correction_yb,
    )
    assert final.objective < initial.objective - 1e-3


def test_stochastic_worker_count_does_not_change_fixed_batch_trajectory():
    rng = np.random.default_rng(1708)
    raw = rng.gamma(shape=0.9, scale=1.0, size=(5, 5, 5))
    raw /= raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(raw)
    problem = sparse_problem_from_dense(
        p_ya, p_yb, p_ab,
        np.ones_like(p_ya, dtype=bool),
        np.ones_like(p_yb, dtype=bool),
    )
    initial = (
        np.log(problem.target_y),
        np.zeros(len(problem.target_ya)),
        np.zeros(len(problem.target_yb)),
    )
    results = [
        stochastic_sparse_dual_approach(
            problem, *initial,
            steps=40, batch_size=1, learning_rate=0.01,
            exact_interval=10, seed=12, trust_radius=3.0,
            sampling="blocks", edge_blocks=8, replicas=4,
            stochastic_workers=workers, variance_reduction=True,
        )
        for workers in (1, 2, 4)
    ]
    reference = results[0]
    for result in results[1:]:
        # Worker-local reduction changes only floating-point association; the
        # fixed replica batch and resulting trajectory remain numerically the
        # same for every worker count.
        np.testing.assert_allclose(
            result.log_base_y, reference.log_base_y, rtol=0.0, atol=1e-13
        )
        np.testing.assert_allclose(
            result.correction_ya, reference.correction_ya,
            rtol=0.0, atol=1e-13,
        )
        np.testing.assert_allclose(
            result.correction_yb, reference.correction_yb,
            rtol=0.0, atol=1e-13,
        )
        assert result.steps == reference.steps
        assert result.stop_reason == reference.stop_reason
        assert result.best_exact_certificate == pytest.approx(
            reference.best_exact_certificate, abs=1e-13
        )


def test_adam_plateau_scheduler_reduces_rate():
    rng = np.random.default_rng(1710)
    raw = rng.gamma(shape=0.8, scale=1.0, size=(5, 5, 5))
    raw /= raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(raw)
    problem = sparse_problem_from_dense(
        p_ya, p_yb, p_ab,
        np.ones_like(p_ya, dtype=bool),
        np.ones_like(p_yb, dtype=bool),
    )
    result = stochastic_sparse_dual_approach(
        problem,
        np.log(problem.target_y),
        np.zeros(len(problem.target_ya)),
        np.zeros(len(problem.target_yb)),
        steps=20, batch_size=1, learning_rate=0.03,
        minimum_learning_rate=0.003, exact_interval=5,
        seed=13, trust_radius=0.02, sampling="blocks",
        edge_blocks=8, replicas=2, variance_reduction=True,
        optimizer="adam_plateau", plateau_patience=1,
        plateau_relative_threshold=0.99,
    )
    reductions = [
        record for record in result.trace
        if record["learning_rate_reduced"]
    ]
    assert reductions
    assert float(reductions[0]["step_size"]) == pytest.approx(0.01)


def test_adam_plateau_stops_when_minimum_rate_is_exhausted():
    rng = np.random.default_rng(1711)
    raw = rng.gamma(shape=0.8, scale=1.0, size=(5, 5, 5))
    raw /= raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(raw)
    problem = sparse_problem_from_dense(
        p_ya, p_yb, p_ab,
        np.ones_like(p_ya, dtype=bool),
        np.ones_like(p_yb, dtype=bool),
    )
    result = stochastic_sparse_dual_approach(
        problem,
        np.log(problem.target_y),
        np.zeros(len(problem.target_ya)),
        np.zeros(len(problem.target_yb)),
        steps=100, batch_size=1, learning_rate=0.003,
        minimum_learning_rate=0.003, exact_interval=5,
        seed=14, trust_radius=0.02, sampling="blocks",
        edge_blocks=8, replicas=2, variance_reduction=True,
        optimizer="adam_plateau", plateau_patience=1,
        plateau_relative_threshold=0.99,
    )
    assert result.steps < 100
    assert result.trace[-1]["scheduler_exhausted"]
    assert result.stop_reason == "plateau"


def test_native_sparse_margins_match_numpy_reference(monkeypatch):
    import product_model_with_memory.graphical_calibration as calibration

    native = calibration._graphical_margin_c
    if native is None:
        pytest.skip("native margin extension is not built")
    rng = np.random.default_rng(1709)
    raw = rng.gamma(shape=0.8, scale=1.0, size=(6, 6, 6))
    raw /= raw.sum()
    p_ya, p_yb, p_ab = _pair_margins(raw)
    problem = sparse_problem_from_dense(
        p_ya, p_yb, p_ab,
        rng.random(p_ya.shape) < 0.7,
        rng.random(p_yb.shape) < 0.7,
    )
    plan = build_sparse_intersection_plan(problem)
    factors = (
        rng.normal(size=6),
        rng.normal(scale=0.2, size=len(problem.target_ya)),
        rng.normal(scale=0.2, size=len(problem.target_yb)),
    )
    actual = sparse_factorized_margins(problem, plan, *factors)
    monkeypatch.setattr(calibration, "_graphical_margin_c", None)
    expected = sparse_factorized_margins(problem, plan, *factors)
    np.testing.assert_allclose(actual.target_y, expected.target_y, rtol=2e-14)
    np.testing.assert_allclose(actual.active_ya, expected.active_ya, rtol=2e-14)
    np.testing.assert_allclose(actual.active_yb, expected.active_yb, rtol=2e-14)
    np.testing.assert_allclose(
        actual.log_normalizer, expected.log_normalizer, rtol=2e-14
    )


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
