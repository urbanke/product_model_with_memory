"""Maximum-entropy calibration from three pairwise marginals.

For variables ``Y, A, B``, suppose the three mutually consistent pair
marginals ``P_ya``, ``P_yb`` and ``P_ab`` are given.  The maximum-entropy
joint has the triangle form

    q(y, a, b) proportional to psi_ya(y, a) psi_yb(y, b) psi_ab(a, b).

Equivalently, fix the third marginal and write

    q(y, a, b) = P_ab(a, b) r(y | a, b),
    r(y | a, b) proportional to f_ya(y, a) f_yb(y, b).

The latter form eliminates ``psi_ab`` analytically.  This module implements
that conditional formulation.  The present dense arrays are a reference for
small alphabets.  They deliberately isolate the equations that a later
background-plus-corrections implementation must reproduce without building
dense probability tables.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


@dataclass(frozen=True)
class ConditionalIPFResult:
    """Result of the three-margin conditional IPF fit."""

    log_f_ya: np.ndarray
    log_f_yb: np.ndarray
    iterations: int
    residual_ya_l1: float
    residual_yb_l1: float
    residual_ab_l1: float
    converged: bool

    def conditional(self) -> np.ndarray:
        """Return ``r[y, a, b]`` for the dense reference problem."""

        score = self.log_f_ya[:, :, None] + self.log_f_yb[:, None, :]
        return np.exp(score - logsumexp(score, axis=0, keepdims=True))

    def joint(self, p_ab: np.ndarray) -> np.ndarray:
        """Return ``q[y, a, b] = p_ab[a, b] r[y | a, b]``."""

        return self.conditional() * np.asarray(p_ab)[None, :, :]


@dataclass(frozen=True)
class GroupedConditionalIPFResult(ConditionalIPFResult):
    """Conditional IPF result with grouped inactive-cell constraints."""

    log_base_y: np.ndarray
    grouped_residual_ya_l1: float
    grouped_residual_yb_l1: float
    residual_y_l1: float

    def conditional(self) -> np.ndarray:
        score = (
            self.log_base_y[:, None, None]
            + self.log_f_ya[:, :, None]
            + self.log_f_yb[:, None, :]
        )
        return np.exp(score - logsumexp(score, axis=0, keepdims=True))


@dataclass(frozen=True)
class GroupedCheckpoint:
    """The three margins and active features at one causal checkpoint."""

    p_ya: np.ndarray
    p_yb: np.ndarray
    p_ab: np.ndarray
    active_ya: np.ndarray
    active_yb: np.ndarray


@dataclass(frozen=True)
class RestrictedMargins:
    """Consistent margins after restricting the context-pair support."""

    p_ya: np.ndarray
    p_yb: np.ndarray
    p_ab: np.ndarray
    retained_ab_mass: float


@dataclass(frozen=True)
class SparseProjectedPair:
    """Pair matrix as an outer-product background plus sparse corrections.

    Matrix orientation is ``(target, context)``.  Its value is

    ``left[context] * right[target] * (background[context] + delta[cell])``.

    ``delta`` is present only at explicitly observed conditional cells.
    """

    vocabulary_size: int
    left: np.ndarray
    right: np.ndarray
    background: np.ndarray
    active_y: np.ndarray
    active_context: np.ndarray
    delta: np.ndarray

    def active_values(self) -> np.ndarray:
        base = self.background[self.active_context] + self.delta
        return (
            self.left[self.active_context] * self.right[self.active_y] * base
        )

    def values(self, target: np.ndarray, context: np.ndarray) -> np.ndarray:
        """Evaluate selected cells without materializing the pair matrix."""

        target = np.asarray(target, dtype=np.int64)
        context = np.asarray(context, dtype=np.int64)
        if target.shape != context.shape:
            raise ValueError("target and context indices must match")
        correction = {
            int(y) * self.vocabulary_size + int(a): float(value)
            for y, a, value in zip(
                self.active_y, self.active_context, self.delta
            )
        }
        keys = target * self.vocabulary_size + context
        extra = np.fromiter(
            (correction.get(int(key), 0.0) for key in keys.flat),
            dtype=np.float64,
            count=keys.size,
        ).reshape(keys.shape)
        return (
            self.left[context] * self.right[target]
            * (self.background[context] + extra)
        )

    def dense(self) -> np.ndarray:
        """Materialize only for small-alphabet validation."""

        matrix = self.right[:, None] * (
            self.left * self.background
        )[None, :]
        matrix[self.active_y, self.active_context] += (
            self.left[self.active_context]
            * self.right[self.active_y]
            * self.delta
        )
        return matrix


@dataclass(frozen=True)
class SparseRestrictedMargins:
    """Observed AB edges and consistent sparse target--lag pairs."""

    p_ya: SparseProjectedPair
    p_yb: SparseProjectedPair
    edge_a: np.ndarray
    edge_b: np.ndarray
    edge_probability: np.ndarray
    retained_ab_mass: float


def _sparse_pair_margins(
    pair: SparseProjectedPair,
) -> tuple[np.ndarray, np.ndarray]:
    v = pair.vocabulary_size
    context = pair.left * (
        pair.background * float(pair.right.sum())
        + np.bincount(
            pair.active_context,
            weights=pair.delta * pair.right[pair.active_y],
            minlength=v,
        )
    )
    target = pair.right * (
        float(pair.left @ pair.background)
        + np.bincount(
            pair.active_y,
            weights=pair.delta * pair.left[pair.active_context],
            minlength=v,
        )
    )
    return target, context


def reproject_sparse_pair(
    pair: SparseProjectedPair,
    target_marginal: np.ndarray,
    context_marginal: np.ndarray,
    *,
    max_iterations: int = 10_000,
    tolerance: float = 1e-12,
) -> SparseProjectedPair:
    """Sinkhorn-project an existing implicit sparse pair to new margins."""

    target_marginal = np.asarray(target_marginal, dtype=np.float64)
    context_marginal = np.asarray(context_marginal, dtype=np.float64)
    v = pair.vocabulary_size
    if target_marginal.shape != (v,) or context_marginal.shape != (v,):
        raise ValueError("new sparse pair margins must both have length V")
    left = pair.left.copy()
    right = pair.right.copy()
    tiny = np.finfo(np.float64).tiny
    for _ in range(max_iterations):
        current_target, current_context = _sparse_pair_margins(
            SparseProjectedPair(
                v, left, right, pair.background,
                pair.active_y, pair.active_context, pair.delta,
            )
        )
        if max(
            float(np.abs(current_target - target_marginal).sum()),
            float(np.abs(current_context - context_marginal).sum()),
        ) < tolerance:
            break
        left *= np.divide(
            context_marginal,
            current_context,
            out=np.zeros(v),
            where=current_context > 0.0,
        )
        current_target, _ = _sparse_pair_margins(
            SparseProjectedPair(
                v, left, right, pair.background,
                pair.active_y, pair.active_context, pair.delta,
            )
        )
        right *= np.divide(
            target_marginal,
            current_target,
            out=np.zeros(v),
            where=current_target > 0.0,
        )
        gauge = max(float(left.max()), tiny)
        left /= gauge
        right *= gauge
    result = SparseProjectedPair(
        v, left, right, pair.background,
        pair.active_y, pair.active_context, pair.delta,
    )
    final_target, final_context = _sparse_pair_margins(result)
    if max(
        float(np.abs(final_target - target_marginal).sum()),
        float(np.abs(final_context - context_marginal).sum()),
    ) >= tolerance * 10:
        raise RuntimeError("sparse pair reprojection did not converge")
    return result


def restrict_sparse_margins_to_observed_contexts(
    p_ya: SparseProjectedPair,
    p_yb: SparseProjectedPair,
    observed_a: np.ndarray,
    observed_b: np.ndarray,
    *,
    max_iterations: int = 10_000,
    tolerance: float = 1e-12,
) -> SparseRestrictedMargins:
    """Sparse counterpart of ``restrict_margins_to_observed_contexts``."""

    if p_ya.vocabulary_size != p_yb.vocabulary_size:
        raise ValueError("the two sparse pairs must use one vocabulary")
    v = p_ya.vocabulary_size
    edge_a = np.asarray(observed_a, dtype=np.int64)
    edge_b = np.asarray(observed_b, dtype=np.int64)
    if edge_a.shape != edge_b.shape or not len(edge_a):
        raise ValueError("observed context edge arrays must match and be nonempty")
    ab = p_ya.values(edge_a, edge_b)
    retained = float(ab.sum())
    if not 0.0 < retained <= 1.0 + 1e-10:
        raise ValueError("observed context support has invalid probability mass")
    edge_probability = ab / retained
    margin_a = np.bincount(edge_a, weights=edge_probability, minlength=v)
    margin_b = np.bincount(edge_b, weights=edge_probability, minlength=v)
    target_y, _ = _sparse_pair_margins(p_ya)
    new_ya = reproject_sparse_pair(
        p_ya, target_y, margin_a,
        max_iterations=max_iterations, tolerance=tolerance,
    )
    new_yb = reproject_sparse_pair(
        p_yb, target_y, margin_b,
        max_iterations=max_iterations, tolerance=tolerance,
    )
    return SparseRestrictedMargins(
        new_ya, new_yb, edge_a, edge_b, edge_probability, retained
    )


def sparse_problem_from_projected(
    margins: SparseRestrictedMargins,
) -> SparseGroupedProblem:
    """Construct calibration constraints without dense pair margins."""

    target_y, _ = _sparse_pair_margins(margins.p_ya)
    return SparseGroupedProblem(
        vocabulary_size=margins.p_ya.vocabulary_size,
        edge_a=margins.edge_a,
        edge_b=margins.edge_b,
        edge_probability=margins.edge_probability,
        target_y=target_y,
        active_ya_y=margins.p_ya.active_y,
        active_ya_a=margins.p_ya.active_context,
        target_ya=margins.p_ya.active_values(),
        active_yb_y=margins.p_yb.active_y,
        active_yb_b=margins.p_yb.active_context,
        target_yb=margins.p_yb.active_values(),
    )


def project_sparse_layered_pair(
    context_marginal: np.ndarray,
    unseen_probability: np.ndarray,
    active_y: np.ndarray,
    active_context: np.ndarray,
    active_probability: np.ndarray,
    target_marginal: np.ndarray,
    projected_context_marginal: np.ndarray | None = None,
    *,
    max_iterations: int = 10_000,
    tolerance: float = 1e-12,
) -> SparseProjectedPair:
    """Project a sparse layered conditional joint to supplied margins.

    The input conditional has one ``unseen_probability`` per context row and
    explicit ``active_probability`` values.  Sinkhorn row/column scaling
    preserves its outer-background-plus-corrections form.
    """

    context_marginal = np.asarray(context_marginal, dtype=np.float64)
    target_marginal = np.asarray(target_marginal, dtype=np.float64)
    unseen_probability = np.asarray(unseen_probability, dtype=np.float64)
    active_y = np.asarray(active_y, dtype=np.int64)
    active_context = np.asarray(active_context, dtype=np.int64)
    active_probability = np.asarray(active_probability, dtype=np.float64)
    v = len(context_marginal)
    desired_context = (
        context_marginal if projected_context_marginal is None
        else np.asarray(projected_context_marginal, dtype=np.float64)
    )
    if any(len(array) != v for array in (
        target_marginal, unseen_probability, desired_context
    )):
        raise ValueError("all dense margins/backgrounds must have length V")
    if not (active_y.shape == active_context.shape == active_probability.shape):
        raise ValueError("active pair arrays must have identical shapes")
    if ((active_y < 0) | (active_y >= v) | (active_context < 0)
            | (active_context >= v)).any():
        raise ValueError("active pair indices are outside the vocabulary")

    background = context_marginal * unseen_probability
    delta = context_marginal[active_context] * (
        active_probability - unseen_probability[active_context]
    )
    left = np.ones(v)
    right = np.ones(v)
    tiny = np.finfo(np.float64).tiny
    for _ in range(max_iterations):
        right_sum = float(right.sum())
        sparse_row = np.bincount(
            active_context,
            weights=delta * right[active_y],
            minlength=v,
        )
        current_context = left * (background * right_sum + sparse_row)
        left *= np.divide(
            desired_context,
            current_context,
            out=np.zeros(v),
            where=current_context > 0.0,
        )
        background_column = float(left @ background)
        sparse_column = np.bincount(
            active_y,
            weights=delta * left[active_context],
            minlength=v,
        )
        current_target = right * (background_column + sparse_column)
        right *= np.divide(
            target_marginal,
            current_target,
            out=np.zeros(v),
            where=current_target > 0.0,
        )
        if max(
            float(np.abs(current_context - desired_context).sum()),
            float(np.abs(current_target - target_marginal).sum()),
        ) < tolerance:
            break
        # Remove the harmless global gauge before it grows numerically.
        gauge = max(float(left.max()), tiny)
        left /= gauge
        right *= gauge
    result = SparseProjectedPair(
        v, left, right, background,
        active_y, active_context, delta,
    )
    final_context = left * (
        background * float(right.sum())
        + np.bincount(
            active_context, weights=delta * right[active_y], minlength=v
        )
    )
    final_target = right * (
        float(left @ background)
        + np.bincount(
            active_y, weights=delta * left[active_context], minlength=v
        )
    )
    if max(
        float(np.abs(final_context - desired_context).sum()),
        float(np.abs(final_target - target_marginal).sum()),
    ) >= tolerance * 10:
        raise RuntimeError("sparse pair-margin projection did not converge")
    return result


@dataclass(frozen=True)
class SparseGroupedProblem:
    """Observed-edge grouped calibration without dense pair tables."""

    vocabulary_size: int
    edge_a: np.ndarray
    edge_b: np.ndarray
    edge_probability: np.ndarray
    target_y: np.ndarray
    active_ya_y: np.ndarray
    active_ya_a: np.ndarray
    target_ya: np.ndarray
    active_yb_y: np.ndarray
    active_yb_b: np.ndarray
    target_yb: np.ndarray


@dataclass(frozen=True)
class SparseGroupedResult:
    """Factors and diagnostics of matrix-free grouped IPF."""

    log_base_y: np.ndarray
    correction_ya: np.ndarray
    correction_yb: np.ndarray
    iterations: int
    grouped_residual_ya_l1: float
    grouped_residual_yb_l1: float
    residual_y_l1: float
    converged: bool


@dataclass(frozen=True)
class SparseGroupedCheckpoint:
    """One sparse calibration problem and its layered unigram baseline."""

    problem: SparseGroupedProblem
    log_base_y: np.ndarray


def star_log_probabilities(
    targets: np.ndarray,
    context_a: np.ndarray,
    context_b: np.ndarray,
    p_ya: np.ndarray,
    p_yb: np.ndarray,
) -> np.ndarray:
    """Score the two-pair maximum-entropy (conditional-independence) model."""

    p_ya = np.asarray(p_ya, dtype=np.float64)
    p_yb = np.asarray(p_yb, dtype=np.float64)
    if p_ya.shape != p_yb.shape or p_ya.ndim != 2:
        raise ValueError("pair margins must have the same matrix shape")
    y = np.asarray(targets, dtype=np.int64)
    a = np.asarray(context_a, dtype=np.int64)
    b = np.asarray(context_b, dtype=np.int64)
    if y.shape != a.shape or y.shape != b.shape:
        raise ValueError("targets and contexts must have identical shapes")
    if y.size == 0:
        return np.empty(y.shape, dtype=np.float64)
    tiny = np.finfo(np.float64).tiny
    py = p_ya.sum(axis=1)
    answer = np.empty(y.shape, dtype=np.float64)
    v = p_ya.shape[1]
    keys = a * v + b
    order = np.argsort(keys, kind="stable")
    ordered_keys = keys[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(ordered_keys))]
    stops = np.r_[starts[1:], len(order)]
    for start, stop in zip(starts, stops):
        selected = order[start:stop]
        key = ordered_keys[start]
        aa, bb = divmod(int(key), v)
        raw = p_ya[:, aa] * p_yb[:, bb] / np.maximum(py, tiny)
        chosen = y[selected]
        answer[selected] = (
            np.log(np.maximum(raw[chosen], tiny))
            - np.log(np.maximum(raw.sum(), tiny))
        )
    return answer


def sparse_star_log_probabilities(
    p_ya: SparseProjectedPair,
    p_yb: SparseProjectedPair,
    targets: np.ndarray,
    context_a: np.ndarray,
    context_b: np.ndarray,
) -> np.ndarray:
    """Two-pair maxent scorer from implicit sparse pair margins."""

    if p_ya.vocabulary_size != p_yb.vocabulary_size:
        raise ValueError("sparse fallback pairs must use one vocabulary")
    v = p_ya.vocabulary_size
    y = np.asarray(targets, dtype=np.int64)
    a = np.asarray(context_a, dtype=np.int64)
    b = np.asarray(context_b, dtype=np.int64)
    if y.shape != a.shape or y.shape != b.shape:
        raise ValueError("targets and contexts must have identical shapes")
    if y.size == 0:
        return np.empty(y.shape, dtype=np.float64)
    target_margin, _ = _sparse_pair_margins(p_ya)
    core = p_ya.right * p_yb.right / np.maximum(
        target_margin, np.finfo(np.float64).tiny
    )
    core_sum = float(core.sum())
    rows1: list[dict[int, float]] = [{} for _ in range(v)]
    rows2: list[dict[int, float]] = [{} for _ in range(v)]
    for yy, aa, value in zip(
        p_ya.active_y, p_ya.active_context, p_ya.delta
    ):
        rows1[int(aa)][int(yy)] = float(value)
    for yy, bb, value in zip(
        p_yb.active_y, p_yb.active_context, p_yb.delta
    ):
        rows2[int(bb)][int(yy)] = float(value)

    keys = a * v + b
    order = np.argsort(keys, kind="stable")
    ordered_keys = keys[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(ordered_keys))]
    stops = np.r_[starts[1:], len(order)]
    answer = np.empty(y.shape, dtype=np.float64)
    tiny = np.finfo(np.float64).tiny
    for start, stop in zip(starts, stops):
        selected = order[start:stop]
        aa, bb = divmod(int(ordered_keys[start]), v)
        background_a = p_ya.background[aa]
        background_b = p_yb.background[bb]
        correction_a = rows1[aa]
        correction_b = rows2[bb]
        normalizer = background_a * background_b * core_sum
        for yy in correction_a.keys() | correction_b.keys():
            old = background_a * background_b
            new = (
                background_a + correction_a.get(yy, 0.0)
            ) * (
                background_b + correction_b.get(yy, 0.0)
            )
            normalizer += core[yy] * (new - old)
        chosen = y[selected]
        raw = core[chosen] * np.array([
            (background_a + correction_a.get(int(yy), 0.0))
            * (background_b + correction_b.get(int(yy), 0.0))
            for yy in chosen
        ])
        answer[selected] = (
            np.log(np.maximum(raw, tiny))
            - np.log(max(normalizer, tiny))
        )
    return answer


def sparse_gated_log_probabilities(
    problem: SparseGroupedProblem,
    result: SparseGroupedResult,
    targets: np.ndarray,
    context_a: np.ndarray,
    context_b: np.ndarray,
    fallback_ya: np.ndarray | SparseProjectedPair,
    fallback_yb: np.ndarray | SparseProjectedPair,
) -> np.ndarray:
    """Score records with calibrated observed edges and star fallback.

    The calibrated normalizer is evaluated as baseline mass plus sparse
    corrections; no dense context-by-target table is constructed.  Context
    pairs outside ``problem.edge_*`` use the two-pair maximum-entropy product
    declared for the gated approximation.
    """

    v = problem.vocabulary_size
    y = np.asarray(targets, dtype=np.int64)
    a = np.asarray(context_a, dtype=np.int64)
    b = np.asarray(context_b, dtype=np.int64)
    if y.shape != a.shape or y.shape != b.shape:
        raise ValueError("targets and contexts must have identical shapes")
    if y.size == 0:
        return np.empty(y.shape, dtype=np.float64)
    if ((y < 0) | (y >= v) | (a < 0) | (a >= v)
            | (b < 0) | (b >= v)).any():
        raise ValueError("record ids are outside the problem vocabulary")
    sparse_fallback = isinstance(fallback_ya, SparseProjectedPair)
    if sparse_fallback != isinstance(fallback_yb, SparseProjectedPair):
        raise ValueError("fallback pairs must use the same representation")
    if sparse_fallback:
        p_ya = fallback_ya
        p_yb = fallback_yb
        if p_ya.vocabulary_size != v or p_yb.vocabulary_size != v:
            raise ValueError("sparse fallback vocabulary does not match")
    else:
        p_ya = np.asarray(fallback_ya, dtype=np.float64)
        p_yb = np.asarray(fallback_yb, dtype=np.float64)
        if p_ya.shape != (v, v) or p_yb.shape != (v, v):
            raise ValueError(
                "fallback pair margins must both have shape (V, V)"
            )

    log_base = result.log_base_y - logsumexp(result.log_base_y)
    base = np.exp(log_base)
    first: dict[int, dict[int, float]] = {}
    for yy, aa, value in zip(
        problem.active_ya_y, problem.active_ya_a, result.correction_ya
    ):
        first.setdefault(int(aa), {})[int(yy)] = float(value)
    second: dict[int, dict[int, float]] = {}
    for yy, bb, value in zip(
        problem.active_yb_y, problem.active_yb_b, result.correction_yb
    ):
        second.setdefault(int(bb), {})[int(yy)] = float(value)
    supported = {
        int(aa) * v + int(bb)
        for aa, bb in zip(problem.edge_a, problem.edge_b)
    }

    answer = np.empty(y.shape, dtype=np.float64)
    context_key = a * v + b
    order = np.argsort(context_key, kind="stable")
    ordered_keys = context_key[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(ordered_keys))]
    stops = np.r_[starts[1:], len(order)]
    for start, stop in zip(starts, stops):
        selected = order[start:stop]
        key = ordered_keys[start]
        aa, bb = divmod(int(key), v)
        if int(key) in supported:
            corrections = dict(first.get(aa, {}))
            for yy, value in second.get(bb, {}).items():
                corrections[yy] = corrections.get(yy, 0.0) + value
            corrected_y = np.fromiter(corrections, dtype=np.int64)
            corrected_mass = float(base[corrected_y].sum())
            background_mass = max(0.0, 1.0 - corrected_mass)
            corrected_log_mass = logsumexp(np.array([
                log_base[yy] + value
                for yy, value in corrections.items()
            ])) if corrections else -np.inf
            log_normalizer = np.logaddexp(
                np.log(background_mass) if background_mass > 0.0 else -np.inf,
                corrected_log_mass,
            )
            chosen = y[selected]
            extra = np.array([
                corrections.get(int(yy), 0.0) for yy in chosen
            ])
            answer[selected] = log_base[chosen] + extra - log_normalizer
        else:
            if sparse_fallback:
                answer[selected] = sparse_star_log_probabilities(
                    p_ya, p_yb, y[selected], a[selected], b[selected]
                )
            else:
                answer[selected] = star_log_probabilities(
                    y[selected], a[selected], b[selected], p_ya, p_yb
                )
    return answer


def transfer_sparse_warm_start(
    previous_problem: SparseGroupedProblem,
    previous_result: SparseGroupedResult,
    new_problem: SparseGroupedProblem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carry factors to a later checkpoint whose active support has grown."""

    if previous_problem.vocabulary_size != new_problem.vocabulary_size:
        raise ValueError("warm-start checkpoints must use one vocabulary")
    first = {
        (int(y), int(a)): float(value)
        for y, a, value in zip(
            previous_problem.active_ya_y,
            previous_problem.active_ya_a,
            previous_result.correction_ya,
        )
    }
    second = {
        (int(y), int(b)): float(value)
        for y, b, value in zip(
            previous_problem.active_yb_y,
            previous_problem.active_yb_b,
            previous_result.correction_yb,
        )
    }
    new_first = np.array([
        first.get((int(y), int(a)), 0.0)
        for y, a in zip(new_problem.active_ya_y, new_problem.active_ya_a)
    ])
    new_second = np.array([
        second.get((int(y), int(b)), 0.0)
        for y, b in zip(new_problem.active_yb_y, new_problem.active_yb_b)
    ])
    return previous_result.log_base_y.copy(), new_first, new_second


def first_pair_warm_start(
    problem: SparseGroupedProblem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embed the exact one-pair ``P(y|a)`` model in the larger model.

    The second pair factor starts neutral.  Constants depending only on
    ``a`` cancel in the conditional normalization, so active corrections
    are simply the log ratio between the requested YA cell and the
    unigram baseline contribution.
    """

    tiny = np.finfo(np.float64).tiny
    base = np.maximum(problem.target_y, tiny)
    pa = np.bincount(
        problem.edge_a,
        weights=problem.edge_probability,
        minlength=problem.vocabulary_size,
    )
    denominator = pa[problem.active_ya_a] * base[problem.active_ya_y]
    first = np.log(np.maximum(problem.target_ya, tiny)) - np.log(
        np.maximum(denominator, tiny)
    )
    return np.log(base), first, np.zeros_like(problem.target_yb)


def pair_product_warm_start(
    problem: SparseGroupedProblem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embed the symmetric product/star model using both predictive pairs."""

    log_base, first, _ = first_pair_warm_start(problem)
    tiny = np.finfo(np.float64).tiny
    base = np.maximum(problem.target_y, tiny)
    pb = np.bincount(
        problem.edge_b,
        weights=problem.edge_probability,
        minlength=problem.vocabulary_size,
    )
    denominator = pb[problem.active_yb_b] * base[problem.active_yb_y]
    second = np.log(np.maximum(problem.target_yb, tiny)) - np.log(
        np.maximum(denominator, tiny)
    )
    return log_base, first, second


def second_pair_warm_start(
    problem: SparseGroupedProblem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embed the exact one-pair ``P(y|b)`` model in the larger model."""

    log_base, first, second = pair_product_warm_start(problem)
    return log_base, np.zeros_like(first), second


def pair_midpoint_warm_start(
    problem: SparseGroupedProblem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average the natural parameters of the two one-pair models."""

    log_base, first, second = pair_product_warm_start(problem)
    return log_base, 0.5 * first, 0.5 * second


def fit_sparse_grouped_checkpoints(
    checkpoints: Sequence[SparseGroupedCheckpoint],
    *,
    initial_results: Sequence[SparseGroupedResult] | None = None,
    interleave: int = 1,
    max_iterations: int = 5_000,
    tolerance: float = 1e-5,
    solver: str = "lbfgs",
    margin_workers: int = 1,
    initialization: str = "unigram",
) -> list[SparseGroupedResult]:
    """Fit causal sparse checkpoints in interleaved warm-start chains.

    Chain ``j`` fits checkpoints ``j, j + interleave, ...``.  A later fit
    inherits factors for features already present and initializes newly
    observed features to zero.  Different chains may run concurrently.
    """

    if interleave < 1:
        raise ValueError("interleave must be positive")
    if initialization not in (
        "unigram", "first_pair", "second_pair", "pair_midpoint",
        "pair_product",
    ):
        raise ValueError(
            "unknown initialization strategy"
        )
    points = list(checkpoints)
    if not points:
        return []
    starts = None if initial_results is None else list(initial_results)
    if starts is not None and len(starts) != len(points):
        raise ValueError("initial_results must match the checkpoints")

    def fit_chain(offset: int) -> list[tuple[int, SparseGroupedResult]]:
        answer = []
        previous_point = None
        previous_result = None
        for index in range(offset, len(points), interleave):
            point = points[index]
            if previous_point is None and starts is not None:
                initial = starts[index]
                warm = (
                    initial.log_base_y,
                    initial.correction_ya,
                    initial.correction_yb,
                )
            elif previous_point is None:
                if initialization == "first_pair":
                    warm = first_pair_warm_start(point.problem)
                elif initialization == "second_pair":
                    warm = second_pair_warm_start(point.problem)
                elif initialization == "pair_midpoint":
                    warm = pair_midpoint_warm_start(point.problem)
                elif initialization == "pair_product":
                    warm = pair_product_warm_start(point.problem)
                else:
                    warm = (point.log_base_y, None, None)
            else:
                warm = transfer_sparse_warm_start(
                    previous_point.problem, previous_result, point.problem
                )
            result = sparse_grouped_ipf(
                point.problem,
                max_iterations=max_iterations,
                tolerance=tolerance,
                log_base_y=warm[0],
                correction_ya=warm[1],
                correction_yb=warm[2],
                solver=solver,
                margin_workers=margin_workers,
            )
            answer.append((index, result))
            previous_point = point
            previous_result = result
        return answer

    workers = min(interleave, len(points))
    if workers == 1:
        chains = [fit_chain(0)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            chains = list(executor.map(fit_chain, range(workers)))
    ordered: list[SparseGroupedResult | None] = [None] * len(points)
    for chain in chains:
        for index, result in chain:
            ordered[index] = result
    if any(result is None for result in ordered):
        raise RuntimeError("an interleaved sparse chain produced no result")
    return [result for result in ordered if result is not None]


def _validated_margins(
    p_ya: np.ndarray,
    p_yb: np.ndarray,
    p_ab: np.ndarray,
    *,
    consistency_tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_ya = np.asarray(p_ya, dtype=np.float64)
    p_yb = np.asarray(p_yb, dtype=np.float64)
    p_ab = np.asarray(p_ab, dtype=np.float64)
    if p_ya.ndim != 2 or p_yb.ndim != 2 or p_ab.ndim != 2:
        raise ValueError("all three margins must be matrices")
    ny, na = p_ya.shape
    ny2, nb = p_yb.shape
    if ny2 != ny or p_ab.shape != (na, nb):
        raise ValueError(
            "shapes must be P_ya=(Y,A), P_yb=(Y,B), P_ab=(A,B)"
        )
    for name, p in (("P_ya", p_ya), ("P_yb", p_yb), ("P_ab", p_ab)):
        if not np.isfinite(p).all() or (p < 0.0).any():
            raise ValueError(f"{name} must be finite and nonnegative")
        if abs(float(p.sum()) - 1.0) > consistency_tol:
            raise ValueError(f"{name} must sum to one")

    errors = {
        "Y": np.abs(p_ya.sum(axis=1) - p_yb.sum(axis=1)).sum(),
        "A": np.abs(p_ya.sum(axis=0) - p_ab.sum(axis=1)).sum(),
        "B": np.abs(p_yb.sum(axis=0) - p_ab.sum(axis=0)).sum(),
    }
    bad = {name: float(err) for name, err in errors.items()
           if err > consistency_tol}
    if bad:
        raise ValueError(f"pair margins are inconsistent: {bad}")
    return p_ya, p_yb, p_ab


def _project_joint_to_margins(
    joint: np.ndarray,
    row_margin: np.ndarray,
    column_margin: np.ndarray,
    *,
    max_iterations: int,
    tolerance: float,
) -> np.ndarray:
    out = np.asarray(joint, dtype=np.float64).copy()
    if (out <= 0.0).any():
        raise ValueError("the joint being projected must be strictly positive")
    for _ in range(max_iterations):
        row_total = out.sum(axis=1)
        row_scale = np.divide(
            row_margin,
            row_total,
            out=np.zeros_like(row_margin),
            where=row_total > 0.0,
        )
        out *= row_scale[:, None]
        column_total = out.sum(axis=0)
        column_scale = np.divide(
            column_margin,
            column_total,
            out=np.zeros_like(column_margin),
            where=column_total > 0.0,
        )
        out *= column_scale[None, :]
        if max(
            np.abs(out.sum(axis=1) - row_margin).sum(),
            np.abs(out.sum(axis=0) - column_margin).sum(),
        ) < tolerance:
            return out
    raise RuntimeError("margin projection did not converge")


def restrict_margins_to_observed_contexts(
    p_ya: np.ndarray,
    p_yb: np.ndarray,
    p_ab: np.ndarray,
    observed_ab: np.ndarray,
    *,
    max_iterations: int = 10_000,
    tolerance: float = 1e-12,
    consistency_tolerance: float = 1e-10,
) -> RestrictedMargins:
    """Condition calibration on context pairs observed in the prefix.

    ``P_ab`` is restricted to the supplied support and renormalized.  This
    changes its A and B marginals.  The two target margins are consequently
    projected to those new context marginals while retaining their common Y
    marginal.  The projection is the minimum-relative-entropy adjustment of
    each positive pair joint.

    Unobserved context pairs are outside this calibrated model and must use a
    separately declared fallback predictor.
    """

    p_ya, p_yb, p_ab = _validated_margins(
        p_ya, p_yb, p_ab, consistency_tol=consistency_tolerance
    )
    observed = np.asarray(observed_ab, dtype=bool)
    if observed.shape != p_ab.shape:
        raise ValueError("observed_ab must have the shape of P_ab")
    retained = float(p_ab[observed].sum())
    if not 0.0 < retained <= 1.0:
        raise ValueError("observed context support has no probability mass")
    restricted_ab = np.where(observed, p_ab / retained, 0.0)
    margin_y = p_ya.sum(axis=1)
    restricted_ya = _project_joint_to_margins(
        p_ya,
        margin_y,
        restricted_ab.sum(axis=1),
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    restricted_yb = _project_joint_to_margins(
        p_yb,
        margin_y,
        restricted_ab.sum(axis=0),
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    return RestrictedMargins(
        p_ya=restricted_ya,
        p_yb=restricted_yb,
        p_ab=restricted_ab,
        retained_ab_mass=retained,
    )


def sparse_problem_from_dense(
    p_ya: np.ndarray,
    p_yb: np.ndarray,
    p_ab: np.ndarray,
    active_ya: np.ndarray,
    active_yb: np.ndarray,
) -> SparseGroupedProblem:
    """Convert a small dense problem to the matrix-free representation."""

    p_ya, p_yb, p_ab = _validated_margins(
        p_ya, p_yb, p_ab, consistency_tol=1e-10
    )
    active_ya = np.asarray(active_ya, dtype=bool)
    active_yb = np.asarray(active_yb, dtype=bool)
    ey, ea = np.nonzero(active_ya)
    fy, fb = np.nonzero(active_yb)
    edge_a, edge_b = np.nonzero(p_ab)
    return SparseGroupedProblem(
        vocabulary_size=p_ya.shape[0],
        edge_a=edge_a,
        edge_b=edge_b,
        edge_probability=p_ab[edge_a, edge_b],
        target_y=p_ya.sum(axis=1),
        active_ya_y=ey,
        active_ya_a=ea,
        target_ya=p_ya[ey, ea],
        active_yb_y=fy,
        active_yb_b=fb,
        target_yb=p_yb[fy, fb],
    )


def sparse_grouped_ipf(
    problem: SparseGroupedProblem,
    *,
    max_iterations: int = 10_000,
    tolerance: float = 1e-10,
    log_base_y: np.ndarray | None = None,
    correction_ya: np.ndarray | None = None,
    correction_yb: np.ndarray | None = None,
    solver: str = "ipf",
    anderson_history: int = 3,
    dense_fallback_mass: float = 0.99,
    margin_workers: int = 1,
) -> SparseGroupedResult:
    """Matrix-free grouped IPF over observed context-pair edges.

    This correctness implementation stores only the global target baseline,
    observed context edges, and active target--lag corrections.  It never
    constructs a ``Y*A``, ``Y*B`` or ``Y*A*B`` probability array.  Python
    dictionaries and loops keep the equations transparent; the production
    version will replace them with sorted arrays and vectorized joins.
    """

    v = int(problem.vocabulary_size)
    arrays = (
        problem.edge_a, problem.edge_b, problem.edge_probability,
        problem.target_y, problem.active_ya_y, problem.active_ya_a,
        problem.target_ya, problem.active_yb_y, problem.active_yb_b,
        problem.target_yb,
    )
    if any(not np.isfinite(np.asarray(x)).all() for x in arrays):
        raise ValueError("problem arrays must be finite")
    if abs(float(np.sum(problem.edge_probability)) - 1.0) > 1e-10:
        raise ValueError("edge probabilities must sum to one")
    if abs(float(np.sum(problem.target_y)) - 1.0) > 1e-10:
        raise ValueError("target_y must sum to one")
    if solver not in ("ipf", "anderson", "lbfgs"):
        raise ValueError("solver must be 'ipf', 'anderson' or 'lbfgs'")
    if anderson_history < 1:
        raise ValueError("anderson_history must be positive")
    if not 0.0 < dense_fallback_mass <= 1.0:
        raise ValueError("dense_fallback_mass must lie in (0, 1]")
    if margin_workers < 1:
        raise ValueError("margin_workers must be positive")

    lb = (np.log(np.maximum(problem.target_y, np.finfo(float).tiny))
          if log_base_y is None
          else np.array(log_base_y, dtype=np.float64, copy=True))
    c1 = (np.zeros(len(problem.target_ya)) if correction_ya is None
          else np.array(correction_ya, dtype=np.float64, copy=True))
    c2 = (np.zeros(len(problem.target_yb)) if correction_yb is None
          else np.array(correction_yb, dtype=np.float64, copy=True))
    if lb.shape != (v,) or c1.shape != problem.target_ya.shape \
            or c2.shape != problem.target_yb.shape:
        raise ValueError("warm-start factor shapes do not match the problem")

    rows1: list[dict[int, int]] = [{} for _ in range(v)]
    rows2: list[dict[int, int]] = [{} for _ in range(v)]
    for i, (y, a) in enumerate(zip(problem.active_ya_y,
                                    problem.active_ya_a)):
        rows1[int(a)][int(y)] = i
    for i, (y, b) in enumerate(zip(problem.active_yb_y,
                                    problem.active_yb_b)):
        rows2[int(b)][int(y)] = i
    pa = np.bincount(problem.edge_a, weights=problem.edge_probability,
                     minlength=v)
    pb = np.bincount(problem.edge_b, weights=problem.edge_probability,
                     minlength=v)
    target_active_a = np.bincount(
        problem.active_ya_a, weights=problem.target_ya, minlength=v
    )
    target_active_b = np.bincount(
        problem.active_yb_b, weights=problem.target_yb, minlength=v
    )
    tiny = np.finfo(np.float64).tiny

    # One-time incidence plan.  Each row is one target symbol in the union
    # of the two active correction rows for an observed context edge.
    union_edge = []
    union_y = []
    union_i1 = []
    union_i2 = []
    for edge, (a_, b_) in enumerate(zip(problem.edge_a, problem.edge_b)):
        r1 = rows1[int(a_)]
        r2 = rows2[int(b_)]
        for y in sorted(set(r1) | set(r2)):
            union_edge.append(edge)
            union_y.append(y)
            union_i1.append(r1.get(y, -1))
            union_i2.append(r2.get(y, -1))
    ue = np.asarray(union_edge, dtype=np.int64)
    uy = np.asarray(union_y, dtype=np.int64)
    ui1 = np.asarray(union_i1, dtype=np.int64)
    ui2 = np.asarray(union_i2, dtype=np.int64)
    edge_count = len(problem.edge_probability)
    edge_weight = np.asarray(problem.edge_probability)

    def margins():
        log_base = lb - logsumexp(lb)
        base = np.exp(log_base)
        if not len(ue):
            return (base.copy(), np.zeros_like(c1), np.zeros_like(c2),
                    np.zeros(edge_count))

        correction = np.zeros(len(ue))
        has1 = ui1 >= 0
        has2 = ui2 >= 0
        correction[has1] += c1[ui1[has1]]
        correction[has2] += c2[ui2[has2]]
        term = log_base[uy] + correction

        union_mass = np.bincount(
            ue, weights=base[uy], minlength=edge_count
        )
        edge_max = np.full(edge_count, -np.inf)
        np.maximum.at(edge_max, ue, term)
        scaled_sum = np.bincount(
            ue, weights=np.exp(term - edge_max[ue]), minlength=edge_count
        )
        log_corrected = edge_max + np.log(np.maximum(scaled_sum, tiny))
        with np.errstate(divide="ignore", invalid="ignore"):
            log_background = np.log1p(-np.minimum(union_mass, 1.0))
        log_z = np.logaddexp(log_background, log_corrected)

        corrected_probability = np.exp(term - log_z[ue])
        m1 = np.bincount(
            ui1[has1],
            weights=edge_weight[ue[has1]] * corrected_probability[has1],
            minlength=len(c1),
        )
        m2 = np.bincount(
            ui2[has2],
            weights=edge_weight[ue[has2]] * corrected_probability[has2],
            minlength=len(c2),
        )

        # Stable dense fallback only where the active union contains almost
        # all baseline mass.  Such edges are common only in tiny diagnostics.
        # The cutoff exposes the tradeoff between the exact dense evaluation
        # of nearly covered rows and the faster analytic background identity.
        fallback = (union_mass > dense_fallback_mass) | (log_z < -500.0)
        analytic = ~fallback
        background_scale = float(np.sum(
            edge_weight[analytic] * np.exp(-log_z[analytic])
        ))
        pos_analytic = analytic[ue]
        uncorrected = np.exp(log_base[uy[pos_analytic]]
                             - log_z[ue[pos_analytic]])
        delta = edge_weight[ue[pos_analytic]] * (
            corrected_probability[pos_analytic] - uncorrected
        )
        my = base * background_scale + np.bincount(
            uy[pos_analytic], weights=delta, minlength=v
        )
        fallback_edges = np.flatnonzero(fallback)
        # Bound the temporary dense block.  This path is for edges whose
        # sparse union is already essentially the whole alphabet.
        edges_per_block = max(1, 1_000_000 // max(1, v))
        blocks = [
            fallback_edges[start:start + edges_per_block]
            for start in range(0, len(fallback_edges), edges_per_block)
        ]

        def dense_contribution(selected):
            score = np.broadcast_to(
                log_base, (len(selected), v)
            ).copy()
            lo_edge = int(selected[0])
            hi_edge = int(selected[-1]) + 1
            positions = (ue >= lo_edge) & (ue < hi_edge) & fallback[ue]
            local_edge = np.searchsorted(selected, ue[positions])
            np.add.at(
                score,
                (local_edge, uy[positions]),
                correction[positions],
            )
            probability = np.exp(
                score - logsumexp(score, axis=1, keepdims=True)
            )
            return edge_weight[selected] @ probability

        if margin_workers == 1 or len(blocks) < 2:
            contributions = map(dense_contribution, blocks)
            for contribution in contributions:
                my += contribution
        else:
            with ThreadPoolExecutor(
                max_workers=min(margin_workers, len(blocks))
            ) as executor:
                for contribution in executor.map(dense_contribution, blocks):
                    my += contribution
        return my, m1, m2, log_z

    def update(correction, target, current, state_ids, state_mass,
               target_active):
        correction += np.log(np.maximum(target, tiny)) - np.log(
            np.maximum(current, tiny)
        )
        current_active = np.bincount(
            state_ids, weights=current, minlength=v
        )
        wanted_inactive = state_mass - target_active
        current_inactive = state_mass - current_active
        shift = np.zeros(v)
        has_inactive = wanted_inactive > tiny
        shift[has_inactive] = (
            np.log(wanted_inactive[has_inactive])
            - np.log(np.maximum(current_inactive[has_inactive], tiny))
        )
        correction -= shift[state_ids]
        # With no inactive class the whole row is explicit.  Fix its harmless
        # row-constant gauge in one segmented maximum.
        full_states = ~has_inactive
        if np.any(full_states[state_ids]):
            row_max = np.full(v, -np.inf)
            np.maximum.at(row_max, state_ids, correction)
            select = full_states[state_ids]
            correction[select] -= row_max[state_ids[select]]

    grouped_ya = grouped_yb = residual_y = float("inf")
    previous_x = None
    previous_f = None
    delta_x: list[np.ndarray] = []
    delta_f: list[np.ndarray] = []
    n_base = len(lb)
    n_first = n_base + len(c1)

    def diagnostics(my, m1, m2):
        residual_y_ = float(np.abs(my - problem.target_y).sum())
        grouped_ya_ = float(np.abs(m1 - problem.target_ya).sum())
        grouped_yb_ = float(np.abs(m2 - problem.target_yb).sum())
        current_active_a = np.bincount(
            problem.active_ya_a, weights=m1, minlength=v
        )
        current_active_b = np.bincount(
            problem.active_yb_b, weights=m2, minlength=v
        )
        grouped_ya_ += float(np.abs(
            (pa - current_active_a) - (pa - target_active_a)
        ).sum())
        grouped_yb_ += float(np.abs(
            (pb - current_active_b) - (pb - target_active_b)
        ).sum())
        return residual_y_, grouped_ya_, grouped_yb_

    if solver == "lbfgs":
        initial = np.concatenate([lb, c1, c2])
        best_parameters = initial.copy()
        best_certificate = float("inf")

        def objective_gradient(parameters):
            nonlocal best_parameters, best_certificate
            lb[:] = parameters[:n_base]
            c1[:] = parameters[n_base:n_first]
            c2[:] = parameters[n_first:]
            my, m1, m2, log_z = margins()
            log_base = lb - logsumexp(lb)
            objective = (
                float(edge_weight @ log_z)
                - float(problem.target_y @ log_base)
                - float(problem.target_ya @ c1)
                - float(problem.target_yb @ c2)
            )
            gradient = np.concatenate([
                my - problem.target_y,
                m1 - problem.target_ya,
                m2 - problem.target_yb,
            ])
            certificate = max(diagnostics(my, m1, m2))
            if certificate < best_certificate:
                best_certificate = certificate
                best_parameters = parameters.copy()
            return objective, gradient

        lbfgs_iterations = min(max_iterations, 1_000)
        optimized = minimize(
            objective_gradient,
            initial,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": lbfgs_iterations,
                # scipy stops on the largest gradient component whereas our
                # certificate is an L1 margin residual over all components.
                "gtol": tolerance / max(10, len(initial)),
                "ftol": 0.0,
                "maxls": 40,
                "maxfun": lbfgs_iterations * 20,
            },
        )
        # The dual objective may still decrease along nearly flat gauge
        # directions after the actual margin certificate has worsened.  The
        # certificate, not scipy's final iterate, chooses the handoff to IPF.
        parameters = best_parameters
        lb[:] = parameters[:n_base]
        c1[:] = parameters[n_base:n_first]
        c2[:] = parameters[n_first:]
        my, m1, m2, _ = margins()
        residual_y, grouped_ya, grouped_yb = diagnostics(my, m1, m2)
        converged = max(residual_y, grouped_ya, grouped_yb) < tolerance
        if not converged:
            polished = sparse_grouped_ipf(
                problem,
                max_iterations=max_iterations,
                tolerance=tolerance,
                log_base_y=lb,
                correction_ya=c1,
                correction_yb=c2,
                solver="ipf",
                dense_fallback_mass=dense_fallback_mass,
                margin_workers=margin_workers,
            )
            return SparseGroupedResult(
                polished.log_base_y,
                polished.correction_ya,
                polished.correction_yb,
                int(optimized.nit) + polished.iterations,
                polished.grouped_residual_ya_l1,
                polished.grouped_residual_yb_l1,
                polished.residual_y_l1,
                polished.converged,
            )
        return SparseGroupedResult(
            lb.copy(), c1.copy(), c2.copy(), int(optimized.nit),
            grouped_ya, grouped_yb, residual_y, converged
        )

    # Alternating projections decrease their own divergence objective, but the
    # user-facing certificate (the largest of three L1 margin errors) need not
    # be monotone.  Preserve the best certified iterate, including the warm
    # start supplied by a preceding L-BFGS phase.
    my, m1, m2, _ = margins()
    residual_y, grouped_ya, grouped_yb = diagnostics(my, m1, m2)
    iteration_best_certificate = max(residual_y, grouped_ya, grouped_yb)
    iteration_best_state = (lb.copy(), c1.copy(), c2.copy())
    iteration_best_residuals = (residual_y, grouped_ya, grouped_yb)

    for iteration in range(1, max_iterations + 1):
        x_before = np.concatenate([lb, c1, c2])
        my, _, _, _ = margins()
        lb += np.log(np.maximum(problem.target_y, tiny)) - np.log(
            np.maximum(my, tiny)
        )
        lb -= lb.max()
        _, m1, _, _ = margins()
        update(c1, problem.target_ya, m1, problem.active_ya_a,
               pa, target_active_a)
        _, _, m2, _ = margins()
        update(c2, problem.target_yb, m2, problem.active_yb_b,
               pb, target_active_b)
        my, m1, m2, _ = margins()
        residual_y, grouped_ya, grouped_yb = diagnostics(my, m1, m2)
        certificate = max(residual_y, grouped_ya, grouped_yb)
        if certificate < iteration_best_certificate:
            iteration_best_certificate = certificate
            iteration_best_state = (lb.copy(), c1.copy(), c2.copy())
            iteration_best_residuals = (residual_y, grouped_ya, grouped_yb)
        if certificate < tolerance:
            return SparseGroupedResult(
                lb, c1, c2, iteration, grouped_ya, grouped_yb,
                residual_y, True
            )
        if solver == "anderson":
            x_mapped = np.concatenate([lb, c1, c2])
            fixed_point_residual = x_mapped - x_before
            if previous_x is not None:
                delta_x.append(x_before - previous_x)
                delta_f.append(fixed_point_residual - previous_f)
                if len(delta_x) > anderson_history:
                    delta_x.pop(0)
                    delta_f.pop(0)
            previous_x = x_before
            previous_f = fixed_point_residual
            if delta_x:
                dx = np.vstack(delta_x)
                df = np.vstack(delta_f)
                gram = df @ df.T
                gram[np.diag_indices_from(gram)] += (
                    1e-8 * max(1.0, float(gram.max()))
                )
                try:
                    gamma = np.linalg.solve(
                        gram, df @ fixed_point_residual
                    )
                    candidate = x_mapped - (dx + df).T @ gamma
                    if np.isfinite(candidate).all():
                        lb[:] = candidate[:n_base]
                        c1[:] = candidate[n_base:n_first]
                        c2[:] = candidate[n_first:]
                except np.linalg.LinAlgError:
                    delta_x.clear()
                    delta_f.clear()
                    previous_x = previous_f = None
    best_lb, best_c1, best_c2 = iteration_best_state
    best_y, best_ya, best_yb = iteration_best_residuals
    return SparseGroupedResult(
        best_lb, best_c1, best_c2, max_iterations, best_ya, best_yb,
        best_y, False
    )


def conditional_ipf(
    p_ya: np.ndarray,
    p_yb: np.ndarray,
    p_ab: np.ndarray,
    *,
    max_iterations: int = 10_000,
    tolerance: float = 1e-11,
    consistency_tolerance: float = 1e-10,
    log_f_ya: np.ndarray | None = None,
    log_f_yb: np.ndarray | None = None,
) -> ConditionalIPFResult:
    """Fit the maximum-entropy triangle while fixing ``P_ab`` exactly.

    The returned conditional is

    ``r(y|a,b) proportional to exp(log_f_ya[y,a] + log_f_yb[y,b])``.

    Alternating multiplicative corrections make its ``(Y,A)`` and ``(Y,B)``
    margins match the requested margins.  ``P_ab`` is exact after every
    update because each conditional distribution is normalized over ``Y``.

    This dense implementation is intended as a correctness reference.  Its
    working joint is cubic in the alphabet sizes.
    """

    p_ya, p_yb, p_ab = _validated_margins(
        p_ya, p_yb, p_ab, consistency_tol=consistency_tolerance
    )
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    lf_ya = (np.zeros_like(p_ya) if log_f_ya is None
             else np.array(log_f_ya, dtype=np.float64, copy=True))
    lf_yb = (np.zeros_like(p_yb) if log_f_yb is None
             else np.array(log_f_yb, dtype=np.float64, copy=True))
    if lf_ya.shape != p_ya.shape or lf_yb.shape != p_yb.shape:
        raise ValueError("initial factor shapes do not match the margins")
    if not np.isfinite(lf_ya).all() or not np.isfinite(lf_yb).all():
        raise ValueError("initial log factors must be finite")

    tiny = np.finfo(np.float64).tiny

    def joint() -> np.ndarray:
        score = lf_ya[:, :, None] + lf_yb[:, None, :]
        cond = np.exp(score - logsumexp(score, axis=0, keepdims=True))
        return cond * p_ab[None, :, :]

    residual_ya = residual_yb = residual_ab = float("inf")
    for iteration in range(1, max_iterations + 1):
        q = joint()
        m_ya = q.sum(axis=2)
        lf_ya += np.log(np.maximum(p_ya, tiny)) - np.log(
            np.maximum(m_ya, tiny)
        )
        # Remove a harmless global gauge before the next exponential.
        lf_ya -= lf_ya.max()

        q = joint()
        m_yb = q.sum(axis=1)
        lf_yb += np.log(np.maximum(p_yb, tiny)) - np.log(
            np.maximum(m_yb, tiny)
        )
        lf_yb -= lf_yb.max()

        q = joint()
        residual_ya = float(np.abs(q.sum(axis=2) - p_ya).sum())
        residual_yb = float(np.abs(q.sum(axis=1) - p_yb).sum())
        residual_ab = float(np.abs(q.sum(axis=0) - p_ab).sum())
        if max(residual_ya, residual_yb, residual_ab) < tolerance:
            return ConditionalIPFResult(
                log_f_ya=lf_ya,
                log_f_yb=lf_yb,
                iterations=iteration,
                residual_ya_l1=residual_ya,
                residual_yb_l1=residual_yb,
                residual_ab_l1=residual_ab,
                converged=True,
            )

    return ConditionalIPFResult(
        log_f_ya=lf_ya,
        log_f_yb=lf_yb,
        iterations=max_iterations,
        residual_ya_l1=residual_ya,
        residual_yb_l1=residual_yb,
        residual_ab_l1=residual_ab,
        converged=False,
    )


def grouped_conditional_ipf(
    p_ya: np.ndarray,
    p_yb: np.ndarray,
    p_ab: np.ndarray,
    active_ya: np.ndarray,
    active_yb: np.ndarray,
    *,
    max_iterations: int = 10_000,
    tolerance: float = 1e-11,
    consistency_tolerance: float = 1e-10,
    log_base_y: np.ndarray | None = None,
    log_f_ya: np.ndarray | None = None,
    log_f_yb: np.ndarray | None = None,
) -> GroupedConditionalIPFResult:
    """Fit an exact background-plus-corrections maximum-entropy model.

    Active ``(Y,A)`` and ``(Y,B)`` cells are matched individually.  For each
    context row, all inactive target cells form one feature and only their
    aggregate probability is constrained.  The factors therefore contain one
    default value per row plus explicit active-cell corrections, exactly the
    representation required by a tables-free large-alphabet implementation.

    This is the exact maximum-entropy solution for the explicitly coarsened
    constraints.  It need not reproduce every inactive cell of the uncoarsened
    triangle separately.
    """

    p_ya, p_yb, p_ab = _validated_margins(
        p_ya, p_yb, p_ab, consistency_tol=consistency_tolerance
    )
    active_ya = np.asarray(active_ya, dtype=bool)
    active_yb = np.asarray(active_yb, dtype=bool)
    if active_ya.shape != p_ya.shape or active_yb.shape != p_yb.shape:
        raise ValueError("active masks must match their pair margins")

    lb_y = (np.zeros(p_ya.shape[0]) if log_base_y is None
            else np.array(log_base_y, dtype=np.float64, copy=True))
    lf_ya = (np.zeros_like(p_ya) if log_f_ya is None
             else np.array(log_f_ya, dtype=np.float64, copy=True))
    lf_yb = (np.zeros_like(p_yb) if log_f_yb is None
             else np.array(log_f_yb, dtype=np.float64, copy=True))
    if (lb_y.shape != (p_ya.shape[0],)
            or lf_ya.shape != p_ya.shape or lf_yb.shape != p_yb.shape):
        raise ValueError("initial factor shapes do not match the margins")
    if (not np.isfinite(lb_y).all() or not np.isfinite(lf_ya).all()
            or not np.isfinite(lf_yb).all()):
        raise ValueError("initial log factors must be finite")
    tiny = np.finfo(np.float64).tiny

    def joint() -> np.ndarray:
        score = lb_y[:, None, None] + lf_ya[:, :, None] + lf_yb[:, None, :]
        cond = np.exp(score - logsumexp(score, axis=0, keepdims=True))
        return cond * p_ab[None, :, :]

    def grouped_update(log_f, target, current, active):
        log_f[active] += (
            np.log(np.maximum(target[active], tiny))
            - np.log(np.maximum(current[active], tiny))
        )
        for state in range(target.shape[1]):
            inactive = ~active[:, state]
            if inactive.any():
                wanted = float(target[inactive, state].sum())
                got = float(current[inactive, state].sum())
                log_f[inactive, state] += (
                    np.log(max(wanted, tiny)) - np.log(max(got, tiny))
                )
        log_f -= log_f.max()

    def grouped_residual(target, current, active):
        residual = float(np.abs(target[active] - current[active]).sum())
        for state in range(target.shape[1]):
            inactive = ~active[:, state]
            if inactive.any():
                residual += abs(
                    float(target[inactive, state].sum())
                    - float(current[inactive, state].sum())
                )
        return residual

    target_y = p_ya.sum(axis=1)
    full_ya = full_yb = residual_ab = residual_y = float("inf")
    grouped_ya = grouped_yb = float("inf")
    for iteration in range(1, max_iterations + 1):
        q = joint()
        current_y = q.sum(axis=(1, 2))
        lb_y += np.log(np.maximum(target_y, tiny)) - np.log(
            np.maximum(current_y, tiny)
        )
        lb_y -= lb_y.max()
        q = joint()
        grouped_update(lf_ya, p_ya, q.sum(axis=2), active_ya)
        q = joint()
        grouped_update(lf_yb, p_yb, q.sum(axis=1), active_yb)
        q = joint()
        m_ya = q.sum(axis=2)
        m_yb = q.sum(axis=1)
        m_ab = q.sum(axis=0)
        residual_y = float(np.abs(q.sum(axis=(1, 2)) - target_y).sum())
        grouped_ya = grouped_residual(p_ya, m_ya, active_ya)
        grouped_yb = grouped_residual(p_yb, m_yb, active_yb)
        full_ya = float(np.abs(m_ya - p_ya).sum())
        full_yb = float(np.abs(m_yb - p_yb).sum())
        residual_ab = float(np.abs(m_ab - p_ab).sum())
        if max(grouped_ya, grouped_yb, residual_ab, residual_y) < tolerance:
            return GroupedConditionalIPFResult(
                log_f_ya=lf_ya,
                log_f_yb=lf_yb,
                iterations=iteration,
                residual_ya_l1=full_ya,
                residual_yb_l1=full_yb,
                residual_ab_l1=residual_ab,
                converged=True,
                log_base_y=lb_y,
                grouped_residual_ya_l1=grouped_ya,
                grouped_residual_yb_l1=grouped_yb,
                residual_y_l1=residual_y,
            )

    return GroupedConditionalIPFResult(
        log_f_ya=lf_ya,
        log_f_yb=lf_yb,
        iterations=max_iterations,
        residual_ya_l1=full_ya,
        residual_yb_l1=full_yb,
        residual_ab_l1=residual_ab,
        converged=False,
        log_base_y=lb_y,
        grouped_residual_ya_l1=grouped_ya,
        grouped_residual_yb_l1=grouped_yb,
        residual_y_l1=residual_y,
    )


def fit_grouped_checkpoints(
    checkpoints: Sequence[GroupedCheckpoint],
    *,
    interleave: int = 1,
    max_iterations: int = 10_000,
    tolerance: float = 1e-11,
    consistency_tolerance: float = 1e-10,
) -> list[GroupedConditionalIPFResult]:
    """Fit warm-started checkpoint chains, optionally interleaved.

    With ``interleave=k``, worker ``j`` fits checkpoints
    ``j, j+k, j+2k, ...`` sequentially and warm-starts each from its
    predecessor in that chain.  The independent chains run concurrently;
    results are returned in chronological checkpoint order.
    """

    if interleave < 1:
        raise ValueError("interleave must be positive")
    points = list(checkpoints)
    if not points:
        return []
    results: list[GroupedConditionalIPFResult | None] = [None] * len(points)

    def run_chain(offset: int) -> None:
        previous = None
        for index in range(offset, len(points), interleave):
            point = points[index]
            fitted = grouped_conditional_ipf(
                point.p_ya,
                point.p_yb,
                point.p_ab,
                point.active_ya,
                point.active_yb,
                max_iterations=max_iterations,
                tolerance=tolerance,
                consistency_tolerance=consistency_tolerance,
                log_base_y=(None if previous is None
                            else previous.log_base_y),
                log_f_ya=(None if previous is None
                          else previous.log_f_ya),
                log_f_yb=(None if previous is None
                          else previous.log_f_yb),
            )
            results[index] = fitted
            previous = fitted

    workers = min(interleave, len(points))
    if workers == 1:
        run_chain(0)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run_chain, range(workers)))
    if any(result is None for result in results):
        raise RuntimeError("an interleaved calibration chain produced no result")
    return [result for result in results if result is not None]
