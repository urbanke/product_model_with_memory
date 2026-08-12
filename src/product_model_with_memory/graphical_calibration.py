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

import json
import multiprocessing as mp
import traceback
from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from threading import Lock
from time import perf_counter

import numpy as np
from scipy.optimize import line_search, linprog, minimize
from scipy.sparse import coo_matrix, csr_matrix
from scipy.special import logsumexp

try:
    from product_model_with_memory import _graphical_margin_c
except ImportError:  # Source checkout before the optional extension is built.
    _graphical_margin_c = None


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


def sparse_layered_pair(
    context_marginal: np.ndarray,
    unseen_probability: np.ndarray,
    active_y: np.ndarray,
    active_context: np.ndarray,
    active_probability: np.ndarray,
) -> SparseProjectedPair:
    """Represent a layered conditional estimate as its unmodified joint law.

    Unlike :func:`project_sparse_layered_pair`, this performs no Sinkhorn
    scaling.  It merely multiplies every conditional row by the supplied
    context marginal and retains the outer-background-plus-corrections form.
    This is the appropriate input to the statistically relaxed calibration:
    finite-data pair estimates are allowed to be mutually incompatible.
    """

    context_marginal = np.asarray(context_marginal, dtype=np.float64)
    unseen_probability = np.asarray(unseen_probability, dtype=np.float64)
    active_y = np.asarray(active_y, dtype=np.int64)
    active_context = np.asarray(active_context, dtype=np.int64)
    active_probability = np.asarray(active_probability, dtype=np.float64)
    v = len(context_marginal)
    if unseen_probability.shape != (v,):
        raise ValueError("layered background must have length V")
    if not (active_y.shape == active_context.shape == active_probability.shape):
        raise ValueError("active pair arrays must have identical shapes")
    if ((active_y < 0) | (active_y >= v) | (active_context < 0)
            | (active_context >= v)).any():
        raise ValueError("active pair indices are outside the vocabulary")
    if not np.isclose(float(context_marginal.sum()), 1.0):
        raise ValueError("context marginal must sum to one")
    return SparseProjectedPair(
        vocabulary_size=v,
        left=np.ones(v),
        right=np.ones(v),
        background=context_marginal * unseen_probability,
        active_y=active_y,
        active_context=active_context,
        delta=context_marginal[active_context] * (
            active_probability - unseen_probability[active_context]
        ),
    )


def sparse_relaxed_problem_from_layered_pairs(
    p_ya: SparseProjectedPair,
    p_yb: SparseProjectedPair,
    observed_a: np.ndarray,
    observed_b: np.ndarray,
    target_y: np.ndarray,
) -> tuple[SparseGroupedProblem, float]:
    """Build the relaxed problem without forcing pair compatibility.

    ``P_AB`` is the stationary, one-step pair law ``P_YA`` restricted to the
    observed context support.  ``P_Y`` and this context law remain hard.  The
    unmodified active coordinates of ``P_YA`` and ``P_YB`` are the soft
    finite-data targets handled by the pair-slack penalty during fitting.
    """

    if p_ya.vocabulary_size != p_yb.vocabulary_size:
        raise ValueError("the two sparse pairs must use one vocabulary")
    v = p_ya.vocabulary_size
    edge_a = np.asarray(observed_a, dtype=np.int64)
    edge_b = np.asarray(observed_b, dtype=np.int64)
    target_y = np.asarray(target_y, dtype=np.float64)
    if edge_a.shape != edge_b.shape or not len(edge_a):
        raise ValueError("observed context edge arrays must match and be nonempty")
    if target_y.shape != (v,) or not np.isclose(float(target_y.sum()), 1.0):
        raise ValueError("target_y must be a probability vector of length V")
    # By stationarity, (A, B) = (X_{t-1}, X_{t-2}) has the same
    # orientation as (Y, A) = (X_t, X_{t-1}): P_AB(a, b) = P_YA(a, b).
    # `values` takes (target, context), hence target=a and context=b.
    ab = p_ya.values(edge_a, edge_b)
    retained = float(ab.sum())
    if not 0.0 < retained <= 1.0 + 1e-10:
        raise ValueError("observed context support has invalid probability mass")
    return SparseGroupedProblem(
        vocabulary_size=v,
        edge_a=edge_a,
        edge_b=edge_b,
        edge_probability=ab / retained,
        target_y=target_y,
        active_ya_y=p_ya.active_y,
        active_ya_a=p_ya.active_context,
        target_ya=p_ya.active_values(),
        active_yb_y=p_yb.active_y,
        active_yb_b=p_yb.active_context,
        target_yb=p_yb.active_values(),
    ), retained


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
    final_residual = max(
        float(np.abs(final_context - desired_context).sum()),
        float(np.abs(final_target - target_marginal).sum()),
    )
    if final_residual >= tolerance * 10:
        raise RuntimeError(
            "sparse pair-margin projection did not converge: "
            f"residual={final_residual:.6g}, tolerance={tolerance:.6g}, "
            f"iterations={max_iterations}"
        )
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
    margin_evaluations: int = 0
    hessian_products: int = 0
    stationarity: float = float("nan")
    objective: float = float("nan")
    final_stationarity: float = float("nan")
    final_objective: float = float("nan")


@dataclass(frozen=True)
class SparseGroupedCheckpoint:
    """One sparse calibration problem and its layered unigram baseline.

    The ``projected_*`` field names are retained for state compatibility.
    They may contain the unprojected layered pair laws used by relaxed fits.
    """

    problem: SparseGroupedProblem
    log_base_y: np.ndarray
    projected_ya: SparseProjectedPair | None = None
    projected_yb: SparseProjectedPair | None = None


@dataclass(frozen=True)
class GroupedFeasibilityResult:
    """Small-problem linear feasibility certificate."""

    feasible: bool
    status: int
    message: str
    variables: int
    equality_constraints: int
    max_equality_residual: float


@dataclass(frozen=True)
class SparseFactorizedMargins:
    """Sufficient-statistic margins from intersection-only factor algebra."""

    target_y: np.ndarray
    active_ya: np.ndarray
    active_yb: np.ndarray
    log_normalizer: np.ndarray


@dataclass(frozen=True)
class SparseReferenceMargins:
    """SVRG block reference with only context-touched corrections."""

    target_y: np.ndarray
    ya_position: np.ndarray
    active_ya: np.ndarray
    yb_position: np.ndarray
    active_yb: np.ndarray


@dataclass(frozen=True)
class SparseDualEvaluation:
    """Conditional-dual value and gradient for one edge distribution."""

    objective: float
    gradient_y: np.ndarray
    gradient_ya: np.ndarray
    gradient_yb: np.ndarray
    certificate: float | None = None
    residual_y_l1: float | None = None
    residual_ya_l1: float | None = None
    residual_yb_l1: float | None = None

    def gradient(self) -> np.ndarray:
        return np.concatenate([
            self.gradient_y, self.gradient_ya, self.gradient_yb,
        ])


def empirical_pair_slack_variances(
    problem: SparseGroupedProblem,
    sample_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return universal plug-in sampling variances for active pair masses.

    Each active target is a joint-event probability estimated from the
    available prefix.  The binomial plug-in variance is ``p(1-p)/N``.  A
    one-count floor, ``1/N**2``, prevents an observed rare event from being
    treated as known exactly.  This is deliberately corpus-independent and
    automatically tightens with the amount of past data.

    The layered estimator is smoother than an empirical count, so these are
    a first conservative uncertainty model rather than a claim that its
    coordinates are independent.  A later covariance-aware version can be
    substituted without changing the relaxed solver.
    """

    if sample_size < 1:
        raise ValueError("sample size must be positive")
    inverse_n = 1.0 / float(sample_size)

    def variance(probability: np.ndarray) -> np.ndarray:
        values = np.asarray(probability, dtype=np.float64)
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("pair probabilities must lie in [0, 1]")
        return np.maximum(
            values * (1.0 - values) * inverse_n,
            inverse_n * inverse_n,
        )

    return variance(problem.target_ya), variance(problem.target_yb)


@dataclass(frozen=True)
class SparseStochasticResult:
    """Factors produced by a stochastic approach phase."""

    log_base_y: np.ndarray
    correction_ya: np.ndarray
    correction_yb: np.ndarray
    steps: int
    sampled_edges: int
    exact_evaluations: int
    best_exact_objective: float
    best_exact_certificate: float
    trace: tuple[dict[str, float | int], ...]
    stop_reason: str = "safety_limit"
    exact_seconds: float = 0.0
    sampled_gradient_seconds: float = 0.0
    optimizer_seconds: float = 0.0
    reference_cache_seconds: float = 0.0
    intersection_plan_bytes: int = 0
    reference_cache_bytes: int = 0
    sampled_phase_timing: dict[str, float] = field(default_factory=dict)
    best_exact_stationarity: float = float("inf")


@dataclass(frozen=True)
class SparseLineSearchResult:
    """Factors produced by exact spectral gradient descent with line search."""

    log_base_y: np.ndarray
    correction_ya: np.ndarray
    correction_yb: np.ndarray
    iterations: int
    evaluations: int
    objective: float
    certificate: float
    converged: bool
    trace: tuple[dict[str, float | int | bool], ...]
    final_log_base_y: np.ndarray
    final_correction_ya: np.ndarray
    final_correction_yb: np.ndarray
    final_objective: float
    final_certificate: float
    stationarity: float = float("nan")
    final_stationarity: float = float("nan")


@dataclass(frozen=True)
class SparseIntersectionPlan:
    """Compact correction intersections indexed by retained AB edge."""

    edge: np.ndarray
    target_y: np.ndarray
    correction_ya: np.ndarray
    correction_yb: np.ndarray
    edge_offset: int = 0


@dataclass(frozen=True)
class SparseEdgeBlock:
    """Contiguous zero-copy AB-edge and intersection-plan view."""

    probability_mass: float
    problem: SparseGroupedProblem
    intersection_plan: SparseIntersectionPlan


@dataclass(frozen=True)
class LayeredIntersectionGraph:
    """One CSR support-triangle layer per checkpoint birth depth.

    Rows are global YA-correction indices.  Within a row, each graph edge
    stores only its YB-correction index and AB-edge index; the YA index and
    target symbol are implicit in the row and ``problem.active_ya_y``.
    """

    row_ptr: tuple[np.ndarray, ...]
    correction_yb: tuple[np.ndarray, ...]
    edge_ab: tuple[np.ndarray, ...]

    @property
    def layers(self) -> int:
        return len(self.row_ptr)

    @property
    def edges(self) -> int:
        return sum(len(values) for values in self.edge_ab)

    @property
    def nbytes(self) -> int:
        return sum(
            array.nbytes
            for family in (self.row_ptr, self.correction_yb, self.edge_ab)
            for array in family
        )


@dataclass(frozen=True)
class ABMajorIntersectionGraph:
    """Shared triangles contiguous by AB edge, with explicit birth depth."""

    edge_ptr: np.ndarray
    correction_ya: np.ndarray
    correction_yb: np.ndarray
    birth: np.ndarray

    @property
    def edges(self) -> int:
        return len(self.correction_ya)

    @property
    def nbytes(self) -> int:
        return sum(array.nbytes for array in (
            self.edge_ptr, self.correction_ya, self.correction_yb, self.birth
        ))


@dataclass(frozen=True)
class SparseIntersectionDelta:
    """One immutable triangle-birth chunk in two sparse CSR orderings.

    ``ya_row`` contains only nonempty global YA rows.  ``ya_ptr`` indexes the
    parallel YB- and AB-index payloads.  The AB-major arrays use the same
    convention for global retained-context rows.  Consequently an old delta
    never needs extending when later checkpoints append support edges.
    """

    checkpoint: int
    ya_row: np.ndarray
    ya_ptr: np.ndarray
    ya_correction_yb: np.ndarray
    ya_edge_ab: np.ndarray
    ab_row: np.ndarray
    ab_ptr: np.ndarray
    ab_correction_ya: np.ndarray
    ab_correction_yb: np.ndarray

    @property
    def triangles(self) -> int:
        return len(self.ya_edge_ab)

    @property
    def nbytes(self) -> int:
        return sum(array.nbytes for array in (
            self.ya_row, self.ya_ptr, self.ya_correction_yb,
            self.ya_edge_ab, self.ab_row, self.ab_ptr,
            self.ab_correction_ya, self.ab_correction_yb,
        ))


def _sparse_csr_order(
    major: np.ndarray,
    secondary: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return stable order, represented major rows, and their CSR pointer."""

    major = np.asarray(major, dtype=np.int32)
    order = (
        np.argsort(major, kind="stable")
        if secondary is None
        else np.lexsort((np.asarray(secondary), major))
    )
    sorted_major = major[order]
    rows, counts = np.unique(sorted_major, return_counts=True)
    ptr = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    return order, np.asarray(rows, dtype=np.int32), ptr


def build_sparse_intersection_delta(
    problem: SparseGroupedProblem,
    birth_ya: np.ndarray,
    birth_yb: np.ndarray,
    birth_ab: np.ndarray,
    checkpoint: int,
) -> SparseIntersectionDelta:
    """Build triangles born at one checkpoint using stable global edge IDs.

    ``problem`` must be ordered by append-only global support ID, and the
    three birth arrays must use those same IDs.  The initial reference builder
    deliberately uses the explicit intersection plan; a direct native delta
    builder can replace it without changing the persisted contract.
    """

    if checkpoint < 0:
        raise ValueError("checkpoint must be nonnegative")
    first_birth = np.asarray(birth_ya, dtype=np.int64)
    second_birth = np.asarray(birth_yb, dtype=np.int64)
    context_birth = np.asarray(birth_ab, dtype=np.int64)
    if (
        first_birth.shape != problem.target_ya.shape
        or second_birth.shape != problem.target_yb.shape
        or context_birth.shape != problem.edge_probability.shape
    ):
        raise ValueError("pair-edge births and sparse problem disagree")
    plan = build_sparse_intersection_plan(problem)
    triangle_birth = np.maximum.reduce((
        first_birth[plan.correction_ya],
        second_birth[plan.correction_yb],
        context_birth[plan.edge],
    ))
    selected = np.flatnonzero(triangle_birth == checkpoint)
    first = np.asarray(plan.correction_ya[selected], dtype=np.int32)
    second = np.asarray(plan.correction_yb[selected], dtype=np.int32)
    edge = np.asarray(plan.edge[selected], dtype=np.int32)

    ya_order, ya_row, ya_ptr = _sparse_csr_order(first, edge)
    ab_order, ab_row, ab_ptr = _sparse_csr_order(
        edge, problem.active_ya_y[first]
    )
    return SparseIntersectionDelta(
        checkpoint=checkpoint,
        ya_row=ya_row,
        ya_ptr=ya_ptr,
        ya_correction_yb=second[ya_order],
        ya_edge_ab=edge[ya_order],
        ab_row=ab_row,
        ab_ptr=ab_ptr,
        ab_correction_ya=first[ab_order],
        ab_correction_yb=second[ab_order],
    )


def build_sparse_intersection_delta_incremental(
    problem: SparseGroupedProblem,
    birth_ya: np.ndarray,
    birth_yb: np.ndarray,
    birth_ab: np.ndarray,
    checkpoint: int,
) -> SparseIntersectionDelta:
    """Generate only triangles newly enabled at ``checkpoint``.

    The three loops are deliberately disjoint.  A triangle is owned first by
    a new YA edge, otherwise by a new YB edge, and otherwise by a new AB edge.
    No triangle born earlier is revisited.  This Python implementation is the
    specification for the production native builder.
    """

    if checkpoint < 0:
        raise ValueError("checkpoint must be nonnegative")
    first_birth = np.asarray(birth_ya, dtype=np.int64)
    second_birth = np.asarray(birth_yb, dtype=np.int64)
    context_birth = np.asarray(birth_ab, dtype=np.int64)
    if (
        first_birth.shape != problem.target_ya.shape
        or second_birth.shape != problem.target_yb.shape
        or context_birth.shape != problem.edge_probability.shape
    ):
        raise ValueError("pair-edge births and sparse problem disagree")
    if any(np.any(values > checkpoint) for values in (
        first_birth, second_birth, context_birth
    )):
        raise ValueError("incremental support contains future edge births")

    v = problem.vocabulary_size
    ab_lookup = {
        int(a) * v + int(b): edge
        for edge, (a, b) in enumerate(zip(problem.edge_a, problem.edge_b))
    }
    yb_by_y: dict[int, list[int]] = {}
    old_ya_by_y: dict[int, list[int]] = {}
    old_ya_by_a: dict[int, list[int]] = {}
    old_yb_by_context_y: dict[int, int] = {}
    for second, y in enumerate(problem.active_yb_y):
        yb_by_y.setdefault(int(y), []).append(second)
        if second_birth[second] < checkpoint:
            key = int(problem.active_yb_b[second]) * v + int(y)
            old_yb_by_context_y[key] = second
    for first, y in enumerate(problem.active_ya_y):
        if first_birth[first] < checkpoint:
            old_ya_by_y.setdefault(int(y), []).append(first)
            old_ya_by_a.setdefault(
                int(problem.active_ya_a[first]), []
            ).append(first)

    out_first: list[int] = []
    out_second: list[int] = []
    out_edge: list[int] = []

    # Case 1: a new YA edge owns every compatible triangle using it.
    for first in np.flatnonzero(first_birth == checkpoint):
        y = int(problem.active_ya_y[first])
        a = int(problem.active_ya_a[first])
        for second in yb_by_y.get(y, ()):
            b = int(problem.active_yb_b[second])
            edge = ab_lookup.get(a * v + b)
            if edge is not None:
                out_first.append(int(first))
                out_second.append(second)
                out_edge.append(edge)

    # Case 2: a new YB edge owns triangles whose YA edge is older.
    for second in np.flatnonzero(second_birth == checkpoint):
        y = int(problem.active_yb_y[second])
        b = int(problem.active_yb_b[second])
        for first in old_ya_by_y.get(y, ()):
            a = int(problem.active_ya_a[first])
            edge = ab_lookup.get(a * v + b)
            if edge is not None:
                out_first.append(first)
                out_second.append(int(second))
                out_edge.append(edge)

    # Case 3: a new AB edge owns triangles whose two pair edges are older.
    for edge in np.flatnonzero(context_birth == checkpoint):
        a = int(problem.edge_a[edge])
        b = int(problem.edge_b[edge])
        for first in old_ya_by_a.get(a, ()):
            y = int(problem.active_ya_y[first])
            second = old_yb_by_context_y.get(b * v + y)
            if second is not None:
                out_first.append(first)
                out_second.append(second)
                out_edge.append(int(edge))

    first = np.asarray(out_first, dtype=np.int32)
    second = np.asarray(out_second, dtype=np.int32)
    edge = np.asarray(out_edge, dtype=np.int32)
    ya_order, ya_row, ya_ptr = _sparse_csr_order(first, edge)
    ab_order, ab_row, ab_ptr = _sparse_csr_order(
        edge, problem.active_ya_y[first]
    )
    return SparseIntersectionDelta(
        checkpoint=checkpoint,
        ya_row=ya_row,
        ya_ptr=ya_ptr,
        ya_correction_yb=second[ya_order],
        ya_edge_ab=edge[ya_order],
        ab_row=ab_row,
        ab_ptr=ab_ptr,
        ab_correction_ya=first[ab_order],
        ab_correction_yb=second[ab_order],
    )


def sparse_deltas_as_layered_graph(
    deltas: Sequence[SparseIntersectionDelta],
    ya_edges: int,
) -> LayeredIntersectionGraph:
    """Expand sparse delta row directories for compatibility testing."""

    row_ptr = []
    correction_yb = []
    edge_ab = []
    for delta in deltas:
        if np.any(delta.ya_row < 0) or np.any(delta.ya_row >= ya_edges):
            raise ValueError("delta YA row lies outside current support")
        counts = np.zeros(ya_edges, dtype=np.int64)
        counts[delta.ya_row] = np.diff(delta.ya_ptr)
        row_ptr.append(np.r_[0, np.cumsum(counts, dtype=np.int64)])
        correction_yb.append(delta.ya_correction_yb)
        edge_ab.append(delta.ya_edge_ab)
    return LayeredIntersectionGraph(
        tuple(row_ptr), tuple(correction_yb), tuple(edge_ab)
    )


def sparse_deltas_as_ab_major_graph(
    deltas: Sequence[SparseIntersectionDelta],
    ab_edges: int,
) -> ABMajorIntersectionGraph:
    """Materialize sparse AB-major deltas for compatibility testing."""

    counts = np.zeros(ab_edges, dtype=np.int64)
    for delta in deltas:
        if np.any(delta.ab_row < 0) or np.any(delta.ab_row >= ab_edges):
            raise ValueError("delta AB row lies outside current support")
        counts[delta.ab_row] += np.diff(delta.ab_ptr)
    edge_ptr = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    first = np.empty(edge_ptr[-1], dtype=np.int32)
    second = np.empty(edge_ptr[-1], dtype=np.int32)
    birth = np.empty(edge_ptr[-1], dtype=np.uint8)
    cursor = edge_ptr[:-1].copy()
    for delta in deltas:
        for local, edge in enumerate(delta.ab_row):
            lo, hi = delta.ab_ptr[local:local + 2]
            count = hi - lo
            out = cursor[edge]
            first[out:out + count] = delta.ab_correction_ya[lo:hi]
            second[out:out + count] = delta.ab_correction_yb[lo:hi]
            birth[out:out + count] = delta.checkpoint
            cursor[edge] += count
    return ABMajorIntersectionGraph(edge_ptr, first, second, birth)


def save_sparse_intersection_delta(
    delta: SparseIntersectionDelta,
    directory: str | Path,
) -> None:
    """Publish one immutable delta; the manifest is the completion marker."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / "manifest.json"
    if manifest.exists():
        raise FileExistsError(f"delta is already published at {destination}")
    arrays = {
        "ya_row": delta.ya_row,
        "ya_ptr": delta.ya_ptr,
        "ya_correction_yb": delta.ya_correction_yb,
        "ya_edge_ab": delta.ya_edge_ab,
        "ab_row": delta.ab_row,
        "ab_ptr": delta.ab_ptr,
        "ab_correction_ya": delta.ab_correction_ya,
        "ab_correction_yb": delta.ab_correction_yb,
    }
    for name, array in arrays.items():
        temporary = destination / f".{name}.npy.tmp"
        with temporary.open("wb") as output:
            np.save(output, array)
        temporary.replace(destination / f"{name}.npy")
    temporary_manifest = destination / ".manifest.json.tmp"
    temporary_manifest.write_text(json.dumps({
        "version": 1,
        "checkpoint": delta.checkpoint,
        "triangles": delta.triangles,
        "bytes": delta.nbytes,
    }, indent=2))
    temporary_manifest.replace(manifest)


def load_sparse_intersection_delta(
    directory: str | Path,
    *,
    mmap_mode: str | None = "r",
) -> SparseIntersectionDelta:
    """Open one completed immutable delta, memory-mapped by default."""

    source = Path(directory)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("version") != 1:
        raise ValueError("unsupported sparse intersection delta manifest")
    names = (
        "ya_row", "ya_ptr", "ya_correction_yb", "ya_edge_ab",
        "ab_row", "ab_ptr", "ab_correction_ya", "ab_correction_yb",
    )
    arrays = tuple(
        np.load(source / f"{name}.npy", mmap_mode=mmap_mode,
                allow_pickle=False)
        for name in names
    )
    delta = SparseIntersectionDelta(int(manifest["checkpoint"]), *arrays)
    if (
        delta.triangles != int(manifest.get("triangles", -1))
        or delta.nbytes != int(manifest.get("bytes", -1))
        or len(delta.ya_ptr) != len(delta.ya_row) + 1
        or len(delta.ab_ptr) != len(delta.ab_row) + 1
        or len(delta.ya_correction_yb) != delta.triangles
        or len(delta.ab_correction_ya) != delta.triangles
    ):
        raise ValueError("sparse intersection delta differs from manifest")
    return delta


def save_ab_major_intersection_graph(
    graph: ABMajorIntersectionGraph,
    directory: str | Path,
) -> None:
    """Persist an AB-major graph as independently memory-mappable arrays."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    for name, array in (
        ("edge_ptr", graph.edge_ptr),
        ("correction_ya", graph.correction_ya),
        ("correction_yb", graph.correction_yb),
        ("birth", graph.birth),
    ):
        np.save(destination / f"{name}.npy", array)
    temporary = destination / "manifest.json.tmp"
    temporary.write_text(json.dumps({
        "version": 1, "edges": graph.edges, "bytes": graph.nbytes,
    }, indent=2))
    temporary.replace(destination / "manifest.json")


def load_ab_major_intersection_graph(
    directory: str | Path,
    *,
    mmap_mode: str | None = "r",
) -> ABMajorIntersectionGraph:
    """Load a persisted AB-major graph, memory-mapped by default."""

    source = Path(directory)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("version") != 1:
        raise ValueError("unsupported AB-major graph manifest")
    graph = ABMajorIntersectionGraph(*(
        np.load(source / f"{name}.npy", mmap_mode=mmap_mode,
                allow_pickle=False)
        for name in ("edge_ptr", "correction_ya", "correction_yb", "birth")
    ))
    if graph.edges != int(manifest.get("edges", -1)):
        raise ValueError("AB-major graph edge count differs from manifest")
    return graph


@dataclass(frozen=True)
class BirthMajorSparseSupport:
    """Final sparse problem ordered so checkpoint supports are prefixes."""

    problem: SparseGroupedProblem
    birth_ya: np.ndarray
    birth_yb: np.ndarray
    birth_ab: np.ndarray


@dataclass(frozen=True)
class AppendOnlySparseSupportState:
    """Stable global pair-edge IDs known through one checkpoint."""

    vocabulary_size: int
    keys_ya: np.ndarray
    keys_yb: np.ndarray
    keys_ab: np.ndarray
    birth_ya: np.ndarray
    birth_yb: np.ndarray
    birth_ab: np.ndarray


def append_checkpoint_support(
    previous: AppendOnlySparseSupportState | None,
    checkpoint_problem: SparseGroupedProblem,
    checkpoint: int,
) -> tuple[AppendOnlySparseSupportState, BirthMajorSparseSupport]:
    """Append newly observed pair edges and align one checkpoint numerically."""

    if checkpoint < 0:
        raise ValueError("checkpoint must be nonnegative")
    v = checkpoint_problem.vocabulary_size
    current_keys = (
        checkpoint_problem.active_ya_y.astype(np.int64) * v
        + checkpoint_problem.active_ya_a,
        checkpoint_problem.active_yb_y.astype(np.int64) * v
        + checkpoint_problem.active_yb_b,
        checkpoint_problem.edge_a.astype(np.int64) * v
        + checkpoint_problem.edge_b,
    )
    current_values = (
        checkpoint_problem.target_ya,
        checkpoint_problem.target_yb,
        checkpoint_problem.edge_probability,
    )
    if previous is None:
        old_keys = tuple(np.empty(0, dtype=np.int64) for _ in range(3))
        old_births = tuple(np.empty(0, dtype=np.uint8) for _ in range(3))
    else:
        if previous.vocabulary_size != v:
            raise ValueError("vocabulary size changes between checkpoints")
        old_keys = (previous.keys_ya, previous.keys_yb, previous.keys_ab)
        old_births = (
            previous.birth_ya, previous.birth_yb, previous.birth_ab,
        )

    ordered_keys = []
    ordered_births = []
    ordered_values = []
    for old, births, keys, values in zip(
        old_keys, old_births, current_keys, current_values
    ):
        order = np.argsort(keys, kind="stable")
        sorted_current = np.asarray(keys[order], dtype=np.int64)
        sorted_values = np.asarray(values)[order]
        if len(np.unique(sorted_current)) != len(sorted_current):
            raise ValueError("checkpoint support contains duplicate pair edges")
        old_positions = np.searchsorted(sorted_current, old)
        if (
            np.any(old_positions == len(sorted_current))
            or not np.array_equal(sorted_current[old_positions], old)
        ):
            raise ValueError("checkpoint support is not append-only")
        is_old = np.zeros(len(sorted_current), dtype=bool)
        is_old[old_positions] = True
        new = sorted_current[~is_old]
        combined = np.r_[old, new]
        combined_births = np.r_[
            births,
            np.full(len(new), checkpoint, dtype=np.uint8),
        ]
        value_positions = np.searchsorted(sorted_current, combined)
        ordered_keys.append(combined)
        ordered_births.append(combined_births)
        ordered_values.append(sorted_values[value_positions])

    state = AppendOnlySparseSupportState(
        v, *ordered_keys, *ordered_births
    )
    ya_y, ya_a = divmod(state.keys_ya, v)
    yb_y, yb_b = divmod(state.keys_yb, v)
    edge_a, edge_b = divmod(state.keys_ab, v)
    ordered_problem = SparseGroupedProblem(
        vocabulary_size=v,
        edge_a=np.asarray(edge_a, dtype=np.int32),
        edge_b=np.asarray(edge_b, dtype=np.int32),
        edge_probability=ordered_values[2],
        target_y=checkpoint_problem.target_y,
        active_ya_y=np.asarray(ya_y, dtype=np.int32),
        active_ya_a=np.asarray(ya_a, dtype=np.int32),
        target_ya=ordered_values[0],
        active_yb_y=np.asarray(yb_y, dtype=np.int32),
        active_yb_b=np.asarray(yb_b, dtype=np.int32),
        target_yb=ordered_values[1],
    )
    return state, BirthMajorSparseSupport(
        ordered_problem, *ordered_births
    )


def save_layered_intersection_graph(
    graph: LayeredIntersectionGraph,
    directory: str | Path,
) -> None:
    """Persist independent uncompressed arrays suitable for memory mapping."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "layers": graph.layers, "edges": graph.edges}
    for depth in range(graph.layers):
        np.save(destination / f"row_ptr_{depth:03d}.npy", graph.row_ptr[depth])
        np.save(
            destination / f"correction_yb_{depth:03d}.npy",
            graph.correction_yb[depth],
        )
        np.save(destination / f"edge_ab_{depth:03d}.npy", graph.edge_ab[depth])
    temporary = destination / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(destination / "manifest.json")


def load_layered_intersection_graph(
    directory: str | Path,
    *,
    mmap_mode: str | None = "r",
) -> LayeredIntersectionGraph:
    """Load a persisted graph, memory-mapped by default."""

    source = Path(directory)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("version") != 1 or int(manifest.get("layers", 0)) < 1:
        raise ValueError("unsupported layered graph manifest")
    layers = int(manifest["layers"])
    graph = LayeredIntersectionGraph(
        tuple(np.load(
            source / f"row_ptr_{depth:03d}.npy",
            mmap_mode=mmap_mode, allow_pickle=False,
        ) for depth in range(layers)),
        tuple(np.load(
            source / f"correction_yb_{depth:03d}.npy",
            mmap_mode=mmap_mode, allow_pickle=False,
        ) for depth in range(layers)),
        tuple(np.load(
            source / f"edge_ab_{depth:03d}.npy",
            mmap_mode=mmap_mode, allow_pickle=False,
        ) for depth in range(layers)),
    )
    if graph.edges != int(manifest.get("edges", -1)):
        raise ValueError("layered graph edge count differs from manifest")
    return graph


def birth_major_sparse_support(
    problem: SparseGroupedProblem,
    birth_ya: np.ndarray,
    birth_yb: np.ndarray,
    birth_ab: np.ndarray,
) -> BirthMajorSparseSupport:
    """Assign stable pair-edge IDs ordered by birth depth then pair key."""

    v = problem.vocabulary_size
    births = tuple(np.asarray(values, dtype=np.uint8) for values in (
        birth_ya, birth_yb, birth_ab
    ))
    if (
        births[0].shape != problem.target_ya.shape
        or births[1].shape != problem.target_yb.shape
        or births[2].shape != problem.edge_probability.shape
    ):
        raise ValueError("pair-edge births and sparse problem disagree")
    keys = (
        problem.active_ya_y.astype(np.int64) * v + problem.active_ya_a,
        problem.active_yb_y.astype(np.int64) * v + problem.active_yb_b,
        problem.edge_a.astype(np.int64) * v + problem.edge_b,
    )
    orders = tuple(
        np.lexsort((key, birth)) for key, birth in zip(keys, births)
    )
    first, second, context = orders
    ordered = SparseGroupedProblem(
        vocabulary_size=v,
        edge_a=problem.edge_a[context],
        edge_b=problem.edge_b[context],
        edge_probability=problem.edge_probability[context],
        target_y=problem.target_y,
        active_ya_y=problem.active_ya_y[first],
        active_ya_a=problem.active_ya_a[first],
        target_ya=problem.target_ya[first],
        active_yb_y=problem.active_yb_y[second],
        active_yb_b=problem.active_yb_b[second],
        target_yb=problem.target_yb[second],
    )
    return BirthMajorSparseSupport(
        ordered, births[0][first], births[1][second], births[2][context]
    )


def checkpoint_in_birth_major_support(
    checkpoint_problem: SparseGroupedProblem,
    support: BirthMajorSparseSupport,
    checkpoint: int,
) -> SparseGroupedProblem:
    """Reorder one checkpoint into the exact active prefixes of a support."""

    final = support.problem
    if checkpoint_problem.vocabulary_size != final.vocabulary_size:
        raise ValueError("checkpoint and final support vocabularies differ")
    v = final.vocabulary_size

    def align(
        current_left, current_right, current_values,
        final_left, final_right, final_birth,
    ):
        count = int(np.searchsorted(final_birth, checkpoint, side="right"))
        if len(current_values) != count:
            raise ValueError("checkpoint support size is not its birth prefix")
        current_key = (
            current_left.astype(np.int64) * v + current_right
        )
        expected_key = (
            final_left[:count].astype(np.int64) * v + final_right[:count]
        )
        order = np.argsort(current_key, kind="stable")
        position = np.searchsorted(current_key[order], expected_key)
        if (
            np.any(position == len(order))
            or not np.array_equal(current_key[order[position]], expected_key)
        ):
            raise ValueError("checkpoint support differs from its birth prefix")
        return np.asarray(current_values)[order[position]], count

    target_ya, n1 = align(
        checkpoint_problem.active_ya_y, checkpoint_problem.active_ya_a,
        checkpoint_problem.target_ya,
        final.active_ya_y, final.active_ya_a, support.birth_ya,
    )
    target_yb, n2 = align(
        checkpoint_problem.active_yb_y, checkpoint_problem.active_yb_b,
        checkpoint_problem.target_yb,
        final.active_yb_y, final.active_yb_b, support.birth_yb,
    )
    edge_probability, ne = align(
        checkpoint_problem.edge_a, checkpoint_problem.edge_b,
        checkpoint_problem.edge_probability,
        final.edge_a, final.edge_b, support.birth_ab,
    )
    return SparseGroupedProblem(
        vocabulary_size=v,
        edge_a=final.edge_a[:ne],
        edge_b=final.edge_b[:ne],
        edge_probability=edge_probability,
        target_y=checkpoint_problem.target_y,
        active_ya_y=final.active_ya_y[:n1],
        active_ya_a=final.active_ya_a[:n1],
        target_ya=target_ya,
        active_yb_y=final.active_yb_y[:n2],
        active_yb_b=final.active_yb_b[:n2],
        target_yb=target_yb,
    )


def layered_intersection_graph_from_plan(
    problem: SparseGroupedProblem,
    plan: SparseIntersectionPlan,
    triangle_birth: np.ndarray,
    *,
    layers: int,
) -> LayeredIntersectionGraph:
    """Compress one explicit plan into birth-layered YA-row CSR arrays."""

    birth = np.asarray(triangle_birth)
    if birth.shape != plan.edge.shape:
        raise ValueError("triangle births and intersection plan disagree")
    if layers < 1 or np.any(birth < 0) or np.any(birth >= layers):
        raise ValueError("triangle birth lies outside the layer range")
    n1 = len(problem.target_ya)
    row_ptr = []
    correction_yb = []
    edge_ab = []
    for depth in range(layers):
        selected = np.flatnonzero(birth == depth)
        first = plan.correction_ya[selected]
        # Stable grouping retains the original within-row order.
        order = np.argsort(first, kind="stable")
        first = first[order]
        counts = np.bincount(first, minlength=n1)
        row_ptr.append(np.concatenate([
            np.array([0], dtype=np.int64),
            np.cumsum(counts, dtype=np.int64),
        ]))
        correction_yb.append(np.asarray(
            plan.correction_yb[selected][order], dtype=np.int32
        ))
        edge_ab.append(np.asarray(plan.edge[selected][order], dtype=np.int32))
    return LayeredIntersectionGraph(
        tuple(row_ptr), tuple(correction_yb), tuple(edge_ab)
    )


def build_layered_intersection_graph(
    problem: SparseGroupedProblem,
    birth_ya: np.ndarray,
    birth_yb: np.ndarray,
    birth_ab: np.ndarray,
    *,
    layers: int,
) -> LayeredIntersectionGraph:
    """Build birth-layered CSR directly, without a four-index plan."""

    first_birth = np.asarray(birth_ya, dtype=np.uint8)
    second_birth = np.asarray(birth_yb, dtype=np.uint8)
    context_birth = np.asarray(birth_ab, dtype=np.uint8)
    if (
        first_birth.shape != problem.target_ya.shape
        or second_birth.shape != problem.target_yb.shape
        or context_birth.shape != problem.edge_probability.shape
    ):
        raise ValueError("pair-edge births and sparse problem disagree")
    if layers < 1 or any(
        np.any(values >= layers)
        for values in (first_birth, second_birth, context_birth)
    ):
        raise ValueError("pair-edge birth lies outside the layer range")
    if not (
        _graphical_margin_c is not None
        and hasattr(_graphical_margin_c, "layered_intersection_graph")
    ):
        plan = build_sparse_intersection_plan(problem)
        triangle_birth = np.maximum.reduce([
            first_birth[plan.correction_ya],
            second_birth[plan.correction_yb],
            context_birth[plan.edge],
        ])
        return layered_intersection_graph_from_plan(
            problem, plan, triangle_birth, layers=layers
        )
    row_ptr, correction_yb, edge_ab = (
        _graphical_margin_c.layered_intersection_graph(
            np.ascontiguousarray(problem.edge_a, dtype=np.int32),
            np.ascontiguousarray(problem.edge_b, dtype=np.int32),
            np.ascontiguousarray(problem.active_ya_y, dtype=np.int32),
            np.ascontiguousarray(problem.active_ya_a, dtype=np.int32),
            np.ascontiguousarray(problem.active_yb_y, dtype=np.int32),
            np.ascontiguousarray(problem.active_yb_b, dtype=np.int32),
            np.ascontiguousarray(first_birth),
            np.ascontiguousarray(second_birth),
            np.ascontiguousarray(context_birth),
            layers,
        )
    )
    return LayeredIntersectionGraph(
        tuple(row_ptr), tuple(correction_yb), tuple(edge_ab)
    )


def build_ab_major_intersection_graph(
    problem: SparseGroupedProblem,
    birth_ya: np.ndarray,
    birth_yb: np.ndarray,
    birth_ab: np.ndarray,
) -> ABMajorIntersectionGraph:
    """Build the shared topology contiguously by retained AB edge."""

    first_birth = np.asarray(birth_ya, dtype=np.uint8)
    second_birth = np.asarray(birth_yb, dtype=np.uint8)
    context_birth = np.asarray(birth_ab, dtype=np.uint8)
    if (
        first_birth.shape != problem.target_ya.shape
        or second_birth.shape != problem.target_yb.shape
        or context_birth.shape != problem.edge_probability.shape
    ):
        raise ValueError("pair-edge births and sparse problem disagree")
    if not (
        _graphical_margin_c is not None
        and hasattr(_graphical_margin_c, "ab_major_intersection_graph")
    ):
        plan = build_sparse_intersection_plan(problem)
        counts = np.bincount(
            plan.edge, minlength=len(problem.edge_probability)
        )
        birth = np.maximum.reduce([
            first_birth[plan.correction_ya],
            second_birth[plan.correction_yb],
            context_birth[plan.edge],
        ])
        return ABMajorIntersectionGraph(
            np.r_[0, np.cumsum(counts, dtype=np.int64)],
            np.asarray(plan.correction_ya, dtype=np.int32),
            np.asarray(plan.correction_yb, dtype=np.int32),
            np.asarray(birth, dtype=np.uint8),
        )
    ptr, first, second, birth = (
        _graphical_margin_c.ab_major_intersection_graph(
            np.ascontiguousarray(problem.edge_a, dtype=np.int32),
            np.ascontiguousarray(problem.edge_b, dtype=np.int32),
            np.ascontiguousarray(problem.active_ya_y, dtype=np.int32),
            np.ascontiguousarray(problem.active_ya_a, dtype=np.int32),
            np.ascontiguousarray(problem.active_yb_y, dtype=np.int32),
            np.ascontiguousarray(problem.active_yb_b, dtype=np.int32),
            np.ascontiguousarray(first_birth),
            np.ascontiguousarray(second_birth),
            np.ascontiguousarray(context_birth),
        )
    )
    return ABMajorIntersectionGraph(ptr, first, second, birth)


def intersection_plan_from_layered_graph(
    problem: SparseGroupedProblem,
    graph: LayeredIntersectionGraph,
    checkpoint: int,
) -> SparseIntersectionPlan:
    """Expand active layers as a correctness reference for the native path."""

    if checkpoint < 0 or checkpoint >= graph.layers:
        raise ValueError("checkpoint lies outside the layered graph")
    n1 = len(problem.target_ya)
    first_parts = []
    second_parts = []
    edge_parts = []
    for depth in range(checkpoint + 1):
        ptr = graph.row_ptr[depth]
        if ptr.ndim != 1 or len(ptr) < n1 + 1:
            raise ValueError("layer row pointer has the wrong shape")
        first_parts.append(np.repeat(
            np.arange(n1, dtype=np.int32), np.diff(ptr[:n1 + 1])
        ))
        active_edges = int(ptr[n1])
        second_parts.append(graph.correction_yb[depth][:active_edges])
        edge_parts.append(graph.edge_ab[depth][:active_edges])
    first = np.concatenate(first_parts)
    second = np.concatenate(second_parts)
    edge = np.concatenate(edge_parts)
    target = problem.active_ya_y[first].astype(np.int32, copy=False)
    # Restore the canonical explicit-plan ordering: AB edge, then target.
    order = np.lexsort((target, edge))
    return SparseIntersectionPlan(
        edge[order], target[order], first[order], second[order]
    )


def sparse_factorized_margins_layered_reference(
    problem: SparseGroupedProblem,
    graph: LayeredIntersectionGraph,
    checkpoint: int,
    log_base_y: np.ndarray,
    correction_ya: np.ndarray,
    correction_yb: np.ndarray,
) -> SparseFactorizedMargins:
    """Evaluate active CSR layers directly, without an explicit plan.

    This intentionally simple NumPy implementation is the specification for
    the native layered traversal.  It assumes ``problem`` and its numerical
    margins describe the requested checkpoint; the graph supplies only the
    shared topology.
    """

    if checkpoint < 0 or checkpoint >= graph.layers:
        raise ValueError("checkpoint lies outside the layered graph")
    v = problem.vocabulary_size
    n1 = len(problem.target_ya)
    n2 = len(problem.target_yb)
    log_base = np.asarray(log_base_y) - logsumexp(log_base_y)
    base = np.exp(log_base)
    r1 = np.expm1(np.asarray(correction_ya))
    r2 = np.expm1(np.asarray(correction_yb))
    if r1.shape != (n1,) or r2.shape != (n2,):
        raise ValueError("layered correction arrays have the wrong shape")
    s1 = np.bincount(
        problem.active_ya_a,
        weights=base[problem.active_ya_y] * r1,
        minlength=v,
    )
    s2 = np.bincount(
        problem.active_yb_b,
        weights=base[problem.active_yb_y] * r2,
        minlength=v,
    )
    cross = np.zeros(len(problem.edge_probability))
    active_layers: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for depth in range(checkpoint + 1):
        ptr = graph.row_ptr[depth]
        first = np.repeat(
            np.arange(n1, dtype=np.int32), np.diff(ptr[:n1 + 1])
        )
        active_edges = int(ptr[n1])
        second = graph.correction_yb[depth][:active_edges]
        edge = graph.edge_ab[depth][:active_edges]
        active_layers.append((first, second, edge))
        cross += np.bincount(
            edge,
            weights=(
                base[problem.active_ya_y[first]] * r1[first] * r2[second]
            ),
            minlength=len(problem.edge_probability),
        )
    z = 1.0 + s1[problem.edge_a] + s2[problem.edge_b] + cross
    edge_mass = problem.edge_probability / z
    row_mass = np.bincount(
        problem.edge_a, weights=edge_mass, minlength=v
    )
    column_mass = np.bincount(
        problem.edge_b, weights=edge_mass, minlength=v
    )
    y1 = problem.active_ya_y
    y2 = problem.active_yb_y
    active_ya = base[y1] * (1.0 + r1) * row_mass[problem.active_ya_a]
    active_yb = base[y2] * (1.0 + r2) * column_mass[problem.active_yb_b]
    target_y = base * float(edge_mass.sum())
    target_y += np.bincount(
        y1, weights=base[y1] * r1 * row_mass[problem.active_ya_a],
        minlength=v,
    )
    target_y += np.bincount(
        y2, weights=base[y2] * r2 * column_mass[problem.active_yb_b],
        minlength=v,
    )
    for first, second, edge in active_layers:
        y = y1[first]
        common = base[y] * edge_mass[edge]
        active_ya += np.bincount(
            first,
            weights=common * (1.0 + r1[first]) * r2[second],
            minlength=n1,
        )
        active_yb += np.bincount(
            second,
            weights=common * (1.0 + r2[second]) * r1[first],
            minlength=n2,
        )
        target_y += np.bincount(
            y, weights=common * r1[first] * r2[second], minlength=v
        )
    return SparseFactorizedMargins(
        target_y, active_ya, active_yb, np.log(z)
    )


def sparse_factorized_margins_layered(
    problem: SparseGroupedProblem,
    graph: LayeredIntersectionGraph,
    checkpoint: int,
    log_base_y: np.ndarray,
    correction_ya: np.ndarray,
    correction_yb: np.ndarray,
    *,
    workers: int = 1,
    max_parallel_scratch_bytes: int = 1 << 30,
    context_rows: tuple[
        tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
    ] | None = None,
    phase_timing: dict[str, float] | None = None,
) -> SparseFactorizedMargins:
    """Evaluate a birth-layered graph with the native sequential kernel."""

    if workers < 1 or max_parallel_scratch_bytes < 1:
        raise ValueError("invalid layered parallel settings")
    scratch_per_worker = 8 * (
        len(problem.edge_probability) + len(problem.target_yb)
        + problem.vocabulary_size
    )
    effective_workers = min(
        workers,
        max(1, max_parallel_scratch_bytes // max(1, scratch_per_worker)),
        max(1, len(problem.target_ya)),
    )
    if not (
        _graphical_margin_c is not None
        and hasattr(_graphical_margin_c, "fused_margins_layered")
    ):
        return sparse_factorized_margins_layered_reference(
            problem, graph, checkpoint,
            log_base_y, correction_ya, correction_yb,
        )
    log_base = np.asarray(log_base_y, dtype=np.float64)
    normalized_log_base = log_base - logsumexp(log_base)
    base = np.exp(normalized_log_base)
    # Extreme solver trials are rejected by the caller after the finite check;
    # overflow here is therefore a control signal, not a user-facing warning.
    with np.errstate(over="ignore", invalid="ignore"):
        r1 = np.expm1(np.asarray(correction_ya, dtype=np.float64))
        r2 = np.expm1(np.asarray(correction_yb, dtype=np.float64))
    native_started = perf_counter()
    target_y, active_ya, active_yb, log_z, unstable = (
        _graphical_margin_c.fused_margins_layered(
            np.ascontiguousarray(base),
            np.ascontiguousarray(r1),
            np.ascontiguousarray(r2),
            np.ascontiguousarray(problem.edge_a, dtype=np.int32),
            np.ascontiguousarray(problem.edge_b, dtype=np.int32),
            np.ascontiguousarray(problem.edge_probability, dtype=np.float64),
            np.ascontiguousarray(problem.active_ya_y, dtype=np.int32),
            np.ascontiguousarray(problem.active_ya_a, dtype=np.int32),
            np.ascontiguousarray(problem.active_yb_y, dtype=np.int32),
            np.ascontiguousarray(problem.active_yb_b, dtype=np.int32),
            tuple(np.ascontiguousarray(x, dtype=np.int64) for x in graph.row_ptr),
            tuple(np.ascontiguousarray(x, dtype=np.int32) for x in graph.correction_yb),
            tuple(np.ascontiguousarray(x, dtype=np.int32) for x in graph.edge_ab),
            checkpoint,
            effective_workers,
        )
    )
    if phase_timing is not None:
        phase_timing["layered_native_seconds"] = (
            phase_timing.get("layered_native_seconds", 0.0)
            + perf_counter() - native_started
        )
    # The expanded 1+S1+S2+cross formula is fast but can lose precision when
    # large signed corrections cancel.  The native kernel omits and flags
    # such AB edges; add them back from their positive log-sum-exp form.
    if context_rows is None:
        order1 = np.argsort(problem.active_ya_a, kind="stable")
        order2 = np.argsort(problem.active_yb_b, kind="stable")
        ptr1 = np.r_[0, np.cumsum(np.bincount(
            problem.active_ya_a, minlength=problem.vocabulary_size
        ), dtype=np.int64)]
        ptr2 = np.r_[0, np.cumsum(np.bincount(
            problem.active_yb_b, minlength=problem.vocabulary_size
        ), dtype=np.int64)]
    else:
        (ptr1, order1), (ptr2, order2) = context_rows
    unstable_edges = np.flatnonzero(unstable)
    repair_started = perf_counter()
    for edge in unstable_edges:
        a = problem.edge_a[edge]
        b = problem.edge_b[edge]
        selected1 = order1[ptr1[a]:ptr1[a + 1]]
        selected2 = order2[ptr2[b]:ptr2[b + 1]]
        score = normalized_log_base.copy()
        score[problem.active_ya_y[selected1]] += np.asarray(
            correction_ya
        )[selected1]
        score[problem.active_yb_y[selected2]] += np.asarray(
            correction_yb
        )[selected2]
        direct_log_z = float(logsumexp(score))
        joint = problem.edge_probability[edge] * np.exp(
            score - direct_log_z
        )
        target_y += joint
        active_ya[selected1] += joint[problem.active_ya_y[selected1]]
        active_yb[selected2] += joint[problem.active_yb_y[selected2]]
        log_z[edge] = direct_log_z
    if phase_timing is not None:
        phase_timing["layered_repair_seconds"] = (
            phase_timing.get("layered_repair_seconds", 0.0)
            + perf_counter() - repair_started
        )
        phase_timing["layered_unstable_edges"] = (
            phase_timing.get("layered_unstable_edges", 0.0)
            + len(unstable_edges)
        )
        phase_timing["layered_evaluations"] = (
            phase_timing.get("layered_evaluations", 0.0) + 1.0
        )
    if not all(np.all(np.isfinite(array)) for array in (
        target_y, active_ya, active_yb, log_z
    )):
        raise FloatingPointError("layered margin evaluation remained nonfinite")
    return SparseFactorizedMargins(target_y, active_ya, active_yb, log_z)


def sparse_factorized_margins_ab_major(
    problem: SparseGroupedProblem,
    graph: ABMajorIntersectionGraph,
    checkpoint: int,
    edge_offset: int,
    log_base_y: np.ndarray,
    correction_ya: np.ndarray,
    correction_yb: np.ndarray,
    *,
    workers: int = 1,
    max_parallel_scratch_bytes: int = 1 << 30,
    prepared_factors: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ] | None = None,
    edge_indices: np.ndarray | None = None,
    context_rows: tuple[
        tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
    ] | None = None,
    phase_timing: dict[str, float] | None = None,
) -> SparseFactorizedMargins:
    """Evaluate a contiguous or indexed AB batch from the shared graph."""

    if checkpoint < 0 or edge_offset < 0 or workers < 1:
        raise ValueError("invalid AB-major checkpoint or edge offset")
    ne = len(problem.edge_probability)
    if edge_indices is None and edge_offset + ne + 1 > len(graph.edge_ptr):
        raise ValueError("AB-major edge range lies outside the graph")
    if edge_indices is not None:
        edge_indices = np.ascontiguousarray(edge_indices, dtype=np.int32)
        if edge_indices.shape != (ne,):
            raise ValueError("indexed AB batch must match its edge law")
    if prepared_factors is None:
        log_base = np.asarray(log_base_y, dtype=np.float64)
        normalized_log_base = log_base - logsumexp(log_base)
        base = np.exp(normalized_log_base)
        c1 = np.asarray(correction_ya, dtype=np.float64)
        c2 = np.asarray(correction_yb, dtype=np.float64)
        r1 = np.expm1(c1)
        r2 = np.expm1(c2)
    else:
        normalized_log_base, base, c1, c2, r1, r2 = prepared_factors
    scratch_per_worker = 8 * (
        3 * problem.vocabulary_size + len(c1) + len(c2)
    )
    effective_workers = min(
        workers,
        max(1, max_parallel_scratch_bytes // max(1, scratch_per_worker)),
        max(1, ne),
    )
    native_arguments = (
            np.ascontiguousarray(base), np.ascontiguousarray(r1),
            np.ascontiguousarray(r2),
            np.ascontiguousarray(problem.edge_a, dtype=np.int32),
            np.ascontiguousarray(problem.edge_b, dtype=np.int32),
            np.ascontiguousarray(problem.edge_probability, dtype=np.float64),
            np.ascontiguousarray(problem.active_ya_y, dtype=np.int32),
            np.ascontiguousarray(problem.active_ya_a, dtype=np.int32),
            np.ascontiguousarray(problem.active_yb_y, dtype=np.int32),
            np.ascontiguousarray(problem.active_yb_b, dtype=np.int32),
            np.ascontiguousarray(graph.edge_ptr, dtype=np.int64),
            np.ascontiguousarray(graph.correction_ya, dtype=np.int32),
            np.ascontiguousarray(graph.correction_yb, dtype=np.int32),
            np.ascontiguousarray(graph.birth, dtype=np.uint8),
            checkpoint, edge_offset, effective_workers,
    )
    if edge_indices is not None:
        native_arguments += (edge_indices,)
    native_started = perf_counter()
    target_y, active_ya, active_yb, log_z, unstable = (
        _graphical_margin_c.fused_margins_ab_major(*native_arguments)
    )
    if phase_timing is not None:
        phase_timing["ab_native_seconds"] = (
            phase_timing.get("ab_native_seconds", 0.0)
            + perf_counter() - native_started
        )
    if context_rows is None:
        order1 = np.argsort(problem.active_ya_a, kind="stable")
        order2 = np.argsort(problem.active_yb_b, kind="stable")
        ptr1 = np.r_[0, np.cumsum(np.bincount(
            problem.active_ya_a, minlength=problem.vocabulary_size
        ), dtype=np.int64)]
        ptr2 = np.r_[0, np.cumsum(np.bincount(
            problem.active_yb_b, minlength=problem.vocabulary_size
        ), dtype=np.int64)]
    else:
        (ptr1, order1), (ptr2, order2) = context_rows
    unstable_edges = np.flatnonzero(unstable)
    repair_started = perf_counter()
    for edge in unstable_edges:
        a = problem.edge_a[edge]
        b = problem.edge_b[edge]
        selected1 = order1[ptr1[a]:ptr1[a + 1]]
        selected2 = order2[ptr2[b]:ptr2[b + 1]]
        score = normalized_log_base.copy()
        score[problem.active_ya_y[selected1]] += c1[selected1]
        score[problem.active_yb_y[selected2]] += c2[selected2]
        direct_log_z = float(logsumexp(score))
        joint = problem.edge_probability[edge] * np.exp(
            score - direct_log_z
        )
        target_y += joint
        active_ya[selected1] += joint[problem.active_ya_y[selected1]]
        active_yb[selected2] += joint[problem.active_yb_y[selected2]]
        log_z[edge] = direct_log_z
    if phase_timing is not None:
        phase_timing["ab_repair_seconds"] = (
            phase_timing.get("ab_repair_seconds", 0.0)
            + perf_counter() - repair_started
        )
        phase_timing["ab_unstable_edges"] = (
            phase_timing.get("ab_unstable_edges", 0.0)
            + len(unstable_edges)
        )
    if not all(np.all(np.isfinite(array)) for array in (
        target_y, active_ya, active_yb, log_z
    )):
        raise FloatingPointError("AB-major margin evaluation remained nonfinite")
    return SparseFactorizedMargins(target_y, active_ya, active_yb, log_z)


def build_sparse_intersection_plan(
    problem: SparseGroupedProblem,
    *,
    edge_chunk_size: int = 100_000,
    max_intersections: int | None = None,
) -> SparseIntersectionPlan:
    """Build correction intersections with compiled sparse row products."""

    if edge_chunk_size < 1:
        raise ValueError("edge_chunk_size must be positive")
    if (
        _graphical_margin_c is not None
        and hasattr(_graphical_margin_c, "intersection_plan")
    ):
        edge, target_y, correction_ya, correction_yb = (
            _graphical_margin_c.intersection_plan(
                np.ascontiguousarray(problem.edge_a, dtype=np.int32),
                np.ascontiguousarray(problem.edge_b, dtype=np.int32),
                np.ascontiguousarray(problem.active_ya_y, dtype=np.int32),
                np.ascontiguousarray(problem.active_ya_a, dtype=np.int32),
                np.ascontiguousarray(problem.active_yb_y, dtype=np.int32),
                np.ascontiguousarray(problem.active_yb_b, dtype=np.int32),
                -1 if max_intersections is None else max_intersections,
            )
        )
        return SparseIntersectionPlan(
            edge, target_y, correction_ya, correction_yb
        )
    v = problem.vocabulary_size
    first = csr_matrix(
        (
            np.arange(len(problem.target_ya), dtype=np.int64) + 1,
            (problem.active_ya_a, problem.active_ya_y),
        ),
        shape=(v, v),
    )
    second = csr_matrix(
        (
            np.arange(len(problem.target_yb), dtype=np.int64) + 1,
            (problem.active_yb_b, problem.active_yb_y),
        ),
        shape=(v, v),
    )
    first.sort_indices()
    second.sort_indices()
    edge_parts = []
    y_parts = []
    first_parts = []
    second_parts = []
    total = 0
    for start in range(0, len(problem.edge_a), edge_chunk_size):
        stop = min(len(problem.edge_a), start + edge_chunk_size)
        rows1 = first[problem.edge_a[start:stop]]
        rows2 = second[problem.edge_b[start:stop]]
        intersection1 = rows1.multiply(rows2.astype(bool)).tocsr()
        intersection2 = rows2.multiply(rows1.astype(bool)).tocsr()
        intersection1.sort_indices()
        intersection2.sort_indices()
        if not (
            np.array_equal(intersection1.indptr, intersection2.indptr)
            and np.array_equal(intersection1.indices, intersection2.indices)
        ):
            raise RuntimeError("sparse intersection products disagree")
        count = intersection1.nnz
        total += count
        if max_intersections is not None and total > max_intersections:
            raise MemoryError(
                f"intersection plan exceeds limit {max_intersections}"
            )
        edge_parts.append(np.repeat(
            np.arange(start, stop, dtype=np.int32),
            np.diff(intersection1.indptr),
        ))
        y_parts.append(intersection1.indices.astype(np.int32, copy=False))
        first_parts.append(
            (intersection1.data - 1).astype(np.int32, copy=False)
        )
        second_parts.append(
            (intersection2.data - 1).astype(np.int32, copy=False)
        )

    def combine(parts):
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int32)

    return SparseIntersectionPlan(
        combine(edge_parts), combine(y_parts),
        combine(first_parts), combine(second_parts),
    )


def sparse_factorized_margins(
    problem: SparseGroupedProblem,
    plan: SparseIntersectionPlan,
    log_base_y: np.ndarray,
    correction_ya: np.ndarray,
    correction_yb: np.ndarray,
    *,
    executor=None,
    intersection_shards: Sequence[tuple[int, int]] | None = None,
) -> SparseFactorizedMargins:
    """Vectorized intersection-only sufficient-statistic evaluation."""

    v = problem.vocabulary_size
    log_base = np.asarray(log_base_y) - logsumexp(log_base_y)
    base = np.exp(log_base)
    r1 = np.expm1(np.asarray(correction_ya))
    r2 = np.expm1(np.asarray(correction_yb))
    if _graphical_margin_c is not None:
        edge = (
            plan.edge if plan.edge_offset == 0
            else plan.edge - plan.edge_offset
        )
        target_y, active_ya, active_yb, log_z = (
            _graphical_margin_c.fused_margins(
                np.ascontiguousarray(base, dtype=np.float64),
                np.ascontiguousarray(r1, dtype=np.float64),
                np.ascontiguousarray(r2, dtype=np.float64),
                np.ascontiguousarray(problem.edge_a, dtype=np.int32),
                np.ascontiguousarray(problem.edge_b, dtype=np.int32),
                np.ascontiguousarray(problem.edge_probability, dtype=np.float64),
                np.ascontiguousarray(problem.active_ya_y, dtype=np.int32),
                np.ascontiguousarray(problem.active_ya_a, dtype=np.int32),
                np.ascontiguousarray(problem.active_yb_y, dtype=np.int32),
                np.ascontiguousarray(problem.active_yb_b, dtype=np.int32),
                np.ascontiguousarray(edge, dtype=np.int32),
                np.ascontiguousarray(plan.target_y, dtype=np.int32),
                np.ascontiguousarray(plan.correction_ya, dtype=np.int32),
                np.ascontiguousarray(plan.correction_yb, dtype=np.int32),
                len(intersection_shards) if intersection_shards else 1,
            )
        )
        if (
            np.all(np.isfinite(target_y))
            and np.all(np.isfinite(active_ya))
            and np.all(np.isfinite(active_yb))
            and np.all(np.isfinite(log_z))
        ):
            return SparseFactorizedMargins(
                target_y, active_ya, active_yb, log_z
            )
        # Rare extreme edges can lose all precision in the expanded
        # ``1 + S1 + S2 + cross`` normalizer.  Fall through to the
        # transparent evaluator, which isolates and recomputes only those
        # edges in log space.
    s1 = np.bincount(
        problem.active_ya_a,
        weights=base[problem.active_ya_y] * r1,
        minlength=v,
    )
    s2 = np.bincount(
        problem.active_yb_b,
        weights=base[problem.active_yb_y] * r2,
        minlength=v,
    )
    edge = (
        plan.edge if plan.edge_offset == 0
        else plan.edge - plan.edge_offset
    )
    y = plan.target_y
    i1 = plan.correction_ya
    i2 = plan.correction_yb
    shards = intersection_shards or [(0, len(edge))]

    def cross_contribution(bounds):
        lo, hi = bounds
        return np.bincount(
            edge[lo:hi],
            weights=base[y[lo:hi]] * r1[i1[lo:hi]] * r2[i2[lo:hi]],
            minlength=len(problem.edge_probability),
        )

    cross_by_edge = np.zeros(len(problem.edge_probability))
    cross_parts = (
        map(cross_contribution, shards)
        if executor is None else executor.map(cross_contribution, shards)
    )
    for part in cross_parts:
        cross_by_edge += part
    z = (
        1.0 + s1[problem.edge_a] + s2[problem.edge_b]
        + cross_by_edge
    )
    scale = (
        1.0 + np.abs(s1[problem.edge_a])
        + np.abs(s2[problem.edge_b]) + np.abs(cross_by_edge)
    )
    unstable = (
        ~np.isfinite(z) | (z <= 0.0)
        | (scale > 1e12 * np.maximum(np.abs(z), np.finfo(float).tiny))
    )
    m_edge = np.zeros_like(problem.edge_probability)
    np.divide(
        problem.edge_probability, z, out=m_edge,
        where=~unstable,
    )
    row_m = np.bincount(problem.edge_a, weights=m_edge, minlength=v)
    col_m = np.bincount(problem.edge_b, weights=m_edge, minlength=v)
    active_ya = (
        base[problem.active_ya_y] * (1.0 + r1)
        * row_m[problem.active_ya_a]
    )
    active_yb = (
        base[problem.active_yb_y] * (1.0 + r2)
        * col_m[problem.active_yb_b]
    )
    def margin_contribution(bounds):
        lo, hi = bounds
        local_edge = edge[lo:hi]
        local_y = y[lo:hi]
        local_i1 = i1[lo:hi]
        local_i2 = i2[lo:hi]
        common = base[local_y] * m_edge[local_edge]
        return (
            np.bincount(
                local_i1,
                weights=(common * (1.0 + r1[local_i1]) * r2[local_i2]),
                minlength=len(r1),
            ),
            np.bincount(
                local_i2,
                weights=(common * (1.0 + r2[local_i2]) * r1[local_i1]),
                minlength=len(r2),
            ),
            np.bincount(
                local_y,
                weights=common * r1[local_i1] * r2[local_i2],
                minlength=v,
            ),
        )

    target_cross = np.zeros(v)
    margin_parts = (
        map(margin_contribution, shards)
        if executor is None else executor.map(margin_contribution, shards)
    )
    for part_ya, part_yb, part_y in margin_parts:
        active_ya += part_ya
        active_yb += part_yb
        target_cross += part_y
    target_y = base * float(m_edge.sum())
    target_y += np.bincount(
        problem.active_ya_y,
        weights=(base[problem.active_ya_y] * r1
                 * row_m[problem.active_ya_a]),
        minlength=v,
    )
    target_y += np.bincount(
        problem.active_yb_y,
        weights=(base[problem.active_yb_y] * r2
                 * col_m[problem.active_yb_b]),
        minlength=v,
    )
    target_y += target_cross
    log_z = np.empty_like(z)
    np.log(z, out=log_z, where=~unstable)
    for bad_edge in np.flatnonzero(unstable):
        a = problem.edge_a[bad_edge]
        b = problem.edge_b[bad_edge]
        selected1 = np.flatnonzero(problem.active_ya_a == a)
        selected2 = np.flatnonzero(problem.active_yb_b == b)
        scores = np.zeros(v)
        scores[problem.active_ya_y[selected1]] += np.asarray(
            correction_ya
        )[selected1]
        scores[problem.active_yb_y[selected2]] += np.asarray(
            correction_yb
        )[selected2]
        direct_log_z = float(logsumexp(log_base + scores))
        joint_mass = (
            problem.edge_probability[bad_edge]
            * np.exp(log_base + scores - direct_log_z)
        )
        target_y += joint_mass
        active_ya[selected1] += joint_mass[
            problem.active_ya_y[selected1]
        ]
        active_yb[selected2] += joint_mass[
            problem.active_yb_y[selected2]
        ]
        log_z[bad_edge] = direct_log_z
    return SparseFactorizedMargins(
        target_y, active_ya, active_yb, log_z
    )


def sparse_factorized_margins_reference(
    problem: SparseGroupedProblem,
    log_base_y: np.ndarray,
    correction_ya: np.ndarray,
    correction_yb: np.ndarray,
) -> SparseFactorizedMargins:
    """Evaluate margins without expanding active unions over AB edges.

    This transparent Python implementation is a correctness reference for
    the production streamed/heavy-light evaluator.  Its work is proportional
    to correction intersections plus feature-edge propagation, not to a
    materialized target union for every AB edge.
    """

    v = problem.vocabulary_size
    log_base = np.asarray(log_base_y) - logsumexp(log_base_y)
    base = np.exp(log_base)
    r1 = np.expm1(np.asarray(correction_ya))
    r2 = np.expm1(np.asarray(correction_yb))
    rows1: list[dict[int, tuple[int, float]]] = [{} for _ in range(v)]
    rows2: list[dict[int, tuple[int, float]]] = [{} for _ in range(v)]
    for index, (y, a, value) in enumerate(zip(
        problem.active_ya_y, problem.active_ya_a, r1
    )):
        rows1[int(a)][int(y)] = (index, float(value))
    for index, (y, b, value) in enumerate(zip(
        problem.active_yb_y, problem.active_yb_b, r2
    )):
        rows2[int(b)][int(y)] = (index, float(value))

    s1 = np.bincount(
        problem.active_ya_a,
        weights=base[problem.active_ya_y] * r1,
        minlength=v,
    )
    s2 = np.bincount(
        problem.active_yb_b,
        weights=base[problem.active_yb_y] * r2,
        minlength=v,
    )
    z = 1.0 + s1[problem.edge_a] + s2[problem.edge_b]
    intersections: list[tuple[int, int, float, float]] = []
    for edge, (a, b) in enumerate(zip(problem.edge_a, problem.edge_b)):
        first = rows1[int(a)]
        second = rows2[int(b)]
        for y in first.keys() & second.keys():
            value1 = first[y][1]
            value2 = second[y][1]
            z[edge] += base[y] * value1 * value2
            intersections.append((edge, y, value1, value2))
    m_edge = problem.edge_probability / z
    row_m = np.bincount(problem.edge_a, weights=m_edge, minlength=v)
    col_m = np.bincount(problem.edge_b, weights=m_edge, minlength=v)

    active_ya = base[problem.active_ya_y] * (1.0 + r1) * row_m[
        problem.active_ya_a
    ]
    active_yb = base[problem.active_yb_y] * (1.0 + r2) * col_m[
        problem.active_yb_b
    ]
    target_y = base * float(m_edge.sum())
    target_y += np.bincount(
        problem.active_ya_y,
        weights=(base[problem.active_ya_y] * r1
                 * row_m[problem.active_ya_a]),
        minlength=v,
    )
    target_y += np.bincount(
        problem.active_yb_y,
        weights=(base[problem.active_yb_y] * r2
                 * col_m[problem.active_yb_b]),
        minlength=v,
    )

    neighbors_a: list[list[tuple[int, float]]] = [[] for _ in range(v)]
    neighbors_b: list[list[tuple[int, float]]] = [[] for _ in range(v)]
    for edge, (a, b) in enumerate(zip(problem.edge_a, problem.edge_b)):
        neighbors_a[int(a)].append((int(b), float(m_edge[edge])))
        neighbors_b[int(b)].append((int(a), float(m_edge[edge])))
    for index, (y, a) in enumerate(zip(
        problem.active_ya_y, problem.active_ya_a
    )):
        extra = sum(
            mass * rows2[b].get(int(y), (-1, 0.0))[1]
            for b, mass in neighbors_a[int(a)]
        )
        active_ya[index] += base[y] * (1.0 + r1[index]) * extra
    for index, (y, b) in enumerate(zip(
        problem.active_yb_y, problem.active_yb_b
    )):
        extra = sum(
            mass * rows1[a].get(int(y), (-1, 0.0))[1]
            for a, mass in neighbors_b[int(b)]
        )
        active_yb[index] += base[y] * (1.0 + r2[index]) * extra
    for edge, y, value1, value2 in intersections:
        target_y[y] += base[y] * m_edge[edge] * value1 * value2

    return SparseFactorizedMargins(
        target_y, active_ya, active_yb, np.log(z)
    )


def sparse_problem_with_edge_distribution(
    problem: SparseGroupedProblem,
    edge_indices: np.ndarray,
    edge_probability: np.ndarray,
) -> SparseGroupedProblem:
    """Restrict a problem to edges carrying a supplied probability law.

    The target sufficient-statistic margins deliberately remain those of the
    complete problem.  Consequently a random edge law whose expectation is
    ``problem.edge_probability`` gives an unbiased estimate of the complete
    conditional-dual gradient.
    """

    indices = np.asarray(edge_indices, dtype=np.int64)
    probability = np.asarray(edge_probability, dtype=np.float64)
    if indices.ndim != 1 or probability.shape != indices.shape:
        raise ValueError("edge indices and probabilities must be 1-D matches")
    if not len(indices):
        raise ValueError("an edge distribution must have nonempty support")
    if np.any(indices < 0) or np.any(indices >= len(problem.edge_probability)):
        raise ValueError("edge index outside the problem")
    if not np.isfinite(probability).all() or np.any(probability < 0.0):
        raise ValueError("edge probabilities must be finite and nonnegative")
    total = float(probability.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("edge probabilities must have positive mass")
    probability = probability / total
    return SparseGroupedProblem(
        vocabulary_size=problem.vocabulary_size,
        edge_a=problem.edge_a[indices],
        edge_b=problem.edge_b[indices],
        edge_probability=probability,
        target_y=problem.target_y,
        active_ya_y=problem.active_ya_y,
        active_ya_a=problem.active_ya_a,
        target_ya=problem.target_ya,
        active_yb_y=problem.active_yb_y,
        active_yb_b=problem.active_yb_b,
        target_yb=problem.target_yb,
    )


def sample_sparse_grouped_edges(
    problem: SparseGroupedProblem,
    sample_size: int,
    rng: np.random.Generator,
) -> SparseGroupedProblem:
    """Draw an empirical AB-edge distribution from the target edge law."""

    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    draws = rng.choice(
        len(problem.edge_probability),
        size=sample_size,
        replace=True,
        p=problem.edge_probability,
    )
    indices, counts = np.unique(draws, return_counts=True)
    return sparse_problem_with_edge_distribution(
        problem, indices, counts.astype(np.float64)
    )


def sparse_edge_minibatch(
    problem: SparseGroupedProblem,
    intersection_plan: SparseIntersectionPlan,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[SparseGroupedProblem, SparseIntersectionPlan]:
    """Sample edges and slice a checkpoint's fixed intersection plan.

    Reusing the plan is essential: rebuilding the sparse row intersections
    on every stochastic step would erase much of the minibatch saving.
    """

    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    draws = rng.choice(
        len(problem.edge_probability), size=sample_size, replace=True,
        p=problem.edge_probability,
    )
    original_edges, counts = np.unique(draws, return_counts=True)
    return sparse_edge_distribution_with_plan(
        problem, intersection_plan, original_edges,
        counts.astype(np.float64),
    )


def sparse_edge_distribution_with_plan(
    problem: SparseGroupedProblem,
    intersection_plan: SparseIntersectionPlan,
    edge_indices: np.ndarray,
    edge_probability: np.ndarray,
) -> tuple[SparseGroupedProblem, SparseIntersectionPlan]:
    """Attach a restricted edge distribution to its sliced fixed plan."""

    original_edges = np.asarray(edge_indices, dtype=np.int64)
    order = np.argsort(original_edges, kind="stable")
    original_edges = original_edges[order]
    weights = np.asarray(edge_probability, dtype=np.float64)[order]
    sampled = sparse_problem_with_edge_distribution(
        problem, original_edges, weights
    )
    plan_edge = intersection_plan.edge
    starts = np.searchsorted(plan_edge, original_edges, side="left")
    stops = np.searchsorted(plan_edge, original_edges, side="right")
    lengths = stops - starts
    positions = (
        np.concatenate([
            np.arange(start, stop, dtype=np.int64)
            for start, stop in zip(starts, stops) if stop > start
        ])
        if int(lengths.sum()) else np.empty(0, dtype=np.int64)
    )
    local_edges = np.repeat(
        np.arange(len(original_edges), dtype=np.int32), lengths
    )
    plan = SparseIntersectionPlan(
        edge=local_edges,
        target_y=intersection_plan.target_y[positions],
        correction_ya=intersection_plan.correction_ya[positions],
        correction_yb=intersection_plan.correction_yb[positions],
    )
    return sampled, plan


def build_sparse_edge_blocks(
    problem: SparseGroupedProblem,
    intersection_plan: SparseIntersectionPlan,
    blocks: int,
) -> list[SparseEdgeBlock]:
    """Partition a fixed plan into approximately equal-work edge blocks."""

    edge_count = len(problem.edge_probability)
    if blocks < 1:
        raise ValueError("blocks must be positive")
    blocks = min(blocks, edge_count)
    edge_boundaries = np.empty(blocks + 1, dtype=np.int64)
    edge_boundaries[0] = 0
    edge_boundaries[-1] = edge_count
    if blocks > 1:
        if len(intersection_plan.edge):
            # Direct lookup at equally spaced plan positions gives cuts with
            # approximately equal intersection work.
            positions = np.linspace(
                0, len(intersection_plan.edge), blocks + 1, dtype=np.int64
            )
            edge_boundaries[1:-1] = intersection_plan.edge[
                np.minimum(positions[1:-1], len(intersection_plan.edge) - 1)
            ]
        else:
            edge_boundaries[1:-1] = np.linspace(
                0, edge_count, blocks + 1, dtype=np.int64
            )[1:-1]
    edge_boundaries = np.unique(edge_boundaries)
    plan_boundaries = np.searchsorted(
        intersection_plan.edge, edge_boundaries, side="left"
    )
    answer = []
    for block_index, (edge_lo, edge_hi) in enumerate(
        pairwise(edge_boundaries)
    ):
        edge_lo = int(edge_lo)
        edge_hi = int(edge_hi)
        if edge_hi <= edge_lo:
            continue
        mass = float(problem.edge_probability[edge_lo:edge_hi].sum())
        if mass <= 0.0:
            continue
        plan_lo = int(plan_boundaries[block_index])
        plan_hi = int(plan_boundaries[block_index + 1])
        block_problem = SparseGroupedProblem(
            vocabulary_size=problem.vocabulary_size,
            edge_a=problem.edge_a[edge_lo:edge_hi],
            edge_b=problem.edge_b[edge_lo:edge_hi],
            edge_probability=(
                problem.edge_probability[edge_lo:edge_hi] / mass
            ),
            target_y=problem.target_y,
            active_ya_y=problem.active_ya_y,
            active_ya_a=problem.active_ya_a,
            target_ya=problem.target_ya,
            active_yb_y=problem.active_yb_y,
            active_yb_b=problem.active_yb_b,
            target_yb=problem.target_yb,
        )
        block_plan = SparseIntersectionPlan(
            edge=intersection_plan.edge[plan_lo:plan_hi],
            target_y=intersection_plan.target_y[plan_lo:plan_hi],
            correction_ya=intersection_plan.correction_ya[plan_lo:plan_hi],
            correction_yb=intersection_plan.correction_yb[plan_lo:plan_hi],
            edge_offset=edge_lo,
        )
        answer.append(SparseEdgeBlock(mass, block_problem, block_plan))
    return answer


def layered_edge_intersection_counts(
    problem: SparseGroupedProblem,
    graph: LayeredIntersectionGraph,
    checkpoint: int,
) -> np.ndarray:
    """Count active intersection triangles per AB edge without a full plan."""

    if checkpoint < 0 or checkpoint >= graph.layers:
        raise ValueError("checkpoint lies outside the layered graph")
    counts = np.zeros(len(problem.edge_probability), dtype=np.int64)
    n1 = len(problem.target_ya)
    for depth in range(checkpoint + 1):
        stop = int(graph.row_ptr[depth][n1])
        counts += np.bincount(
            graph.edge_ab[depth][:stop], minlength=len(counts)
        )
    return counts


def sparse_edge_block_from_bounds(
    problem: SparseGroupedProblem,
    edge_lo: int,
    edge_hi: int,
    *,
    build_plan: bool = True,
) -> SparseEdgeBlock:
    """Construct one local block and only that block's intersections."""

    if not 0 <= edge_lo < edge_hi <= len(problem.edge_probability):
        raise ValueError("invalid sparse edge block bounds")
    mass = float(problem.edge_probability[edge_lo:edge_hi].sum())
    if mass <= 0.0:
        raise ValueError("sparse edge block has no probability mass")
    block_problem = SparseGroupedProblem(
        vocabulary_size=problem.vocabulary_size,
        edge_a=problem.edge_a[edge_lo:edge_hi],
        edge_b=problem.edge_b[edge_lo:edge_hi],
        edge_probability=problem.edge_probability[edge_lo:edge_hi] / mass,
        target_y=problem.target_y,
        active_ya_y=problem.active_ya_y,
        active_ya_a=problem.active_ya_a,
        target_ya=problem.target_ya,
        active_yb_y=problem.active_yb_y,
        active_yb_b=problem.active_yb_b,
        target_yb=problem.target_yb,
    )
    plan = (
        build_sparse_intersection_plan(block_problem) if build_plan
        else SparseIntersectionPlan(
            *(np.empty(0, dtype=np.int32) for _ in range(4))
        )
    )
    return SparseEdgeBlock(mass, block_problem, plan)


def stratified_sparse_edge_minibatch(
    problem: SparseGroupedProblem,
    intersection_plan: SparseIntersectionPlan,
    sample_size: int,
    rng: np.random.Generator,
    *,
    strata: int = 16,
) -> tuple[SparseGroupedProblem, SparseIntersectionPlan]:
    """Unbiased probability-rank-stratified AB-edge minibatch.

    Edges are divided into equal-count strata after sorting by target edge
    probability.  Sampling remains proportional to probability within each
    stratum, while every nonempty stratum receives draws.  Each stratum's
    empirical weights sum exactly to its true mass, so the combined objective
    and gradient remain unbiased without self-normalization.
    """

    edge_count = len(problem.edge_probability)
    if sample_size < 1 or strata < 1:
        raise ValueError("sample size and strata must be positive")
    strata = min(strata, edge_count, sample_size)
    ranked = np.argsort(problem.edge_probability, kind="stable")
    groups = np.array_split(ranked, strata)
    allocation = np.full(strata, sample_size // strata, dtype=np.int64)
    allocation[:sample_size % strata] += 1
    accumulated: dict[int, float] = {}
    for group, draws_in_group in zip(groups, allocation):
        group_probability = problem.edge_probability[group]
        group_mass = float(group_probability.sum())
        draws = rng.choice(
            group, size=int(draws_in_group), replace=True,
            p=group_probability / group_mass,
        )
        indices, counts = np.unique(draws, return_counts=True)
        unit_weight = group_mass / float(draws_in_group)
        for edge, count in zip(indices, counts):
            key = int(edge)
            accumulated[key] = (
                accumulated.get(key, 0.0) + float(count) * unit_weight
            )
    indices = np.fromiter(accumulated, dtype=np.int64)
    weights = np.fromiter(accumulated.values(), dtype=np.float64)
    return sparse_edge_distribution_with_plan(
        problem, intersection_plan, indices, weights
    )


def sparse_factorized_dual_evaluation(
    problem: SparseGroupedProblem,
    log_base_y: np.ndarray,
    correction_ya: np.ndarray,
    correction_yb: np.ndarray,
    *,
    intersection_plan: SparseIntersectionPlan | None = None,
    compute_certificate: bool = False,
    executor=None,
    intersection_shards: Sequence[tuple[int, int]] | None = None,
    layered_graph: LayeredIntersectionGraph | None = None,
    layered_checkpoint: int | None = None,
    margin_workers: int = 1,
) -> SparseDualEvaluation:
    """Evaluate the conditional dual for a full or sampled edge law.

    Sampling changes only the expectation of ``log Z_ab`` and its model
    sufficient statistics.  The target linear terms are kept exact; this is
    what makes the sampled gradient unbiased for the complete dual.
    """

    lb = np.asarray(log_base_y, dtype=np.float64)
    c1 = np.asarray(correction_ya, dtype=np.float64)
    c2 = np.asarray(correction_yb, dtype=np.float64)
    if lb.shape != problem.target_y.shape:
        raise ValueError("baseline shape does not match target_y")
    if c1.shape != problem.target_ya.shape:
        raise ValueError("first correction shape does not match target_ya")
    if c2.shape != problem.target_yb.shape:
        raise ValueError("second correction shape does not match target_yb")
    if (layered_graph is None) != (layered_checkpoint is None):
        raise ValueError("layered graph and checkpoint must be supplied together")
    if layered_graph is not None and intersection_plan is not None:
        raise ValueError("choose either an intersection plan or layered graph")
    if layered_graph is not None:
        margins = sparse_factorized_margins_layered(
            problem, layered_graph, layered_checkpoint,
            lb, c1, c2, workers=margin_workers,
        )
    else:
        plan = intersection_plan or build_sparse_intersection_plan(problem)
        margins = sparse_factorized_margins(
            problem, plan, lb, c1, c2,
            executor=executor,
            intersection_shards=intersection_shards,
        )
    normalized_base = lb - logsumexp(lb)
    objective = (
        float(problem.edge_probability @ margins.log_normalizer)
        - float(problem.target_y @ normalized_base)
        - float(problem.target_ya @ c1)
        - float(problem.target_yb @ c2)
    )
    residual_y = residual_ya = residual_yb = certificate = None
    if compute_certificate:
        v = problem.vocabulary_size
        pa = np.bincount(
            problem.edge_a, weights=problem.edge_probability, minlength=v
        )
        pb = np.bincount(
            problem.edge_b, weights=problem.edge_probability, minlength=v
        )
        current_a = np.bincount(
            problem.active_ya_a, weights=margins.active_ya, minlength=v
        )
        current_b = np.bincount(
            problem.active_yb_b, weights=margins.active_yb, minlength=v
        )
        target_a = np.bincount(
            problem.active_ya_a, weights=problem.target_ya, minlength=v
        )
        target_b = np.bincount(
            problem.active_yb_b, weights=problem.target_yb, minlength=v
        )
        residual_y = float(np.abs(
            margins.target_y - problem.target_y
        ).sum())
        residual_ya = float(np.abs(
            margins.active_ya - problem.target_ya
        ).sum() + np.abs((pa - current_a) - (pa - target_a)).sum())
        residual_yb = float(np.abs(
            margins.active_yb - problem.target_yb
        ).sum() + np.abs((pb - current_b) - (pb - target_b)).sum())
        certificate = max(residual_y, residual_ya, residual_yb)
    return SparseDualEvaluation(
        objective=objective,
        gradient_y=margins.target_y - problem.target_y,
        gradient_ya=margins.active_ya - problem.target_ya,
        gradient_yb=margins.active_yb - problem.target_yb,
        certificate=certificate,
        residual_y_l1=residual_y,
        residual_ya_l1=residual_ya,
        residual_yb_l1=residual_yb,
    )


def sparse_factorized_dual_hessian_product(
    problem: SparseGroupedProblem,
    log_base_y: np.ndarray,
    correction_ya: np.ndarray,
    correction_yb: np.ndarray,
    direction: np.ndarray,
    *,
    intersection_plan: SparseIntersectionPlan | None = None,
) -> np.ndarray:
    """Apply the exact dual Hessian using directional factor algebra.

    This is the forward directional derivative of
    :func:`sparse_factorized_margins`.  It forms neither a Hessian nor a
    dense ``V x V`` table and is the standard primitive required by a
    truncated-Newton/Newton--CG solver.
    """

    v = problem.vocabulary_size
    n1 = len(problem.target_ya)
    n2 = len(problem.target_yb)
    vector = np.asarray(direction, dtype=np.float64)
    if vector.shape != (v + n1 + n2,):
        raise ValueError("Hessian direction has the wrong shape")
    d0 = vector[:v]
    d1 = vector[v:v + n1]
    d2 = vector[v + n1:]
    plan = intersection_plan or build_sparse_intersection_plan(problem)

    log_base = np.asarray(log_base_y, dtype=np.float64)
    log_base = log_base - logsumexp(log_base)
    base = np.exp(log_base)
    # The normalized baseline has one global gauge.  Differentiating its
    # softmax representation removes that gauge exactly.
    dbase = base * (d0 - float(base @ d0))
    r1 = np.expm1(np.asarray(correction_ya, dtype=np.float64))
    r2 = np.expm1(np.asarray(correction_yb, dtype=np.float64))
    dr1 = (1.0 + r1) * d1
    dr2 = (1.0 + r2) * d2

    a1 = problem.active_ya_a
    y1 = problem.active_ya_y
    b2 = problem.active_yb_b
    y2 = problem.active_yb_y
    s1 = np.bincount(a1, weights=base[y1] * r1, minlength=v)
    s2 = np.bincount(b2, weights=base[y2] * r2, minlength=v)
    ds1 = np.bincount(
        a1, weights=dbase[y1] * r1 + base[y1] * dr1, minlength=v
    )
    ds2 = np.bincount(
        b2, weights=dbase[y2] * r2 + base[y2] * dr2, minlength=v
    )

    edge = plan.edge - plan.edge_offset
    py = plan.target_y
    i1 = plan.correction_ya
    i2 = plan.correction_yb
    cross_weight = base[py] * r1[i1] * r2[i2]
    dcross_weight = (
        dbase[py] * r1[i1] * r2[i2]
        + base[py] * dr1[i1] * r2[i2]
        + base[py] * r1[i1] * dr2[i2]
    )
    edge_count = len(problem.edge_probability)
    cross = np.bincount(edge, weights=cross_weight, minlength=edge_count)
    dcross = np.bincount(
        edge, weights=dcross_weight, minlength=edge_count
    )
    z = 1.0 + s1[problem.edge_a] + s2[problem.edge_b] + cross
    dz = ds1[problem.edge_a] + ds2[problem.edge_b] + dcross
    edge_mass = problem.edge_probability / z
    dedge_mass = -edge_mass * dz / z
    row = np.bincount(problem.edge_a, weights=edge_mass, minlength=v)
    col = np.bincount(problem.edge_b, weights=edge_mass, minlength=v)
    drow = np.bincount(problem.edge_a, weights=dedge_mass, minlength=v)
    dcol = np.bincount(problem.edge_b, weights=dedge_mass, minlength=v)

    e1 = 1.0 + r1
    e2 = 1.0 + r2
    dmargin1 = (
        (dbase[y1] * e1 + base[y1] * dr1) * row[a1]
        + base[y1] * e1 * drow[a1]
    )
    dmargin2 = (
        (dbase[y2] * e2 + base[y2] * dr2) * col[b2]
        + base[y2] * e2 * dcol[b2]
    )
    common = base[py] * edge_mass[edge]
    dcommon = (
        dbase[py] * edge_mass[edge]
        + base[py] * dedge_mass[edge]
    )
    dmargin1 += np.bincount(
        i1,
        weights=(
            dcommon * e1[i1] * r2[i2]
            + common * dr1[i1] * r2[i2]
            + common * e1[i1] * dr2[i2]
        ),
        minlength=n1,
    )
    dmargin2 += np.bincount(
        i2,
        weights=(
            dcommon * e2[i2] * r1[i1]
            + common * dr2[i2] * r1[i1]
            + common * e2[i2] * dr1[i1]
        ),
        minlength=n2,
    )

    dtarget = dbase * float(edge_mass.sum())
    dtarget += base * float(dedge_mass.sum())
    dtarget += np.bincount(
        y1,
        weights=(
            (dbase[y1] * r1 + base[y1] * dr1) * row[a1]
            + base[y1] * r1 * drow[a1]
        ),
        minlength=v,
    )
    dtarget += np.bincount(
        y2,
        weights=(
            (dbase[y2] * r2 + base[y2] * dr2) * col[b2]
            + base[y2] * r2 * dcol[b2]
        ),
        minlength=v,
    )
    dtarget += np.bincount(
        py,
        weights=(
            dcommon * r1[i1] * r2[i2]
            + common * dr1[i1] * r2[i2]
            + common * r1[i1] * dr2[i2]
        ),
        minlength=v,
    )
    return np.concatenate([dtarget, dmargin1, dmargin2])


def sparse_factorized_dual_hessian_product_layered_reference(
    problem: SparseGroupedProblem,
    graph: LayeredIntersectionGraph,
    checkpoint: int,
    log_base_y: np.ndarray,
    correction_ya: np.ndarray,
    correction_yb: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Apply the layered dual Hessian in one verified reference traversal.

    This mirrors :func:`sparse_factorized_margins_layered_reference` and
    differentiates its algebra analytically.  Edges whose expanded
    normalizer is cancellation-prone are differentiated directly in the
    positive log domain.  It is the correctness oracle for the native
    parallel kernel, not the eventual production implementation.
    """

    v = problem.vocabulary_size
    n1, n2 = len(problem.target_ya), len(problem.target_yb)
    vector = np.asarray(direction, dtype=np.float64)
    if vector.shape != (v + n1 + n2,):
        raise ValueError("Hessian direction has the wrong shape")
    if not 0 <= checkpoint < graph.layers:
        raise ValueError("invalid layered checkpoint")
    d0, d1, d2 = vector[:v], vector[v:v + n1], vector[v + n1:]
    log_base = np.asarray(log_base_y, dtype=np.float64)
    log_base = log_base - logsumexp(log_base)
    base = np.exp(log_base)
    centered_d0 = d0 - float(base @ d0)
    dbase = base * centered_d0
    r1 = np.expm1(np.asarray(correction_ya, dtype=np.float64))
    r2 = np.expm1(np.asarray(correction_yb, dtype=np.float64))
    dr1, dr2 = (1.0 + r1) * d1, (1.0 + r2) * d2
    y1, a1 = problem.active_ya_y, problem.active_ya_a
    y2, b2 = problem.active_yb_y, problem.active_yb_b

    s1 = np.bincount(a1, weights=base[y1] * r1, minlength=v)
    s2 = np.bincount(b2, weights=base[y2] * r2, minlength=v)
    ds1 = np.bincount(
        a1, weights=dbase[y1] * r1 + base[y1] * dr1, minlength=v,
    )
    ds2 = np.bincount(
        b2, weights=dbase[y2] * r2 + base[y2] * dr2, minlength=v,
    )
    edge_count = len(problem.edge_probability)
    cross = np.zeros(edge_count)
    dcross = np.zeros(edge_count)
    active_layers = []
    for depth in range(checkpoint + 1):
        ptr = graph.row_ptr[depth]
        first = np.repeat(
            np.arange(n1, dtype=np.int32), np.diff(ptr[:n1 + 1])
        )
        active_edges = int(ptr[n1])
        second = graph.correction_yb[depth][:active_edges]
        edge = graph.edge_ab[depth][:active_edges]
        active_layers.append((first, second, edge))
        y = y1[first]
        cross += np.bincount(
            edge, weights=base[y] * r1[first] * r2[second],
            minlength=edge_count,
        )
        dcross += np.bincount(
            edge,
            weights=(
                dbase[y] * r1[first] * r2[second]
                + base[y] * dr1[first] * r2[second]
                + base[y] * r1[first] * dr2[second]
            ),
            minlength=edge_count,
        )

    z = 1.0 + s1[problem.edge_a] + s2[problem.edge_b] + cross
    dz = ds1[problem.edge_a] + ds2[problem.edge_b] + dcross
    scale = (
        1.0 + np.abs(s1[problem.edge_a]) + np.abs(s2[problem.edge_b])
        + np.abs(cross)
    )
    unstable = ~np.isfinite(z) | (z <= 0.0) | (
        scale > 1e10 * np.maximum(np.abs(z), np.finfo(float).tiny)
    )
    edge_mass = np.zeros(edge_count)
    dedge_mass = np.zeros(edge_count)
    stable = ~unstable
    edge_mass[stable] = problem.edge_probability[stable] / z[stable]
    dedge_mass[stable] = -edge_mass[stable] * dz[stable] / z[stable]
    row = np.bincount(problem.edge_a, weights=edge_mass, minlength=v)
    col = np.bincount(problem.edge_b, weights=edge_mass, minlength=v)
    drow = np.bincount(problem.edge_a, weights=dedge_mass, minlength=v)
    dcol = np.bincount(problem.edge_b, weights=dedge_mass, minlength=v)

    e1, e2 = 1.0 + r1, 1.0 + r2
    dm1 = (
        (dbase[y1] * e1 + base[y1] * dr1) * row[a1]
        + base[y1] * e1 * drow[a1]
    )
    dm2 = (
        (dbase[y2] * e2 + base[y2] * dr2) * col[b2]
        + base[y2] * e2 * dcol[b2]
    )
    dy = dbase * float(edge_mass.sum()) + base * float(dedge_mass.sum())
    dy += np.bincount(
        y1, weights=(dbase[y1] * r1 + base[y1] * dr1) * row[a1]
        + base[y1] * r1 * drow[a1], minlength=v,
    )
    dy += np.bincount(
        y2, weights=(dbase[y2] * r2 + base[y2] * dr2) * col[b2]
        + base[y2] * r2 * dcol[b2], minlength=v,
    )
    for first, second, edge in active_layers:
        y = y1[first]
        common = base[y] * edge_mass[edge]
        dcommon = dbase[y] * edge_mass[edge] + base[y] * dedge_mass[edge]
        dm1 += np.bincount(
            first, weights=dcommon * e1[first] * r2[second]
            + common * dr1[first] * r2[second]
            + common * e1[first] * dr2[second], minlength=n1,
        )
        dm2 += np.bincount(
            second, weights=dcommon * e2[second] * r1[first]
            + common * dr2[second] * r1[first]
            + common * e2[second] * dr1[first], minlength=n2,
        )
        dy += np.bincount(
            y, weights=dcommon * r1[first] * r2[second]
            + common * dr1[first] * r2[second]
            + common * r1[first] * dr2[second], minlength=v,
        )

    # Replace each omitted unstable edge by the derivative of its direct
    # positive conditional distribution.
    order1 = np.argsort(a1, kind="stable")
    order2 = np.argsort(b2, kind="stable")
    ptr1 = np.r_[0, np.cumsum(np.bincount(a1, minlength=v), dtype=np.int64)]
    ptr2 = np.r_[0, np.cumsum(np.bincount(b2, minlength=v), dtype=np.int64)]
    for edge in np.flatnonzero(unstable):
        selected1 = order1[ptr1[problem.edge_a[edge]]:ptr1[problem.edge_a[edge] + 1]]
        selected2 = order2[ptr2[problem.edge_b[edge]]:ptr2[problem.edge_b[edge] + 1]]
        score = log_base.copy()
        directional_score = centered_d0.copy()
        score[y1[selected1]] += np.asarray(correction_ya)[selected1]
        score[y2[selected2]] += np.asarray(correction_yb)[selected2]
        directional_score[y1[selected1]] += d1[selected1]
        directional_score[y2[selected2]] += d2[selected2]
        probability = np.exp(score - logsumexp(score))
        joint = problem.edge_probability[edge] * probability
        djoint = joint * (
            directional_score - float(probability @ directional_score)
        )
        dy += djoint
        dm1[selected1] += djoint[y1[selected1]]
        dm2[selected2] += djoint[y2[selected2]]
    return np.concatenate([dy, dm1, dm2])


def sparse_factorized_dual_hessian_product_layered(
    problem: SparseGroupedProblem,
    graph: LayeredIntersectionGraph,
    checkpoint: int,
    log_base_y: np.ndarray,
    correction_ya: np.ndarray,
    correction_yb: np.ndarray,
    direction: np.ndarray,
    *,
    workers: int = 1,
) -> np.ndarray:
    """Apply the layered Hessian with the native parallel traversal."""

    if not (
        _graphical_margin_c is not None
        and hasattr(_graphical_margin_c, "fused_hessian_layered")
    ):
        return sparse_factorized_dual_hessian_product_layered_reference(
            problem, graph, checkpoint, log_base_y, correction_ya,
            correction_yb, direction,
        )
    v = problem.vocabulary_size
    n1, n2 = len(problem.target_ya), len(problem.target_yb)
    vector = np.asarray(direction, dtype=np.float64)
    if vector.shape != (v + n1 + n2,):
        raise ValueError("Hessian direction has the wrong shape")
    d0, d1, d2 = vector[:v], vector[v:v + n1], vector[v + n1:]
    log_base = np.asarray(log_base_y, dtype=np.float64)
    log_base = log_base - logsumexp(log_base)
    base = np.exp(log_base)
    centered_d0 = d0 - float(base @ d0)
    dbase = base * centered_d0
    r1 = np.expm1(np.asarray(correction_ya, dtype=np.float64))
    r2 = np.expm1(np.asarray(correction_yb, dtype=np.float64))
    dr1, dr2 = (1.0 + r1) * d1, (1.0 + r2) * d2
    dy, dm1, dm2, unstable = _graphical_margin_c.fused_hessian_layered(
        np.ascontiguousarray(base), np.ascontiguousarray(dbase),
        np.ascontiguousarray(r1), np.ascontiguousarray(dr1),
        np.ascontiguousarray(r2), np.ascontiguousarray(dr2),
        np.ascontiguousarray(problem.edge_a, dtype=np.int32),
        np.ascontiguousarray(problem.edge_b, dtype=np.int32),
        np.ascontiguousarray(problem.edge_probability, dtype=np.float64),
        np.ascontiguousarray(problem.active_ya_y, dtype=np.int32),
        np.ascontiguousarray(problem.active_ya_a, dtype=np.int32),
        np.ascontiguousarray(problem.active_yb_y, dtype=np.int32),
        np.ascontiguousarray(problem.active_yb_b, dtype=np.int32),
        tuple(np.ascontiguousarray(x, dtype=np.int64) for x in graph.row_ptr),
        tuple(np.ascontiguousarray(x, dtype=np.int32) for x in graph.correction_yb),
        tuple(np.ascontiguousarray(x, dtype=np.int32) for x in graph.edge_ab),
        checkpoint, workers,
    )
    unstable_edges = np.flatnonzero(unstable)
    if len(unstable_edges):
        y1, a1 = problem.active_ya_y, problem.active_ya_a
        y2, b2 = problem.active_yb_y, problem.active_yb_b
        order1 = np.argsort(a1, kind="stable")
        order2 = np.argsort(b2, kind="stable")
        ptr1 = np.r_[
            0, np.cumsum(np.bincount(a1, minlength=v), dtype=np.int64)
        ]
        ptr2 = np.r_[
            0, np.cumsum(np.bincount(b2, minlength=v), dtype=np.int64)
        ]
    for edge in unstable_edges:
        a, b = problem.edge_a[edge], problem.edge_b[edge]
        selected1 = order1[ptr1[a]:ptr1[a + 1]]
        selected2 = order2[ptr2[b]:ptr2[b + 1]]
        score = log_base.copy()
        directional_score = centered_d0.copy()
        score[y1[selected1]] += np.asarray(correction_ya)[selected1]
        score[y2[selected2]] += np.asarray(correction_yb)[selected2]
        directional_score[y1[selected1]] += d1[selected1]
        directional_score[y2[selected2]] += d2[selected2]
        probability = np.exp(score - logsumexp(score))
        joint = problem.edge_probability[edge] * probability
        djoint = joint * (
            directional_score - float(probability @ directional_score)
        )
        dy += djoint
        dm1[selected1] += djoint[y1[selected1]]
        dm2[selected2] += djoint[y2[selected2]]
    return np.concatenate([dy, dm1, dm2])


def diagnose_sparse_factorized_normalizers(
    problem: SparseGroupedProblem,
    plan: SparseIntersectionPlan,
    log_base_y: np.ndarray,
    correction_ya: np.ndarray,
    correction_yb: np.ndarray,
) -> dict[str, float | int]:
    """Describe the worst cancellation in the expanded normalizers."""

    v = problem.vocabulary_size
    log_base = np.asarray(log_base_y, dtype=np.float64)
    log_base -= logsumexp(log_base)
    base = np.exp(log_base)
    c1 = np.asarray(correction_ya, dtype=np.float64)
    c2 = np.asarray(correction_yb, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        r1 = np.expm1(c1)
        r2 = np.expm1(c2)
    s1 = np.bincount(
        problem.active_ya_a,
        weights=base[problem.active_ya_y] * r1,
        minlength=v,
    )
    s2 = np.bincount(
        problem.active_yb_b,
        weights=base[problem.active_yb_y] * r2,
        minlength=v,
    )
    edge = plan.edge - plan.edge_offset
    cross = np.bincount(
        edge,
        weights=(
            base[plan.target_y]
            * r1[plan.correction_ya]
            * r2[plan.correction_yb]
        ),
        minlength=len(problem.edge_probability),
    )
    first_term = s1[problem.edge_a]
    second_term = s2[problem.edge_b]
    z = 1.0 + first_term + second_term + cross
    scale = 1.0 + np.abs(first_term) + np.abs(second_term) + np.abs(cross)
    ratio = np.divide(
        scale, np.abs(z), out=np.full_like(scale, np.inf),
        where=np.isfinite(z) & (z != 0.0),
    )
    bad = np.flatnonzero(~np.isfinite(z) | (z <= 0.0))
    worst = int(bad[0] if len(bad) else np.argmax(ratio))
    a = int(problem.edge_a[worst])
    b = int(problem.edge_b[worst])
    selected1 = np.flatnonzero(problem.active_ya_a == a)
    selected2 = np.flatnonzero(problem.active_yb_b == b)
    map1 = {
        int(problem.active_ya_y[index]): float(c1[index])
        for index in selected1
    }
    map2 = {
        int(problem.active_yb_y[index]): float(c2[index])
        for index in selected2
    }
    union = np.array(sorted(set(map1) | set(map2)), dtype=np.int64)
    active_logs = np.array([
        log_base[y] + map1.get(int(y), 0.0) + map2.get(int(y), 0.0)
        for y in union
    ])
    background = max(0.0, 1.0 - float(base[union].sum()))
    pieces = active_logs
    if background > 0.0:
        pieces = np.append(pieces, np.log(background))
    direct_log_z = float(logsumexp(pieces))
    return {
        "bad_normalizer_count": len(bad),
        "worst_edge": worst,
        "worst_a": a,
        "worst_b": b,
        "term_one": 1.0,
        "term_first": float(first_term[worst]),
        "term_second": float(second_term[worst]),
        "term_cross": float(cross[worst]),
        "expanded_z": float(z[worst]),
        "cancellation_ratio": float(ratio[worst]),
        "direct_log_z": direct_log_z,
        "direct_z": float(np.exp(direct_log_z)),
        "active_union_size": len(union),
    }


def sparse_grouped_newton_cg(
    problem: SparseGroupedProblem,
    *,
    log_base_y: np.ndarray,
    correction_ya: np.ndarray,
    correction_yb: np.ndarray,
    max_iterations: int = 200,
    tolerance: float = 1e-5,
    jacobi_precondition: bool = True,
    precondition_floor: float = 1e-8,
    precondition_max_scale: float = 10.0,
    max_hessian_products: int | None = None,
    pair_slack_precision: float = float("inf"),
    pair_slack_variance_ya: np.ndarray | None = None,
    pair_slack_variance_yb: np.ndarray | None = None,
    layered_graph: LayeredIntersectionGraph | None = None,
    layered_checkpoint: int | None = None,
    margin_workers: int = 1,
) -> SparseGroupedResult:
    """Fit the exact or statistically relaxed dual with Newton--CG.

    The Hessian is never materialized.  ``scipy``'s truncated conjugate
    gradient iteration receives exact products from
    :func:`sparse_factorized_dual_hessian_product`.
    """

    if max_iterations < 1 or tolerance <= 0.0:
        raise ValueError("invalid Newton-CG convergence settings")
    if precondition_floor <= 0.0 or precondition_max_scale < 1.0:
        raise ValueError("invalid Newton-CG preconditioner settings")
    if max_hessian_products is not None and max_hessian_products < 1:
        raise ValueError("max_hessian_products must be positive")
    if pair_slack_precision <= 0.0 or np.isnan(pair_slack_precision):
        raise ValueError("pair slack precision must be positive")
    if (layered_graph is None) != (layered_checkpoint is None):
        raise ValueError("layered graph and checkpoint must be supplied together")
    v = problem.vocabulary_size
    n1 = len(problem.target_ya)
    n2 = len(problem.target_yb)
    full_initial = np.concatenate([
        np.asarray(log_base_y, dtype=np.float64),
        np.asarray(correction_ya, dtype=np.float64),
        np.asarray(correction_yb, dtype=np.float64),
    ])
    if full_initial.shape != (v + n1 + n2,):
        raise ValueError("initial Newton-CG factors have the wrong shape")
    plan = None if layered_graph is not None else build_sparse_intersection_plan(problem)
    relaxed = np.isfinite(pair_slack_precision)
    variance_ya = np.ones(n1, dtype=np.float64)
    variance_yb = np.ones(n2, dtype=np.float64)
    for supplied, destination, name in (
        (pair_slack_variance_ya, variance_ya, "YA slack variance"),
        (pair_slack_variance_yb, variance_yb, "YB slack variance"),
    ):
        if supplied is not None:
            values = np.asarray(supplied, dtype=np.float64)
            if values.shape != destination.shape or not np.isfinite(values).all():
                raise ValueError(f"invalid {name}")
            destination[:] = values
    if np.any(variance_ya <= 0.0) or np.any(variance_yb <= 0.0):
        raise ValueError("pair slack variances must be positive")
    regularizer_diagonal = np.concatenate([
        np.zeros(v), variance_ya, variance_yb,
    ]) / pair_slack_precision if relaxed else np.zeros_like(full_initial)

    # Remove the same exact gauges used by the L-BFGS solver.  Besides the
    # global baseline constant, a correction row containing all targets has
    # one constant gauge because it appears in every conditional outcome.
    free = np.ones(len(full_initial), dtype=bool)
    free[0] = False
    if not relaxed:
        for states, offset in (
            (problem.active_ya_a, v),
            (problem.active_yb_b, v + n1),
        ):
            counts = np.bincount(states, minlength=v)
            for state in np.flatnonzero(counts == v):
                free[offset + np.flatnonzero(states == state)[0]] = False
    fixed = full_initial.copy()

    # Optimize in diagonally whitened coordinates.  The entries below are
    # the Bernoulli/Fisher diagonal for each target feature, with the active
    # pair features conditioned on their context.  It is a cheap Jacobi
    # approximation to the exact Hessian diagonal and uses only the target
    # margins already resident in the problem.  Trust-NCG still receives
    # exact Hessian-vector products; this is solely a linear change of
    # variables that removes the worst rare-feature scaling.
    scale = np.ones_like(full_initial)
    if jacobi_precondition:
        pa = np.bincount(
            problem.edge_a,
            weights=problem.edge_probability,
            minlength=v,
        )
        pb = np.bincount(
            problem.edge_b,
            weights=problem.edge_probability,
            minlength=v,
        )
        conditional_ya = np.divide(
            problem.target_ya,
            pa[problem.active_ya_a],
            out=np.zeros_like(problem.target_ya),
            where=pa[problem.active_ya_a] > 0.0,
        )
        conditional_yb = np.divide(
            problem.target_yb,
            pb[problem.active_yb_b],
            out=np.zeros_like(problem.target_yb),
            where=pb[problem.active_yb_b] > 0.0,
        )
        diagonal = np.concatenate([
            problem.target_y * (1.0 - problem.target_y),
            problem.target_ya * np.maximum(0.0, 1.0 - conditional_ya),
            problem.target_yb * np.maximum(0.0, 1.0 - conditional_yb),
        ])
        diagonal += regularizer_diagonal
        scale = np.minimum(
            precondition_max_scale,
            1.0 / np.sqrt(np.maximum(diagonal, precondition_floor)),
        )
    best_parameters = full_initial.copy()
    best_certificate = float("inf")
    best_stationarity = float("inf")
    minimum_stationarity = float("inf")
    best_objective = float("inf")
    last_stationarity = float("inf")
    last_objective = float("inf")
    best_residuals = (float("inf"),) * 3
    evaluations = 0
    hessian_products = 0

    class _ConvergenceReached(Exception):
        pass

    class _HessianBudgetReached(Exception):
        pass

    def expand(reduced: np.ndarray) -> np.ndarray:
        full = fixed.copy()
        full[free] += scale[free] * reduced
        return full

    def objective_gradient(reduced: np.ndarray):
        nonlocal best_parameters, best_certificate, best_stationarity
        nonlocal minimum_stationarity
        nonlocal best_objective, best_residuals, evaluations
        nonlocal last_stationarity, last_objective
        full = expand(reduced)
        evaluation = sparse_factorized_dual_evaluation(
            problem,
            full[:v], full[v:v + n1], full[v + n1:],
            intersection_plan=plan,
            compute_certificate=True,
            layered_graph=layered_graph,
            layered_checkpoint=layered_checkpoint,
            margin_workers=margin_workers,
        )
        evaluations += 1
        if relaxed:
            evaluation = SparseDualEvaluation(
                objective=float(evaluation.objective + 0.5 * np.dot(
                    regularizer_diagonal, full * full,
                )),
                gradient_y=evaluation.gradient_y,
                gradient_ya=(evaluation.gradient_ya
                             + regularizer_diagonal[v:v + n1] * full[v:v + n1]),
                gradient_yb=(evaluation.gradient_yb
                             + regularizer_diagonal[v + n1:] * full[v + n1:]),
                certificate=evaluation.certificate,
                residual_y_l1=evaluation.residual_y_l1,
                residual_ya_l1=evaluation.residual_ya_l1,
                residual_yb_l1=evaluation.residual_yb_l1,
            )
        certificate = float(evaluation.certificate)
        current_stationarity = float(np.max(np.abs(evaluation.gradient())))
        minimum_stationarity = min(minimum_stationarity, current_stationarity)
        last_stationarity = current_stationarity
        last_objective = float(evaluation.objective)
        selection = float(evaluation.objective) if relaxed else certificate
        best_selection = best_objective if relaxed else best_certificate
        if selection < best_selection:
            best_certificate = certificate
            best_stationarity = current_stationarity
            best_objective = float(evaluation.objective)
            best_parameters = full.copy()
            best_residuals = (
                float(evaluation.residual_y_l1),
                float(evaluation.residual_ya_l1),
                float(evaluation.residual_yb_l1),
            )
        return evaluation.objective, scale[free] * evaluation.gradient()[free]

    def hessian_product(reduced: np.ndarray, direction: np.ndarray):
        nonlocal hessian_products
        if (
            max_hessian_products is not None
            and hessian_products >= max_hessian_products
        ):
            raise _HessianBudgetReached
        hessian_products += 1
        full = expand(reduced)
        full_direction = np.zeros_like(full)
        full_direction[free] = scale[free] * direction
        if layered_graph is None:
            product = sparse_factorized_dual_hessian_product(
                problem,
                full[:v], full[v:v + n1], full[v + n1:],
                full_direction,
                intersection_plan=plan,
            )
            product += regularizer_diagonal * full_direction
        else:
            product = sparse_factorized_dual_hessian_product_layered(
                problem, layered_graph, layered_checkpoint,
                full[:v], full[v:v + n1], full[v + n1:], full_direction,
                workers=margin_workers,
            )
            product += regularizer_diagonal * full_direction
        return scale[free] * product[free]

    accepted_iterations = 0

    def stop_at_convergence(_reduced):
        nonlocal accepted_iterations
        accepted_iterations += 1
        value = best_stationarity if relaxed else best_certificate
        if relaxed:
            value = minimum_stationarity
        if value <= tolerance:
            raise _ConvergenceReached

    initial_reduced = np.zeros(np.count_nonzero(free), dtype=np.float64)
    objective_gradient(initial_reduced)
    initial_stopping_value = (
        minimum_stationarity if relaxed else best_certificate
    )
    if initial_stopping_value <= tolerance:
        iterations = 0
    else:
        try:
            optimized = minimize(
                objective_gradient,
                initial_reduced,
                method="trust-ncg",
                jac=True,
                hessp=hessian_product,
                callback=stop_at_convergence,
                options={"maxiter": max_iterations, "gtol": tolerance / 10.0},
            )
            iterations = int(optimized.nit)
        except _ConvergenceReached:
            iterations = accepted_iterations
        except _HessianBudgetReached:
            iterations = accepted_iterations
    return SparseGroupedResult(
        best_parameters[:v],
        best_parameters[v:v + n1],
        best_parameters[v + n1:],
        iterations,
        best_residuals[1],
        best_residuals[2],
        best_residuals[0],
        (minimum_stationarity if relaxed else best_certificate) <= tolerance,
        evaluations,
        hessian_products,
        best_stationarity,
        best_objective,
        last_stationarity,
        last_objective,
    )


def exact_sparse_dual_wolfe(
    problem: SparseGroupedProblem,
    log_base_y: np.ndarray,
    correction_ya: np.ndarray,
    correction_yb: np.ndarray,
    *,
    max_iterations: int = 1_000,
    tolerance: float = 1e-2,
    initial_step: float = 1.0,
    armijo: float = 1e-4,
    curvature: float = 0.9,
    minimum_step: float = 1e-12,
    maximum_step: float = 1e12,
    margin_workers: int = 1,
    layered_graph: LayeredIntersectionGraph | None = None,
    layered_checkpoint: int | None = None,
    pair_slack_precision: float = float("inf"),
    pair_slack_variance_ya: np.ndarray | None = None,
    pair_slack_variance_yb: np.ndarray | None = None,
    progress_callback: Callable[[dict[str, float | int | bool]], None]
    | None = None,
) -> SparseLineSearchResult:
    """Minimize the exact or soft-margin dual with a strong Wolfe search.

    A finite ``pair_slack_precision`` solves the statistically relaxed primal

    ``KL(q || q0) + precision/2 * sum_i slack_i**2 / variance_i``.

    Its dual is the ordinary maximum-entropy dual plus
    ``sum_i variance_i * theta_i**2 / (2 * precision)``.  Thus incompatible
    finite-data YA and YB margins no longer send factors to infinity.  The Y
    margin remains hard: together with the fixed AB law it is always feasible.
    Infinite precision recovers exact margin matching.

    The first iteration starts from ``initial_step``.  Later iterations
    propose the Barzilai--Borwein spectral step.
    Every accepted step satisfies both sufficient decrease and the strong
    Wolfe curvature condition, preventing nearly flat spectral overshoots.
    Nonfinite factorized trials are rejected like ordinary failed trials.
    """

    if max_iterations < 0 or tolerance <= 0.0:
        raise ValueError("invalid line-search iteration limit or tolerance")
    if not 0.0 < armijo < 1.0:
        raise ValueError("Armijo constant must lie in (0, 1)")
    if not armijo < curvature < 1.0:
        raise ValueError("Wolfe curvature constant must lie in (armijo, 1)")
    if not 0.0 < minimum_step <= initial_step <= maximum_step:
        raise ValueError("invalid line-search step bounds")
    if (layered_graph is None) != (layered_checkpoint is None):
        raise ValueError("layered graph and checkpoint must be supplied together")
    if pair_slack_precision <= 0.0 or np.isnan(pair_slack_precision):
        raise ValueError("pair slack precision must be positive")

    first = len(log_base_y)
    second = first + len(correction_ya)
    parameters = np.concatenate([
        np.asarray(log_base_y, dtype=np.float64),
        np.asarray(correction_ya, dtype=np.float64),
        np.asarray(correction_yb, dtype=np.float64),
    ])
    parameters[:first] -= logsumexp(parameters[:first])
    evaluations = 0

    relaxed = np.isfinite(pair_slack_precision)
    variance_ya = np.ones(len(correction_ya), dtype=np.float64)
    variance_yb = np.ones(len(correction_yb), dtype=np.float64)
    for supplied, destination, name in (
        (pair_slack_variance_ya, variance_ya, "YA slack variance"),
        (pair_slack_variance_yb, variance_yb, "YB slack variance"),
    ):
        if supplied is not None:
            values = np.asarray(supplied, dtype=np.float64)
            if values.shape != destination.shape or not np.isfinite(values).all():
                raise ValueError(f"invalid {name}")
            destination[:] = values
    if np.any(variance_ya <= 0.0) or np.any(variance_yb <= 0.0):
        raise ValueError("pair slack variances must be positive")

    def regularize(
        value: SparseDualEvaluation, vector: np.ndarray,
    ) -> SparseDualEvaluation:
        if not relaxed:
            return value
        d1 = vector[first:second]
        d2 = vector[second:]
        scale = 1.0 / pair_slack_precision
        return SparseDualEvaluation(
            objective=float(value.objective + 0.5 * scale * (
                np.dot(variance_ya, d1 * d1)
                + np.dot(variance_yb, d2 * d2)
            )),
            gradient_y=value.gradient_y,
            gradient_ya=value.gradient_ya + scale * variance_ya * d1,
            gradient_yb=value.gradient_yb + scale * variance_yb * d2,
            certificate=value.certificate,
            residual_y_l1=value.residual_y_l1,
            residual_ya_l1=value.residual_ya_l1,
            residual_yb_l1=value.residual_yb_l1,
        )

    def evaluate(vector: np.ndarray) -> SparseDualEvaluation:
        nonlocal evaluations
        evaluations += 1
        value = sparse_factorized_dual_evaluation(
            problem, vector[:first], vector[first:second], vector[second:],
            compute_certificate=True,
            layered_graph=layered_graph,
            layered_checkpoint=layered_checkpoint,
            margin_workers=margin_workers,
        )
        return regularize(value, vector)

    current = evaluate(parameters)
    gradient = current.gradient()
    best_parameters = parameters.copy()
    best = current
    trace: list[dict[str, float | int | bool]] = []
    previous_parameters = None
    previous_gradient = None
    previous_step = initial_step

    def stationarity(value: SparseDualEvaluation) -> float:
        return float(np.max(np.abs(value.gradient())))

    best_stationarity = stationarity(current)

    for iteration in range(max_iterations + 1):
        record: dict[str, float | int | bool] = {
            "iteration": iteration,
            "objective": float(current.objective),
            "certificate": float(current.certificate),
            "gradient_l2": float(np.linalg.norm(gradient)),
            "stationarity": stationarity(current),
            "evaluations": evaluations,
            "factor_abs_max": float(np.max(np.abs(parameters))),
        }
        if iteration > 0:
            record["previous_accepted_step"] = float(previous_step)
        trace.append(record)
        if progress_callback is not None:
            progress_callback(record)
        stopping_value = (
            stationarity(current) if relaxed else float(current.certificate)
        )
        if stopping_value <= tolerance or iteration == max_iterations:
            break

        if previous_parameters is None:
            proposal = initial_step
        else:
            displacement = parameters - previous_parameters
            gradient_change = gradient - previous_gradient
            secant_curvature = float(displacement @ gradient_change)
            proposal = (
                float(displacement @ displacement) / secant_curvature
                if secant_curvature > 0.0 and np.isfinite(secant_curvature)
                else previous_step
            )
            proposal = float(np.clip(proposal, minimum_step, maximum_step))

        # Scaling the descent direction by the spectral proposal lets the
        # standard Wolfe search operate around the local curvature estimate.
        direction = -proposal * gradient
        cached_vector = None
        cached_evaluation = None

        def cached_trial(vector: np.ndarray):
            nonlocal cached_vector, cached_evaluation
            candidate = np.array(vector, dtype=np.float64, copy=True)
            candidate[:first] -= logsumexp(candidate[:first])
            if cached_vector is not None and np.array_equal(
                candidate, cached_vector
            ):
                return cached_evaluation
            try:
                value = evaluate(candidate)
            except (FloatingPointError, OverflowError):
                value = None
            if value is not None and not (
                np.isfinite(value.objective)
                and np.isfinite(value.certificate)
                and np.all(np.isfinite(value.gradient()))
            ):
                value = None
            cached_vector = candidate
            cached_evaluation = value
            return value

        def objective_at(vector: np.ndarray) -> float:
            value = cached_trial(vector)
            return float("inf") if value is None else float(value.objective)

        def gradient_at(vector: np.ndarray) -> np.ndarray:
            value = cached_trial(vector)
            return (
                np.zeros_like(parameters)
                if value is None else value.gradient()
            )

        alpha, _, _, _, _, _ = line_search(
            objective_at, gradient_at, parameters, direction,
            gfk=gradient, old_fval=current.objective,
            c1=armijo, c2=curvature,
            amax=maximum_step / proposal,
            maxiter=25,
        )
        if alpha is None or alpha * proposal < minimum_step:
            record["line_search_failed"] = True
            break
        step = float(alpha * proposal)
        candidate = parameters + alpha * direction
        candidate[:first] -= logsumexp(candidate[:first])
        candidate_value = cached_trial(candidate)
        if candidate_value is None:
            record["line_search_failed"] = True
            break
        record["accepted_step"] = float(step)
        previous_parameters = parameters
        previous_gradient = gradient
        previous_step = step
        parameters = candidate
        current = candidate_value
        gradient = current.gradient()
        current_stationarity = stationarity(current)
        if (
            (relaxed and current_stationarity < best_stationarity)
            or (not relaxed and current.certificate < best.certificate)
        ):
            best_parameters = parameters.copy()
            best = current
            best_stationarity = current_stationarity

    return SparseLineSearchResult(
        best_parameters[:first].copy(),
        best_parameters[first:second].copy(),
        best_parameters[second:].copy(),
        len(trace) - 1,
        evaluations,
        float(best.objective),
        float(best.certificate),
        bool((best_stationarity if relaxed else best.certificate) <= tolerance),
        tuple(trace),
        parameters[:first].copy(),
        parameters[first:second].copy(),
        parameters[second:].copy(),
        float(current.objective),
        float(current.certificate),
        float(best_stationarity),
        float(stationarity(current)),
    )


def stochastic_sparse_dual_approach(
    problem: SparseGroupedProblem,
    log_base_y: np.ndarray,
    correction_ya: np.ndarray,
    correction_yb: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float = 0.03,
    exact_interval: int = 50,
    seed: int = 0,
    trust_radius: float = 8.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    sampling: str = "iid",
    strata: int = 16,
    replicas: int = 1,
    stochastic_workers: int | None = None,
    edge_blocks: int = 256,
    variance_reduction: bool = False,
    certificate_tolerance: float | None = None,
    exact_margin_workers: int = 1,
    optimizer: str = "adam",
    minimum_learning_rate: float = 0.003,
    plateau_patience: int = 3,
    plateau_factor: float = 1.0 / 3.0,
    plateau_relative_threshold: float = 1e-4,
    bb_min_step: float = 1e-7,
    bb_max_step: float = 1.0,
    exact_layered_graph: LayeredIntersectionGraph | None = None,
    exact_layered_checkpoint: int | None = None,
    lazy_block_cache: int = 16,
    sampled_ab_major_graph: ABMajorIntersectionGraph | None = None,
    fused_ab_batch: bool = False,
    block_replica_schedule: str = "independent",
    persistent_reference_positions: bool = False,
    process_shards: int = 1,
    pair_slack_precision: float = float("inf"),
    pair_slack_variance_ya: np.ndarray | None = None,
    pair_slack_variance_yb: np.ndarray | None = None,
    exact_record_callback: Callable[
        [int, float, np.ndarray, np.ndarray, np.ndarray], None
    ] | None = None,
) -> SparseStochasticResult:
    """Use minibatch Adam only to approach the exact dual optimum.

    Exact evaluations select the returned iterate.  This routine never
    claims convergence; its output is intended as a warm start for the exact
    certified solver.
    """

    if steps < 0 or batch_size < 1 or exact_interval < 1 or replicas < 1:
        raise ValueError("invalid stochastic schedule")
    if stochastic_workers is None:
        stochastic_workers = replicas
    if stochastic_workers < 1:
        raise ValueError("stochastic_workers must be positive")
    if process_shards < 1:
        raise ValueError("process_shards must be positive")
    if process_shards > 1 and process_shards > replicas:
        raise ValueError("process_shards cannot exceed replicas")
    if process_shards > 1 and sampled_ab_major_graph is None:
        raise ValueError(
            "process shards require sampled_ab_major_graph"
        )
    if process_shards > 1 and fused_ab_batch:
        raise ValueError(
            "process shards do not support fused_ab_batch"
        )
    if learning_rate <= 0.0 or trust_radius <= 0.0:
        raise ValueError("learning rate and trust radius must be positive")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("Adam decay factors must lie in [0, 1)")
    if sampling not in ("iid", "stratified", "blocks"):
        raise ValueError("sampling must be 'iid', 'stratified' or 'blocks'")
    if block_replica_schedule not in ("independent", "systematic"):
        raise ValueError(
            "block_replica_schedule must be 'independent' or 'systematic'"
        )
    if edge_blocks < 1:
        raise ValueError("edge_blocks must be positive")
    if lazy_block_cache < 1:
        raise ValueError("lazy_block_cache must be positive")
    if certificate_tolerance is not None and certificate_tolerance <= 0.0:
        raise ValueError("certificate_tolerance must be positive")
    if exact_margin_workers < 1:
        raise ValueError("exact_margin_workers must be positive")
    if optimizer not in ("adam", "adam_cosine", "adam_plateau", "svrg_bb"):
        raise ValueError(
            "optimizer must be 'adam', 'adam_cosine', 'adam_plateau', "
            "or 'svrg_bb'"
        )
    if not 0.0 < minimum_learning_rate <= learning_rate:
        raise ValueError("minimum learning rate must lie in (0, learning_rate]")
    if plateau_patience < 1 or not 0.0 < plateau_factor < 1.0:
        raise ValueError("invalid plateau schedule")
    if not 0.0 <= plateau_relative_threshold < 1.0:
        raise ValueError("invalid plateau relative threshold")
    if not 0.0 < bb_min_step <= bb_max_step:
        raise ValueError("invalid BB step bounds")
    if (exact_layered_graph is None) != (exact_layered_checkpoint is None):
        raise ValueError(
            "exact layered graph and checkpoint must be supplied together"
        )
    if sampled_ab_major_graph is not None and sampling != "blocks":
        raise ValueError("AB-major sampled graph requires block sampling")
    if sampled_ab_major_graph is not None and exact_layered_checkpoint is None:
        raise ValueError("AB-major sampled graph requires checkpoint depth")
    if pair_slack_precision <= 0.0 or np.isnan(pair_slack_precision):
        raise ValueError("pair slack precision must be positive")

    lb = np.array(log_base_y, dtype=np.float64, copy=True)
    c1 = np.array(correction_ya, dtype=np.float64, copy=True)
    c2 = np.array(correction_yb, dtype=np.float64, copy=True)
    origin = np.concatenate([lb, c1, c2])
    parameters = origin.copy()
    process_context = None
    shared_parameters_owner = None
    if process_shards > 1:
        # Prototype execution engine: forked workers inherit the read-only
        # memory-mapped graph, while the frequently changing parameters stay
        # in one shared buffer rather than crossing a Python queue.
        process_context = mp.get_context("fork")
        shared_parameters_owner = process_context.RawArray(
            "d", len(parameters)
        )
        shared_parameters = np.frombuffer(
            shared_parameters_owner, dtype=np.float64
        )
        shared_parameters[:] = parameters
        parameters = shared_parameters
    first = len(lb)
    second = first + len(c1)
    relaxed = np.isfinite(pair_slack_precision)
    variance_ya = np.ones(len(c1), dtype=np.float64)
    variance_yb = np.ones(len(c2), dtype=np.float64)
    for supplied, destination, name in (
        (pair_slack_variance_ya, variance_ya, "YA slack variance"),
        (pair_slack_variance_yb, variance_yb, "YB slack variance"),
    ):
        if supplied is not None:
            values = np.asarray(supplied, dtype=np.float64)
            if values.shape != destination.shape or not np.isfinite(values).all():
                raise ValueError(f"invalid {name}")
            destination[:] = values
    if np.any(variance_ya <= 0.0) or np.any(variance_yb <= 0.0):
        raise ValueError("pair slack variances must be positive")

    def slack_gradient(vector: np.ndarray) -> np.ndarray:
        gradient = np.zeros_like(vector)
        if relaxed:
            scale = 1.0 / pair_slack_precision
            gradient[first:second] = scale * variance_ya * vector[first:second]
            gradient[second:] = scale * variance_yb * vector[second:]
        return gradient

    def relaxed_objective(objective: float, vector: np.ndarray) -> float:
        if not relaxed:
            return float(objective)
        scale = 1.0 / pair_slack_precision
        return float(objective + 0.5 * scale * (
            np.dot(variance_ya, vector[first:second] ** 2)
            + np.dot(variance_yb, vector[second:] ** 2)
        ))

    def prepare_ab_factors(vector: np.ndarray):
        local_lb = vector[:first]
        normalized = local_lb - logsumexp(local_lb)
        local_c1 = vector[first:second]
        local_c2 = vector[second:]
        return (
            np.ascontiguousarray(normalized),
            np.ascontiguousarray(np.exp(normalized)),
            local_c1,
            local_c2,
            np.ascontiguousarray(np.expm1(local_c1)),
            np.ascontiguousarray(np.expm1(local_c2)),
        )

    moment = np.zeros_like(parameters)
    square = np.zeros_like(parameters)
    rngs = [np.random.default_rng(seed + 1_000_003 * replica)
            for replica in range(replicas)]
    schedule_rng = np.random.default_rng(seed + 97_000_291)
    scheduled_blocks: np.ndarray | None = None
    direct_ab_blocks = sampled_ab_major_graph is not None
    ab_context_rows = None
    if direct_ab_blocks:
        ab_context_rows = tuple(
            (
                np.r_[0, np.cumsum(np.bincount(
                    contexts, minlength=problem.vocabulary_size
                ), dtype=np.int64)],
                np.argsort(contexts, kind="stable"),
            )
            for contexts in (problem.active_ya_a, problem.active_yb_b)
        )
    lazy_blocks = sampling == "blocks" and (
        exact_layered_graph is not None or direct_ab_blocks
    )
    full_plan = (
        None if lazy_blocks else build_sparse_intersection_plan(problem)
    )
    exact_executor = (
        ThreadPoolExecutor(max_workers=exact_margin_workers)
        if exact_margin_workers > 1 and exact_layered_graph is None else None
    )
    exact_boundaries = np.linspace(
        0, 0 if full_plan is None else len(full_plan.edge),
        exact_margin_workers + 1, dtype=np.int64
    )
    exact_shards = [
        (int(lo), int(hi))
        for lo, hi in pairwise(np.unique(exact_boundaries)) if hi > lo
    ]
    edge_cdf = np.cumsum(problem.edge_probability)
    edge_cdf[-1] = 1.0
    ranked_groups: list[np.ndarray] = []
    group_cdfs: list[np.ndarray] = []
    group_masses: list[float] = []
    prepared_blocks: list[SparseEdgeBlock] = []
    block_bounds: list[tuple[int, int]] = []
    block_masses = np.empty(0)
    block_cache: OrderedDict[int, SparseEdgeBlock] = OrderedDict()
    block_cache_lock = Lock()
    peak_cached_plan_bytes = 0
    block_cdf = np.empty(0)
    direct_block_edges: list[np.ndarray] = []
    direct_block_probabilities: list[np.ndarray] = []
    if sampling == "stratified":
        effective_strata = min(
            strata, len(problem.edge_probability), batch_size
        )
        ranked = np.argsort(problem.edge_probability, kind="stable")
        ranked_groups = list(np.array_split(ranked, effective_strata))
        for group in ranked_groups:
            probabilities = problem.edge_probability[group]
            mass = float(probabilities.sum())
            cdf = np.cumsum(probabilities / mass)
            cdf[-1] = 1.0
            group_cdfs.append(cdf)
            group_masses.append(mass)
    elif sampling == "blocks":
        if lazy_blocks:
            if direct_ab_blocks:
                active = np.r_[0, np.cumsum(
                    sampled_ab_major_graph.birth
                    <= exact_layered_checkpoint,
                    dtype=np.int64,
                )]
                ptr = sampled_ab_major_graph.edge_ptr[:(
                    len(problem.edge_probability) + 1
                )]
                work = active[ptr[1:]] - active[ptr[:-1]] + 1
            else:
                work = layered_edge_intersection_counts(
                    problem, exact_layered_graph, exact_layered_checkpoint
                ) + 1
            cumulative = np.r_[0, np.cumsum(work, dtype=np.int64)]
            targets = np.linspace(0, cumulative[-1], edge_blocks + 1)
            boundaries = np.unique(np.searchsorted(cumulative, targets))
            boundaries[0] = 0
            boundaries[-1] = len(problem.edge_probability)
            block_bounds = [
                (int(lo), int(hi)) for lo, hi in pairwise(boundaries)
                if hi > lo
            ]
            block_masses = np.asarray([
                problem.edge_probability[lo:hi].sum()
                for lo, hi in block_bounds
            ])
            if direct_ab_blocks:
                direct_block_edges = [
                    np.arange(lo, hi, dtype=np.int32)
                    for lo, hi in block_bounds
                ]
                direct_block_probabilities = [
                    np.ascontiguousarray(
                        problem.edge_probability[lo:hi] / block_masses[index]
                    )
                    for index, (lo, hi) in enumerate(block_bounds)
                ]
        else:
            prepared_blocks = build_sparse_edge_blocks(
                problem, full_plan, edge_blocks
            )
            block_masses = np.asarray([
                block.probability_mass for block in prepared_blocks
            ])
        block_cdf = np.cumsum(block_masses)
        block_cdf[-1] = 1.0

    def get_block(index: int) -> SparseEdgeBlock:
        nonlocal peak_cached_plan_bytes
        if not lazy_blocks:
            return prepared_blocks[index]
        with block_cache_lock:
            cached = block_cache.get(index)
            if cached is not None:
                block_cache.move_to_end(index)
                return cached
        # Construct outside the lock: independent cache misses should use the
        # available stochastic workers rather than serializing all builders.
        lo, hi = block_bounds[index]
        candidate = sparse_edge_block_from_bounds(
            problem, lo, hi, build_plan=not direct_ab_blocks
        )
        with block_cache_lock:
            cached = block_cache.get(index)
            if cached is not None:
                block_cache.move_to_end(index)
                return cached
            block = candidate
            block_cache[index] = block
            while len(block_cache) > lazy_block_cache:
                block_cache.popitem(last=False)
            peak_cached_plan_bytes = max(
                peak_cached_plan_bytes,
                sum(
                    array.nbytes
                    for cached_block in block_cache.values()
                    for array in (
                        cached_block.intersection_plan.edge,
                        cached_block.intersection_plan.target_y,
                        cached_block.intersection_plan.correction_ya,
                        cached_block.intersection_plan.correction_yb,
                    )
                ),
            )
            return block
    best = parameters.copy()
    best_objective = float("inf")
    best_certificate = float("inf")
    best_stationarity = float("inf")
    snapshot_parameters = parameters.copy()
    snapshot_gradient = np.zeros_like(parameters)
    shared_snapshot_parameters_owner = None
    shared_snapshot_gradient_owner = None
    if process_shards > 1:
        shared_snapshot_parameters_owner = process_context.RawArray(
            "d", len(parameters)
        )
        shared_snapshot_gradient_owner = process_context.RawArray(
            "d", len(parameters)
        )
        shared_snapshot_parameters = np.frombuffer(
            shared_snapshot_parameters_owner, dtype=np.float64
        )
        shared_snapshot_gradient = np.frombuffer(
            shared_snapshot_gradient_owner, dtype=np.float64
        )
        shared_snapshot_parameters[:] = snapshot_parameters
        shared_snapshot_gradient[:] = snapshot_gradient
        snapshot_parameters = shared_snapshot_parameters
        snapshot_gradient = shared_snapshot_gradient
    current_ab_factors = None
    snapshot_ab_factors = (
        prepare_ab_factors(snapshot_parameters) if direct_ab_blocks else None
    )
    exact_evaluations = 0
    exact_seconds = 0.0
    sampled_gradient_seconds = 0.0
    optimizer_seconds = 0.0
    reference_cache_seconds = 0.0
    ab_phase_timing: dict[str, float] = {}
    rejected_nonfinite_steps = 0
    trace: list[dict[str, float | int]] = []
    current_step_size = learning_rate
    scheduler_best = float("inf")
    scheduler_bad_records = 0
    scheduler_exhausted = False
    adam_step = 0
    stop_reason = "safety_limit"

    def exact_record(step: int) -> bool:
        nonlocal best, best_objective, best_certificate, best_stationarity
        nonlocal exact_evaluations
        nonlocal snapshot_parameters, snapshot_gradient, exact_seconds
        nonlocal current_step_size
        nonlocal scheduler_best, scheduler_bad_records
        nonlocal scheduler_exhausted
        nonlocal adam_step
        nonlocal stop_reason
        nonlocal snapshot_ab_factors
        started = perf_counter()
        evaluation = sparse_factorized_dual_evaluation(
            problem, parameters[:first], parameters[first:second],
            parameters[second:],
            intersection_plan=(
                full_plan if exact_layered_graph is None else None
            ),
            compute_certificate=True,
            executor=exact_executor,
            intersection_shards=exact_shards,
            layered_graph=exact_layered_graph,
            layered_checkpoint=exact_layered_checkpoint,
            margin_workers=exact_margin_workers,
        )
        exact_evaluations += 1
        gradient = evaluation.gradient() + slack_gradient(parameters)
        observed_gradient = gradient.copy()
        objective = relaxed_objective(evaluation.objective, parameters)
        stationarity = float(np.max(np.abs(gradient)))
        certificate = float(evaluation.certificate)
        exact_seconds += perf_counter() - started
        if not (
            np.isfinite(objective)
            and np.isfinite(certificate)
            and np.all(np.isfinite(gradient))
        ):
            diagnostic = diagnose_sparse_factorized_normalizers(
                problem, (
                    full_plan or build_sparse_intersection_plan(problem)
                ),
                parameters[:first], parameters[first:second],
                parameters[second:],
            )
            parameters[:] = snapshot_parameters
            moment.fill(0.0)
            square.fill(0.0)
            adam_step = 0
            old_step_size = current_step_size
            current_step_size = max(
                minimum_learning_rate,
                current_step_size * plateau_factor,
            )
            scheduler_exhausted = old_step_size <= minimum_learning_rate
            trace.append({
                "step": step,
                "exact_objective": float("inf"),
                "exact_gradient_l2": float("inf"),
                "exact_gradient_linf": float("inf"),
                "exact_certificate": float("inf"),
                "residual_y_l1": float("inf"),
                "residual_ya_l1": float("inf"),
                "residual_yb_l1": float("inf"),
                "step_size": current_step_size,
                "learning_rate_reduced": True,
                "scheduler_exhausted": scheduler_exhausted,
                "rejected_nonfinite": True,
                **diagnostic,
            })
            if scheduler_exhausted:
                stop_reason = "nonfinite"
            return scheduler_exhausted
        selection_value = stationarity if relaxed else certificate
        if selection_value < (
            best_stationarity if relaxed else best_certificate
        ):
            best_certificate = certificate
            best_stationarity = stationarity
            best_objective = objective
            best = parameters.copy()
        learning_rate_reduced = False
        if optimizer == "adam_plateau":
            # ReduceLROnPlateau monitors the smooth quantity being optimized,
            # not the nonsmooth maximum-L1 certificate used for termination.
            # Individual margin residuals can temporarily trade off while the
            # exact convex dual objective decreases steadily.
            scheduler_value = objective
            if scheduler_value < scheduler_best * (
                1.0 - plateau_relative_threshold
            ):
                scheduler_best = scheduler_value
                scheduler_bad_records = 0
            elif step > 0:
                scheduler_bad_records += 1
                if scheduler_bad_records >= plateau_patience:
                    if current_step_size > minimum_learning_rate:
                        current_step_size = max(
                            minimum_learning_rate,
                            current_step_size * plateau_factor,
                        )
                        learning_rate_reduced = True
                    else:
                        scheduler_exhausted = True
                    scheduler_bad_records = 0
        if optimizer == "svrg_bb" and variance_reduction and step > 0:
            displacement = parameters - snapshot_parameters
            gradient_change = gradient - snapshot_gradient
            curvature = float(displacement @ gradient_change)
            squared_distance = float(displacement @ displacement)
            if (
                curvature > 0.0
                and squared_distance > 0.0
                and np.isfinite(curvature)
                and np.isfinite(squared_distance)
            ):
                proposal = (
                    squared_distance / curvature / exact_interval
                )
                current_step_size = float(np.clip(
                    proposal, bb_min_step, bb_max_step
                ))
        if variance_reduction:
            if process_shards > 1:
                snapshot_parameters[:] = parameters
                snapshot_gradient[:] = gradient
            else:
                snapshot_parameters = parameters.copy()
                snapshot_gradient = gradient.copy()
            if direct_ab_blocks:
                snapshot_ab_factors = prepare_ab_factors(snapshot_parameters)
        trace.append({
            "step": step,
            "exact_objective": objective,
            "exact_gradient_l2": float(np.linalg.norm(observed_gradient)),
            "exact_gradient_linf": float(np.max(np.abs(observed_gradient))),
            "exact_certificate": certificate,
            "regularized_stationarity": stationarity,
            "residual_y_l1": float(evaluation.residual_y_l1),
            "residual_ya_l1": float(evaluation.residual_ya_l1),
            "residual_yb_l1": float(evaluation.residual_yb_l1),
            "step_size": current_step_size,
            "scheduler_value": objective,
            "learning_rate_reduced": learning_rate_reduced,
            "scheduler_exhausted": scheduler_exhausted,
            "rejected_nonfinite": False,
        })
        if exact_record_callback is not None:
            exact_record_callback(
                step, certificate,
                parameters[:first], parameters[first:second],
                parameters[second:],
            )
        if (
            certificate_tolerance is not None
            and (stationarity if relaxed else certificate)
            <= certificate_tolerance
        ):
            stop_reason = "tolerance"
            return True
        if scheduler_exhausted:
            stop_reason = "plateau"
            return True
        return False

    reached_certificate = exact_record(0)
    replica_executor = (
        ThreadPoolExecutor(max_workers=stochastic_workers)
        if process_shards == 1 and stochastic_workers > 1 else None
    )
    replica_groups = [
        group for group in np.array_split(
            np.arange(replicas, dtype=np.int64),
            min(stochastic_workers, replicas),
        ) if len(group)
    ]
    ab_phase_timing["configured_workers"] = float(stochastic_workers)
    ab_phase_timing["configured_replicas"] = float(replicas)
    ab_phase_timing["configured_worker_groups"] = float(len(replica_groups))
    ab_phase_timing["configured_process_shards"] = float(process_shards)
    if replica_groups:
        ab_phase_timing["largest_replica_group"] = float(
            max(len(group) for group in replica_groups)
        )
    reference_block_margins: list[SparseReferenceMargins] = []
    reference_positions = [] if lazy_blocks else [
        (
            np.flatnonzero(np.isin(
                problem.active_ya_a,
                np.unique(get_block(index).problem.edge_a)
            )).astype(np.int32),
            np.flatnonzero(np.isin(
                problem.active_yb_b,
                np.unique(get_block(index).problem.edge_b)
            )).astype(np.int32),
        )
        for index in range(len(block_masses))
    ]
    lazy_reference_cache: OrderedDict[int, SparseReferenceMargins] = (
        OrderedDict()
    )
    lazy_reference_lock = Lock()
    peak_lazy_reference_bytes = 0
    reference_position_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    reference_position_cache_bytes = 0

    def reference_for_block(index: int) -> SparseReferenceMargins:
        nonlocal peak_lazy_reference_bytes, reference_cache_seconds
        nonlocal reference_position_cache_bytes
        if not lazy_blocks:
            return reference_block_margins[index]
        with lazy_reference_lock:
            cached = lazy_reference_cache.get(index)
            if cached is not None:
                lazy_reference_cache.move_to_end(index)
                return cached
        # As above, the numerical reference calculation is independent and
        # should run concurrently; only insertion/eviction is serialized.
        reference_started = perf_counter()
        block = get_block(index)
        positions_started = perf_counter()
        with lazy_reference_lock:
            cached_positions = reference_position_cache.get(index)
        if cached_positions is None:
            ya_position = np.flatnonzero(np.isin(
                problem.active_ya_a, np.unique(block.problem.edge_a)
            )).astype(np.int32)
            yb_position = np.flatnonzero(np.isin(
                problem.active_yb_b, np.unique(block.problem.edge_b)
            )).astype(np.int32)
            if persistent_reference_positions:
                with lazy_reference_lock:
                    existing = reference_position_cache.get(index)
                    if existing is None:
                        reference_position_cache[index] = (
                            ya_position, yb_position
                        )
                        reference_position_cache_bytes += (
                            ya_position.nbytes + yb_position.nbytes
                        )
                    else:
                        ya_position, yb_position = existing
        else:
            ya_position, yb_position = cached_positions
        positions_seconds = perf_counter() - positions_started
        native_started = perf_counter()
        margins = (
            sparse_factorized_margins_ab_major(
                block.problem, sampled_ab_major_graph,
                exact_layered_checkpoint, block_bounds[index][0],
                snapshot_parameters[:first],
                snapshot_parameters[first:second],
                snapshot_parameters[second:],
                prepared_factors=snapshot_ab_factors,
                context_rows=ab_context_rows,
            ) if direct_ab_blocks else sparse_factorized_margins(
                block.problem, block.intersection_plan,
                snapshot_parameters[:first],
                snapshot_parameters[first:second],
                snapshot_parameters[second:],
            )
        )
        native_seconds = perf_counter() - native_started
        candidate = SparseReferenceMargins(
            margins.target_y, ya_position,
            margins.active_ya[ya_position], yb_position,
            margins.active_yb[yb_position],
        )
        with lazy_reference_lock:
            cached = lazy_reference_cache.get(index)
            ab_phase_timing["reference_position_seconds"] = (
                ab_phase_timing.get("reference_position_seconds", 0.0)
                + positions_seconds
            )
            ab_phase_timing["reference_native_seconds"] = (
                ab_phase_timing.get("reference_native_seconds", 0.0)
                + native_seconds
            )
            ab_phase_timing["reference_build_attempts"] = (
                ab_phase_timing.get("reference_build_attempts", 0.0) + 1.0
            )
            if cached is not None:
                ab_phase_timing["reference_duplicate_builds"] = (
                    ab_phase_timing.get("reference_duplicate_builds", 0.0)
                    + 1.0
                )
                lazy_reference_cache.move_to_end(index)
                return cached
            reference = candidate
            reference_cache_seconds += perf_counter() - reference_started
            lazy_reference_cache[index] = reference
            while len(lazy_reference_cache) > lazy_block_cache:
                lazy_reference_cache.popitem(last=False)
            peak_lazy_reference_bytes = max(
                peak_lazy_reference_bytes,
                sum(
                    array.nbytes
                    for item in lazy_reference_cache.values()
                    for array in (
                        item.target_y, item.ya_position, item.active_ya,
                        item.yb_position, item.active_yb,
                    )
                ),
            )
            return reference

    def refresh_reference_blocks() -> None:
        """Cache the fixed SVRG side once instead of once per draw."""
        nonlocal reference_block_margins, reference_cache_seconds
        if not (variance_reduction and sampling == "blocks"):
            return
        if lazy_blocks:
            with lazy_reference_lock:
                lazy_reference_cache.clear()
            return

        def evaluate_reference(item):
            index, (ya_position, yb_position) = item
            block = get_block(index)
            margins = sparse_factorized_margins(
                block.problem, block.intersection_plan,
                snapshot_parameters[:first],
                snapshot_parameters[first:second],
                snapshot_parameters[second:],
            )
            return SparseReferenceMargins(
                margins.target_y,
                ya_position,
                margins.active_ya[ya_position],
                yb_position,
                margins.active_yb[yb_position],
            )

        started = perf_counter()
        evaluations = (
            map(evaluate_reference, enumerate(reference_positions))
            if replica_executor is None
            else replica_executor.map(
                evaluate_reference, enumerate(reference_positions)
            )
        )
        reference_block_margins = list(evaluations)
        reference_cache_seconds += perf_counter() - started

    refresh_reference_blocks()

    def sampled_gradient(
        replica: int,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        # Worker-time components: block lookup, current native margins,
        # reference lookup/construction, and gradient assembly.
        phases = np.zeros(4, dtype=np.float64)
        if sampling == "blocks":
            chosen = (
                int(scheduled_blocks[replica])
                if scheduled_blocks is not None
                else int(np.searchsorted(
                    block_cdf, rngs[replica].random(), side="right"
                ))
            )
            phase_started = perf_counter()
            block = get_block(chosen)
            phases[0] = perf_counter() - phase_started
            phase_started = perf_counter()
            margins = (
                sparse_factorized_margins_ab_major(
                    block.problem, sampled_ab_major_graph,
                    exact_layered_checkpoint, block_bounds[chosen][0],
                    parameters[:first], parameters[first:second],
                    parameters[second:],
                    prepared_factors=current_ab_factors,
                    context_rows=ab_context_rows,
                ) if direct_ab_blocks else sparse_factorized_margins(
                    block.problem, block.intersection_plan,
                    parameters[:first], parameters[first:second],
                    parameters[second:],
                )
            )
            phases[1] = perf_counter() - phase_started
            phase_started = perf_counter()
            if variance_reduction:
                reference = reference_for_block(chosen)
                phases[2] = perf_counter() - phase_started
                phase_started = perf_counter()
                # Targets cancel in the SVRG difference.  Forming full dual
                # evaluations here used to perform two unnecessary target
                # subtractions, objectives, and gradient concatenations per
                # replica.  Keep only the changing model margins.
                gradient = snapshot_gradient.copy()
                gradient[:first] += margins.target_y - reference.target_y
                margins.active_ya[reference.ya_position] -= (
                    reference.active_ya
                )
                margins.active_yb[reference.yb_position] -= (
                    reference.active_yb
                )
                gradient[first:second] += margins.active_ya
                gradient[second:] += margins.active_yb
            else:
                phases[2] = perf_counter() - phase_started
                phase_started = perf_counter()
                gradient = np.concatenate([
                    margins.target_y - block.problem.target_y,
                    margins.active_ya - block.problem.target_ya,
                    margins.active_yb - block.problem.target_yb,
                ])
            phases[3] = perf_counter() - phase_started
            edges_used = len(block.problem.edge_probability)
            return (
                gradient,
                edges_used * (2 if variance_reduction else 1),
                phases,
            )
        if sampling == "stratified":
            allocation = np.full(
                len(ranked_groups), batch_size // len(ranked_groups),
                dtype=np.int64,
            )
            allocation[:batch_size % len(ranked_groups)] += 1
            all_edges = []
            all_weights = []
            for group, cdf, mass, count in zip(
                ranked_groups, group_cdfs, group_masses, allocation
            ):
                draws = group[np.searchsorted(
                    cdf, rngs[replica].random(int(count)), side="right"
                )]
                edges, multiplicity = np.unique(draws, return_counts=True)
                all_edges.append(edges)
                all_weights.append(multiplicity * (mass / float(count)))
            edges = np.concatenate(all_edges)
            weights = np.concatenate(all_weights)
        else:
            draws = np.searchsorted(
                edge_cdf, rngs[replica].random(batch_size), side="right"
            )
            edges, counts = np.unique(draws, return_counts=True)
            weights = counts.astype(np.float64)
        sampled, sampled_plan = sparse_edge_distribution_with_plan(
            problem, full_plan, edges, weights
        )
        gradient = sparse_factorized_dual_evaluation(
            sampled, parameters[:first], parameters[first:second],
            parameters[second:], intersection_plan=sampled_plan,
        ).gradient()
        if variance_reduction:
            reference = sparse_factorized_dual_evaluation(
                sampled,
                snapshot_parameters[:first],
                snapshot_parameters[first:second],
                snapshot_parameters[second:],
                intersection_plan=sampled_plan,
            ).gradient()
            gradient += snapshot_gradient - reference
        return gradient, batch_size * (2 if variance_reduction else 1), phases

    def sampled_gradient_group(
        replica_group: np.ndarray,
    ) -> tuple[np.ndarray, int, float, np.ndarray]:
        """Evaluate and reduce several replicas inside one worker."""

        group_started = perf_counter()
        local_gradient = np.zeros_like(parameters)
        local_edges = 0
        local_phases = np.zeros(4, dtype=np.float64)
        for replica in replica_group:
            contribution, edges_used, phases = sampled_gradient(int(replica))
            local_gradient += contribution
            local_edges += edges_used
            local_phases += phases
        return (
            local_gradient, local_edges,
            perf_counter() - group_started, local_phases,
        )

    def sampled_gradient_fused_ab() -> tuple[np.ndarray, int]:
        """Fuse the fixed replica batch into one indexed native traversal."""

        chosen = [
            int(np.searchsorted(
                block_cdf, rngs[replica].random(), side="right"
            ))
            for replica in range(replicas)
        ]
        selected_edges = np.ascontiguousarray(np.concatenate([
            direct_block_edges[index] for index in chosen
        ]), dtype=np.int32)
        selected_probability = np.ascontiguousarray(np.concatenate([
            direct_block_probabilities[index] / replicas for index in chosen
        ]))
        sampled = SparseGroupedProblem(
            vocabulary_size=problem.vocabulary_size,
            edge_a=problem.edge_a[selected_edges],
            edge_b=problem.edge_b[selected_edges],
            edge_probability=selected_probability,
            target_y=problem.target_y,
            active_ya_y=problem.active_ya_y,
            active_ya_a=problem.active_ya_a,
            target_ya=problem.target_ya,
            active_yb_y=problem.active_yb_y,
            active_yb_b=problem.active_yb_b,
            target_yb=problem.target_yb,
        )
        current_margins = sparse_factorized_margins_ab_major(
            sampled, sampled_ab_major_graph, exact_layered_checkpoint, 0,
            parameters[:first], parameters[first:second], parameters[second:],
            workers=stochastic_workers,
            prepared_factors=current_ab_factors,
            edge_indices=selected_edges,
            context_rows=ab_context_rows,
            phase_timing=ab_phase_timing,
        )
        gradient = snapshot_gradient.copy()
        gradient[:first] += current_margins.target_y
        gradient[first:second] += current_margins.active_ya
        gradient[second:] += current_margins.active_yb
        inverse_replicas = 1.0 / replicas
        for index in chosen:
            reference = reference_for_block(index)
            gradient[:first] -= inverse_replicas * reference.target_y
            gradient[first + reference.ya_position] -= (
                inverse_replicas * reference.active_ya
            )
            gradient[second + reference.yb_position] -= (
                inverse_replicas * reference.active_yb
            )
        return gradient, 2 * len(selected_edges)

    process_workers = []
    process_connections = []
    shared_process_gradient_owners = []
    shared_process_gradients = []

    def process_replica_worker(connection, replica_group, gradient_owner):
        """Persist one process-local threaded replica group."""

        nonlocal current_ab_factors, snapshot_ab_factors
        output = np.frombuffer(gradient_owner, dtype=np.float64)
        local_executor = ThreadPoolExecutor(max_workers=len(replica_group))
        try:
            while True:
                command = connection.recv()
                if command == "stop":
                    return
                if command == "refresh":
                    with lazy_reference_lock:
                        lazy_reference_cache.clear()
                    snapshot_ab_factors = prepare_ab_factors(
                        snapshot_parameters
                    )
                    connection.send(("refreshed",))
                    continue
                if command != "evaluate":
                    raise ValueError(f"unknown process command {command!r}")
                started = perf_counter()
                current_ab_factors = prepare_ab_factors(parameters)
                results = list(local_executor.map(
                    sampled_gradient, replica_group
                ))
                output.fill(0.0)
                local_edges = 0
                local_phases = np.zeros(4, dtype=np.float64)
                for contribution, edges_used, phases in results:
                    output += contribution
                    local_edges += edges_used
                    local_phases += phases
                connection.send((
                    "evaluated", local_edges,
                    perf_counter() - started, local_phases,
                ))
        except BaseException as error:
            try:
                connection.send((
                    "error", type(error).__name__, str(error),
                    traceback.format_exc(),
                ))
            finally:
                raise
        finally:
            local_executor.shutdown()
            connection.close()

    if process_shards > 1:
        process_replica_groups = [
            np.asarray(group, dtype=np.int64)
            for group in np.array_split(
                np.arange(replicas, dtype=np.int64), process_shards
            ) if len(group)
        ]
        for replica_group in process_replica_groups:
            gradient_owner = process_context.RawArray("d", len(parameters))
            parent_connection, child_connection = process_context.Pipe()
            worker = process_context.Process(
                target=process_replica_worker,
                args=(child_connection, replica_group, gradient_owner),
            )
            worker.start()
            child_connection.close()
            process_workers.append(worker)
            process_connections.append(parent_connection)
            shared_process_gradient_owners.append(gradient_owner)
            shared_process_gradients.append(np.frombuffer(
                gradient_owner, dtype=np.float64
            ))

    def refresh_process_references() -> None:
        if process_shards == 1:
            return
        for connection in process_connections:
            connection.send("refresh")
        for connection in process_connections:
            message = connection.recv()
            if message != ("refreshed",):
                raise RuntimeError("process reference refresh failed")

    def receive_process_result(connection):
        message = connection.recv()
        if message and message[0] == "error":
            _, kind, detail, worker_traceback = message
            raise RuntimeError(
                f"replica worker failed with {kind}: {detail}\n"
                f"{worker_traceback}"
            )
        return message

    sampled_edge_evaluations = 0
    completed_steps = 0
    for step in (range(1, steps + 1) if not reached_certificate else ()):
        completed_steps = step
        if sampling == "blocks" and block_replica_schedule == "systematic":
            offset = schedule_rng.random() / replicas
            positions = offset + np.arange(replicas) / replicas
            scheduled_blocks = np.searchsorted(
                block_cdf, positions, side="right"
            ).astype(np.int64)
            scheduled_blocks = scheduled_blocks[
                schedule_rng.permutation(replicas)
            ]
        if optimizer == "adam_cosine":
            progress = (step - 1) / max(1, steps - 1)
            current_step_size = minimum_learning_rate + 0.5 * (
                learning_rate - minimum_learning_rate
            ) * (1.0 + np.cos(np.pi * progress))
        sampled_started = perf_counter()
        if direct_ab_blocks and process_shards == 1:
            preparation_started = perf_counter()
            current_ab_factors = prepare_ab_factors(parameters)
            ab_phase_timing["factor_prepare_seconds"] = (
                ab_phase_timing.get("factor_prepare_seconds", 0.0)
                + perf_counter() - preparation_started
            )
        if direct_ab_blocks and variance_reduction and fused_ab_batch:
            gradient, edges_used = sampled_gradient_fused_ab()
            sampled_edge_evaluations += edges_used
        elif process_shards > 1:
            evaluation_started = perf_counter()
            for connection in process_connections:
                connection.send("evaluate")
            process_results = [
                receive_process_result(connection)
                for connection in process_connections
            ]
            gradients = []
            for shared_gradient, result in zip(
                shared_process_gradients, process_results
            ):
                status, edges_used, timing, phases = result
                if status != "evaluated":
                    raise RuntimeError("process replica evaluation failed")
                gradients.append((shared_gradient, edges_used, timing, phases))
            ab_phase_timing["replica_evaluation_seconds"] = (
                ab_phase_timing.get("replica_evaluation_seconds", 0.0)
                + perf_counter() - evaluation_started
            )
            group_seconds = [timing for _, _, timing, _ in gradients]
            ab_phase_timing["worker_task_seconds"] = (
                ab_phase_timing.get("worker_task_seconds", 0.0)
                + sum(group_seconds)
            )
            ab_phase_timing["worker_critical_path_seconds"] = (
                ab_phase_timing.get("worker_critical_path_seconds", 0.0)
                + max(group_seconds)
            )
            ab_phase_timing["worker_fastest_seconds"] = (
                ab_phase_timing.get("worker_fastest_seconds", 0.0)
                + min(group_seconds)
            )
            ab_phase_timing["worker_tasks"] = (
                ab_phase_timing.get("worker_tasks", 0.0) + len(gradients)
            )
            ab_phase_timing["worker_timed_updates"] = (
                ab_phase_timing.get("worker_timed_updates", 0.0) + 1.0
            )
            phase_totals = np.sum(
                [phases for _, _, _, phases in gradients], axis=0
            )
            for name, value in zip(
                ("block_lookup", "current_native", "reference", "assembly"),
                phase_totals,
            ):
                key = f"worker_{name}_seconds"
                ab_phase_timing[key] = (
                    ab_phase_timing.get(key, 0.0) + float(value)
                )
            reduction_started = perf_counter()
            gradient = np.zeros_like(parameters)
            for contribution, edges_used, _, _ in gradients:
                gradient += contribution
                sampled_edge_evaluations += edges_used
            gradient /= replicas
            ab_phase_timing["replica_reduction_seconds"] = (
                ab_phase_timing.get("replica_reduction_seconds", 0.0)
                + perf_counter() - reduction_started
            )
        else:
            evaluation_started = perf_counter()
            gradients = list(
                map(sampled_gradient_group, replica_groups)
                if replica_executor is None
                else replica_executor.map(
                    sampled_gradient_group, replica_groups
                )
            )
            ab_phase_timing["replica_evaluation_seconds"] = (
                ab_phase_timing.get("replica_evaluation_seconds", 0.0)
                + perf_counter() - evaluation_started
            )
            group_seconds = [timing for _, _, timing, _ in gradients]
            if group_seconds:
                ab_phase_timing["worker_task_seconds"] = (
                    ab_phase_timing.get("worker_task_seconds", 0.0)
                    + sum(group_seconds)
                )
                ab_phase_timing["worker_critical_path_seconds"] = (
                    ab_phase_timing.get("worker_critical_path_seconds", 0.0)
                    + max(group_seconds)
                )
                ab_phase_timing["worker_fastest_seconds"] = (
                    ab_phase_timing.get("worker_fastest_seconds", 0.0)
                    + min(group_seconds)
                )
                ab_phase_timing["worker_tasks"] = (
                    ab_phase_timing.get("worker_tasks", 0.0)
                    + len(group_seconds)
                )
                ab_phase_timing["worker_timed_updates"] = (
                    ab_phase_timing.get("worker_timed_updates", 0.0) + 1.0
                )
                phase_totals = np.sum(
                    [phases for _, _, _, phases in gradients], axis=0
                )
                for name, value in zip(
                    ("block_lookup", "current_native", "reference", "assembly"),
                    phase_totals,
                ):
                    key = f"worker_{name}_seconds"
                    ab_phase_timing[key] = (
                        ab_phase_timing.get(key, 0.0) + float(value)
                    )
            reduction_started = perf_counter()
            gradient = np.zeros_like(parameters)
            for contribution, edges_used, _, _ in gradients:
                gradient += contribution
                sampled_edge_evaluations += edges_used
            gradient /= replicas
            ab_phase_timing["replica_reduction_seconds"] = (
                ab_phase_timing.get("replica_reduction_seconds", 0.0)
                + perf_counter() - reduction_started
            )
        if relaxed:
            gradient += slack_gradient(parameters)
            if variance_reduction:
                gradient -= slack_gradient(snapshot_parameters)
        sampled_gradient_seconds += perf_counter() - sampled_started
        if not np.all(np.isfinite(gradient)):
            rejected_nonfinite_steps += 1
            parameters[:] = snapshot_parameters
            moment.fill(0.0)
            square.fill(0.0)
            adam_step = 0
            current_step_size = max(
                minimum_learning_rate,
                current_step_size * plateau_factor,
            )
            if current_step_size <= minimum_learning_rate:
                stop_reason = "nonfinite"
                break
            continue
        optimizer_started = perf_counter()
        if optimizer == "svrg_bb":
            parameters -= current_step_size * gradient
        else:
            adam_step += 1
            moment *= beta1
            moment += (1.0 - beta1) * gradient
            square *= beta2
            square += (1.0 - beta2) * gradient * gradient
            corrected_moment = moment / (1.0 - beta1 ** adam_step)
            corrected_square = square / (1.0 - beta2 ** adam_step)
            parameters -= (
                current_step_size * corrected_moment
                / (np.sqrt(corrected_square) + 1e-8)
            )
        if not np.all(np.isfinite(parameters)):
            rejected_nonfinite_steps += 1
            parameters[:] = snapshot_parameters
            moment.fill(0.0)
            square.fill(0.0)
            adam_step = 0
            current_step_size = max(
                minimum_learning_rate,
                current_step_size * plateau_factor,
            )
            if current_step_size <= minimum_learning_rate:
                stop_reason = "nonfinite"
                break
            continue
        if optimizer != "adam_plateau":
            np.clip(
                parameters, origin - trust_radius, origin + trust_radius,
                out=parameters,
            )
        # Remove the harmless global baseline gauge before the next step.
        parameters[:first] -= logsumexp(parameters[:first])
        optimizer_seconds += perf_counter() - optimizer_started
        if step % exact_interval == 0 or step == steps:
            if exact_record(step):
                reached_certificate = True
                break
            refresh_reference_blocks()
            refresh_process_references()

    if replica_executor is not None:
        replica_executor.shutdown()
    for connection in process_connections:
        connection.send("stop")
    for worker in process_workers:
        worker.join()
        if worker.exitcode != 0:
            raise RuntimeError(
                f"replica worker exited with status {worker.exitcode}"
            )
    for connection in process_connections:
        connection.close()
    if exact_executor is not None:
        exact_executor.shutdown()

    intersection_plan_bytes = (
        peak_cached_plan_bytes if full_plan is None else sum(
            array.nbytes for array in (
                full_plan.edge,
                full_plan.target_y,
                full_plan.correction_ya,
                full_plan.correction_yb,
            )
        )
    )
    reference_cache_bytes = (
        peak_lazy_reference_bytes if lazy_blocks else sum(
            array.nbytes
            for reference in reference_block_margins
            for array in (
                reference.target_y,
                reference.ya_position,
                reference.active_ya,
                reference.yb_position,
                reference.active_yb,
            )
        )
    )
    reference_cache_bytes += reference_position_cache_bytes
    ab_phase_timing["reference_position_cache_bytes"] = float(
        reference_position_cache_bytes
    )
    ab_phase_timing["reference_position_cache_entries"] = float(
        len(reference_position_cache)
    )

    return SparseStochasticResult(
        log_base_y=best[:first].copy(),
        correction_ya=best[first:second].copy(),
        correction_yb=best[second:].copy(),
        steps=completed_steps,
        sampled_edges=sampled_edge_evaluations,
        exact_evaluations=exact_evaluations,
        best_exact_objective=best_objective,
        best_exact_certificate=best_certificate,
        trace=tuple(trace),
        stop_reason=stop_reason,
        exact_seconds=exact_seconds,
        sampled_gradient_seconds=sampled_gradient_seconds,
        optimizer_seconds=optimizer_seconds,
        reference_cache_seconds=reference_cache_seconds,
        intersection_plan_bytes=intersection_plan_bytes,
        reference_cache_bytes=reference_cache_bytes,
        sampled_phase_timing=dict(ab_phase_timing),
        best_exact_stationarity=best_stationarity,
    )


def check_grouped_feasibility_lp(
    problem: SparseGroupedProblem,
    *,
    max_variables: int = 2_000_000,
    time_limit: float | None = None,
    progress: bool = False,
) -> GroupedFeasibilityResult:
    """Check grouped-margin feasibility by a small-reference linear program.

    There is one nonnegative variable for every target symbol on every
    retained AB edge.  This is intentionally a diagnostic reference, not a
    production large-vocabulary algorithm.
    """

    v = problem.vocabulary_size
    edges = len(problem.edge_probability)
    variables = v * edges
    if variables > max_variables:
        raise ValueError(
            f"feasibility LP has {variables} variables; limit is "
            f"{max_variables}"
        )
    index_dtype = np.int32 if variables <= np.iinfo(np.int32).max else np.int64
    variable_index = np.arange(variables, dtype=index_dtype)
    row_parts = [
        np.repeat(np.arange(edges, dtype=index_dtype), v),
        edges + np.tile(np.arange(v, dtype=index_dtype), edges),
    ]
    column_parts = [variable_index, variable_index]
    next_row = edges + v

    def append_grouped_constraints(
        active_y: np.ndarray,
        active_context: np.ndarray,
        edge_context: np.ndarray,
    ) -> None:
        nonlocal next_row
        order = np.argsort(edge_context, kind="stable")
        ptr = np.r_[0, np.cumsum(np.bincount(
            edge_context, minlength=v
        ), dtype=np.int64)]
        local_rows = []
        local_columns = []
        for offset, (y, context) in enumerate(zip(
            active_y, active_context
        )):
            selected = order[ptr[context]:ptr[context + 1]]
            if len(selected):
                local_rows.append(np.full(
                    len(selected), next_row + offset, dtype=index_dtype
                ))
                local_columns.append(np.asarray(
                    selected * v + y, dtype=index_dtype
                ))
        if local_rows:
            row_parts.append(np.concatenate(local_rows))
            column_parts.append(np.concatenate(local_columns))
        next_row += len(active_y)

    append_grouped_constraints(
        problem.active_ya_y, problem.active_ya_a, problem.edge_a
    )
    append_grouped_constraints(
        problem.active_yb_y, problem.active_yb_b, problem.edge_b
    )
    rows = np.concatenate(row_parts)
    columns = np.concatenate(column_parts)
    matrix = coo_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, columns)),
        shape=(next_row, variables),
    ).tocsr()
    rhs = np.concatenate([
        problem.edge_probability, problem.target_y,
        problem.target_ya, problem.target_yb,
    ])
    options = {"disp": progress}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    result = linprog(
        np.zeros(variables),
        A_eq=matrix,
        b_eq=rhs,
        bounds=(0.0, None),
        method="highs",
        options=options,
    )
    residual = (
        float(np.max(np.abs(matrix @ result.x - rhs)))
        if result.success else float("inf")
    )
    return GroupedFeasibilityResult(
        feasible=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        variables=variables,
        equality_constraints=len(rhs),
        max_equality_residual=residual,
    )


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


def sparse_pair_log_probabilities(
    pair: SparseProjectedPair,
    targets: np.ndarray,
    contexts: np.ndarray,
) -> np.ndarray:
    """Score the honest one-lag conditional represented by a sparse joint.

    The stored orientation is ``(target, context)``, so this returns
    ``log p(target | context)`` without constructing a dense pair table.
    """

    y = np.asarray(targets, dtype=np.int64)
    a = np.asarray(contexts, dtype=np.int64)
    if y.shape != a.shape:
        raise ValueError("targets and contexts must have identical shapes")
    if y.size == 0:
        return np.empty(y.shape, dtype=np.float64)
    joint = pair.values(y, a)
    _, context_margin = _sparse_pair_margins(pair)
    tiny = np.finfo(np.float64).tiny
    return (
        np.log(np.maximum(joint, tiny))
        - np.log(np.maximum(context_margin[a], tiny))
    )


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
    unsupported = []
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
            # Subtracting corrected_mass from one is inaccurate when the
            # corrected targets carry almost all baseline mass.  In that
            # regime the rounding residue can exceed the true complement
            # (or invent a background when every target is corrected), and
            # the resulting conditional no longer normalizes.  Use
            # subtraction only when it is well conditioned; otherwise sum
            # the usually small complement directly in log space.
            if corrected_mass <= 0.5:
                log_background_mass = np.log1p(-corrected_mass)
            elif len(corrected_y) == v:
                log_background_mass = -np.inf
            else:
                uncorrected = np.ones(v, dtype=bool)
                uncorrected[corrected_y] = False
                log_background_mass = logsumexp(log_base[uncorrected])
            corrected_log_mass = logsumexp(np.array([
                log_base[yy] + value
                for yy, value in corrections.items()
            ])) if corrections else -np.inf
            log_normalizer = np.logaddexp(
                log_background_mass,
                corrected_log_mass,
            )
            chosen = y[selected]
            extra = np.array([
                corrections.get(int(yy), 0.0) for yy in chosen
            ])
            answer[selected] = log_base[chosen] + extra - log_normalizer
        else:
            if sparse_fallback:
                unsupported.append(selected)
            else:
                answer[selected] = star_log_probabilities(
                    y[selected], a[selected], b[selected], p_ya, p_yb
                )
    if unsupported:
        selected = np.concatenate(unsupported)
        answer[selected] = sparse_star_log_probabilities(
            p_ya, p_yb, y[selected], a[selected], b[selected]
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


def _projected_pair_factor(pair: SparseProjectedPair) -> np.ndarray:
    """Return exact active log corrections relative to the pair background."""

    row_background = pair.background[pair.active_context]
    reference = np.where(row_background > 0.0, row_background, 1.0)
    multiplier = row_background + pair.delta
    if np.any(multiplier <= 0.0):
        raise ValueError("projected active pair cells must be positive")
    return np.log(multiplier) - np.log(reference)


def projected_pair_warm_start(
    p_ya: SparseProjectedPair,
    p_yb: SparseProjectedPair,
    target_y: np.ndarray,
    initialization: str,
    *,
    canonicalize: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct exact grouped tree factors from retained sparse pair data."""

    if p_ya.vocabulary_size != p_yb.vocabulary_size:
        raise ValueError("projected initializer pairs must share a vocabulary")
    tiny = np.finfo(np.float64).tiny
    first = _projected_pair_factor(p_ya)
    second = _projected_pair_factor(p_yb)
    log_first_base = np.log(np.maximum(p_ya.right, tiny))
    log_second_base = np.log(np.maximum(p_yb.right, tiny))
    if initialization == "first_pair":
        warm = (log_first_base, first, np.zeros_like(second))
    elif initialization == "second_pair":
        warm = (log_second_base, np.zeros_like(first), second)
    elif initialization == "pair_midpoint":
        warm = (
            0.5 * (log_first_base + log_second_base),
            0.5 * first,
            0.5 * second,
        )
    elif initialization == "pair_product":
        warm = (
            log_first_base + log_second_base
            - np.log(np.maximum(target_y, tiny)),
            first,
            second,
        )
    else:
        raise ValueError("projected pair start requires a pair initialization")

    log_base, correction_ya, correction_yb = (
        np.array(component, copy=True) for component in warm
    )
    if canonicalize:
        # Tree differences require one explicit gauge across checkpoints.
        # Ordinary initialization does not: changing its coordinates can
        # alter the finite-precision L-BFGS trajectory despite representing
        # exactly the same conditional distribution.
        log_base -= logsumexp(log_base)
        for correction, states in (
            (correction_ya, p_ya.active_context),
            (correction_yb, p_yb.active_context),
        ):
            counts = np.bincount(states, minlength=p_ya.vocabulary_size)
            for state in np.flatnonzero(counts == p_ya.vocabulary_size):
                selected = states == state
                correction[selected] -= np.max(correction[selected])
    return log_base, correction_ya, correction_yb


def fit_sparse_grouped_checkpoints(
    checkpoints: Sequence[SparseGroupedCheckpoint],
    *,
    initial_results: Sequence[SparseGroupedResult] | None = None,
    interleave: int = 1,
    max_iterations: int = 5_000,
    tolerance: float = 1e-5,
    solver: str = "lbfgs",
    margin_workers: int = 1,
    evaluator: str = "union",
    lbfgs_trust_radius: float = 16.0,
    lbfgs_precondition: bool = False,
    lbfgs_precondition_max_scale: float = 10.0,
    initialization: str = "unigram",
    checkpoint_transfer: str = "copy",
    stochastic_replicas: int = 12,
    stochastic_edge_blocks: int = 128,
    stochastic_learning_rate: float = 3e-2,
    stochastic_minimum_learning_rate: float = 3e-3,
    stochastic_exact_interval: int = 50,
    stochastic_trust_radius: float = 8.0,
    stochastic_seed: int = 71,
    layered_graph: LayeredIntersectionGraph | None = None,
    layered_checkpoint_indices: Sequence[int] | None = None,
    sampled_ab_major_graph: ABMajorIntersectionGraph | None = None,
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
    if checkpoint_transfer not in ("copy", "tree_delta"):
        raise ValueError("checkpoint_transfer must be 'copy' or 'tree_delta'")
    points = list(checkpoints)
    if not points:
        return []
    if (layered_graph is None) != (layered_checkpoint_indices is None):
        raise ValueError(
            "layered graph and checkpoint indices must be supplied together"
        )
    layered_indices = (
        None if layered_checkpoint_indices is None
        else list(layered_checkpoint_indices)
    )
    if layered_indices is not None and len(layered_indices) != len(points):
        raise ValueError("layered checkpoint indices must match checkpoints")
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
                if (
                    initialization != "unigram"
                    and point.projected_ya is not None
                    and point.projected_yb is not None
                ):
                    warm = projected_pair_warm_start(
                        point.projected_ya,
                        point.projected_yb,
                        point.problem.target_y,
                        initialization,
                    )
                elif initialization == "first_pair":
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
                if (
                    checkpoint_transfer == "tree_delta"
                    and previous_point.projected_ya is not None
                    and previous_point.projected_yb is not None
                    and point.projected_ya is not None
                    and point.projected_yb is not None
                ):
                    previous_tree = projected_pair_warm_start(
                        previous_point.projected_ya,
                        previous_point.projected_yb,
                        previous_point.problem.target_y,
                        "pair_product",
                        canonicalize=True,
                    )
                    tree_result = SparseGroupedResult(
                        previous_tree[0], previous_tree[1], previous_tree[2],
                        0, np.nan, np.nan, np.nan, False,
                    )
                    transferred_tree = transfer_sparse_warm_start(
                        previous_point.problem, tree_result, point.problem
                    )
                    current_tree = projected_pair_warm_start(
                        point.projected_ya,
                        point.projected_yb,
                        point.problem.target_y,
                        "pair_product",
                        canonicalize=True,
                    )
                    warm = tuple(
                        fitted + current - old
                        for fitted, current, old in zip(
                            warm, current_tree, transferred_tree
                        )
                    )
            if solver == "stochastic":
                stochastic = stochastic_sparse_dual_approach(
                    point.problem, *warm,
                    steps=max_iterations,
                    batch_size=1,
                    learning_rate=stochastic_learning_rate,
                    minimum_learning_rate=stochastic_minimum_learning_rate,
                    exact_interval=stochastic_exact_interval,
                    seed=stochastic_seed + index,
                    trust_radius=stochastic_trust_radius,
                    sampling="blocks",
                    replicas=stochastic_replicas,
                    stochastic_workers=margin_workers,
                    edge_blocks=stochastic_edge_blocks,
                    variance_reduction=True,
                    certificate_tolerance=tolerance,
                    exact_margin_workers=margin_workers,
                    optimizer="adam_plateau",
                    exact_layered_graph=layered_graph,
                    exact_layered_checkpoint=(
                        None if layered_indices is None
                        else layered_indices[index]
                    ),
                    sampled_ab_major_graph=sampled_ab_major_graph,
                )
                record = min(stochastic.trace, key=lambda item: float(
                    item["exact_certificate"]
                ))
                result = SparseGroupedResult(
                    stochastic.log_base_y,
                    stochastic.correction_ya,
                    stochastic.correction_yb,
                    stochastic.steps,
                    float(record["residual_ya_l1"]),
                    float(record["residual_yb_l1"]),
                    float(record["residual_y_l1"]),
                    stochastic.best_exact_certificate <= tolerance,
                    stochastic.exact_evaluations,
                )
            else:
                result = sparse_grouped_ipf(
                    point.problem,
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                    log_base_y=warm[0],
                    correction_ya=warm[1],
                    correction_yb=warm[2],
                    solver=solver,
                    margin_workers=margin_workers,
                    evaluator=evaluator,
                    lbfgs_trust_radius=lbfgs_trust_radius,
                    lbfgs_precondition=lbfgs_precondition,
                    lbfgs_precondition_max_scale=(
                        lbfgs_precondition_max_scale
                    ),
                    _layered_graph=layered_graph,
                    _layered_checkpoint=(
                        None if layered_indices is None
                        else layered_indices[index]
                    ),
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
    trace: list[dict[str, float | int | str]] | None = None,
    trace_interval: int = 25,
    reduce_gauge: bool = True,
    phase_timing: dict[str, float] | None = None,
    evaluator: str = "union",
    lbfgs_trust_radius: float = 16.0,
    lbfgs_precondition: bool = False,
    lbfgs_precondition_max_scale: float = 10.0,
    _intersection_plan: SparseIntersectionPlan | None = None,
    _layered_graph: LayeredIntersectionGraph | None = None,
    _layered_checkpoint: int | None = None,
    _trace_phase: str | None = None,
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
    if evaluator not in ("union", "factorized", "layered", "auto"):
        raise ValueError(
            "evaluator must be 'union', 'factorized', 'layered' or 'auto'"
        )
    if evaluator == "layered" and (
        _layered_graph is None or _layered_checkpoint is None
    ):
        raise ValueError("layered evaluator requires a graph and checkpoint")
    if not np.isfinite(lbfgs_trust_radius) or lbfgs_trust_radius <= 0.0:
        raise ValueError("lbfgs_trust_radius must be finite and positive")
    if lbfgs_precondition_max_scale < 1.0:
        raise ValueError("L-BFGS precondition scale must be at least one")
    if trace_interval < 1:
        raise ValueError("trace_interval must be positive")

    if evaluator == "auto":
        active_per_a = np.bincount(problem.active_ya_a, minlength=v)
        active_per_b = np.bincount(problem.active_yb_b, minlength=v)
        union_incidence_upper = int(np.sum(
            active_per_a[problem.edge_a] + active_per_b[problem.edge_b]
        ))
        # For small sparse problems the positive union calculation is both
        # cheap and immune to cancellation in the intersection identity.
        # Larger problems use the factorized plan that makes scaling possible.
        evaluator = (
            "union" if union_incidence_upper <= 5_000_000 else "factorized"
        )

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

    intersection_plan = (
        _intersection_plan or build_sparse_intersection_plan(problem)
        if evaluator == "factorized" else None
    )
    layered_context_rows = None
    if evaluator == "layered":
        layered_context_rows = []
        for states in (problem.active_ya_a, problem.active_yb_b):
            order = np.argsort(states, kind="stable")
            ptr = np.r_[0, np.cumsum(
                np.bincount(states, minlength=v), dtype=np.int64
            )]
            layered_context_rows.append((ptr, order))
        layered_context_rows = tuple(layered_context_rows)
    rows1: list[dict[int, int]] = [{} for _ in range(v)]
    rows2: list[dict[int, int]] = [{} for _ in range(v)]
    if evaluator == "union":
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
    if evaluator == "union":
        for edge, (a_, b_) in enumerate(zip(
            problem.edge_a, problem.edge_b
        )):
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
    margin_evaluations = 0
    edge_ptr = np.r_[
        0, np.cumsum(np.bincount(ue, minlength=edge_count), dtype=np.int64)
    ]
    edge_executor = (
        ThreadPoolExecutor(max_workers=margin_workers)
        if margin_workers > 1 else None
    )
    if edge_executor is None or evaluator != "union":
        edge_shards = [(0, edge_count)]
    else:
        desired = np.linspace(0, len(ue), margin_workers + 1)
        boundaries = np.searchsorted(edge_ptr, desired, side="left")
        boundaries[0] = 0
        boundaries[-1] = edge_count
        boundaries = np.unique(boundaries)
        edge_shards = [
            (int(lo), int(hi))
            for lo, hi in pairwise(boundaries)
            if hi > lo
        ]
    if intersection_plan is None or edge_executor is None:
        intersection_shards = None
    else:
        boundaries = np.linspace(
            0, len(intersection_plan.edge), margin_workers + 1,
            dtype=np.int64,
        )
        intersection_shards = [
            (int(lo), int(hi))
            for lo, hi in pairwise(np.unique(boundaries))
            if hi > lo
        ]

    def reduction_shards(ids, valid, output_size):
        positions = np.flatnonzero(valid)
        if edge_executor is None or not len(positions):
            return []
        order = positions[np.argsort(ids[positions], kind="stable")]
        ordered_ids = ids[order]
        desired = np.linspace(0, len(order), margin_workers + 1)
        boundaries = np.searchsorted(
            np.bincount(ordered_ids, minlength=output_size).cumsum(),
            desired,
            side="left",
        )
        boundaries[0] = 0
        boundaries[-1] = output_size
        boundaries = np.unique(boundaries)
        answer = []
        for lo, hi in pairwise(boundaries):
            start = np.searchsorted(ordered_ids, lo, side="left")
            stop = np.searchsorted(ordered_ids, hi, side="left")
            if hi > lo:
                answer.append((int(lo), int(hi), order[start:stop]))
        return answer

    has1_plan = ui1 >= 0
    has2_plan = ui2 >= 0
    reduction_shards_1 = reduction_shards(ui1, has1_plan, len(c1))
    reduction_shards_2 = reduction_shards(ui2, has2_plan, len(c2))
    reduction_shards_y = reduction_shards(
        uy, np.ones(len(uy), dtype=bool), v
    )

    def shutdown_executor() -> None:
        nonlocal edge_executor
        if edge_executor is not None:
            edge_executor.shutdown()
            edge_executor = None

    def finish(result: SparseGroupedResult) -> SparseGroupedResult:
        shutdown_executor()
        return result

    def margins():
        nonlocal margin_evaluations
        margin_evaluations += 1
        margin_started = perf_counter() if phase_timing is not None else 0.0

        def mark(name, started):
            if phase_timing is not None:
                phase_timing[name] = (
                    phase_timing.get(name, 0.0) + perf_counter() - started
                )

        if evaluator == "layered":
            factorized = sparse_factorized_margins_layered(
                problem, _layered_graph, _layered_checkpoint,
                lb, c1, c2, workers=margin_workers,
                context_rows=layered_context_rows,
                phase_timing=phase_timing,
            )
            mark("layered_seconds", margin_started)
            mark("margin_total_seconds", margin_started)
            return (
                factorized.target_y,
                factorized.active_ya,
                factorized.active_yb,
                factorized.log_normalizer,
            )

        if intersection_plan is not None:
            factorized = sparse_factorized_margins(
                problem, intersection_plan, lb, c1, c2,
                executor=edge_executor,
                intersection_shards=intersection_shards,
            )
            mark("factorized_seconds", margin_started)
            mark("margin_total_seconds", margin_started)
            return (
                factorized.target_y,
                factorized.active_ya,
                factorized.active_yb,
                factorized.log_normalizer,
            )

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

        def normalize_shard(bounds):
            lo, hi = bounds
            start = edge_ptr[lo]
            stop = edge_ptr[hi]
            local_edge = ue[start:stop] - lo
            local_count = hi - lo
            local_union_mass = np.bincount(
                local_edge, weights=base[uy[start:stop]],
                minlength=local_count,
            )
            local_max = np.full(local_count, -np.inf)
            np.maximum.at(local_max, local_edge, term[start:stop])
            local_scaled = np.bincount(
                local_edge,
                weights=np.exp(term[start:stop] - local_max[local_edge]),
                minlength=local_count,
            )
            local_corrected = local_max + np.log(
                np.maximum(local_scaled, tiny)
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                local_background = np.log1p(
                    -np.minimum(local_union_mass, 1.0)
                )
            local_log_z = np.logaddexp(
                local_background, local_corrected
            )
            probability = np.exp(
                term[start:stop] - local_log_z[local_edge]
            )
            return lo, hi, start, stop, local_union_mass, local_log_z, probability

        union_mass = np.empty(edge_count)
        log_z = np.empty(edge_count)
        corrected_probability = np.empty(len(ue))
        phase_started = perf_counter() if phase_timing is not None else 0.0
        normalized = (
            map(normalize_shard, edge_shards)
            if edge_executor is None
            else edge_executor.map(normalize_shard, edge_shards)
        )
        for lo, hi, start, stop, local_mass, local_z, probability in normalized:
            union_mass[lo:hi] = local_mass
            log_z[lo:hi] = local_z
            corrected_probability[start:stop] = probability
        mark("normalization_seconds", phase_started)
        phase_started = perf_counter() if phase_timing is not None else 0.0
        if edge_executor is None:
            m1 = np.bincount(
                ui1[has1],
                weights=edge_weight[ue[has1]]
                * corrected_probability[has1],
                minlength=len(c1),
            )
            m2 = np.bincount(
                ui2[has2],
                weights=edge_weight[ue[has2]]
                * corrected_probability[has2],
                minlength=len(c2),
            )
        else:
            def reduce_features(task):
                ids, lo, hi, positions = task
                return lo, hi, np.bincount(
                    ids[positions] - lo,
                    weights=(
                        edge_weight[ue[positions]]
                        * corrected_probability[positions]
                    ),
                    minlength=hi - lo,
                )

            tasks = [
                (ui1, lo, hi, positions)
                for lo, hi, positions in reduction_shards_1
            ] + [
                (ui2, lo, hi, positions)
                for lo, hi, positions in reduction_shards_2
            ]
            reduced = list(edge_executor.map(reduce_features, tasks))
            m1 = np.zeros(len(c1))
            m2 = np.zeros(len(c2))
            split = len(reduction_shards_1)
            for lo, hi, contribution in reduced[:split]:
                m1[lo:hi] = contribution
            for lo, hi, contribution in reduced[split:]:
                m2[lo:hi] = contribution
        mark("feature_reduction_seconds", phase_started)

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
        phase_started = perf_counter() if phase_timing is not None else 0.0
        if edge_executor is None:
            uncorrected = np.exp(
                log_base[uy[pos_analytic]] - log_z[ue[pos_analytic]]
            )
            delta = edge_weight[ue[pos_analytic]] * (
                corrected_probability[pos_analytic] - uncorrected
            )
            my = base * background_scale + np.bincount(
                uy[pos_analytic], weights=delta, minlength=v
            )
        else:
            def reduce_target(task):
                lo, hi, positions = task
                selected = positions[pos_analytic[positions]]
                uncorrected = np.exp(
                    log_base[uy[selected]] - log_z[ue[selected]]
                )
                delta = edge_weight[ue[selected]] * (
                    corrected_probability[selected] - uncorrected
                )
                return lo, hi, np.bincount(
                    uy[selected] - lo, weights=delta, minlength=hi - lo
                )

            my = base * background_scale
            for lo, hi, contribution in edge_executor.map(
                reduce_target, reduction_shards_y
            ):
                my[lo:hi] += contribution
        mark("target_reduction_seconds", phase_started)
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

        phase_started = perf_counter() if phase_timing is not None else 0.0
        if margin_workers == 1 or len(blocks) < 2:
            contributions = map(dense_contribution, blocks)
            for contribution in contributions:
                my += contribution
        else:
            for contribution in edge_executor.map(dense_contribution, blocks):
                    my += contribution
        mark("dense_fallback_seconds", phase_started)
        mark("margin_total_seconds", margin_started)
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
        # Whether an inactive class exists is a structural fact.  Inferring
        # it from ``state_mass - target_active`` is unstable for saturated
        # rows: cancellation can leave a fictitious mass around 1e-13 and
        # produce corrections of hundreds when the current difference
        # rounds to zero.
        active_count = np.bincount(state_ids, minlength=v)
        saturated = active_count == v
        has_inactive = (~saturated) & (wanted_inactive > tiny)
        shift[has_inactive] = (
            np.log(wanted_inactive[has_inactive])
            - np.log(np.maximum(current_inactive[has_inactive], tiny))
        )
        correction -= shift[state_ids]
        # With no inactive class the whole row is explicit.  Fix its harmless
        # row-constant gauge in one segmented maximum.
        if np.any(saturated[state_ids]):
            row_max = np.full(v, -np.inf)
            np.maximum.at(row_max, state_ids, correction)
            select = saturated[state_ids]
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

    def record_trace(phase, iteration, my, m1, m2, log_z):
        if trace is None:
            return
        residuals = diagnostics(my, m1, m2)
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
        names = ("y", "ya", "yb")
        all_factors = np.concatenate([log_base, c1, c2])
        absolute = np.abs(all_factors)
        trace.append({
            "phase": phase,
            "iteration": int(iteration),
            "objective": objective,
            "gradient_l2": float(np.linalg.norm(gradient)),
            "gradient_linf": float(np.max(np.abs(gradient))),
            "residual_y_l1": residuals[0],
            "residual_ya_l1": residuals[1],
            "residual_yb_l1": residuals[2],
            "certificate": max(residuals),
            "limiting_margin": names[int(np.argmax(residuals))],
            "factor_abs_p50": float(np.quantile(absolute, 0.5)),
            "factor_abs_p90": float(np.quantile(absolute, 0.9)),
            "factor_abs_p99": float(np.quantile(absolute, 0.99)),
            "factor_abs_max": float(np.max(absolute)),
        })

    if solver == "lbfgs":
        # A stochastic warm point can be mathematically finite while its
        # factorized expm1 expansion overflows in floating point.  Recover by
        # moving on the straight line from the neutral model to the proposed
        # warm point.  This changes only the initialization, never the target
        # or the final certificate.  It is also the natural backtracking rule:
        # retain as much of the warm start as the exact evaluator can certify.
        proposed_lb = lb.copy()
        proposed_c1 = c1.copy()
        proposed_c2 = c2.copy()
        try:
            preflight_margins = margins()
        except FloatingPointError:
            neutral_lb = np.log(np.maximum(problem.target_y, tiny))
            recovered = False
            for exponent in range(1, 21):
                fraction = 2.0 ** -exponent
                lb[:] = neutral_lb + fraction * (proposed_lb - neutral_lb)
                c1[:] = fraction * proposed_c1
                c2[:] = fraction * proposed_c2
                try:
                    preflight_margins = margins()
                except FloatingPointError:
                    continue
                recovered = True
                break
            if not recovered:
                raise FloatingPointError(
                    "no finite exact margin point on the neutral-to-warm "
                    "initialization segment"
                )
        full_initial = np.concatenate([lb, c1, c2])
        free = np.ones(len(full_initial), dtype=bool)
        if reduce_gauge:
            # Global baseline constants cancel from every conditional.
            free[0] = False
            # A constant added to a structurally saturated correction row
            # also cancels.  Anchor one explicitly present target per row.
            for states, offset in (
                (problem.active_ya_a, n_base),
                (problem.active_yb_b, n_first),
            ):
                counts = np.bincount(states, minlength=v)
                for state in np.flatnonzero(counts == v):
                    free[offset + np.flatnonzero(states == state)[0]] = False
        fixed_parameters = full_initial.copy()
        parameter_scale = np.ones_like(full_initial)
        if lbfgs_precondition:
            conditional_ya = np.divide(
                problem.target_ya,
                pa[problem.active_ya_a],
                out=np.zeros_like(problem.target_ya),
                where=pa[problem.active_ya_a] > 0.0,
            )
            conditional_yb = np.divide(
                problem.target_yb,
                pb[problem.active_yb_b],
                out=np.zeros_like(problem.target_yb),
                where=pb[problem.active_yb_b] > 0.0,
            )
            diagonal = np.concatenate([
                problem.target_y * (1.0 - problem.target_y),
                problem.target_ya * np.maximum(0.0, 1.0 - conditional_ya),
                problem.target_yb * np.maximum(0.0, 1.0 - conditional_yb),
            ])
            parameter_scale = np.minimum(
                lbfgs_precondition_max_scale,
                1.0 / np.sqrt(np.maximum(diagonal, 1e-8)),
            )
            initial = np.zeros(np.count_nonzero(free), dtype=np.float64)
        else:
            initial = full_initial[free]
        best_parameters = full_initial.copy()
        best_certificate = float("inf")
        best_objective = float("inf")
        function_evaluations = 0
        accepted_iterations = 0
        last_reduced: np.ndarray | None = None
        last_certificate = float("inf")

        class _CertificateReached(Exception):
            pass

        def expand(reduced_parameters):
            parameters = fixed_parameters.copy()
            if lbfgs_precondition:
                parameters[free] += (
                    parameter_scale[free] * reduced_parameters
                )
            else:
                parameters[free] = reduced_parameters
            return parameters

        def objective_gradient(reduced_parameters):
            nonlocal best_parameters, best_certificate, best_objective
            nonlocal function_evaluations
            nonlocal last_reduced, last_certificate
            parameters = expand(reduced_parameters)
            lb[:] = parameters[:n_base]
            c1[:] = parameters[n_base:n_first]
            c2[:] = parameters[n_first:]
            try:
                if (
                    function_evaluations == 0
                    and np.array_equal(reduced_parameters, initial)
                ):
                    my, m1, m2, log_z = preflight_margins
                else:
                    my, m1, m2, log_z = margins()
            except FloatingPointError:
                # L-BFGS line searches are allowed to propose invalid points.
                # Return a smooth finite barrier pointing back toward the
                # certified initialization instead of aborting the run.  The
                # optimizer will shorten/reject this trial in the usual way.
                displacement = reduced_parameters - initial
                scale = 1e12 / max(1, len(displacement))
                function_evaluations += 1
                last_reduced = np.array(reduced_parameters, copy=True)
                last_certificate = float("inf")
                return (
                    1e12 + scale * float(displacement @ displacement),
                    2.0 * scale * displacement,
                )
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
            last_reduced = np.array(reduced_parameters, copy=True)
            last_certificate = certificate
            if trace is not None and function_evaluations % trace_interval == 0:
                record_trace(
                    _trace_phase or "lbfgs", function_evaluations,
                    my, m1, m2, log_z,
                )
            function_evaluations += 1
            if (
                certificate < best_certificate
                or (
                    certificate == best_certificate
                    and objective < best_objective
                )
            ):
                best_certificate = certificate
                best_objective = objective
                best_parameters = parameters.copy()
            reduced_gradient = gradient[free]
            if lbfgs_precondition:
                reduced_gradient = parameter_scale[free] * reduced_gradient
            return objective, reduced_gradient

        def stop_at_certificate(accepted_parameters):
            nonlocal accepted_iterations
            accepted_iterations += 1
            if (
                last_reduced is None
                or not np.array_equal(accepted_parameters, last_reduced)
            ):
                objective_gradient(accepted_parameters)
            if last_certificate < tolerance:
                raise _CertificateReached

        lbfgs_iterations = min(max_iterations, 1_000)
        try:
            optimized = minimize(
                objective_gradient,
                initial,
                method="L-BFGS-B",
                jac=True,
                callback=stop_at_certificate,
                # L-BFGS is an approach phase followed, when necessary, by
                # unrestricted IPF.  A finite displacement region prevents a
                # line-search trial from changing log factors by hundreds and
                # overflowing the intersection expansion.  It does not bound
                # the final estimator: IPF polishing remains unconstrained.
                bounds=(
                    [
                        (-lbfgs_trust_radius / value,
                         lbfgs_trust_radius / value)
                        for value in parameter_scale[free]
                    ]
                    if lbfgs_precondition else
                    [
                        (value - lbfgs_trust_radius,
                         value + lbfgs_trust_radius)
                        for value in initial
                    ]
                ),
                options={
                    "maxiter": lbfgs_iterations,
                    # scipy stops on the largest gradient component whereas
                    # our certificate is an L1 residual over all components.
                    "gtol": tolerance / max(10, len(initial)),
                    "ftol": 0.0,
                    "maxls": 40,
                    "maxfun": lbfgs_iterations * 20,
                },
            )
            optimized_iterations = int(optimized.nit)
        except _CertificateReached:
            optimized_iterations = accepted_iterations
        # The dual objective may still decrease along nearly flat gauge
        # directions after the actual margin certificate has worsened.  The
        # certificate, not scipy's final iterate, chooses the handoff to IPF.
        parameters = best_parameters
        lb[:] = parameters[:n_base]
        c1[:] = parameters[n_base:n_first]
        c2[:] = parameters[n_first:]
        my, m1, m2, log_z = margins()
        residual_y, grouped_ya, grouped_yb = diagnostics(my, m1, m2)
        record_trace(
            "lbfgs_best", optimized_iterations, my, m1, m2, log_z
        )
        converged = max(residual_y, grouped_ya, grouped_yb) < tolerance
        if not converged:
            # The polishing solver creates its own persistent pool.  Release
            # the now-idle L-BFGS pool first rather than retaining two worker
            # teams per warm chain.
            shutdown_executor()
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
                trace=trace,
                trace_interval=trace_interval,
                reduce_gauge=reduce_gauge,
                phase_timing=phase_timing,
                evaluator=evaluator,
                lbfgs_trust_radius=lbfgs_trust_radius,
                lbfgs_precondition=lbfgs_precondition,
                lbfgs_precondition_max_scale=lbfgs_precondition_max_scale,
                _intersection_plan=intersection_plan,
                _layered_graph=_layered_graph,
                _layered_checkpoint=_layered_checkpoint,
                _trace_phase="ipf_polish",
            )
            return finish(SparseGroupedResult(
                polished.log_base_y,
                polished.correction_ya,
                polished.correction_yb,
                optimized_iterations + polished.iterations,
                polished.grouped_residual_ya_l1,
                polished.grouped_residual_yb_l1,
                polished.residual_y_l1,
                polished.converged,
                margin_evaluations + polished.margin_evaluations,
            ))
        return finish(SparseGroupedResult(
            lb.copy(), c1.copy(), c2.copy(), optimized_iterations,
            grouped_ya, grouped_yb, residual_y, converged,
            margin_evaluations,
        ))

    # Alternating projections decrease their own divergence objective, but the
    # user-facing certificate (the largest of three L1 margin errors) need not
    # be monotone.  Preserve the best certified iterate, including the warm
    # start supplied by a preceding L-BFGS phase.
    my, m1, m2, log_z = margins()
    residual_y, grouped_ya, grouped_yb = diagnostics(my, m1, m2)
    record_trace(_trace_phase or solver, 0, my, m1, m2, log_z)
    iteration_best_certificate = max(residual_y, grouped_ya, grouped_yb)
    iteration_best_state = (lb.copy(), c1.copy(), c2.copy())
    iteration_best_residuals = (residual_y, grouped_ya, grouped_yb)

    def finite_mixed_update(array, before, before_margins):
        """Accept a full IPF sub-update or a finite old/new mixture."""

        proposed = array.copy()
        delta = proposed - before
        for exponent in range(21):
            if exponent:
                array[:] = before + (2.0 ** -exponent) * delta
            try:
                return margins()
            except FloatingPointError:
                continue
        # Reject this sub-update.  The previous state was certified finite.
        array[:] = before
        return before_margins

    for iteration in range(1, max_iterations + 1):
        x_before = np.concatenate([lb, c1, c2])
        before_margins = margins()
        my = before_margins[0]
        lb_before = lb.copy()
        lb += np.log(np.maximum(problem.target_y, tiny)) - np.log(
            np.maximum(my, tiny)
        )
        lb -= lb.max()
        after_lb = finite_mixed_update(lb, lb_before, before_margins)
        m1 = after_lb[1]
        c1_before = c1.copy()
        update(c1, problem.target_ya, m1, problem.active_ya_a,
               pa, target_active_a)
        after_c1 = finite_mixed_update(c1, c1_before, after_lb)
        m2 = after_c1[2]
        c2_before = c2.copy()
        update(c2, problem.target_yb, m2, problem.active_yb_b,
               pb, target_active_b)
        my, m1, m2, log_z = finite_mixed_update(
            c2, c2_before, after_c1
        )
        residual_y, grouped_ya, grouped_yb = diagnostics(my, m1, m2)
        if iteration % trace_interval == 0 or iteration == max_iterations:
            record_trace(
                _trace_phase or solver, iteration, my, m1, m2, log_z
            )
        certificate = max(residual_y, grouped_ya, grouped_yb)
        if certificate < iteration_best_certificate:
            iteration_best_certificate = certificate
            iteration_best_state = (lb.copy(), c1.copy(), c2.copy())
            iteration_best_residuals = (residual_y, grouped_ya, grouped_yb)
        if certificate < tolerance:
            if trace is not None and iteration % trace_interval:
                record_trace(
                    _trace_phase or solver, iteration, my, m1, m2, log_z
                )
            return finish(SparseGroupedResult(
                lb, c1, c2, iteration, grouped_ya, grouped_yb,
                residual_y, True, margin_evaluations
            ))
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
    return finish(SparseGroupedResult(
        best_lb, best_c1, best_c2, max_iterations, best_ya, best_yb,
        best_y, False, margin_evaluations
    ))


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
