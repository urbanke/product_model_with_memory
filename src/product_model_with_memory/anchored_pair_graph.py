"""Reference implementation of the hard-YA relaxed pair graph.

This module deliberately starts with dense, small-alphabet operations.  It is
the numerical oracle for the sparse production implementation, not the
paper-scale evaluator.

The model is

    q(y,a,b) = p_YA(y,a) r_AB(b|a) exp(u(y,b) + v(a,b)) / Z(y,a).

Consequently the fitted joint has the supplied YA marginal exactly for every
finite value of the soft-factor parameters.  ``u`` and ``v`` are available to
fit the YB and AB constraints by the relaxed stochastic objective.
"""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass


ANCHORED_PAIR_GRAPH_MODEL = "anchored_ya_relaxed_pair_graph_v1"
ANCHORED_PAIR_GRAPH_INITIALIZER = "cold_pair_midpoint_v1"
ANCHORED_FULL_PAIR_REFERENCE = "anchored_ya_full_layered_ab_reference_v1"


def _probability_table(name: str, values: np.ndarray) -> np.ndarray:
    table = np.asarray(values, dtype=np.float64)
    if table.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional table")
    if not np.all(np.isfinite(table)) or np.any(table < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    total = float(np.sum(table))
    if not total > 0.0:
        raise ValueError(f"{name} must have positive mass")
    return table / total


def ab_conditional(p_ab: np.ndarray) -> np.ndarray:
    """Return ``p(b|a)``; zero-mass A rows receive a uniform conditional."""

    ab = _probability_table("p_ab", p_ab)
    row_mass = np.sum(ab, axis=1, keepdims=True)
    result = np.empty_like(ab)
    np.divide(ab, row_mass, out=result, where=row_mass > 0.0)
    zero = row_mass[:, 0] == 0.0
    if np.any(zero):
        result[zero] = 1.0 / ab.shape[1]
    return result


def anchored_joint(
    p_ya: np.ndarray,
    p_ab: np.ndarray,
    correction_yb: np.ndarray,
    correction_ab: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the dense anchored joint ``q[y,a,b]``.

    Normalization occurs over B separately for each fixed ``(y,a)``.  This is
    the operation that makes the YA constraint exact rather than penalized.
    """

    ya = _probability_table("p_ya", p_ya)
    r_ab = ab_conditional(p_ab)
    u = np.asarray(correction_yb, dtype=np.float64)
    if ya.shape[0] != u.shape[0] or r_ab.shape[1] != u.shape[1]:
        raise ValueError("correction_yb must have shape (Y, B)")
    if ya.shape[1] != r_ab.shape[0]:
        raise ValueError("YA and AB tables disagree on alphabet A")
    if correction_ab is None:
        v = np.zeros_like(r_ab)
    else:
        v = np.asarray(correction_ab, dtype=np.float64)
        if v.shape != r_ab.shape:
            raise ValueError("correction_ab must have shape (A, B)")
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
        raise ValueError("corrections must be finite")

    logits = u[:, None, :] + v[None, :, :]
    logits -= np.max(logits, axis=2, keepdims=True)
    weights = r_ab[None, :, :] * np.exp(logits)
    normalizer = np.sum(weights, axis=2, keepdims=True)
    if np.any(normalizer <= 0.0):
        raise FloatingPointError("anchored conditional has zero normalizer")
    return ya[:, :, None] * weights / normalizer


def dense_layered_pair(pair) -> np.ndarray:
    """Materialize a projected/layered pair only for small-alphabet gates."""

    v = int(pair.vocabulary_size)
    dense = (
        np.asarray(pair.right)[:, None]
        * np.asarray(pair.left)[None, :]
        * np.asarray(pair.background)[None, :]
    )
    dense = np.array(dense, dtype=np.float64, copy=True)
    np.add.at(
        dense, (np.asarray(pair.active_y), np.asarray(pair.active_context)),
        np.asarray(pair.right)[pair.active_y]
        * np.asarray(pair.left)[pair.active_context]
        * np.asarray(pair.delta),
    )
    if dense.shape != (v, v) or not np.isclose(np.sum(dense), 1.0):
        raise ValueError("layered pair does not define a normalized dense law")
    return dense


def dense_full_pair_dual(
    p_ya: np.ndarray,
    p_ab: np.ndarray,
    yb_y: np.ndarray,
    yb_b: np.ndarray,
    target_yb: np.ndarray,
    ab_a: np.ndarray,
    ab_b: np.ndarray,
    target_ab: np.ndarray,
    correction_yb: np.ndarray,
    correction_ab: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Dense oracle retaining implicit YA and AB mass exactly."""

    ya = _probability_table("p_ya", p_ya)
    ab = _probability_table("p_ab", p_ab)
    u = np.asarray(correction_yb, dtype=np.float64)
    v = np.asarray(correction_ab, dtype=np.float64)
    dense_u = np.zeros((ya.shape[0], ab.shape[1]))
    dense_v = np.zeros_like(ab)
    dense_u[np.asarray(yb_y), np.asarray(yb_b)] = u
    dense_v[np.asarray(ab_a), np.asarray(ab_b)] = v
    r = ab_conditional(ab)
    weights = r[None, :, :] * np.exp(dense_u[:, None, :] + dense_v[None, :, :])
    z = np.sum(weights, axis=2)
    joint = ya[:, :, None] * weights / z[:, :, None]
    objective = float(np.sum(ya * np.log(z)))
    objective -= float(np.dot(target_yb, u) + np.dot(target_ab, v))
    model_yb = np.sum(joint, axis=1)
    model_ab = np.sum(joint, axis=0)
    return (
        objective,
        model_yb[np.asarray(yb_y), np.asarray(yb_b)] - target_yb,
        model_ab[np.asarray(ab_a), np.asarray(ab_b)] - target_ab,
    )


def cold_pair_midpoint(
    p_ya: np.ndarray,
    p_yb: np.ndarray,
    p_ab: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the cold midpoint initialization in anchored coordinates.

    The historical midpoint averages the natural parameters of the exact
    ``Y|A`` and ``Y|B`` pair models.  Here ``p_YA`` is already the hard base
    measure, so its natural parameter is present in full.  The remaining
    half-strength ``Y|B`` likelihood ratio is placed in ``u`` and the AB soft
    correction starts neutral.  Each checkpoint recomputes this initializer
    solely from its own prefix; no fitted state is transferred.
    """

    ya = _probability_table("p_ya", p_ya)
    yb = _probability_table("p_yb", p_yb)
    ab = _probability_table("p_ab", p_ab)
    if ya.shape[0] != yb.shape[0]:
        raise ValueError("YA and YB tables disagree on alphabet Y")
    if ya.shape[1] != ab.shape[0] or yb.shape[1] != ab.shape[1]:
        raise ValueError("pair tables use incompatible alphabets")

    p_y = np.sum(ya, axis=1)
    p_b = np.sum(yb, axis=0)
    independent = p_y[:, None] * p_b[None, :]
    tiny = np.finfo(np.float64).tiny
    correction_yb = 0.5 * (
        np.log(np.maximum(yb, tiny)) - np.log(np.maximum(independent, tiny))
    )
    correction_ab = np.zeros_like(ab)
    return correction_yb, correction_ab


@dataclass(frozen=True)
class AnchoredSparseProblem:
    """Sparse edge representation of the anchored convex dual.

    ``row_ptr`` indexes compatible B values for every active YA row.  YB and
    AB factors have stable sparse edge IDs, so the representation can later
    be backed directly by the repository's persisted intersection graph.
    """

    ya_y: np.ndarray
    ya_a: np.ndarray
    ya_probability: np.ndarray
    row_ptr: np.ndarray
    triple_b: np.ndarray
    triple_yb: np.ndarray
    triple_ab: np.ndarray
    log_r_ab: np.ndarray
    yb_y: np.ndarray
    yb_b: np.ndarray
    target_yb: np.ndarray
    ab_a: np.ndarray
    ab_b: np.ndarray
    target_ab: np.ndarray


@dataclass(frozen=True)
class AnchoredSgdResult:
    """Deterministic result of the relaxed cold-start SGD approach."""

    correction_yb: np.ndarray
    correction_ab: np.ndarray
    steps: int
    batch_size: int
    seed: int
    initial_objective: float
    best_objective: float
    final_objective: float


@dataclass(frozen=True)
class FullImplicitSgdResult:
    correction_yb: np.ndarray
    correction_ab: np.ndarray
    steps: int
    batch_size: int
    training_seed: int
    validation_seed: int
    validation_samples: int
    initial_validation_objective: float
    best_validation_objective: float
    final_validation_objective: float


@dataclass(frozen=True)
class AnchoredIntersectionProblem:
    """Hard-YA problem backed by the existing correction intersections."""

    edge_a: np.ndarray
    edge_probability: np.ndarray
    ya_a: np.ndarray
    target_ya: np.ndarray
    target_yb: np.ndarray
    triangle_ya: np.ndarray
    triangle_yb: np.ndarray
    triangle_ab: np.ndarray
    ab_order: np.ndarray
    ab_ptr: np.ndarray
    triangle_order: np.ndarray
    triangle_ptr: np.ndarray
    edge_local_position: np.ndarray


@dataclass(frozen=True)
class ImplicitYaIntersectionProblem:
    """Anchored topology retaining the layered YA background exactly."""

    vocabulary_size: int
    edge_a: np.ndarray
    edge_b: np.ndarray
    edge_probability: np.ndarray
    target_yb: np.ndarray
    yb_y: np.ndarray
    yb_b: np.ndarray
    ya_background: np.ndarray
    ya_active_y: np.ndarray
    ya_active_a: np.ndarray
    ya_delta: np.ndarray
    ya_context_probability: np.ndarray
    row_y: np.ndarray
    row_a: np.ndarray
    row_probability: np.ndarray
    triangle_row: np.ndarray
    triangle_yb: np.ndarray
    triangle_ab: np.ndarray
    ab_order: np.ndarray
    ab_ptr: np.ndarray
    yb_order: np.ndarray
    yb_ptr: np.ndarray


@dataclass(frozen=True)
class FullImplicitPairProblem:
    """Full layered YA/AB laws with only sparse active-factor coordinates."""

    vocabulary_size: int
    ya_background: np.ndarray
    ya_active_y: np.ndarray
    ya_active_a: np.ndarray
    ya_delta: np.ndarray
    ab_background: np.ndarray
    ab_active_a: np.ndarray
    ab_active_b: np.ndarray
    ab_delta: np.ndarray
    yb_y: np.ndarray
    yb_b: np.ndarray
    target_yb: np.ndarray
    factor_ab_a: np.ndarray
    factor_ab_b: np.ndarray
    target_ab: np.ndarray
    ab_delta_order: np.ndarray
    ab_delta_ptr: np.ndarray
    yb_order: np.ndarray
    yb_ptr: np.ndarray
    factor_ab_order: np.ndarray
    factor_ab_ptr: np.ndarray


def full_implicit_pair_problem(problem, p_ya, p_yb) -> FullImplicitPairProblem:
    """Build the production target with full stationary layered AB support."""

    v = int(problem.vocabulary_size)
    for pair, name in ((p_ya, "YA"), (p_yb, "YB")):
        if pair.vocabulary_size != v:
            raise ValueError(f"{name} pair uses a different alphabet")
        if not (np.array_equal(pair.left, np.ones(v)) and np.array_equal(pair.right, np.ones(v))):
            raise ValueError("full implicit pair v1 requires raw layered gauge")
    if np.any(p_ya.delta < 0.0):
        raise ValueError("full implicit pair v1 requires nonnegative layered deltas")
    # Stationarity orientation: P_AB(a,b) = P_YA(y=a, context=b).
    ab_a = np.asarray(p_ya.active_y, dtype=np.int32)
    ab_b = np.asarray(p_ya.active_context, dtype=np.int32)
    edge_a = np.asarray(problem.edge_a, dtype=np.int32)
    edge_b = np.asarray(problem.edge_b, dtype=np.int32)
    target_ab = np.asarray(p_ya.values(edge_a, edge_b), dtype=np.float64)

    def csr_order(major: np.ndarray, secondary: np.ndarray):
        order = np.lexsort((secondary, major))
        ptr = np.r_[0, np.cumsum(np.bincount(major, minlength=v), dtype=np.int64)]
        return order, ptr

    delta_order, delta_ptr = csr_order(ab_a, ab_b)
    yb_y = np.asarray(problem.active_yb_y, dtype=np.int32)
    yb_b = np.asarray(problem.active_yb_b, dtype=np.int32)
    yb_order, yb_ptr = csr_order(yb_y, yb_b)
    factor_order, factor_ptr = csr_order(edge_a, edge_b)
    return FullImplicitPairProblem(
        vocabulary_size=v,
        ya_background=np.asarray(p_ya.background, dtype=np.float64),
        ya_active_y=np.asarray(p_ya.active_y, dtype=np.int32),
        ya_active_a=np.asarray(p_ya.active_context, dtype=np.int32),
        ya_delta=np.asarray(p_ya.delta, dtype=np.float64),
        ab_background=np.asarray(p_ya.background, dtype=np.float64),
        ab_active_a=ab_a, ab_active_b=ab_b,
        ab_delta=np.asarray(p_ya.delta, dtype=np.float64),
        yb_y=yb_y, yb_b=yb_b,
        target_yb=np.asarray(problem.target_yb, dtype=np.float64),
        factor_ab_a=edge_a, factor_ab_b=edge_b, target_ab=target_ab,
        ab_delta_order=delta_order, ab_delta_ptr=delta_ptr,
        yb_order=yb_order, yb_ptr=yb_ptr,
        factor_ab_order=factor_order, factor_ab_ptr=factor_ptr,
    )


def full_implicit_pair_problem_explicit_ab(
    problem, p_ya, p_yb, p_ab,
) -> FullImplicitPairProblem:
    """Build the full-background topology from an explicitly estimated AB law.

    Unequal lag maps break the stationary identity ``P_AB(a,b)=P_YA(a,b)``:
    ``Y`` uses the emission alphabet while ``A`` and ``B`` use their own
    nested state maps.  The AB data-bearing sequence is therefore estimated
    separately at its natural target alphabet and padded with zero-mass
    states only when embedded in this square numerical graph.  This function
    deliberately does not replace the equal-alphabet production entry point.
    """

    v = int(problem.vocabulary_size)
    for pair, name in ((p_ya, "YA"), (p_yb, "YB"), (p_ab, "AB")):
        if pair.vocabulary_size != v:
            raise ValueError(f"{name} pair is not embedded in graph alphabet V")
        if not (np.array_equal(pair.left, np.ones(v))
                and np.array_equal(pair.right, np.ones(v))):
            raise ValueError("explicit AB topology requires raw layered gauge")
        if np.any(pair.delta < 0.0):
            raise ValueError(f"{name} layered deltas must be nonnegative")
    edge_a = np.asarray(problem.edge_a, dtype=np.int32)
    edge_b = np.asarray(problem.edge_b, dtype=np.int32)
    target_ab = np.asarray(p_ab.values(edge_a, edge_b), dtype=np.float64)

    def csr_order(major: np.ndarray, secondary: np.ndarray):
        order = np.lexsort((secondary, major))
        ptr = np.r_[0, np.cumsum(np.bincount(major, minlength=v), dtype=np.int64)]
        return order, ptr

    ab_a = np.asarray(p_ab.active_y, dtype=np.int32)
    ab_b = np.asarray(p_ab.active_context, dtype=np.int32)
    delta_order, delta_ptr = csr_order(ab_a, ab_b)
    yb_y = np.asarray(problem.active_yb_y, dtype=np.int32)
    yb_b = np.asarray(problem.active_yb_b, dtype=np.int32)
    yb_order, yb_ptr = csr_order(yb_y, yb_b)
    factor_order, factor_ptr = csr_order(edge_a, edge_b)
    return FullImplicitPairProblem(
        vocabulary_size=v,
        ya_background=np.asarray(p_ya.background, dtype=np.float64),
        ya_active_y=np.asarray(p_ya.active_y, dtype=np.int32),
        ya_active_a=np.asarray(p_ya.active_context, dtype=np.int32),
        ya_delta=np.asarray(p_ya.delta, dtype=np.float64),
        ab_background=np.asarray(p_ab.background, dtype=np.float64),
        ab_active_a=ab_a, ab_active_b=ab_b,
        ab_delta=np.asarray(p_ab.delta, dtype=np.float64),
        yb_y=yb_y, yb_b=yb_b,
        target_yb=np.asarray(problem.target_yb, dtype=np.float64),
        factor_ab_a=edge_a, factor_ab_b=edge_b, target_ab=target_ab,
        ab_delta_order=delta_order, ab_delta_ptr=delta_ptr,
        yb_order=yb_order, yb_ptr=yb_ptr,
        factor_ab_order=factor_order, factor_ab_ptr=factor_ptr,
    )


def full_implicit_pair_sampled_gradient(
    problem: FullImplicitPairProblem,
    correction_yb: np.ndarray,
    correction_ab: np.ndarray,
    *, batch_size: int, seed: int, step: int, workers: int = 1,
    slack_precision: float = float("inf"),
    variance_yb: np.ndarray | None = None,
    variance_ab: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Unbiased O(batch_size * V) gradient for full layered YA and AB."""

    if batch_size < 1 or step < 0 or workers < 1:
        raise ValueError("invalid sampled-gradient schedule")
    u, v = np.asarray(correction_yb), np.asarray(correction_ab)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(step)]))
    bg_mass = problem.vocabulary_size * float(np.sum(problem.ya_background))
    delta_mass = float(np.sum(problem.ya_delta))
    if not np.isclose(bg_mass + delta_mass, 1.0):
        raise ValueError("full YA sampling masses do not sum to one")
    choose_delta = rng.random(batch_size) >= bg_mass
    ys = np.empty(batch_size, dtype=np.int32)
    aa = np.empty(batch_size, dtype=np.int32)
    count = int(np.sum(~choose_delta))
    if count:
        aa[~choose_delta] = rng.choice(
            problem.vocabulary_size, count,
            p=problem.ya_background / np.sum(problem.ya_background),
        )
        ys[~choose_delta] = rng.integers(problem.vocabulary_size, size=count)
    count = int(np.sum(choose_delta))
    if count:
        selected = rng.choice(
            len(problem.ya_delta), count, p=problem.ya_delta / delta_mass,
        )
        ys[choose_delta] = problem.ya_active_y[selected]
        aa[choose_delta] = problem.ya_active_a[selected]
    target_yb_mass = float(np.sum(problem.target_yb))
    target_ab_mass = float(np.sum(problem.target_ab))
    negative_yb = rng.choice(
        len(problem.target_yb), batch_size, p=problem.target_yb / target_yb_mass,
    )
    negative_ab = rng.choice(
        len(problem.target_ab), batch_size, p=problem.target_ab / target_ab_mass,
    )
    uniforms = rng.random(batch_size)
    gu, gv = np.zeros_like(u), np.zeros_like(v)
    scale = 1.0 / batch_size
    np.add.at(gu, negative_yb, -target_yb_mass * scale)
    np.add.at(gv, negative_ab, -target_ab_mass * scale)
    for y, a, uniform in zip(ys, aa, uniforms):
        weights = problem.ab_background.copy()
        lo, hi = problem.ab_delta_ptr[a:a + 2]
        delta_features = problem.ab_delta_order[lo:hi]
        np.add.at(weights, problem.ab_active_b[delta_features], problem.ab_delta[delta_features])
        ylo, yhi = problem.yb_ptr[y:y + 2]
        yfeatures = problem.yb_order[ylo:yhi]
        if len(yfeatures):
            weights[problem.yb_b[yfeatures]] *= np.exp(u[yfeatures])
        alo, ahi = problem.factor_ab_ptr[a:a + 2]
        afeatures = problem.factor_ab_order[alo:ahi]
        if len(afeatures):
            weights[problem.factor_ab_b[afeatures]] *= np.exp(v[afeatures])
        probabilities = weights / np.sum(weights)
        b = min(
            int(np.searchsorted(np.cumsum(probabilities), uniform, side="right")),
            problem.vocabulary_size - 1,
        )
        if len(yfeatures):
            match = np.flatnonzero(problem.yb_b[yfeatures] == b)
            if len(match):
                gu[yfeatures[match[0]]] += scale
        if len(afeatures):
            match = np.flatnonzero(problem.factor_ab_b[afeatures] == b)
            if len(match):
                gv[afeatures[match[0]]] += scale
    if np.isfinite(slack_precision):
        vy = np.ones_like(u) if variance_yb is None else np.asarray(variance_yb)
        va = np.ones_like(v) if variance_ab is None else np.asarray(variance_ab)
        gu += vy * u / slack_precision
        gv += va * v / slack_precision
    return gu, gv


def _sample_full_ya(
    problem: FullImplicitPairProblem, size: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    bg_mass = problem.vocabulary_size * float(np.sum(problem.ya_background))
    delta_mass = float(np.sum(problem.ya_delta))
    choose_delta = rng.random(size) >= bg_mass
    ys, aa = np.empty(size, dtype=np.int32), np.empty(size, dtype=np.int32)
    count = int(np.sum(~choose_delta))
    if count:
        aa[~choose_delta] = rng.choice(
            problem.vocabulary_size, count,
            p=problem.ya_background / np.sum(problem.ya_background),
        )
        ys[~choose_delta] = rng.integers(problem.vocabulary_size, size=count)
    count = int(np.sum(choose_delta))
    if count:
        selected = rng.choice(
            len(problem.ya_delta), count, p=problem.ya_delta / delta_mass,
        )
        ys[choose_delta] = problem.ya_active_y[selected]
        aa[choose_delta] = problem.ya_active_a[selected]
    return ys, aa


def full_implicit_validation_objective(
    problem: FullImplicitPairProblem,
    correction_yb: np.ndarray,
    correction_ab: np.ndarray,
    *, validation_seed: int, validation_samples: int,
    slack_precision: float = float("inf"),
    variance_yb: np.ndarray | None = None,
    variance_ab: np.ndarray | None = None,
) -> float:
    """Evaluate the dual on one fixed independent Monte Carlo YA sample."""

    if validation_samples < 1:
        raise ValueError("validation_samples must be positive")
    u, v = np.asarray(correction_yb), np.asarray(correction_ab)
    rng = np.random.default_rng(validation_seed)
    ys, aa = _sample_full_ya(problem, validation_samples, rng)
    ab_context_mass = (
        np.full(problem.vocabulary_size, float(np.sum(problem.ab_background)))
        + np.bincount(
            problem.ab_active_a, weights=problem.ab_delta,
            minlength=problem.vocabulary_size,
        )
    )
    total = 0.0
    for y, a in zip(ys, aa):
        weights = problem.ab_background.copy()
        lo, hi = problem.ab_delta_ptr[a:a + 2]
        delta_features = problem.ab_delta_order[lo:hi]
        np.add.at(weights, problem.ab_active_b[delta_features], problem.ab_delta[delta_features])
        ylo, yhi = problem.yb_ptr[y:y + 2]
        yfeatures = problem.yb_order[ylo:yhi]
        if len(yfeatures):
            weights[problem.yb_b[yfeatures]] *= np.exp(u[yfeatures])
        alo, ahi = problem.factor_ab_ptr[a:a + 2]
        afeatures = problem.factor_ab_order[alo:ahi]
        if len(afeatures):
            weights[problem.factor_ab_b[afeatures]] *= np.exp(v[afeatures])
        total += np.log(np.sum(weights)) - np.log(ab_context_mass[a])
    objective = total / validation_samples
    objective -= float(np.dot(problem.target_yb, u) + np.dot(problem.target_ab, v))
    if np.isfinite(slack_precision):
        vy = np.ones_like(u) if variance_yb is None else np.asarray(variance_yb)
        va = np.ones_like(v) if variance_ab is None else np.asarray(variance_ab)
        objective += 0.5 / slack_precision * (
            float(np.dot(vy, u * u)) + float(np.dot(va, v * v))
        )
    return float(objective)


def full_implicit_pair_sgd(
    problem: FullImplicitPairProblem,
    initial_yb: np.ndarray,
    initial_ab: np.ndarray,
    *, steps: int, batch_size: int, training_seed: int,
    validation_seed: int, validation_samples: int,
    workers: int = 1, learning_rate: float = 0.03,
    validation_interval: int = 50, slack_precision: float = 1.0,
    variance_yb: np.ndarray | None = None,
    variance_ab: np.ndarray | None = None,
) -> FullImplicitSgdResult:
    """Cold Adam selected by a fixed independent validation sample."""

    if steps < 0 or validation_interval < 1 or learning_rate <= 0.0:
        raise ValueError("invalid full implicit SGD schedule")
    u, v = np.array(initial_yb, copy=True), np.array(initial_ab, copy=True)
    evaluate = lambda: full_implicit_validation_objective(
        problem, u, v, validation_seed=validation_seed,
        validation_samples=validation_samples, slack_precision=slack_precision,
        variance_yb=variance_yb, variance_ab=variance_ab,
    )
    initial = evaluate()
    best, best_u, best_v = initial, u.copy(), v.copy()
    mu, mv, su, sv = np.zeros_like(u), np.zeros_like(v), np.zeros_like(u), np.zeros_like(v)
    beta1, beta2 = 0.9, 0.999
    for step in range(1, steps + 1):
        gu, gv = full_implicit_pair_sampled_gradient(
            problem, u, v, batch_size=batch_size, seed=training_seed,
            step=step, workers=workers, slack_precision=slack_precision,
            variance_yb=variance_yb, variance_ab=variance_ab,
        )
        mu, mv = beta1 * mu + (1 - beta1) * gu, beta1 * mv + (1 - beta1) * gv
        su, sv = beta2 * su + (1 - beta2) * gu * gu, beta2 * sv + (1 - beta2) * gv * gv
        c1, c2 = 1 - beta1 ** step, 1 - beta2 ** step
        u -= learning_rate * (mu / c1) / (np.sqrt(su / c2) + 1e-8)
        v -= learning_rate * (mv / c1) / (np.sqrt(sv / c2) + 1e-8)
        if step % validation_interval == 0 or step == steps:
            value = evaluate()
            if value < best:
                best, best_u, best_v = value, u.copy(), v.copy()
    final = evaluate()
    return FullImplicitSgdResult(
        best_u, best_v, steps, batch_size, training_seed, validation_seed,
        validation_samples, initial, best, final,
    )


def full_implicit_log_probabilities(
    problem: FullImplicitPairProblem,
    correction_yb: np.ndarray,
    correction_ab: np.ndarray,
    target: np.ndarray,
    lag1: np.ndarray,
    lag2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Literal sparse-row scoring for the complete layered YA/AB model."""

    u, v = np.asarray(correction_yb), np.asarray(correction_ab)
    target = np.asarray(target, dtype=np.int64)
    lag1 = np.asarray(lag1, dtype=np.int64)
    lag2 = np.asarray(lag2, dtype=np.int64)
    if not (target.shape == lag1.shape == lag2.shape):
        raise ValueError("score arrays must have identical shapes")
    vocab = problem.vocabulary_size
    ya_context = vocab * problem.ya_background + np.bincount(
        problem.ya_active_a, weights=problem.ya_delta, minlength=vocab,
    )
    ya_delta = {
        int(y) * vocab + int(a): float(value)
        for y, a, value in zip(
            problem.ya_active_y, problem.ya_active_a, problem.ya_delta,
        )
    }
    yb_factor = {
        int(y) * vocab + int(b): index
        for index, (y, b) in enumerate(zip(problem.yb_y, problem.yb_b))
    }
    ab_factor = {
        int(a) * vocab + int(b): index
        for index, (a, b) in enumerate(zip(problem.factor_ab_a, problem.factor_ab_b))
    }
    z_cache: dict[int, np.ndarray] = {}
    denominator_cache: dict[tuple[int, int], float] = {}

    def ya_value(y: int, a: int) -> float:
        return float(problem.ya_background[a] + ya_delta.get(y * vocab + a, 0.0))

    def ab_row(a: int) -> np.ndarray:
        row = problem.ab_background.copy()
        lo, hi = problem.ab_delta_ptr[a:a + 2]
        features = problem.ab_delta_order[lo:hi]
        np.add.at(row, problem.ab_active_b[features], problem.ab_delta[features])
        alo, ahi = problem.factor_ab_ptr[a:a + 2]
        active = problem.factor_ab_order[alo:ahi]
        if len(active):
            row[problem.factor_ab_b[active]] *= np.exp(v[active])
        return row

    def normalizers(a: int) -> np.ndarray:
        cached = z_cache.get(a)
        if cached is not None:
            return cached
        base = ab_row(a)
        values = np.empty(vocab)
        for y in range(vocab):
            row = base.copy()
            lo, hi = problem.yb_ptr[y:y + 2]
            active = problem.yb_order[lo:hi]
            if len(active):
                row[problem.yb_b[active]] *= np.exp(u[active])
            values[y] = np.sum(row)
        z_cache[a] = values
        return values

    anchored = np.empty(len(target), dtype=np.float64)
    markov = np.empty(len(target), dtype=np.float64)
    for index, (y, a, b) in enumerate(zip(target, lag1, lag2)):
        py = ya_value(int(y), int(a))
        markov[index] = np.log(py) - np.log(ya_context[a])
        key = (int(a), int(b))
        denominator = denominator_cache.get(key)
        z = normalizers(int(a))
        if denominator is None:
            terms = np.asarray([ya_value(symbol, int(a)) for symbol in range(vocab)]) / z
            for symbol in range(vocab):
                feature = yb_factor.get(symbol * vocab + int(b))
                if feature is not None:
                    terms[symbol] *= np.exp(u[feature])
            denominator = float(np.sum(terms))
            denominator_cache[key] = denominator
        feature = yb_factor.get(int(y) * vocab + int(b))
        log_feature = 0.0 if feature is None else float(u[feature])
        anchored[index] = np.log(py) + log_feature - np.log(z[y]) - np.log(denominator)
    return anchored, markov


def implicit_ya_problem_from_grouped(problem, p_ya) -> ImplicitYaIntersectionProblem:
    """Join active YB features to AB edges while keeping implicit YA mass."""

    v = int(problem.vocabulary_size)
    if p_ya.vocabulary_size != v:
        raise ValueError("YA fallback and grouped problem use different alphabets")
    if not (
        np.array_equal(p_ya.active_y, problem.active_ya_y)
        and np.array_equal(p_ya.active_context, problem.active_ya_a)
    ):
        raise ValueError("YA fallback support differs from grouped problem")
    if not (
        np.array_equal(p_ya.left, np.ones(v))
        and np.array_equal(p_ya.right, np.ones(v))
    ):
        raise ValueError("anchored YA v1 requires the raw layered pair gauge")
    background = np.asarray(p_ya.background, dtype=np.float64)
    delta = np.asarray(p_ya.delta, dtype=np.float64)
    ya_context = v * background + np.bincount(
        p_ya.active_context, weights=delta, minlength=v,
    )
    if not np.isclose(np.sum(ya_context), 1.0):
        raise ValueError("implicit YA law must have unit mass")
    delta_by_key = {
        int(y) * v + int(a): float(value)
        for y, a, value in zip(p_ya.active_y, p_ya.active_context, delta)
    }
    ab_by_b: dict[int, list[int]] = {}
    for edge, b in enumerate(np.asarray(problem.edge_b, dtype=np.int32)):
        ab_by_b.setdefault(int(b), []).append(edge)
    row_index: dict[int, int] = {}
    row_y: list[int] = []
    row_a: list[int] = []
    triangle_row: list[int] = []
    triangle_yb: list[int] = []
    triangle_ab: list[int] = []
    for yb, (y, b) in enumerate(zip(problem.active_yb_y, problem.active_yb_b)):
        for edge in ab_by_b.get(int(b), ()):
            a = int(problem.edge_a[edge])
            key = int(y) * v + a
            row = row_index.get(key)
            if row is None:
                row = len(row_y)
                row_index[key] = row
                row_y.append(int(y))
                row_a.append(a)
            triangle_row.append(row)
            triangle_yb.append(yb)
            triangle_ab.append(edge)
    row_y_array = np.asarray(row_y, dtype=np.int32)
    row_a_array = np.asarray(row_a, dtype=np.int32)
    row_probability = np.asarray([
        background[a] + delta_by_key.get(int(y) * v + int(a), 0.0)
        for y, a in zip(row_y_array, row_a_array)
    ])
    if np.any(row_probability <= 0.0):
        raise ValueError("affected YA rows must have positive hard mass")
    if np.any(delta < 0.0):
        raise ValueError("anchored YA v1 requires nonnegative layered deltas")
    edge_b = np.asarray(problem.edge_b, dtype=np.int32)
    edge_a = np.asarray(problem.edge_a, dtype=np.int32)
    ab_order = np.lexsort((edge_b, edge_a))
    ab_ptr = np.r_[0, np.cumsum(
        np.bincount(edge_a, minlength=v), dtype=np.int64,
    )]
    yb_y = np.asarray(problem.active_yb_y, dtype=np.int32)
    yb_b = np.asarray(problem.active_yb_b, dtype=np.int32)
    yb_order = np.lexsort((yb_b, yb_y))
    yb_ptr = np.r_[0, np.cumsum(
        np.bincount(yb_y, minlength=v), dtype=np.int64,
    )]
    return ImplicitYaIntersectionProblem(
        vocabulary_size=v,
        edge_a=edge_a,
        edge_b=edge_b,
        edge_probability=np.asarray(problem.edge_probability, dtype=np.float64),
        target_yb=np.asarray(problem.target_yb, dtype=np.float64),
        yb_y=yb_y,
        yb_b=yb_b,
        ya_background=background,
        ya_active_y=np.asarray(p_ya.active_y, dtype=np.int32),
        ya_active_a=np.asarray(p_ya.active_context, dtype=np.int32),
        ya_delta=delta,
        ya_context_probability=ya_context,
        row_y=row_y_array,
        row_a=row_a_array,
        row_probability=row_probability,
        triangle_row=np.asarray(triangle_row, dtype=np.int32),
        triangle_yb=np.asarray(triangle_yb, dtype=np.int32),
        triangle_ab=np.asarray(triangle_ab, dtype=np.int32),
        ab_order=ab_order,
        ab_ptr=ab_ptr,
        yb_order=yb_order,
        yb_ptr=yb_ptr,
    )


def implicit_ya_intersection_dual(
    problem: ImplicitYaIntersectionProblem,
    correction_yb: np.ndarray,
    correction_ab: np.ndarray,
    *,
    slack_precision: float = float("inf"),
    variance_yb: np.ndarray | None = None,
    variance_ab: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Exact relaxed dual with the full implicit hard-YA marginal."""

    u = np.asarray(correction_yb, dtype=np.float64)
    v = np.asarray(correction_ab, dtype=np.float64)
    if u.shape != problem.target_yb.shape or v.shape != problem.edge_probability.shape:
        raise ValueError("factor shapes do not match implicit-YA problem")
    contexts = problem.vocabulary_size
    p_a = np.bincount(
        problem.edge_a, weights=problem.edge_probability, minlength=contexts,
    )
    base_ab = problem.edge_probability / p_a[problem.edge_a] * np.exp(v)
    base_z = np.bincount(problem.edge_a, weights=base_ab, minlength=contexts)
    if np.any(base_z[problem.ya_context_probability > 0.0] <= 0.0):
        raise FloatingPointError("hard-YA context has no AB baseline mass")
    row_z = base_z[problem.row_a].copy()
    expm1_u = np.expm1(u)
    adjustment = (
        base_ab[problem.triangle_ab] * expm1_u[problem.triangle_yb]
    )
    np.add.at(row_z, problem.triangle_row, adjustment)
    if np.any(row_z <= 0.0) or not np.all(np.isfinite(row_z)):
        raise FloatingPointError("implicit-YA row normalizer is invalid")
    objective = float(np.dot(problem.ya_context_probability, np.log(base_z)))
    objective += float(np.dot(
        problem.row_probability,
        np.log(row_z) - np.log(base_z[problem.row_a]),
    ))
    objective -= float(
        np.dot(problem.target_yb, u) + np.dot(problem.edge_probability, v)
    )
    inverse = problem.row_probability / row_z
    grad_u = -problem.target_yb.copy()
    np.add.at(
        grad_u, problem.triangle_yb,
        inverse[problem.triangle_row]
        * base_ab[problem.triangle_ab]
        * np.exp(u[problem.triangle_yb]),
    )
    grad_v = (
        base_ab
        * (problem.ya_context_probability / base_z)[problem.edge_a]
    )
    context_denominator_change = np.bincount(
        problem.row_a,
        weights=problem.row_probability * (
            1.0 / row_z - 1.0 / base_z[problem.row_a]
        ),
        minlength=contexts,
    )
    grad_v += base_ab * context_denominator_change[problem.edge_a]
    np.add.at(
        grad_v, problem.triangle_ab,
        inverse[problem.triangle_row]
        * base_ab[problem.triangle_ab]
        * expm1_u[problem.triangle_yb],
    )
    grad_v -= problem.edge_probability
    if np.isfinite(slack_precision):
        if slack_precision <= 0.0:
            raise ValueError("slack_precision must be positive")
        vy = np.ones_like(u) if variance_yb is None else np.asarray(variance_yb)
        va = np.ones_like(v) if variance_ab is None else np.asarray(variance_ab)
        scale = 1.0 / slack_precision
        objective += 0.5 * scale * (
            float(np.dot(vy, u * u)) + float(np.dot(va, v * v))
        )
        grad_u += scale * vy * u
        grad_v += scale * va * v
    return objective, grad_u, grad_v


def implicit_ya_sampled_gradient(
    problem: ImplicitYaIntersectionProblem,
    correction_yb: np.ndarray,
    correction_ab: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    step: int,
    workers: int = 1,
    slack_precision: float = float("inf"),
    variance_yb: np.ndarray | None = None,
    variance_ab: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Unbiased gradient sampling the complete implicit hard-YA law."""

    if batch_size < 1 or step < 0 or workers < 1:
        raise ValueError("invalid sampled-gradient schedule")
    u = np.asarray(correction_yb, dtype=np.float64)
    v = np.asarray(correction_ab, dtype=np.float64)
    if u.shape != problem.target_yb.shape or v.shape != problem.edge_probability.shape:
        raise ValueError("factor shapes do not match implicit-YA problem")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(step)]))
    background_mass = problem.vocabulary_size * float(np.sum(problem.ya_background))
    delta_mass = float(np.sum(problem.ya_delta))
    if not np.isclose(background_mass + delta_mass, 1.0):
        raise ValueError("implicit YA sampling masses do not sum to one")
    choose_delta = rng.random(batch_size) >= background_mass
    y = np.empty(batch_size, dtype=np.int32)
    a = np.empty(batch_size, dtype=np.int32)
    background_count = int(np.sum(~choose_delta))
    if background_count:
        a[~choose_delta] = rng.choice(
            problem.vocabulary_size, background_count,
            p=problem.ya_background / np.sum(problem.ya_background),
        )
        y[~choose_delta] = rng.integers(
            problem.vocabulary_size, size=background_count,
        )
    delta_count = int(np.sum(choose_delta))
    if delta_count:
        selected = rng.choice(
            len(problem.ya_delta), delta_count,
            p=problem.ya_delta / delta_mass,
        )
        y[choose_delta] = problem.ya_active_y[selected]
        a[choose_delta] = problem.ya_active_a[selected]
    target_yb_mass = float(np.sum(problem.target_yb))
    target_u = rng.choice(
        len(problem.target_yb), batch_size,
        p=problem.target_yb / target_yb_mass,
    )
    target_v = rng.choice(
        len(problem.edge_probability), batch_size,
        p=problem.edge_probability,
    )
    uniforms = rng.random(batch_size)
    scale = 1.0 / float(batch_size)
    grad_u = np.zeros_like(u)
    grad_v = np.zeros_like(v)
    np.add.at(grad_u, target_u, -target_yb_mass * scale)
    np.add.at(grad_v, target_v, -scale)
    p_a = np.bincount(
        problem.edge_a, weights=problem.edge_probability,
        minlength=problem.vocabulary_size,
    )
    for symbol_y, context_a, uniform in zip(y, a, uniforms):
        elo, ehi = problem.ab_ptr[context_a:context_a + 2]
        edges = problem.ab_order[elo:ehi]
        logits = np.log(problem.edge_probability[edges] / p_a[context_a]) + v[edges]
        flo, fhi = problem.yb_ptr[symbol_y:symbol_y + 2]
        features = problem.yb_order[flo:fhi]
        if len(features):
            positions = np.searchsorted(problem.edge_b[edges], problem.yb_b[features])
            valid = positions < len(edges)
            valid &= problem.edge_b[edges[np.minimum(positions, len(edges) - 1)]] == problem.yb_b[features]
            logits[positions[valid]] += u[features[valid]]
        logits -= np.max(logits)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities)
        chosen_local = min(
            int(np.searchsorted(np.cumsum(probabilities), uniform, side="right")),
            len(edges) - 1,
        )
        chosen_edge = int(edges[chosen_local])
        grad_v[chosen_edge] += scale
        if len(features):
            match = np.flatnonzero(valid & (positions == chosen_local))
            if len(match):
                grad_u[features[match[0]]] += scale
    if np.isfinite(slack_precision):
        if slack_precision <= 0.0:
            raise ValueError("slack_precision must be positive")
        vy = np.ones_like(u) if variance_yb is None else np.asarray(variance_yb)
        va = np.ones_like(v) if variance_ab is None else np.asarray(variance_ab)
        grad_u += vy * u / slack_precision
        grad_v += va * v / slack_precision
    return grad_u, grad_v


def implicit_ya_intersection_sgd(
    problem: ImplicitYaIntersectionProblem,
    initial_yb: np.ndarray,
    initial_ab: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    workers: int = 1,
    learning_rate: float = 0.03,
    exact_interval: int = 50,
    slack_precision: float = 1.0,
    variance_yb: np.ndarray | None = None,
    variance_ab: np.ndarray | None = None,
) -> AnchoredSgdResult:
    """Cold Adam for the complete implicit-YA anchored objective."""

    if steps < 0 or exact_interval < 1 or learning_rate <= 0.0:
        raise ValueError("invalid SGD schedule")
    u, v = np.array(initial_yb, copy=True), np.array(initial_ab, copy=True)
    initial = implicit_ya_intersection_dual(
        problem, u, v, slack_precision=slack_precision,
        variance_yb=variance_yb, variance_ab=variance_ab,
    )[0]
    best_value, best_u, best_v = initial, u.copy(), v.copy()
    mu, mv, su, sv = np.zeros_like(u), np.zeros_like(v), np.zeros_like(u), np.zeros_like(v)
    beta1, beta2 = 0.9, 0.999
    for step in range(1, steps + 1):
        gu, gv = implicit_ya_sampled_gradient(
            problem, u, v, batch_size=batch_size, seed=seed, step=step,
            workers=workers, slack_precision=slack_precision,
            variance_yb=variance_yb, variance_ab=variance_ab,
        )
        mu, mv = beta1 * mu + (1 - beta1) * gu, beta1 * mv + (1 - beta1) * gv
        su, sv = beta2 * su + (1 - beta2) * gu * gu, beta2 * sv + (1 - beta2) * gv * gv
        c1, c2 = 1 - beta1 ** step, 1 - beta2 ** step
        u -= learning_rate * (mu / c1) / (np.sqrt(su / c2) + 1e-8)
        v -= learning_rate * (mv / c1) / (np.sqrt(sv / c2) + 1e-8)
        if step % exact_interval == 0 or step == steps:
            value = implicit_ya_intersection_dual(
                problem, u, v, slack_precision=slack_precision,
                variance_yb=variance_yb, variance_ab=variance_ab,
            )[0]
            if value < best_value:
                best_value, best_u, best_v = value, u.copy(), v.copy()
    final = implicit_ya_intersection_dual(
        problem, u, v, slack_precision=slack_precision,
        variance_yb=variance_yb, variance_ab=variance_ab,
    )[0]
    return AnchoredSgdResult(
        best_u, best_v, steps, batch_size, seed, initial, best_value, final,
    )


def implicit_ya_log_probabilities(
    problem: ImplicitYaIntersectionProblem,
    correction_yb: np.ndarray,
    correction_ab: np.ndarray,
    target: np.ndarray,
    lag1: np.ndarray,
    lag2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score anchored and Markov-1 rules; return log probabilities and coverage."""

    u = np.asarray(correction_yb, dtype=np.float64)
    v = np.asarray(correction_ab, dtype=np.float64)
    y = np.asarray(target, dtype=np.int64)
    a = np.asarray(lag1, dtype=np.int64)
    b = np.asarray(lag2, dtype=np.int64)
    if not (y.shape == a.shape == b.shape):
        raise ValueError("target and lag arrays must have identical shapes")
    vocab = problem.vocabulary_size

    def positions(keys: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        found = np.searchsorted(sorted_keys, query)
        valid = found < len(sorted_keys)
        valid &= sorted_keys[np.minimum(found, len(sorted_keys) - 1)] == query
        return order[np.minimum(found, len(order) - 1)], valid

    p_a_ab = np.bincount(
        problem.edge_a, weights=problem.edge_probability, minlength=vocab,
    )
    base_ab = problem.edge_probability / p_a_ab[problem.edge_a] * np.exp(v)
    base_z = np.bincount(problem.edge_a, weights=base_ab, minlength=vocab)
    row_z = base_z[problem.row_a].copy()
    np.add.at(
        row_z, problem.triangle_row,
        base_ab[problem.triangle_ab] * np.expm1(u[problem.triangle_yb]),
    )
    context_denominator = problem.ya_context_probability / base_z
    np.add.at(
        context_denominator, problem.row_a,
        problem.row_probability * (1.0 / row_z - 1.0 / base_z[problem.row_a]),
    )
    edge_denominator = context_denominator[problem.edge_a] * np.ones_like(base_ab)
    np.add.at(
        edge_denominator, problem.triangle_ab,
        problem.row_probability[problem.triangle_row]
        / row_z[problem.triangle_row]
        * np.expm1(u[problem.triangle_yb]),
    )

    ya_keys = problem.ya_active_y.astype(np.int64) * vocab + problem.ya_active_a
    ya_pos, ya_valid = positions(ya_keys, y * vocab + a)
    p_ya = problem.ya_background[a].copy()
    p_ya[ya_valid] += problem.ya_delta[ya_pos[ya_valid]]
    markov = np.log(p_ya) - np.log(problem.ya_context_probability[a])

    ab_keys = problem.edge_a.astype(np.int64) * vocab + problem.edge_b
    ab_pos, covered = positions(ab_keys, a * vocab + b)
    anchored = markov.copy()
    if np.any(covered):
        selected = np.flatnonzero(covered)
        row_keys = problem.row_y.astype(np.int64) * vocab + problem.row_a
        row_pos, row_valid = positions(row_keys, y[selected] * vocab + a[selected])
        z = base_z[a[selected]].copy()
        z[row_valid] = row_z[row_pos[row_valid]]
        yb_keys = problem.yb_y.astype(np.int64) * vocab + problem.yb_b
        yb_pos, yb_valid = positions(yb_keys, y[selected] * vocab + b[selected])
        feature = np.zeros(len(selected), dtype=np.float64)
        feature[yb_valid] = u[yb_pos[yb_valid]]
        anchored[selected] = (
            np.log(p_ya[selected]) + feature - np.log(z)
            - np.log(edge_denominator[ab_pos[selected]])
        )
    if not np.all(np.isfinite(anchored)) or not np.all(np.isfinite(markov)):
        raise FloatingPointError("anchored scoring produced nonfinite probabilities")
    return anchored, markov, covered


def anchored_problem_from_intersection(problem, plan) -> AnchoredIntersectionProblem:
    """Adapt a ``SparseGroupedProblem`` and its intersection plan.

    The adapter stores no dense alphabet-square object.  Inactive YB factors
    are represented analytically by the AB baseline; ``plan`` contains only
    the active YB corrections that must be added to that baseline.
    """

    edge_a = np.asarray(problem.edge_a, dtype=np.int32)
    edge_probability = np.asarray(problem.edge_probability, dtype=np.float64)
    ya_a = np.asarray(problem.active_ya_a, dtype=np.int32)
    target_ya = np.asarray(problem.target_ya, dtype=np.float64)
    target_yb = np.asarray(problem.target_yb, dtype=np.float64)
    triangle_ya = np.asarray(plan.correction_ya, dtype=np.int32)
    triangle_yb = np.asarray(plan.correction_yb, dtype=np.int32)
    triangle_ab = np.asarray(plan.edge, dtype=np.int32)
    if not np.isclose(np.sum(target_ya), 1.0):
        raise ValueError("hard YA target must have unit mass")
    if not np.isclose(np.sum(target_yb), 1.0):
        raise ValueError("soft YB target must have unit mass")
    if not np.isclose(np.sum(edge_probability), 1.0):
        raise ValueError("soft AB target must have unit mass")
    if not (
        len(triangle_ya) == len(triangle_yb) == len(triangle_ab)
        and np.all(triangle_ya < len(target_ya))
        and np.all(triangle_yb < len(target_yb))
        and np.all(triangle_ab < len(edge_probability))
    ):
        raise ValueError("intersection plan does not match sparse problem")
    contexts = int(max(
        np.max(edge_a, initial=-1), np.max(ya_a, initial=-1),
    )) + 1
    ab_order = np.argsort(edge_a, kind="stable")
    ab_ptr = np.r_[0, np.cumsum(
        np.bincount(edge_a, minlength=contexts), dtype=np.int64,
    )]
    triangle_order = np.argsort(triangle_ya, kind="stable")
    triangle_ptr = np.r_[0, np.cumsum(
        np.bincount(triangle_ya, minlength=len(target_ya)), dtype=np.int64,
    )]
    edge_local_position = np.empty(len(edge_a), dtype=np.int32)
    for a in range(contexts):
        lo, hi = ab_ptr[a:a + 2]
        edge_local_position[ab_order[lo:hi]] = np.arange(hi - lo)
    return AnchoredIntersectionProblem(
        edge_a=edge_a,
        edge_probability=edge_probability,
        ya_a=ya_a,
        target_ya=target_ya,
        target_yb=target_yb,
        triangle_ya=triangle_ya,
        triangle_yb=triangle_yb,
        triangle_ab=triangle_ab,
        ab_order=ab_order,
        ab_ptr=ab_ptr,
        triangle_order=triangle_order,
        triangle_ptr=triangle_ptr,
        edge_local_position=edge_local_position,
    )


def cold_pair_midpoint_from_grouped(problem) -> tuple[np.ndarray, np.ndarray]:
    """Construct the historical midpoint start without dense pair tables."""

    v = int(problem.vocabulary_size)
    p_y = np.bincount(
        np.asarray(problem.active_ya_y, dtype=np.int32),
        weights=np.asarray(problem.target_ya, dtype=np.float64), minlength=v,
    )
    p_b = np.bincount(
        np.asarray(problem.active_yb_b, dtype=np.int32),
        weights=np.asarray(problem.target_yb, dtype=np.float64), minlength=v,
    )
    y = np.asarray(problem.active_yb_y, dtype=np.int32)
    b = np.asarray(problem.active_yb_b, dtype=np.int32)
    target = np.asarray(problem.target_yb, dtype=np.float64)
    tiny = np.finfo(np.float64).tiny
    u = 0.5 * (
        np.log(np.maximum(target, tiny))
        - np.log(np.maximum(p_y[y] * p_b[b], tiny))
    )
    return u, np.zeros(len(problem.edge_probability), dtype=np.float64)


def cold_pair_midpoint_from_projected(problem, p_ya, p_yb) -> tuple[np.ndarray, np.ndarray]:
    """Cold midpoint using complete implicit layered pair marginals."""

    v = int(problem.vocabulary_size)
    p_y = np.full(v, float(np.sum(p_ya.background)))
    np.add.at(p_y, p_ya.active_y, p_ya.delta)
    p_b = v * np.asarray(p_yb.background, dtype=np.float64).copy()
    np.add.at(p_b, p_yb.active_context, p_yb.delta)
    y = np.asarray(problem.active_yb_y, dtype=np.int32)
    b = np.asarray(problem.active_yb_b, dtype=np.int32)
    tiny = np.finfo(np.float64).tiny
    u = 0.5 * (
        np.log(np.maximum(problem.target_yb, tiny))
        - np.log(np.maximum(p_y[y] * p_b[b], tiny))
    )
    return u, np.zeros(len(problem.edge_probability), dtype=np.float64)


def anchored_intersection_dual(
    problem: AnchoredIntersectionProblem,
    correction_yb: np.ndarray,
    correction_ab: np.ndarray,
    *,
    slack_precision: float = float("inf"),
    variance_yb: np.ndarray | None = None,
    variance_ab: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate the anchored dual using baseline-plus-intersection algebra."""

    u = np.asarray(correction_yb, dtype=np.float64)
    v = np.asarray(correction_ab, dtype=np.float64)
    if u.shape != problem.target_yb.shape or v.shape != problem.edge_probability.shape:
        raise ValueError("factor shapes do not match intersection problem")
    if slack_precision <= 0.0 or np.isnan(slack_precision):
        raise ValueError("slack_precision must be positive")
    contexts = int(max(
        np.max(problem.edge_a, initial=-1), np.max(problem.ya_a, initial=-1),
    )) + 1
    p_a = np.bincount(
        problem.edge_a, weights=problem.edge_probability, minlength=contexts,
    )
    if np.any(p_a[problem.edge_a] <= 0.0):
        raise ValueError("AB edge belongs to a zero-mass A context")
    base_ab = (
        problem.edge_probability / p_a[problem.edge_a]
    ) * np.exp(v)
    base_z = np.bincount(problem.edge_a, weights=base_ab, minlength=contexts)
    z = base_z[problem.ya_a].copy()
    expm1_u = np.expm1(u)
    triangle_adjustment = (
        base_ab[problem.triangle_ab] * expm1_u[problem.triangle_yb]
    )
    np.add.at(z, problem.triangle_ya, triangle_adjustment)
    if np.any(z <= 0.0) or not np.all(np.isfinite(z)):
        raise FloatingPointError("anchored normalizer is nonpositive or nonfinite")

    weighted_inverse_z = problem.target_ya / z
    objective = float(np.dot(problem.target_ya, np.log(z)))
    objective -= float(
        np.dot(problem.target_yb, u)
        + np.dot(problem.edge_probability, v)
    )
    grad_u = -problem.target_yb.copy()
    triangle_u_mass = (
        weighted_inverse_z[problem.triangle_ya]
        * base_ab[problem.triangle_ab]
        * np.exp(u[problem.triangle_yb])
    )
    np.add.at(grad_u, problem.triangle_yb, triangle_u_mass)

    context_inverse = np.bincount(
        problem.ya_a, weights=weighted_inverse_z, minlength=contexts,
    )
    grad_v = base_ab * context_inverse[problem.edge_a]
    triangle_v_adjustment = (
        weighted_inverse_z[problem.triangle_ya]
        * base_ab[problem.triangle_ab]
        * expm1_u[problem.triangle_yb]
    )
    np.add.at(grad_v, problem.triangle_ab, triangle_v_adjustment)
    grad_v -= problem.edge_probability

    if np.isfinite(slack_precision):
        vy = np.ones_like(u) if variance_yb is None else np.asarray(variance_yb)
        va = np.ones_like(v) if variance_ab is None else np.asarray(variance_ab)
        if vy.shape != u.shape or va.shape != v.shape:
            raise ValueError("slack variance shapes do not match factors")
        scale = 1.0 / slack_precision
        objective += 0.5 * scale * (
            float(np.dot(vy, u * u)) + float(np.dot(va, v * v))
        )
        grad_u += scale * vy * u
        grad_v += scale * va * v
    return objective, grad_u, grad_v


def anchored_intersection_sampled_gradient(
    problem: AnchoredIntersectionProblem,
    correction_yb: np.ndarray,
    correction_ab: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    step: int,
    workers: int = 1,
    slack_precision: float = float("inf"),
    variance_yb: np.ndarray | None = None,
    variance_ab: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an unbiased, schedule-invariant minibatch gradient.

    The random batch is a pure function of ``(seed, step, batch_size)``.
    ``workers`` may change how a caller evaluates the already-declared batch,
    but never its samples or canonical accumulation order.
    """

    if batch_size < 1 or step < 0 or workers < 1:
        raise ValueError("invalid sampled-gradient schedule")
    u = np.asarray(correction_yb, dtype=np.float64)
    v = np.asarray(correction_ab, dtype=np.float64)
    if u.shape != problem.target_yb.shape or v.shape != problem.edge_probability.shape:
        raise ValueError("factor shapes do not match intersection problem")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(step)]))
    rows = rng.choice(len(problem.target_ya), batch_size, p=problem.target_ya)
    target_u = rng.choice(len(problem.target_yb), batch_size, p=problem.target_yb)
    target_v = rng.choice(
        len(problem.edge_probability), batch_size, p=problem.edge_probability,
    )
    uniforms = rng.random(batch_size)
    grad_u = np.zeros_like(u)
    grad_v = np.zeros_like(v)
    scale = 1.0 / float(batch_size)
    np.add.at(grad_u, target_u, -scale)
    np.add.at(grad_v, target_v, -scale)

    p_a = np.bincount(
        problem.edge_a, weights=problem.edge_probability,
        minlength=len(problem.ab_ptr) - 1,
    )
    for row, uniform in zip(rows, uniforms):
        a = int(problem.ya_a[row])
        elo, ehi = problem.ab_ptr[a:a + 2]
        edges = problem.ab_order[elo:ehi]
        logits = (
            np.log(problem.edge_probability[edges] / p_a[a]) + v[edges]
        )
        tlo, thi = problem.triangle_ptr[row:row + 2]
        triangles = problem.triangle_order[tlo:thi]
        local = problem.edge_local_position[problem.triangle_ab[triangles]]
        logits[local] += u[problem.triangle_yb[triangles]]
        logits -= np.max(logits)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities)
        chosen_local = int(np.searchsorted(
            np.cumsum(probabilities), uniform, side="right",
        ))
        chosen_local = min(chosen_local, len(edges) - 1)
        chosen_edge = int(edges[chosen_local])
        grad_v[chosen_edge] += scale
        match = np.flatnonzero(local == chosen_local)
        if len(match):
            grad_u[problem.triangle_yb[triangles[match[0]]]] += scale
    if np.isfinite(slack_precision):
        if slack_precision <= 0.0:
            raise ValueError("slack_precision must be positive")
        vy = np.ones_like(u) if variance_yb is None else np.asarray(variance_yb)
        va = np.ones_like(v) if variance_ab is None else np.asarray(variance_ab)
        if vy.shape != u.shape or va.shape != v.shape:
            raise ValueError("slack variance shapes do not match factors")
        grad_u += vy * u / slack_precision
        grad_v += va * v / slack_precision
    return grad_u, grad_v


def anchored_intersection_sgd(
    problem: AnchoredIntersectionProblem,
    initial_yb: np.ndarray,
    initial_ab: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    workers: int = 1,
    learning_rate: float = 0.03,
    exact_interval: int = 50,
    slack_precision: float = 1.0,
    variance_yb: np.ndarray | None = None,
    variance_ab: np.ndarray | None = None,
) -> AnchoredSgdResult:
    """Run cold-start Adam and select only among exact evaluated iterates."""

    if steps < 0 or batch_size < 1 or exact_interval < 1 or workers < 1:
        raise ValueError("invalid SGD schedule")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    u = np.array(initial_yb, dtype=np.float64, copy=True)
    v = np.array(initial_ab, dtype=np.float64, copy=True)
    initial, _, _ = anchored_intersection_dual(
        problem, u, v, slack_precision=slack_precision,
        variance_yb=variance_yb, variance_ab=variance_ab,
    )
    best_value = initial
    best_u, best_v = u.copy(), v.copy()
    moment_u, moment_v = np.zeros_like(u), np.zeros_like(v)
    square_u, square_v = np.zeros_like(u), np.zeros_like(v)
    beta1, beta2 = 0.9, 0.999
    for step in range(1, steps + 1):
        gu, gv = anchored_intersection_sampled_gradient(
            problem, u, v, batch_size=batch_size, seed=seed, step=step,
            workers=workers, slack_precision=slack_precision,
            variance_yb=variance_yb, variance_ab=variance_ab,
        )
        moment_u = beta1 * moment_u + (1.0 - beta1) * gu
        moment_v = beta1 * moment_v + (1.0 - beta1) * gv
        square_u = beta2 * square_u + (1.0 - beta2) * gu * gu
        square_v = beta2 * square_v + (1.0 - beta2) * gv * gv
        correction1 = 1.0 - beta1 ** step
        correction2 = 1.0 - beta2 ** step
        u -= learning_rate * (moment_u / correction1) / (
            np.sqrt(square_u / correction2) + 1e-8
        )
        v -= learning_rate * (moment_v / correction1) / (
            np.sqrt(square_v / correction2) + 1e-8
        )
        if step % exact_interval == 0 or step == steps:
            value, _, _ = anchored_intersection_dual(
                problem, u, v, slack_precision=slack_precision,
                variance_yb=variance_yb, variance_ab=variance_ab,
            )
            if value < best_value:
                best_value = value
                best_u, best_v = u.copy(), v.copy()
    final, _, _ = anchored_intersection_dual(
        problem, u, v, slack_precision=slack_precision,
        variance_yb=variance_yb, variance_ab=variance_ab,
    )
    return AnchoredSgdResult(
        correction_yb=best_u, correction_ab=best_v,
        steps=steps, batch_size=batch_size, seed=seed,
        initial_objective=initial, best_objective=best_value,
        final_objective=final,
    )


def sparse_problem_from_dense(
    p_ya: np.ndarray,
    p_yb: np.ndarray,
    p_ab: np.ndarray,
) -> AnchoredSparseProblem:
    """Build the small-alphabet sparse oracle from three pair tables."""

    ya = _probability_table("p_ya", p_ya)
    yb = _probability_table("p_yb", p_yb)
    ab = _probability_table("p_ab", p_ab)
    if ya.shape[0] != yb.shape[0] or ya.shape[1] != ab.shape[0]:
        raise ValueError("pair tables use incompatible alphabets")
    if yb.shape[1] != ab.shape[1]:
        raise ValueError("YB and AB tables disagree on alphabet B")

    ya_y, ya_a = np.nonzero(ya)
    yb_y, yb_b = np.nonzero(yb)
    ab_a, ab_b = np.nonzero(ab)
    yb_index = {(int(y), int(b)): i for i, (y, b) in enumerate(zip(yb_y, yb_b))}
    ab_index = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(ab_a, ab_b))}
    r_ab = ab_conditional(ab)
    ptr = [0]
    triple_b: list[int] = []
    triple_yb: list[int] = []
    triple_ab: list[int] = []
    log_r: list[float] = []
    for y, a in zip(ya_y, ya_a):
        for b in np.flatnonzero(r_ab[a] > 0.0):
            yi = yb_index.get((int(y), int(b)))
            ai = ab_index.get((int(a), int(b)))
            # A missing YB target is still a valid zero-target factor.  The
            # dense oracle keeps it explicitly so model mass can be penalized.
            if yi is None:
                yi = len(yb_y)
                yb_y = np.append(yb_y, y)
                yb_b = np.append(yb_b, b)
                yb_index[(int(y), int(b))] = yi
            triple_b.append(int(b))
            triple_yb.append(yi)
            triple_ab.append(ai)
            log_r.append(float(np.log(r_ab[a, b])))
        ptr.append(len(triple_b))
    target_yb = np.zeros(len(yb_y), dtype=np.float64)
    for (y, b), i in yb_index.items():
        target_yb[i] = yb[y, b]
    return AnchoredSparseProblem(
        ya_y=np.asarray(ya_y, dtype=np.int32),
        ya_a=np.asarray(ya_a, dtype=np.int32),
        ya_probability=ya[ya_y, ya_a],
        row_ptr=np.asarray(ptr, dtype=np.int64),
        triple_b=np.asarray(triple_b, dtype=np.int32),
        triple_yb=np.asarray(triple_yb, dtype=np.int32),
        triple_ab=np.asarray(triple_ab, dtype=np.int32),
        log_r_ab=np.asarray(log_r),
        yb_y=np.asarray(yb_y, dtype=np.int32),
        yb_b=np.asarray(yb_b, dtype=np.int32),
        target_yb=target_yb,
        ab_a=np.asarray(ab_a, dtype=np.int32),
        ab_b=np.asarray(ab_b, dtype=np.int32),
        target_ab=ab[ab_a, ab_b],
    )


def anchored_sparse_dual(
    problem: AnchoredSparseProblem,
    correction_yb: np.ndarray,
    correction_ab: np.ndarray,
    *,
    slack_precision: float = float("inf"),
    variance_yb: np.ndarray | None = None,
    variance_ab: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate the anchored relaxed dual and its exact sparse gradient."""

    u = np.asarray(correction_yb, dtype=np.float64)
    v = np.asarray(correction_ab, dtype=np.float64)
    if u.shape != problem.target_yb.shape or v.shape != problem.target_ab.shape:
        raise ValueError("factor shapes do not match sparse problem")
    if slack_precision <= 0.0 or np.isnan(slack_precision):
        raise ValueError("slack_precision must be positive")
    grad_u = -problem.target_yb.copy()
    grad_v = -problem.target_ab.copy()
    objective = -float(np.dot(problem.target_yb, u) + np.dot(problem.target_ab, v))
    for row, mass in enumerate(problem.ya_probability):
        lo, hi = problem.row_ptr[row:row + 2]
        yb_ids = problem.triple_yb[lo:hi]
        ab_ids = problem.triple_ab[lo:hi]
        logits = problem.log_r_ab[lo:hi] + u[yb_ids] + v[ab_ids]
        maximum = float(np.max(logits))
        weights = np.exp(logits - maximum)
        total = float(np.sum(weights))
        probabilities = weights / total
        objective += float(mass) * (maximum + np.log(total))
        np.add.at(grad_u, yb_ids, mass * probabilities)
        np.add.at(grad_v, ab_ids, mass * probabilities)
    if np.isfinite(slack_precision):
        vy = np.ones_like(u) if variance_yb is None else np.asarray(variance_yb)
        va = np.ones_like(v) if variance_ab is None else np.asarray(variance_ab)
        if vy.shape != u.shape or va.shape != v.shape:
            raise ValueError("slack variance shapes do not match factors")
        scale = 1.0 / slack_precision
        objective += 0.5 * scale * (float(np.dot(vy, u * u)) + float(np.dot(va, v * v)))
        grad_u += scale * vy * u
        grad_v += scale * va * v
    return objective, grad_u, grad_v


def anchored_sparse_sgd(
    problem: AnchoredSparseProblem,
    initial_yb: np.ndarray,
    initial_ab: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    learning_rate: float = 0.03,
    slack_precision: float = 1.0,
    variance_yb: np.ndarray | None = None,
    variance_ab: np.ndarray | None = None,
) -> AnchoredSgdResult:
    """Fit the relaxed anchored dual with reproducible minibatch Adam.

    One minibatch independently samples hard-YA rows, YB target edges, and AB
    target edges.  Their one-hot differences are an unbiased gradient of the
    sparse dual.  ``batch_size`` is the total scientific batch size and is not
    derived from a worker count; a future parallel implementation must merely
    partition these same sampled indices.
    """

    if steps < 0 or batch_size < 1:
        raise ValueError("invalid SGD schedule")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    u = np.array(initial_yb, dtype=np.float64, copy=True)
    v = np.array(initial_ab, dtype=np.float64, copy=True)
    initial, _, _ = anchored_sparse_dual(
        problem, u, v, slack_precision=slack_precision,
        variance_yb=variance_yb, variance_ab=variance_ab,
    )
    best_value = initial
    best_u, best_v = u.copy(), v.copy()
    rng = np.random.default_rng(seed)
    moment_u = np.zeros_like(u)
    moment_v = np.zeros_like(v)
    square_u = np.zeros_like(u)
    square_v = np.zeros_like(v)
    vy = np.ones_like(u) if variance_yb is None else np.asarray(variance_yb)
    va = np.ones_like(v) if variance_ab is None else np.asarray(variance_ab)
    if vy.shape != u.shape or va.shape != v.shape:
        raise ValueError("slack variance shapes do not match factors")
    beta1, beta2 = 0.9, 0.999
    for step in range(1, steps + 1):
        rows = rng.choice(
            len(problem.ya_probability), size=batch_size,
            p=problem.ya_probability,
        )
        target_yb = rng.choice(
            len(problem.target_yb), size=batch_size, p=problem.target_yb,
        )
        target_ab = rng.choice(
            len(problem.target_ab), size=batch_size, p=problem.target_ab,
        )
        gu = np.zeros_like(u)
        gv = np.zeros_like(v)
        scale = 1.0 / float(batch_size)
        np.add.at(gu, target_yb, -scale)
        np.add.at(gv, target_ab, -scale)
        for row in rows:
            lo, hi = problem.row_ptr[row:row + 2]
            yi = problem.triple_yb[lo:hi]
            ai = problem.triple_ab[lo:hi]
            logits = problem.log_r_ab[lo:hi] + u[yi] + v[ai]
            logits -= np.max(logits)
            probabilities = np.exp(logits)
            probabilities /= np.sum(probabilities)
            np.add.at(gu, yi, scale * probabilities)
            np.add.at(gv, ai, scale * probabilities)
        if np.isfinite(slack_precision):
            gu += vy * u / slack_precision
            gv += va * v / slack_precision
        moment_u = beta1 * moment_u + (1.0 - beta1) * gu
        moment_v = beta1 * moment_v + (1.0 - beta1) * gv
        square_u = beta2 * square_u + (1.0 - beta2) * gu * gu
        square_v = beta2 * square_v + (1.0 - beta2) * gv * gv
        correction1 = 1.0 - beta1 ** step
        correction2 = 1.0 - beta2 ** step
        u -= learning_rate * (moment_u / correction1) / (
            np.sqrt(square_u / correction2) + 1e-8
        )
        v -= learning_rate * (moment_v / correction1) / (
            np.sqrt(square_v / correction2) + 1e-8
        )
        # Exact evaluations select the published iterate; sampled loss is
        # never used to claim improvement or convergence.
        value, _, _ = anchored_sparse_dual(
            problem, u, v, slack_precision=slack_precision,
            variance_yb=vy, variance_ab=va,
        )
        if value < best_value:
            best_value = value
            best_u, best_v = u.copy(), v.copy()
    final, _, _ = anchored_sparse_dual(
        problem, u, v, slack_precision=slack_precision,
        variance_yb=vy, variance_ab=va,
    )
    return AnchoredSgdResult(
        correction_yb=best_u,
        correction_ab=best_v,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        initial_objective=initial,
        best_objective=best_value,
        final_objective=final,
    )
