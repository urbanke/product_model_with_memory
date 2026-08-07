"""Portable, dimensionless work estimates for checkpoint scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class AnalyticCheckpointWork:
    checkpoint: int
    prefix: int
    construction: float
    graph: float
    fitting: float
    evaluation: float | None
    expected_pairs: float
    expected_triangles: float


@dataclass(frozen=True)
class MoldableTask:
    task_id: str
    work: float
    dependencies: tuple[str, ...] = ()
    maximum_workers: int = 4


@dataclass(frozen=True)
class PlannedTask:
    task_id: str
    start: float
    finish: float
    workers: int
    work: float


def checkpoint_tasks(
    profile: tuple[AnalyticCheckpointWork, ...],
    *,
    construction_maximum_workers: int = 4,
    fitting_maximum_workers: int = 4,
) -> tuple[MoldableTask, ...]:
    """Translate checkpoint work into the actual C/G/F/E dependency graph."""

    tasks: list[MoldableTask] = []
    for row in profile:
        k = row.checkpoint
        tasks.append(MoldableTask(
            f"C{k}", max(row.construction, 1.0),
            () if k == 0 else (f"C{k - 1}",),
            construction_maximum_workers,
        ))
        graph_dependencies = [f"C{k}"]
        if k:
            graph_dependencies.append(f"G{k - 1}")
        tasks.append(MoldableTask(
            f"G{k}", max(row.graph, 1.0), tuple(graph_dependencies), 1,
        ))
        fitting_dependencies = [f"G{k}"]
        if k:
            fitting_dependencies.append(f"F{k - 1}")
        tasks.append(MoldableTask(
            f"F{k}", max(row.fitting, 1.0), tuple(fitting_dependencies),
            fitting_maximum_workers,
        ))
        if row.evaluation is not None:
            tasks.append(MoldableTask(
                f"E{k}", max(row.evaluation, 1.0),
                (f"F{k}", f"C{k + 1}"), 1,
            ))
    return tuple(tasks)


def worker_speedup(workers: int) -> float:
    """Initial portable speed law: useful to four workers, then saturated.

    This is a prior, not a machine benchmark.  Completion records may later
    replace it during the same run.  The values encode the observed qualitative
    regime: nearly twofold at two workers, then sharply diminishing returns.
    """

    if workers < 1:
        raise ValueError("workers must be positive")
    return (1.0, 1.9, 2.25, 2.4)[min(workers, 4) - 1]


def worker_modes(maximum_workers: int) -> tuple[int, ...]:
    """Return the non-dominated initial worker choices."""

    if maximum_workers < 1:
        raise ValueError("maximum_workers must be positive")
    return tuple(range(1, min(maximum_workers, 4) + 1))


def geometric_prefixes(
    stop: int, count: int, *, first_prefix: int = 2_050,
) -> np.ndarray:
    """Return the causal geometric checkpoint prefixes used by experiments."""

    if stop < 1 or count < 1 or first_prefix < 1:
        raise ValueError("stop, count, and first_prefix must be positive")
    first = min(first_prefix, max(1, stop // count))
    lo, hi = 1.0, 4.0
    for _ in range(200):
        ratio = (lo + hi) / 2.0
        total = first * (
            count if abs(ratio - 1.0) < 1e-12
            else (ratio**count - 1.0) / (ratio - 1.0)
        )
        if total < stop:
            lo = ratio
        else:
            hi = ratio
    ratio = (lo + hi) / 2.0
    accumulated = 0.0
    prefixes = []
    for k in range(count):
        accumulated += first * ratio**k
        prefixes.append(min(stop, round(accumulated)))
    prefixes[-1] = stop
    result = np.unique(np.asarray(prefixes, dtype=np.int64))
    if len(result) != count:
        raise ValueError("requested checkpoints collapse at this stream length")
    return result


def _probability_histogram(
    probabilities: np.ndarray, bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probabilities, dtype=np.float64)
    p = p[p > 0]
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all():
        raise ValueError("invalid unigram probabilities")
    p = p / p.sum()
    bins = max(1, min(int(bins), len(p)))
    edges = np.linspace(np.log(p.min()), np.log(p.max()), bins + 1)
    membership = np.minimum(
        np.searchsorted(edges, np.log(p), side="right") - 1, bins - 1
    )
    counts = np.bincount(membership, minlength=bins).astype(np.float64)
    mass = np.bincount(membership, weights=p, minlength=bins)
    occupied = counts > 0
    return counts[occupied], mass[occupied] / counts[occupied]


def expected_pair_support(
    observations: int, probabilities: np.ndarray, *, bins: int = 64,
) -> float:
    """Estimate distinct ordered pairs without constructing a V by V table."""

    if observations <= 0:
        return 0.0
    counts, representative = _probability_histogram(probabilities, bins)
    rate = np.multiply.outer(representative, representative)
    multiplicity = np.multiply.outer(counts, counts)
    return float(np.sum(multiplicity * -np.expm1(-observations * rate)))


def expected_compatible_triangles(
    observations: int, probabilities: np.ndarray, *, bins: int = 48,
) -> float:
    """Estimate triples whose YA, YB, and AB pair supports all occur."""

    if observations <= 0:
        return 0.0
    counts, p = _probability_histogram(probabilities, bins)
    pair_seen = -np.expm1(
        -observations * np.multiply.outer(p, p)
    )
    total = 0.0
    # The loop is over at most 48 probability classes, not vocabulary items.
    for y in range(len(p)):
        compatibility = (
            pair_seen[y, :, None]
            * pair_seen[y, None, :]
            * pair_seen
        )
        total += counts[y] * np.sum(
            counts[:, None] * counts[None, :] * compatibility
        )
    return float(total)


def checkpoint_work_profile(
    prefixes: np.ndarray,
    probabilities: np.ndarray,
    *,
    stochastic_steps: int,
    replicas: int,
    blocks: int,
    exact_interval: int,
) -> tuple[AnalyticCheckpointWork, ...]:
    """Build dimensionless C/G/F/E work estimates for every checkpoint."""

    edges = np.asarray(prefixes, dtype=np.int64)
    if (
        edges.ndim != 1 or not len(edges) or np.any(edges <= 0)
        or np.any(edges[1:] <= edges[:-1])
    ):
        raise ValueError("checkpoint prefixes must increase")
    if min(stochastic_steps, replicas, blocks, exact_interval) < 1:
        raise ValueError("optimizer work parameters must be positive")
    pairs = np.asarray([
        expected_pair_support(int(n), probabilities) for n in edges
    ])
    triangles = np.asarray([
        expected_compatible_triangles(int(n), probabilities) for n in edges
    ])
    previous_edges = np.r_[0, edges[:-1]]
    previous_pairs = np.r_[0.0, pairs[:-1]]
    previous_triangles = np.r_[0.0, triangles[:-1]]
    # One full pass at initialization plus periodic exact certificates, and
    # sampled blocks covering approximately 1/blocks of the triangles.
    full_passes = 1 + int(np.ceil(stochastic_steps / exact_interval))
    sampled_passes = stochastic_steps * replicas / blocks
    triangle_passes = full_passes + sampled_passes
    rows = []
    for k, prefix in enumerate(edges):
        delta_n = int(prefix - previous_edges[k])
        construction = delta_n + 3.0 * max(0.0, pairs[k] - previous_pairs[k])
        graph = max(0.0, triangles[k] - previous_triangles[k])
        fitting = max(1.0, triangles[k]) * triangle_passes
        evaluation = (
            float(edges[k + 1] - prefix) if k + 1 < len(edges) else None
        )
        rows.append(AnalyticCheckpointWork(
            k, int(prefix), construction, graph, fitting, evaluation,
            float(pairs[k]), float(triangles[k]),
        ))
    return tuple(rows)


def plan_moldable_tasks(
    tasks: tuple[MoldableTask, ...], maximum_workers: int,
) -> tuple[PlannedTask, ...]:
    """Event-driven critical-path list schedule in dimensionless time.

    Every ready task first receives one worker.  Remaining workers are assigned
    by the largest marginal reduction in predicted completion time.  Worker
    counts are fixed after launch, so the plan is moldable rather than
    malleable.  This is the portable phase-one policy; it consumes no measured
    wall-clock durations.
    """

    if maximum_workers < 1:
        raise ValueError("maximum_workers must be positive")
    by_id = {task.task_id: task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("task IDs must be unique")
    for task in tasks:
        if task.work <= 0 or not np.isfinite(task.work):
            raise ValueError(f"task {task.task_id} has invalid work")
        if task.maximum_workers < 1:
            raise ValueError(f"task {task.task_id} has no worker mode")
        unknown = set(task.dependencies) - set(by_id)
        if unknown:
            raise ValueError(f"task {task.task_id} has unknown dependencies")

    children = {task.task_id: [] for task in tasks}
    for task in tasks:
        for parent in task.dependencies:
            children[parent].append(task.task_id)

    visiting: set[str] = set()

    @lru_cache(maxsize=None)
    def bottom_level(task_id: str) -> float:
        if task_id in visiting:
            raise ValueError("task dependencies contain a cycle")
        visiting.add(task_id)
        task = by_id[task_id]
        own = task.work / worker_speedup(
            min(task.maximum_workers, maximum_workers, 4)
        )
        tail = max(
            (bottom_level(child) for child in children[task_id]),
            default=0.0,
        )
        visiting.remove(task_id)
        return own + tail

    for task in tasks:
        bottom_level(task.task_id)

    time = 0.0
    completed: set[str] = set()
    launched: set[str] = set()
    running: list[PlannedTask] = []
    result: list[PlannedTask] = []
    while len(completed) < len(tasks):
        just_finished = [row for row in running if row.finish <= time + 1e-12]
        for row in just_finished:
            completed.add(row.task_id)
        running = [row for row in running if row.finish > time + 1e-12]
        occupied = sum(row.workers for row in running)
        available = maximum_workers - occupied
        ready = [
            task for task in tasks
            if task.task_id not in launched
            and set(task.dependencies) <= completed
        ]
        ready.sort(key=lambda task: (-bottom_level(task.task_id), task.task_id))
        selected = ready[:available]
        allocations = {task.task_id: 1 for task in selected}
        spare = available - len(selected)
        while spare and selected:
            candidates = []
            for task in selected:
                current = allocations[task.task_id]
                limit = min(task.maximum_workers, maximum_workers, 4)
                if current < limit:
                    gain = task.work * (
                        1.0 / worker_speedup(current)
                        - 1.0 / worker_speedup(current + 1)
                    )
                    candidates.append((gain, bottom_level(task.task_id), task))
            if not candidates:
                break
            _, _, chosen = max(
                candidates, key=lambda row: (row[0], row[1], row[2].task_id)
            )
            allocations[chosen.task_id] += 1
            spare -= 1
        for task in selected:
            workers = allocations[task.task_id]
            finish = time + task.work / worker_speedup(workers)
            row = PlannedTask(task.task_id, time, finish, workers, task.work)
            running.append(row)
            result.append(row)
            launched.add(task.task_id)
        if not running and len(completed) < len(tasks):
            raise ValueError("task graph cannot make progress")
        if not running:
            break
        time = min(row.finish for row in running)
    return tuple(result)
