"""Memoized frontier search for honest order-two models.

The reference rate is fixed once: it is the original honest memory-one
result.  A point is a *winner* precisely when it beats that fixed reference,
not when it merely beats the point from which it was reached.

The original memory-one point is the initial expansion center even though it
cannot beat itself.  Search has two stages.  At that reference ``(V, M1, 0)`` it first follows the
declared positive ``M2`` grid (normally 16, 64, 256, ...) one point at a time,
and stops that ray at its first reversal.  It then exhausts every valid point
at lattice distance one or two from every winner.  Newly discovered winners
expand the frontier; a memoized result map makes repeated work impossible.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True, order=True)
class MemoryTwoPoint:
    vocabulary_size: int
    first_lag_states: int
    second_lag_states: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (
            self.vocabulary_size,
            self.first_lag_states,
            self.second_lag_states,
        )


def declared_triplet_grid(
    *,
    vocabulary_grid: Sequence[int],
    first_lag_grid: Sequence[int],
    second_lag_grid: Sequence[int],
    minimum_second_lag: int = 0,
) -> tuple[MemoryTwoPoint, ...]:
    """Build the finite production grid, enforcing ``M2 < M1 <= V``.

    The returned order is deterministic and duplicates in any input grid are
    removed.  A declared finite grid is important: selecting the best measured
    triplet costs ``log2(len(grid))`` bits and must not be reconstructed after
    looking at the answers.
    """

    points = {
        MemoryTwoPoint(v, m1, m2)
        for v in map(int, vocabulary_grid)
        for m1 in map(int, first_lag_grid)
        for m2 in map(int, second_lag_grid)
        if minimum_second_lag <= m2 < m1 <= v
    }
    if not points:
        raise ValueError("declared triplet grid is empty")
    return tuple(sorted(points))


def enumerative_subset_bits(universe_size: int, subset_size: int) -> float:
    """Exact enumerative description length of an unordered subset."""

    if not 0 <= subset_size <= universe_size:
        raise ValueError("invalid subset size")
    return (
        math.lgamma(universe_size + 1)
        - math.lgamma(subset_size + 1)
        - math.lgamma(universe_size - subset_size + 1)
    ) / math.log(2.0)


def nested_frequency_subset_bits(point: MemoryTwoPoint) -> float:
    """Enumerative description of nested frequency-selected lag states.

    The two lag maps use the same frequency order, hence the second-lag set is
    a subset of the first-lag set when ``M2 < M1``.  We transmit the first set
    from ``V`` symbols and then the second from those ``M1`` symbols.  Charging
    two independent subsets from ``V`` would be valid but needlessly wasteful.
    """

    v, m1, m2 = point.as_tuple()
    if not 0 <= m2 < m1 <= v:
        raise ValueError("nested accounting requires 0 <= M2 < M1 <= V")

    return (
        enumerative_subset_bits(v, m1)
        + enumerative_subset_bits(m1, m2)
    )


class MemoryTwoFrontier:
    """State machine for the sparse memory-two search agreed for production."""

    def __init__(
        self,
        *,
        vocabulary_grid: Sequence[int],
        state_grid: Sequence[int],
        baseline_point: MemoryTwoPoint,
        baseline_bpc: float,
        identity_tolerance: float = 1e-10,
        improvement_tolerance: float = 0.0,
    ) -> None:
        self.vocabulary_grid = tuple(sorted(set(map(int, vocabulary_grid))))
        self.state_grid = tuple(sorted(set(map(int, state_grid))))
        if not self.vocabulary_grid or not self.state_grid:
            raise ValueError("vocabulary_grid and state_grid must be nonempty")
        if 0 not in self.state_grid or 16 not in self.state_grid:
            raise ValueError("state_grid must contain the seed values 0 and 16")
        self.baseline_point = baseline_point
        self.baseline_bpc = float(baseline_bpc)
        self.identity_tolerance = float(identity_tolerance)
        self.improvement_tolerance = float(improvement_tolerance)
        if not self._valid(baseline_point):
            raise ValueError("baseline point is outside the declared grids")

        self.results: dict[MemoryTwoPoint, float] = {}
        self.history: list[dict] = []
        self._arm = tuple(
            MemoryTwoPoint(
                baseline_point.vocabulary_size,
                baseline_point.first_lag_states,
                m2,
            )
            for m2 in self.state_grid
            if m2 <= baseline_point.vocabulary_size
        )
        self._arm_closed = False

    def _valid(self, point: MemoryTwoPoint) -> bool:
        return (
            point.vocabulary_size in self.vocabulary_grid
            and point.first_lag_states in self.state_grid
            and point.second_lag_states in self.state_grid
            and 0 <= point.first_lag_states <= point.vocabulary_size
            and 0 <= point.second_lag_states <= point.vocabulary_size
        )

    @property
    def winners(self) -> tuple[MemoryTwoPoint, ...]:
        cutoff = self.baseline_bpc - self.improvement_tolerance
        return tuple(sorted(
            point for point, bpc in self.results.items()
            if point != self.baseline_point and bpc < cutoff
        ))

    @property
    def expansion_centers(self) -> tuple[MemoryTwoPoint, ...]:
        """Original memory one plus every strict winner found thereafter."""

        centers = set(self.winners)
        if self.baseline_point in self.results:
            centers.add(self.baseline_point)
        return tuple(sorted(centers))

    def record(self, point: MemoryTwoPoint, bpc: float) -> None:
        """Record one honest result, rejecting repeats and identity failures."""

        if not self._valid(point):
            raise ValueError(f"invalid search point {point.as_tuple()}")
        if point in self.results:
            raise ValueError(f"point already evaluated: {point.as_tuple()}")
        bpc = float(bpc)
        if point == self.baseline_point and (
            abs(bpc - self.baseline_bpc) > self.identity_tolerance
        ):
            raise RuntimeError(
                "M2=0 identity gate failed: "
                f"memory-two={bpc:.12g}, memory-one={self.baseline_bpc:.12g}"
            )
        self.results[point] = bpc
        self.history.append({
            "point": point.as_tuple(),
            "bpc": bpc,
            "beats_original_memory_one": (
                point != self.baseline_point
                and bpc < self.baseline_bpc - self.improvement_tolerance
            ),
        })

    def _next_arm_point(self) -> MemoryTwoPoint | None:
        if self._arm_closed:
            return None
        for index, point in enumerate(self._arm):
            if point not in self.results:
                if index == 0:
                    return point
                previous = self._arm[index - 1]
                if previous not in self.results:
                    return previous
                # Advance only while the preceding step improved.  The first
                # reversal is retained (and may itself be a winner), but ends
                # this special one-dimensional ray.
                if index >= 2:
                    before_previous = self._arm[index - 2]
                    if (
                        self.results[previous]
                        >= self.results[before_previous]
                        - self.improvement_tolerance
                    ):
                        self._arm_closed = True
                        return None
                return point
        self._arm_closed = True
        return None

    @staticmethod
    def _factor_neighbors(value: int, grid: tuple[int, ...]) -> tuple[int, ...]:
        """One step is exactly x2 or /2; zero has the declared 0<->16 edge."""

        candidates = {16} if value == 0 else {value * 2}
        if value == 16 and 0 in grid:
            candidates.add(0)
        if value > 0 and value % 2 == 0:
            candidates.add(value // 2)
        return tuple(sorted(candidates.intersection(grid)))

    def _one_step_neighbors(self, point: MemoryTwoPoint) -> set[MemoryTwoPoint]:
        found: set[MemoryTwoPoint] = set()
        for value in self._factor_neighbors(
            point.vocabulary_size, self.vocabulary_grid
        ):
            found.add(MemoryTwoPoint(
                value, point.first_lag_states, point.second_lag_states
            ))
        for value in self._factor_neighbors(
            point.first_lag_states, self.state_grid
        ):
            found.add(MemoryTwoPoint(
                point.vocabulary_size, value, point.second_lag_states
            ))
        for value in self._factor_neighbors(
            point.second_lag_states, self.state_grid
        ):
            found.add(MemoryTwoPoint(
                point.vocabulary_size, point.first_lag_states, value
            ))
        return {candidate for candidate in found if self._valid(candidate)}

    def _radius_two_neighbors(
        self, point: MemoryTwoPoint
    ) -> set[MemoryTwoPoint]:
        one = self._one_step_neighbors(point)
        two_candidates = {
            candidate
            for intermediate in one
            for candidate in self._one_step_neighbors(intermediate)
        }
        # "Order two" means one elementary move on each of two distinct
        # coordinates.  It never means applying x2 twice to obtain x4 (or
        # /2 twice to obtain /4) on a single coordinate.
        two = {
            candidate for candidate in two_candidates
            if sum(a != b for a, b in zip(
                candidate.as_tuple(), point.as_tuple()
            )) == 2
        }
        return (one | two) - {point}

    def next_points(self) -> tuple[MemoryTwoPoint, ...]:
        """Return the next unevaluated point(s), in deterministic order."""

        arm = self._next_arm_point()
        if arm is not None:
            return (arm,)
        candidates: set[MemoryTwoPoint] = set()
        for center in self.expansion_centers:
            candidates.update(self._radius_two_neighbors(center))
        candidates.difference_update(self.results)
        return tuple(sorted(candidates))

    @property
    def complete(self) -> bool:
        return not self.next_points()

    def audit(self) -> dict:
        """Serializable state and stopping evidence."""

        pending = self.next_points()
        return {
            "version": 1,
            "baseline_point": self.baseline_point.as_tuple(),
            "baseline_bpc": self.baseline_bpc,
            "winner_definition": "strictly_below_original_memory_one_bpc",
            "neighborhood": (
                "one_move_or_one_move_on_each_of_two_distinct_coordinates;_"
                "each_x2_or_div2;_0_adjacent_16"
            ),
            "vocabulary_grid": self.vocabulary_grid,
            "state_grid": self.state_grid,
            "evaluated": {
                str(point.as_tuple()): bpc
                for point, bpc in sorted(self.results.items())
            },
            "winners": [point.as_tuple() for point in self.winners],
            "expansion_centers": [
                point.as_tuple() for point in self.expansion_centers
            ],
            "pending": [point.as_tuple() for point in pending],
            "complete": not pending,
            "history": list(self.history),
        }
