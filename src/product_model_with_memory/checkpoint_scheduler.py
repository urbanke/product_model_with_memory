"""Deterministic coarse-job scheduler used before online optimization."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from product_model_with_memory.production_coding import (
    PRODUCTION_SEQUENCE_ESTIMATOR,
    require_production_sequence_estimator,
)


@dataclass(frozen=True)
class CheckpointJob:
    """One restartable construction, graph, fitting, or evaluation job."""

    job_id: str
    job_type: str
    checkpoint: int | None
    command: tuple[str, ...]
    dependencies: tuple[str, ...]
    outputs: tuple[str, ...]
    completion_manifest: str
    workers: int
    private_memory_bytes: int = 0
    minimum_workers: int = 1


@dataclass(frozen=True)
class FixedSchedule:
    jobs: tuple[CheckpointJob, ...]
    waves: tuple[tuple[str, ...], ...]
    maximum_workers: int
    maximum_private_memory_bytes: int


def load_fixed_schedule(
    path: str | Path, *, enforce_wave_capacity: bool = True,
) -> FixedSchedule:
    """Load and validate the declarative fixed-schedule format."""

    source = Path(path)
    payload = json.loads(source.read_text())
    require_production_sequence_estimator(
        payload.get("sequence_estimator", PRODUCTION_SEQUENCE_ESTIMATOR),
        source=str(source),
    )
    jobs = tuple(CheckpointJob(
        job_id=row["id"],
        job_type=row["type"],
        checkpoint=row.get("checkpoint"),
        command=tuple(row["command"]),
        dependencies=tuple(row.get("dependencies", ())),
        outputs=tuple(row.get("outputs", ())),
        completion_manifest=row["completion_manifest"],
        workers=int(row.get("workers", 1)),
        private_memory_bytes=int(row.get("private_memory_bytes", 0)),
        minimum_workers=int(row.get("minimum_workers", 1)),
    ) for row in payload["jobs"])
    schedule = FixedSchedule(
        jobs=jobs,
        waves=tuple(tuple(wave) for wave in payload["waves"]),
        maximum_workers=int(payload["maximum_workers"]),
        maximum_private_memory_bytes=int(
            payload.get("maximum_private_memory_bytes", 0)
        ),
    )
    validate_fixed_schedule(
        schedule, enforce_wave_capacity=enforce_wave_capacity,
    )
    return schedule


def validate_fixed_schedule(
    schedule: FixedSchedule, *, enforce_wave_capacity: bool = True,
) -> None:
    """Reject missing jobs, dependency inversions, and resource overflow."""

    if schedule.maximum_workers < 1:
        raise ValueError("maximum_workers must be positive")
    by_id = {job.job_id: job for job in schedule.jobs}
    if len(by_id) != len(schedule.jobs):
        raise ValueError("job IDs must be unique")
    scheduled = [job_id for wave in schedule.waves for job_id in wave]
    if len(scheduled) != len(set(scheduled)):
        raise ValueError("a job appears in more than one wave")
    if set(scheduled) != set(by_id):
        raise ValueError("waves must contain every declared job exactly once")
    completed: set[str] = set()
    for wave_number, wave in enumerate(schedule.waves):
        unknown = set(wave) - set(by_id)
        if unknown:
            raise ValueError(f"wave {wave_number} has unknown jobs {unknown}")
        jobs = [by_id[job_id] for job_id in wave]
        for job in jobs:
            missing = set(job.dependencies) - completed
            if missing:
                raise ValueError(
                    f"job {job.job_id} is scheduled before {sorted(missing)}"
                )
            if job.workers < 1:
                raise ValueError(f"job {job.job_id} has no workers")
            if not 1 <= job.minimum_workers <= job.workers:
                raise ValueError(
                    f"job {job.job_id} has invalid minimum workers"
                )
        if (
            enforce_wave_capacity
            and sum(job.workers for job in jobs) > schedule.maximum_workers
        ):
            raise ValueError(f"wave {wave_number} exceeds worker capacity")
        memory = sum(job.private_memory_bytes for job in jobs)
        if (
            enforce_wave_capacity
            and schedule.maximum_private_memory_bytes > 0
            and memory > schedule.maximum_private_memory_bytes
        ):
            raise ValueError(f"wave {wave_number} exceeds memory capacity")
        completed.update(wave)


def _completed(job: CheckpointJob) -> bool:
    manifest = Path(job.completion_manifest)
    if not manifest.exists():
        return False
    try:
        payload = json.loads(manifest.read_text())
    except (OSError, ValueError):
        return False
    return (
        payload.get("status") == "completed"
        and payload.get("job_id") == job.job_id
        and all(Path(path).exists() for path in job.outputs)
    )


def _publish_completion(job: CheckpointJob, record: dict) -> None:
    destination = Path(job.completion_manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(destination)


def command_with_workers(job: CheckpointJob, workers: int) -> tuple[str, ...]:
    """Set the phase's worker option without changing its other arguments."""

    option = "--jobs" if job.job_type in {"C", "M", "A", "B"} else "--workers"
    if job.job_type not in {"C", "M", "A", "B", "F"}:
        return job.command
    command = list(job.command)
    # C meant the parallel monolithic constructor in version-1 schedules;
    # it is the one-worker assembler in split schedules.  Retain resumability
    # of old schedules without inventing an option for the new assembler.
    if job.job_type == "C" and option not in command:
        return job.command
    try:
        position = command.index(option)
    except ValueError as error:
        raise ValueError(f"{job.job_id} command lacks {option}") from error
    command[position + 1] = str(workers)
    return tuple(command)


def run_planned_schedule(
    jobs: tuple[CheckpointJob, ...],
    planned_workers: dict[str, int],
    priorities: dict[str, float],
    *,
    maximum_workers: int,
    maximum_private_memory_bytes: int = 0,
    worker_caps: dict[str, int] | None = None,
    worker_floors: dict[str, int] | None = None,
    working_directory: str | Path,
    environment: dict[str, str] | None = None,
    poll_seconds: float = 0.1,
    event_callback: Callable[[dict], None] | None = None,
) -> tuple[dict, ...]:
    """Execute a dependency graph whenever workers become available."""

    if maximum_workers < 1 or maximum_private_memory_bytes < 0 or poll_seconds <= 0:
        raise ValueError("invalid executor resource setting")
    root = Path(working_directory)
    by_id = {job.job_id: job for job in jobs}
    if set(planned_workers) != set(by_id):
        raise ValueError("planned worker map must contain every job")
    if set(priorities) != set(by_id):
        raise ValueError("priority map must contain every job")
    if worker_caps is not None and set(worker_caps) != set(by_id):
        raise ValueError("worker cap map must contain every job")
    if worker_floors is not None and set(worker_floors) != set(by_id):
        raise ValueError("worker floor map must contain every job")
    for job in jobs:
        unknown = set(job.dependencies) - set(by_id)
        if unknown:
            raise ValueError(f"{job.job_id} has unknown dependencies")
        if not 1 <= planned_workers[job.job_id] <= maximum_workers:
            raise ValueError(f"invalid worker allocation for {job.job_id}")
        if worker_caps is not None and not (
            1 <= worker_caps[job.job_id] <= maximum_workers
        ):
            raise ValueError(f"invalid worker cap for {job.job_id}")
        if worker_floors is not None and not (
            1 <= worker_floors[job.job_id] <= maximum_workers
            and (
                worker_caps is None
                or worker_floors[job.job_id] <= worker_caps[job.job_id]
            )
        ):
            raise ValueError(f"invalid worker floor for {job.job_id}")

    completed = {job.job_id for job in jobs if _completed(job)}
    launched = set(completed)
    running: dict[str, tuple[CheckpointJob, subprocess.Popen, float, int]] = {}
    records: list[dict] = []
    base_environment = os.environ.copy()
    if environment:
        base_environment.update(environment)

    while len(completed) < len(jobs):
        for job_id, (job, process, started, workers) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            finished = time.time()
            record = {
                "version": 1,
                "status": "completed" if return_code == 0 else "failed",
                "job_id": job.job_id,
                "job_type": job.job_type,
                "checkpoint": job.checkpoint,
                "workers": workers,
                "started_unix_seconds": started,
                "finished_unix_seconds": finished,
                "elapsed_seconds": finished - started,
                "return_code": return_code,
                "command": list(command_with_workers(job, workers)),
                "dependencies": list(job.dependencies),
                "outputs": list(job.outputs),
            }
            absent = [path for path in job.outputs if not Path(path).exists()]
            if return_code or absent:
                record["status"] = "failed"
                if absent:
                    record["missing_outputs"] = absent
                _publish_completion(job, record)
                if event_callback:
                    event_callback({
                        "event": "failed", "job_id": job_id,
                        "elapsed_seconds": record["elapsed_seconds"],
                        "return_code": return_code,
                        "missing_outputs": absent,
                        "assigned_workers": sum(
                            row[3] for key, row in running.items()
                            if key != job_id
                        ),
                    })
                for _, other, _, _ in running.values():
                    if other.poll() is None:
                        other.terminate()
                raise RuntimeError(f"planned job {job_id} failed")
            completed.add(job_id)
            _publish_completion(job, record)
            records.append(record)
            del running[job_id]
            if event_callback:
                event_callback({
                    "event": "finished", "job_id": job_id,
                    "elapsed_seconds": record["elapsed_seconds"],
                    "assigned_workers": sum(
                        row[3] for row in running.values()
                    ),
                    "running": sorted(running),
                })

        available = maximum_workers - sum(row[3] for row in running.values())
        memory_available = (
            maximum_private_memory_bytes
            - sum(row[0].private_memory_bytes for row in running.values())
            if maximum_private_memory_bytes else None
        )
        ready = [
            job for job in jobs
            if job.job_id not in launched
            and set(job.dependencies) <= completed
            and (worker_caps is not None or planned_workers[job.job_id] <= available)
            and (
                memory_available is None
                or job.private_memory_bytes <= memory_available
            )
        ]
        # PERFORMANCE INVARIANT: priorities choose among dependency-ready
        # jobs; they must not become synchronization barriers.  The schedule
        # generator deliberately uses anchor-causal priorities so multi-core
        # pair work cannot be starved behind a corpus-wide set of serial jobs.
        # Keep resource admission live and dependency-driven; changes require
        # an explicit utilization replay because a superficially harmless
        # priority change previously reduced a 12-core workload to 4-5 cores.
        #
        # E jobs have no descendants.  They fill otherwise idle capacity but
        # must never delay a ready causal-chain job because of an imperfect
        # analytic duration estimate.
        # Once construction for an anchor is ready, finish its short
        # topology/fit/score tail before opening still more construction.  A
        # prior priority-only order let pair construction for all 64 anchors
        # run first, then drained scoring at four cores.  Stage rank is only a
        # tie-break among dependency-ready jobs and therefore creates no
        # barrier: unused capacity still advances later anchors.
        tail_rank = {
            "S": 0, "F": 1, "T": 2, "C": 3,
            "A": 4, "B": 4, "M": 5, "U": 6, "D": 7, "R": 8,
        }
        ready.sort(key=lambda job: (
            job.job_type == "E",
            tail_rank.get(job.job_type, 4),
            priorities[job.job_id], job.job_id,
        ))
        launched_now = False
        assignments: dict[str, int] = {}
        if worker_caps is not None:
            # Breadth before depth, subject to phase resource contracts.
            # In particular a fit must not be launched serially merely because
            # the machine happened to be full at that instant: worker counts
            # cannot be changed after launch, so that would strand a long fit
            # at one core when the rest of the DAG drains.
            selected = []
            remaining = available
            remaining_memory = memory_available
            for job in ready:
                floor = 1 if worker_floors is None else worker_floors[job.job_id]
                if (
                    floor <= remaining
                    and (
                        remaining_memory is None
                        or job.private_memory_bytes <= remaining_memory
                    )
                ):
                    selected.append(job)
                    assignments[job.job_id] = floor
                    remaining -= floor
                    if remaining_memory is not None:
                        remaining_memory -= job.private_memory_bytes
            while remaining and any(
                assignments[job.job_id] < worker_caps[job.job_id]
                for job in selected
            ):
                for job in selected:
                    if (
                        remaining
                        and assignments[job.job_id] < worker_caps[job.job_id]
                    ):
                        assignments[job.job_id] += 1
                        remaining -= 1
        for job in ready:
            workers = (
                assignments.get(job.job_id, 0)
                if worker_caps is not None else planned_workers[job.job_id]
            )
            if workers < 1 or workers > available:
                continue
            command = command_with_workers(job, workers)
            started = time.time()
            process = subprocess.Popen(command, cwd=root, env=base_environment)
            running[job.job_id] = (job, process, started, workers)
            launched.add(job.job_id)
            available -= workers
            launched_now = True
            if event_callback:
                event_callback({
                    "event": "launched", "job_id": job.job_id,
                    "workers": workers,
                    "assigned_workers": maximum_workers - available,
                    "running": sorted(running),
                })
        if not running and not launched_now and len(completed) < len(jobs):
            raise RuntimeError("planned schedule cannot make progress")
        if running:
            time.sleep(poll_seconds)
    return tuple(records)


def run_fixed_schedule(
    schedule: FixedSchedule,
    *,
    working_directory: str | Path,
    environment: dict[str, str] | None = None,
) -> tuple[dict, ...]:
    """Execute explicit concurrent waves and return one record per job."""

    root = Path(working_directory)
    by_id = {job.job_id: job for job in schedule.jobs}
    records = []
    completed = {job.job_id for job in schedule.jobs if _completed(job)}
    base_environment = os.environ.copy()
    if environment:
        base_environment.update(environment)

    for wave_number, wave in enumerate(schedule.waves):
        pending = [by_id[job_id] for job_id in wave if job_id not in completed]
        for job in pending:
            missing = set(job.dependencies) - completed
            if missing:
                raise RuntimeError(
                    f"job {job.job_id} lacks completed dependencies "
                    f"{sorted(missing)}"
                )
        running = []
        for job in pending:
            started = time.time()
            process = subprocess.Popen(
                job.command,
                cwd=root,
                env=base_environment,
            )
            running.append((job, process, started))
        failed = []
        for job, process, started in running:
            return_code = process.wait()
            finished = time.time()
            record = {
                "version": 1,
                "status": "completed" if return_code == 0 else "failed",
                "job_id": job.job_id,
                "job_type": job.job_type,
                "checkpoint": job.checkpoint,
                "wave": wave_number,
                "workers": job.workers,
                "private_memory_bytes_requested": job.private_memory_bytes,
                "started_unix_seconds": started,
                "finished_unix_seconds": finished,
                "elapsed_seconds": finished - started,
                "return_code": return_code,
                "command": list(job.command),
                "dependencies": list(job.dependencies),
                "outputs": list(job.outputs),
            }
            if return_code == 0:
                absent = [path for path in job.outputs if not Path(path).exists()]
                if absent:
                    record["status"] = "failed"
                    record["missing_outputs"] = absent
                    failed.append(job.job_id)
                else:
                    completed.add(job.job_id)
            else:
                failed.append(job.job_id)
            _publish_completion(job, record)
            records.append(record)
        if failed:
            raise RuntimeError(
                f"wave {wave_number} failed for jobs {sorted(failed)}"
            )
    return tuple(records)


def run_dependency_schedule(
    schedule: FixedSchedule,
    *,
    working_directory: str | Path,
    maximum_workers: int | None = None,
    environment: dict[str, str] | None = None,
    poll_seconds: float = 0.1,
    event_callback: Callable[[dict], None] | None = None,
) -> tuple[dict, ...]:
    """Run a fixed schedule's DAG without treating its waves as barriers.

    Wave order supplies only a stable priority hint.  Every job launches as
    soon as its declared dependencies and live worker capacity permit.
    """

    capacity = schedule.maximum_workers if maximum_workers is None else maximum_workers
    if capacity < 1:
        raise ValueError("maximum_workers must be positive")
    priority = {
        job_id: float(wave_number)
        for wave_number, wave in enumerate(schedule.waves)
        for job_id in wave
    }
    return run_planned_schedule(
        schedule.jobs,
        {job.job_id: min(job.workers, capacity) for job in schedule.jobs},
        priority,
        maximum_workers=capacity,
        maximum_private_memory_bytes=getattr(
            schedule, "maximum_private_memory_bytes", 0
        ),
        worker_caps={
            job.job_id: min(job.workers, capacity) for job in schedule.jobs
        },
        worker_floors={
            job.job_id: min(job.minimum_workers, capacity)
            for job in schedule.jobs
        },
        working_directory=working_directory,
        environment=environment,
        poll_seconds=poll_seconds,
        event_callback=event_callback,
    )
