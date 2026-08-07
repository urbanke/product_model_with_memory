"""Deterministic coarse-job scheduler used before online optimization."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


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


@dataclass(frozen=True)
class FixedSchedule:
    jobs: tuple[CheckpointJob, ...]
    waves: tuple[tuple[str, ...], ...]
    maximum_workers: int
    maximum_private_memory_bytes: int


def load_fixed_schedule(path: str | Path) -> FixedSchedule:
    """Load and validate the declarative fixed-schedule format."""

    source = Path(path)
    payload = json.loads(source.read_text())
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
    ) for row in payload["jobs"])
    schedule = FixedSchedule(
        jobs=jobs,
        waves=tuple(tuple(wave) for wave in payload["waves"]),
        maximum_workers=int(payload["maximum_workers"]),
        maximum_private_memory_bytes=int(
            payload.get("maximum_private_memory_bytes", 0)
        ),
    )
    validate_fixed_schedule(schedule)
    return schedule


def validate_fixed_schedule(schedule: FixedSchedule) -> None:
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
        if sum(job.workers for job in jobs) > schedule.maximum_workers:
            raise ValueError(f"wave {wave_number} exceeds worker capacity")
        memory = sum(job.private_memory_bytes for job in jobs)
        if (
            schedule.maximum_private_memory_bytes > 0
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
