"""Tests for the deterministic coarse checkpoint scheduler."""

import json
import sys

import pytest

from product_model_with_memory.checkpoint_scheduler import (
    CheckpointJob,
    command_with_workers,
    load_fixed_schedule,
    run_fixed_schedule,
    run_planned_schedule,
)


def _job(tmp_path, name, dependencies=()):
    output = tmp_path / f"{name}.txt"
    manifest = tmp_path / "manifests" / f"{name}.json"
    return {
        "id": name,
        "type": name[0],
        "checkpoint": int(name[1:]),
        "command": [
            sys.executable, "-c",
            f"from pathlib import Path; Path({str(output)!r}).write_text('ok')",
        ],
        "dependencies": list(dependencies),
        "outputs": [str(output)],
        "completion_manifest": str(manifest),
        "workers": 1,
        "private_memory_bytes": 100,
    }


def test_fixed_schedule_runs_waves_and_resumes(tmp_path):
    payload = {
        "maximum_workers": 2,
        "maximum_private_memory_bytes": 200,
        "jobs": [
            _job(tmp_path, "C0"),
            _job(tmp_path, "F0", ("C0",)),
            _job(tmp_path, "C1", ("C0",)),
            _job(tmp_path, "E0", ("F0",)),
        ],
        "waves": [["C0"], ["F0", "C1"], ["E0"]],
    }
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps(payload))
    schedule = load_fixed_schedule(path)
    records = run_fixed_schedule(schedule, working_directory=tmp_path)
    assert [record["job_id"] for record in records] == ["C0", "F0", "C1", "E0"]
    assert all(record["status"] == "completed" for record in records)
    assert run_fixed_schedule(schedule, working_directory=tmp_path) == ()


def test_fixed_schedule_rejects_dependency_in_same_wave(tmp_path):
    payload = {
        "maximum_workers": 2,
        "maximum_private_memory_bytes": 200,
        "jobs": [_job(tmp_path, "C0"), _job(tmp_path, "F0", ("C0",))],
        "waves": [["C0", "F0"]],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="scheduled before"):
        load_fixed_schedule(path)


def test_analytic_load_may_ignore_unused_fixed_wave_capacity(tmp_path):
    payload = {
        "maximum_workers": 1,
        "maximum_private_memory_bytes": 200,
        "jobs": [_job(tmp_path, "C0"), _job(tmp_path, "F0")],
        "waves": [["C0", "F0"]],
    }
    path = tmp_path / "oversized.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="exceeds worker capacity"):
        load_fixed_schedule(path)
    schedule = load_fixed_schedule(path, enforce_wave_capacity=False)
    assert len(schedule.jobs) == 2


def test_command_with_workers_changes_parallel_phase_option(tmp_path):
    common = dict(
        checkpoint=0, dependencies=(), outputs=(),
        completion_manifest=str(tmp_path / "done.json"), workers=1,
    )
    c = CheckpointJob(
        "C0", "C", command=("python", "x", "--jobs", "2"), **common,
    )
    f = CheckpointJob(
        "F0", "F", command=("python", "x", "--workers", "2"), **common,
    )
    assert command_with_workers(c, 4)[-1] == "4"
    assert command_with_workers(f, 3)[-1] == "3"


def test_planned_executor_launches_dependency_graph(tmp_path):
    def job(job_id, dependencies):
        output = tmp_path / f"{job_id}.txt"
        return CheckpointJob(
            job_id, "E", 0,
            (
                sys.executable, "-c",
                f"from pathlib import Path; Path({str(output)!r}).write_text('ok')",
            ),
            dependencies, (str(output),),
            str(tmp_path / "manifests" / f"{job_id}.json"), 1,
        )

    jobs = (job("A", ()), job("B", ()), job("D", ("A", "B")))
    records = run_planned_schedule(
        jobs, {name: 1 for name in ("A", "B", "D")},
        {"A": 0.0, "B": 0.0, "D": 1.0},
        maximum_workers=2, working_directory=tmp_path, poll_seconds=0.01,
    )
    assert {row["job_id"] for row in records} == {"A", "B", "D"}


def test_planned_executor_uses_evaluation_only_as_capacity_filler(tmp_path):
    events = []

    def job(job_id, kind):
        output = tmp_path / f"{job_id}.txt"
        return CheckpointJob(
            job_id, kind, 0,
            (
                sys.executable, "-c",
                f"from pathlib import Path; Path({str(output)!r}).write_text('ok')",
            ),
            (), (str(output),),
            str(tmp_path / "manifests" / f"{job_id}.json"), 1,
        )

    jobs = (job("E0", "E"), job("G0", "G"))
    run_planned_schedule(
        jobs, {"E0": 1, "G0": 1}, {"E0": 0.0, "G0": 1.0},
        maximum_workers=1, working_directory=tmp_path, poll_seconds=0.01,
        event_callback=events.append,
    )
    launches = [row["job_id"] for row in events if row["event"] == "launched"]
    assert launches == ["G0", "E0"]


def test_planned_executor_expands_job_to_available_cap(tmp_path):
    events = []
    output = tmp_path / "C0.txt"
    job = CheckpointJob(
        "C0", "E", 0,
        (
            sys.executable, "-c",
            f"from pathlib import Path; Path({str(output)!r}).write_text('ok')",
        ),
        (), (str(output),), str(tmp_path / "C0.json"), 1,
    )
    run_planned_schedule(
        (job,), {"C0": 2}, {"C0": 0.0}, maximum_workers=8,
        worker_caps={"C0": 4}, working_directory=tmp_path,
        poll_seconds=0.01, event_callback=events.append,
    )
    launch = next(row for row in events if row["event"] == "launched")
    assert launch["workers"] == 4


def test_planned_executor_never_launches_a_zero_worker_job(tmp_path):
    events = []

    def job(job_id, delay):
        output = tmp_path / f"{job_id}.txt"
        return CheckpointJob(
            job_id, "G", 0,
            (
                sys.executable, "-c",
                "import time; from pathlib import Path; "
                f"time.sleep({delay}); Path({str(output)!r}).write_text('ok')",
            ),
            (), (str(output),), str(tmp_path / f"{job_id}.json"), 1,
        )

    run_planned_schedule(
        (job("A", 0.05), job("B", 0.0)), {"A": 1, "B": 1},
        {"A": 0.0, "B": 1.0}, maximum_workers=1,
        worker_caps={"A": 1, "B": 1}, working_directory=tmp_path,
        poll_seconds=0.01, event_callback=events.append,
    )
    launches = [row for row in events if row["event"] == "launched"]
    assert [row["job_id"] for row in launches] == ["A", "B"]
    assert all(row["workers"] == 1 for row in launches)
