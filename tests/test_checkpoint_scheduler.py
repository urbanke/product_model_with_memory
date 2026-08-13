"""Tests for the deterministic coarse checkpoint scheduler."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np

from product_model_with_memory.checkpoint_scheduler import (
    CheckpointJob,
    command_with_workers,
    load_fixed_schedule,
    run_dependency_schedule,
    run_fixed_schedule,
    run_planned_schedule,
)

REPOSITORY = Path(__file__).resolve().parent.parent


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


def test_generated_fit_preserves_fixed_batch_geometry(tmp_path):
    schedule_path = tmp_path / "jobs.json"
    stream = tmp_path / "stream"
    stream.mkdir()
    np.save(stream / "ids.npy", np.arange(1000, dtype=np.int32) % 32)
    (stream / "stream.json").write_text(json.dumps({
        "representation": "bpe", "encoding": "cl100k_base",
        "source_file": "test", "n_bytes": 1000, "n_tokens": 1000,
        "alphabet": 100277, "fixed_bits": 0,
    }))
    subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts" / "make_fixed_checkpoint_schedule.py"),
            "--root", str(tmp_path / "run"),
            "--ids", str(stream),
            "--top-k", "31",
            "--n", "1000",
            "--checkpoints", "2",
            "--first-checkpoint", "100",
            "--policy", "pipeline",
            "--out", str(schedule_path),
        ],
        check=True,
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
    )
    payload = json.loads(schedule_path.read_text())
    assert payload["sequence_estimator"] == (
        "layered_depth_averaged_product_simplex_v1"
    )
    command = next(row["command"] for row in payload["jobs"] if row["id"] == "F0")

    def option(name):
        return command[command.index(name) + 1]

    assert option("--replicas") == "12"
    assert option("--blocks") == "128"
    assert option("--cache") == "128"
    assert option("--exact-interval") == "50"

    evaluation = next(
        row["command"] for row in payload["jobs"] if row["id"] == "E0"
    )
    assert evaluation[evaluation.index("--reduced-stream") + 1] == str(
        tmp_path / "run" / "reduced_stream"
    )


def test_schedule_rejects_explicit_nonproduction_estimator(tmp_path):
    schedule_path = tmp_path / "jobs.json"
    schedule_path.write_text(json.dumps({
        "sequence_estimator": "kt",
        "maximum_workers": 1,
        "maximum_private_memory_bytes": 0,
        "jobs": [],
        "waves": [],
    }))
    with pytest.raises(RuntimeError, match="production requires"):
        load_fixed_schedule(schedule_path)


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


def test_dependency_executor_crosses_fixed_wave_boundaries(tmp_path):
    events = []

    def job(job_id, dependencies=(), delay=0.05):
        output = tmp_path / f"{job_id}.txt"
        return CheckpointJob(
            job_id, "E", 0,
            (
                sys.executable, "-c",
                "import time; from pathlib import Path; "
                f"time.sleep({delay!r}); Path({str(output)!r}).write_text('ok')",
            ),
            dependencies, (str(output),),
            str(tmp_path / "manifests" / f"{job_id}.json"), 1,
        )

    jobs = (
        job("A0", delay=0.01), job("A1", delay=0.15),
        job("B0", ("A0",), delay=0.01), job("B1", ("A1",), delay=0.01),
    )
    schedule = type("Schedule", (), {
        "jobs": jobs, "waves": (("A0", "A1"), ("B0", "B1")),
        "maximum_workers": 2,
    })()
    run_dependency_schedule(
        schedule, working_directory=tmp_path, poll_seconds=0.005,
        event_callback=events.append,
    )
    launches = [row for row in events if row["event"] == "launched"]
    assert {row["job_id"] for row in launches} == {"A0", "A1", "B0", "B1"}
    b0 = next(row for row in launches if row["job_id"] == "B0")
    assert "A1" in b0["running"]


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


def test_dependency_executor_finishes_ready_anchor_tail_before_more_construction(tmp_path):
    events = []

    def job(job_id, kind):
        output = tmp_path / f"{job_id}.txt"
        worker_option = (
            ("--jobs", "1") if kind == "M" else
            (("--workers", "1") if kind == "F" else ())
        )
        return CheckpointJob(
            job_id, kind, 0,
            (sys.executable, "-c",
             f"from pathlib import Path; Path({str(output)!r}).write_text('ok')",
             *worker_option),
            (), (str(output),),
            str(tmp_path / "manifests" / f"{job_id}.json"), 1,
        )

    # All jobs are ready.  S/F/T represent an existing anchor's tail, while
    # M is later construction with an earlier legacy wave priority.
    jobs = (job("M9", "M"), job("T1", "T"), job("F1", "F"), job("S1", "S"))
    run_planned_schedule(
        jobs, {job.job_id: 1 for job in jobs},
        {"M9": 0.0, "T1": 10.0, "F1": 11.0, "S1": 12.0},
        maximum_workers=1, working_directory=tmp_path, poll_seconds=0.005,
        event_callback=events.append,
    )
    launches = [row["job_id"] for row in events if row["event"] == "launched"]
    assert launches == ["S1", "F1", "T1", "M9"]


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


def test_planned_executor_honors_worker_floor_at_launch(tmp_path):
    events = []

    def job(job_id):
        output = tmp_path / f"{job_id}.txt"
        return CheckpointJob(
            job_id, "F", 0,
            (
                sys.executable, "-c",
                f"from pathlib import Path; Path({str(output)!r}).write_text('ok')",
                "--workers", "4",
            ),
            (), (str(output),), str(tmp_path / f"{job_id}.json"), 4,
            minimum_workers=2,
        )

    jobs = (job("F0"), job("F1"), job("F2"), job("F3"))
    run_planned_schedule(
        jobs, {row.job_id: 4 for row in jobs},
        {row.job_id: float(index) for index, row in enumerate(jobs)},
        maximum_workers=13,
        worker_caps={row.job_id: 4 for row in jobs},
        worker_floors={row.job_id: row.minimum_workers for row in jobs},
        working_directory=tmp_path, poll_seconds=0.01,
        event_callback=events.append,
    )
    launches = [row for row in events if row["event"] == "launched"]
    assert all(row["workers"] >= 2 for row in launches)
    assert max(row["assigned_workers"] for row in launches) >= 12


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
