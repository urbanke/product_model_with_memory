"""Tests for the deterministic coarse checkpoint scheduler."""

import json
import sys

import pytest

from product_model_with_memory.checkpoint_scheduler import (
    load_fixed_schedule,
    run_fixed_schedule,
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
