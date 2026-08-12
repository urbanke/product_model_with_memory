import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from product_model_with_memory.checkpoint_scheduler import load_fixed_schedule
from product_model_with_memory.production_coding import PRODUCTION_SEQUENCE_ESTIMATOR


REPOSITORY = Path(__file__).resolve().parents[1]


def write_problem(path: Path, prefix: int) -> None:
    np.savez(
        path, prefix=np.asarray(prefix),
        sequence_estimator=np.asarray(PRODUCTION_SEQUENCE_ESTIMATOR),
        edge_a=np.array([0]), edge_b=np.array([0]), edge_probability=np.array([1.0]),
        target_y=np.array([1.0]), active_ya_y=np.array([0]),
        active_ya_a=np.array([0]), target_ya=np.array([1.0]),
        active_yb_y=np.array([0]), active_yb_b=np.array([0]),
        target_yb=np.array([1.0]),
        fallback_ya_left=np.ones(1), fallback_ya_right=np.ones(1),
        fallback_ya_background=np.zeros(1), fallback_ya_active_y=np.array([0]),
        fallback_ya_active_context=np.array([0]), fallback_ya_delta=np.ones(1),
        fallback_yb_left=np.ones(1), fallback_yb_right=np.ones(1),
        fallback_yb_background=np.zeros(1), fallback_yb_active_y=np.array([0]),
        fallback_yb_active_context=np.array([0]), fallback_yb_delta=np.ones(1),
    )


def test_topology_is_immutable_and_fit_schedule_is_valid(tmp_path):
    problems = tmp_path / "problems" / "states"
    problems.mkdir(parents=True)
    write_problem(problems / "checkpoint_000.npz", 10)
    write_problem(problems / "checkpoint_001.npz", 20)
    topology = tmp_path / "one_topology"
    topology_command = [
        sys.executable, str(REPOSITORY / "scripts/prepare_anchored_ya_topology.py"),
        "--problem", str(problems / "checkpoint_000.npz"),
        "--out", str(topology),
    ]
    first = subprocess.run(topology_command, cwd=REPOSITORY, check=True, capture_output=True, text=True)
    second = subprocess.run(topology_command, cwd=REPOSITORY, check=True, capture_output=True, text=True)
    assert json.loads(first.stdout)["reused"] is False
    assert json.loads(second.stdout)["reused"] is True

    schedule_path = tmp_path / "schedule.json"
    subprocess.run([
        sys.executable, str(REPOSITORY / "scripts/make_anchored_fit_schedule.py"),
        "--problems", str(tmp_path / "problems"), "--root", str(tmp_path / "run"),
        "--maximum-workers", "8", "--fitting-workers", "4",
        "--out", str(schedule_path),
    ], cwd=REPOSITORY, check=True, capture_output=True, text=True)
    schedule = load_fixed_schedule(schedule_path)
    assert len(schedule.jobs) == 4
    payload = json.loads(schedule_path.read_text())
    fit = next(job for job in payload["jobs"] if job["id"] == "F0")
    assert fit["dependencies"] == ["T0"]
    assert "--topology" in fit["command"]
    assert payload["waves"][-1] == ["F0", "F1"]
