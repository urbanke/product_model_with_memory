import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from product_model_with_memory.checkpoint_scheduler import (
    command_with_workers,
    load_fixed_schedule,
)
from product_model_with_memory.production_coding import PRODUCTION_SEQUENCE_ESTIMATOR


REPOSITORY = Path(__file__).resolve().parent.parent


def test_unequal_schedule_preserves_anchor_causal_resource_policy(tmp_path):
    reduced = tmp_path / "reduced"
    reduced.mkdir()
    stream = np.arange(20_000, dtype=np.uint16) % 16
    np.save(reduced / "stream.npy", stream)
    (reduced / "manifest.json").write_text(json.dumps({"vocabulary_size": 16}))
    plan = tmp_path / "anchors.json"
    subprocess.run([
        sys.executable, str(REPOSITORY / "scripts/plan_anchored_potential_anchors.py"),
        "--stream", str(reduced / "stream.npy"), "--minimum-prefix", "100",
        "--windows", "16,64", "--early-strata", "2", "--late-strata", "2",
        "--samples-per-stratum", "2", "--seed", "9", "--out", str(plan),
    ], cwd=REPOSITORY, check=True, capture_output=True, text=True)
    output, root = tmp_path / "schedule.json", tmp_path / "run"
    subprocess.run([
        sys.executable, str(REPOSITORY / "scripts/make_unequal_anchored_schedule.py"),
        "--plan", str(plan), "--root", str(root), "--m1", "16", "--m2", "8",
        "--maximum-workers", "13", "--pair-workers", "4", "--out", str(output),
    ], cwd=REPOSITORY, check=True, capture_output=True, text=True)
    schedule = load_fixed_schedule(output)
    payload = json.loads(output.read_text())
    anchors = sorted(json.loads(plan.read_text())["anchors"], key=lambda row: row["prefix"])
    assert len(schedule.jobs) == 1 + 12 * len(anchors) + 1
    assert payload["sequence_estimator"] == PRODUCTION_SEQUENCE_ESTIMATOR
    assert payload["resource_profile"] == "laptop"
    assert (payload["emission_vocabulary_size"], payload["first_lag_parameter"],
            payload["second_lag_parameter"]) == (16, 16, 8)
    priority = {job: wave for wave, jobs in enumerate(payload["waves"]) for job in jobs}
    first, second = anchors[0]["anchor_id"], anchors[1]["anchor_id"]
    for pair in ("PYA", "PYB", "PAB"):
        assert priority[f"{pair}{first}"] < priority[f"MY{second}"]
    pair_jobs = [job for job in payload["jobs"] if job["id"].startswith("P")]
    assert all(job["workers"] == job["minimum_workers"] == 4 for job in pair_jobs)
    assert all(job["private_memory_bytes"] == int(3.5 * 1024**3) for job in pair_jobs)
    scoring_jobs = [job for job in payload["jobs"] if job["id"].startswith("S")]
    fitting_jobs = [job for job in payload["jobs"] if job["id"].startswith("F")]
    assert all(job["private_memory_bytes"] == int(0.75 * 1024**3)
               for job in scoring_jobs)
    assert all(job["private_memory_bytes"] == int(1.0 * 1024**3)
               for job in fitting_jobs)
    loaded_pairs = [job for job in schedule.jobs if job.job_id.startswith("P")]
    assert all("--jobs" in command_with_workers(job, 4) for job in loaded_pairs)


def test_unequal_schedule_cpu64_profile_exposes_anchor_breadth(tmp_path):
    reduced = tmp_path / "reduced"
    reduced.mkdir()
    np.save(reduced / "stream.npy", np.arange(20_000, dtype=np.uint16) % 16)
    (reduced / "manifest.json").write_text(json.dumps({"vocabulary_size": 16}))
    plan = tmp_path / "anchors.json"
    subprocess.run([
        sys.executable, str(REPOSITORY / "scripts/plan_anchored_potential_anchors.py"),
        "--stream", str(reduced / "stream.npy"), "--minimum-prefix", "100",
        "--windows", "16,64", "--early-strata", "2", "--late-strata", "2",
        "--samples-per-stratum", "2", "--seed", "9", "--out", str(plan),
    ], cwd=REPOSITORY, check=True, capture_output=True, text=True)
    output = tmp_path / "schedule.json"
    subprocess.run([
        sys.executable, str(REPOSITORY / "scripts/make_unequal_anchored_schedule.py"),
        "--plan", str(plan), "--root", str(tmp_path / "run"),
        "--m1", "16", "--m2", "8", "--resource-profile", "cpu64",
        "--out", str(output),
    ], cwd=REPOSITORY, check=True, capture_output=True, text=True)
    payload = json.loads(output.read_text())
    assert payload["resource_profile"] == "cpu64"
    assert payload["maximum_workers"] == 64
    assert payload["maximum_private_memory_bytes"] == int(192 * 1024**3)
    scores = [job for job in payload["jobs"] if job["id"].startswith("S")]
    assert sum(job["private_memory_bytes"] for job in scores) <= int(192 * 1024**3)


def test_declared_machine_profiles_are_distinct():
    import importlib.util
    source = REPOSITORY / "scripts/make_unequal_anchored_schedule.py"
    spec = importlib.util.spec_from_file_location("unequal_schedule_profiles", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profiles = module.RESOURCE_PROFILES
    assert profiles["laptop"]["maximum_workers"] == 13
    assert profiles["m4pro"]["maximum_workers"] == 14
    assert profiles["m4pro"]["maximum_private_memory_gib"] == 24.0
    assert profiles["cpu64"]["maximum_workers"] == 64
    assert profiles["scitas"]["maximum_workers"] == 72
    assert profiles["scitas"]["marginal_max_memory_gib"] == 16.0
    assert "slurm72" not in profiles
