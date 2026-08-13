import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from product_model_with_memory.checkpoint_scheduler import load_fixed_schedule
from product_model_with_memory.production_coding import (
    PRODUCTION_SEQUENCE_ESTIMATOR, sha256_file,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def mark_production_reduced(stream: Path, vocabulary_size: int) -> None:
    (stream.parent / "manifest.json").write_text(json.dumps({
        "version": 2, "kind": "production_reduced_stream",
        "n": len(np.load(stream)), "vocabulary_size": vocabulary_size,
        "representation": "bpe", "encoding": "cl100k_base",
        "complete_source": True, "source_n_tokens": len(np.load(stream)),
        "source_n_bytes": 1, "source_manifest_sha256": "test",
        "source_ids_sha256": "test",
        "sequence_estimator": PRODUCTION_SEQUENCE_ESTIMATOR,
        "stream_sha256": sha256_file(stream),
    }))


def test_random_anchor_ids_are_constructed_in_sorted_prefix_order(tmp_path):
    stream = tmp_path / "stream.npy"
    np.save(stream, np.arange(20_000, dtype=np.uint16) % 16)
    mark_production_reduced(stream, 16)
    plan = tmp_path / "anchors.json"
    subprocess.run([
        sys.executable, str(REPOSITORY / "scripts/plan_anchored_potential_anchors.py"),
        "--stream", str(stream), "--minimum-prefix", "100",
        "--windows", "16,64", "--early-strata", "2", "--late-strata", "2",
        "--samples-per-stratum", "2", "--seed", "9", "--out", str(plan),
    ], cwd=REPOSITORY, check=True, capture_output=True, text=True)
    schedule_path = tmp_path / "schedule.json"
    root = tmp_path / "run"
    subprocess.run([
        sys.executable, str(REPOSITORY / "scripts/make_anchored_potential_schedule.py"),
        "--plan", str(plan), "--root", str(root), "--maximum-workers", "4",
        "--unigram-workers", "1", "--pair-workers", "4",
        "--fitting-workers", "2",
        "--out", str(schedule_path),
    ], cwd=REPOSITORY, check=True, capture_output=True, text=True)
    schedule = load_fixed_schedule(schedule_path)
    payload = json.loads(schedule_path.read_text())
    anchors = json.loads(plan.read_text())["anchors"]
    assert len(schedule.jobs) == 1 + 9 * len(anchors) + 1
    assert payload["waves"][0] == ["R"]
    assert payload["waves"][-1] == ["Z"]
    # Materialize only the causal manifest: build order is sorted, but its
    # parallel anchor IDs retain the original randomized design identities.
    subprocess.run([
        sys.executable, str(REPOSITORY / "scripts/prepare_anchor_construction_stream.py"),
        "--plan", str(plan), "--out", str(root / "construction_stream"),
    ], cwd=REPOSITORY, check=True, capture_output=True, text=True)
    causal = json.loads((root / "construction_stream" / "manifest.json").read_text())
    expected = sorted(anchors, key=lambda row: row["prefix"])
    assert causal["edges"] == [row["prefix"] for row in expected]
    assert causal["anchor_ids"] == [row["anchor_id"] for row in expected]
    priority = {
        job_id: wave_number
        for wave_number, wave in enumerate(payload["waves"])
        for job_id in wave
    }
    first_id = expected[0]["anchor_id"]
    second_id = expected[1]["anchor_id"]
    # Protect the measured utilization policy: ready pair work from the first
    # anchor must outrank the next serial unigram job.
    assert priority[f"A{first_id}"] < priority[f"M{second_id}"]
    assert priority[f"B{first_id}"] < priority[f"M{second_id}"]
    fit = next(job for job in payload["jobs"] if job["id"] == "F0")
    assert fit["command"][fit["command"].index("--seed") + 1] == "31081"
    unigrams = [job for job in payload["jobs"] if job["id"].startswith("M")]
    unigram = unigrams[0]
    pair = next(job for job in payload["jobs"] if job["id"] == "A0")
    assert unigram["workers"] == unigram["minimum_workers"] == 1
    assert pair["workers"] == pair["minimum_workers"] == 4
    unigram_memory = [job["private_memory_bytes"] for job in unigrams]
    assert unigram_memory == sorted(unigram_memory)
    assert unigram_memory[0] >= int(0.5 * 1024 ** 3)
    assert unigram_memory[-1] <= int(1.5 * 1024 ** 3)
    assert pair["private_memory_bytes"] == int(3.5 * 1024 ** 3)


def test_scitas_profile_reserves_measured_full_v_unigram_memory():
    import importlib.util
    source = REPOSITORY / "scripts/make_anchored_potential_schedule.py"
    spec = importlib.util.spec_from_file_location("anchored_profiles", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profile = module.RESOURCE_PROFILES["scitas"]
    assert profile["maximum_workers"] == 72
    assert profile["maximum_private_memory_gib"] == 384.0
    assert profile["unigram_min_private_memory_gib"] == 1.0
    assert profile["unigram_private_memory_gib"] == 16.0
