import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from product_model_with_memory.anchored_pair_graph import (
    ANCHORED_PAIR_GRAPH_INITIALIZER,
    ANCHORED_PAIR_GRAPH_MODEL,
)
from product_model_with_memory.production_coding import PRODUCTION_SEQUENCE_ESTIMATOR


REPOSITORY = Path(__file__).resolve().parents[1]


def test_fit_artifact_is_honest_atomic_and_restartable(tmp_path):
    problem = tmp_path / "problem.npz"
    p_ya = np.array([[0.3, 0.1], [0.2, 0.4]])
    p_yb = np.array([[0.25, 0.15], [0.25, 0.35]])
    p_ab = np.array([[0.15, 0.25], [0.3, 0.3]])
    ya_y, ya_a = np.nonzero(p_ya)
    yb_y, yb_b = np.nonzero(p_yb)
    ab_a, ab_b = np.nonzero(p_ab)
    fallback = {}
    for label, pair, yy, cc in (
        ("ya", p_ya, ya_y, ya_a), ("yb", p_yb, yb_y, yb_b),
    ):
        fallback.update({
            f"fallback_{label}_left": np.ones(2),
            f"fallback_{label}_right": np.ones(2),
            f"fallback_{label}_background": np.zeros(2),
            f"fallback_{label}_active_y": yy,
            f"fallback_{label}_active_context": cc,
            f"fallback_{label}_delta": pair[yy, cc],
        })
    np.savez(
        problem, prefix=np.asarray(1000),
        sequence_estimator=np.asarray(PRODUCTION_SEQUENCE_ESTIMATOR),
        edge_a=ab_a, edge_b=ab_b, edge_probability=p_ab[ab_a, ab_b],
        target_y=p_ya.sum(axis=1),
        active_ya_y=ya_y, active_ya_a=ya_a, target_ya=p_ya[ya_y, ya_a],
        active_yb_y=yb_y, active_yb_b=yb_b, target_yb=p_yb[yb_y, yb_b],
        **fallback,
    )
    out = tmp_path / "fit"
    command = [
        sys.executable, str(REPOSITORY / "scripts/fit_anchored_ya_checkpoint.py"),
        "--problem", str(problem), "--out", str(out), "--steps", "40",
        "--batch-size", "64", "--seed", "7", "--exact-interval", "5",
        "--slack-precision", "3",
    ]
    first = subprocess.run(command, cwd=REPOSITORY, check=True, capture_output=True, text=True)
    assert json.loads(first.stdout)["reused"] is False
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["model"] == ANCHORED_PAIR_GRAPH_MODEL
    assert manifest["initialization"] == ANCHORED_PAIR_GRAPH_INITIALIZER
    assert manifest["warm_start"] is False
    assert manifest["sequence_estimator"] == PRODUCTION_SEQUENCE_ESTIMATOR
    assert manifest["best_validation_objective"] <= manifest["initial_validation_objective"]
    with np.load(out / "state.npz", allow_pickle=False) as state:
        assert str(state["model"]) == ANCHORED_PAIR_GRAPH_MODEL
        assert str(state["sequence_estimator"]) == PRODUCTION_SEQUENCE_ESTIMATOR
    topology = tmp_path / "topology"
    subprocess.run([
        sys.executable, str(REPOSITORY / "scripts/prepare_anchored_ya_topology.py"),
        "--problem", str(problem), "--out", str(topology),
    ], cwd=REPOSITORY, check=True, capture_output=True, text=True)
    stream = tmp_path / "stream.npy"
    np.save(stream, np.arange(1100, dtype=np.uint16) % 2)
    score = tmp_path / "score.json"
    subprocess.run([
        sys.executable, str(REPOSITORY / "scripts/score_anchored_ya_windows.py"),
        "--problem", str(problem), "--topology", str(topology),
        "--fit", str(out), "--stream", str(stream),
        "--windows", "16,64", "--out", str(score),
    ], cwd=REPOSITORY, check=True, capture_output=True, text=True)
    scored = json.loads(score.read_text())
    assert scored["fallback"] == "none_full_layered_ab_v1"
    assert [row["tokens"] for row in scored["rings"]] == [16, 48]
    assert np.isclose(
        sum(row["delta_bits"] for row in scored["rings"]),
        scored["cumulative"][-1]["delta_bits"],
    )
    second = subprocess.run(command, cwd=REPOSITORY, check=True, capture_output=True, text=True)
    assert json.loads(second.stdout)["reused"] is True
    # Worker allocation is execution provenance, not a scientific change.
    different_workers = command + ["--workers", "4"]
    third = subprocess.run(
        different_workers, cwd=REPOSITORY, check=True,
        capture_output=True, text=True,
    )
    assert json.loads(third.stdout)["reused"] is True


def test_fit_artifact_rejects_nonproduction_sequence_estimator(tmp_path):
    problem = tmp_path / "wrong.npz"
    np.savez(
        problem, prefix=np.asarray(10), sequence_estimator=np.asarray("KT"),
        edge_a=np.array([0]), edge_b=np.array([0]),
        edge_probability=np.array([1.0]), target_y=np.array([1.0]),
        active_ya_y=np.array([0]), active_ya_a=np.array([0]),
        target_ya=np.array([1.0]), active_yb_y=np.array([0]),
        active_yb_b=np.array([0]), target_yb=np.array([1.0]),
    )
    command = [
        sys.executable, str(REPOSITORY / "scripts/fit_anchored_ya_checkpoint.py"),
        "--problem", str(problem), "--out", str(tmp_path / "fit"),
        "--steps", "0",
    ]
    failed = subprocess.run(command, cwd=REPOSITORY, capture_output=True, text=True)
    assert failed.returncode != 0
    assert "KT" in failed.stderr
