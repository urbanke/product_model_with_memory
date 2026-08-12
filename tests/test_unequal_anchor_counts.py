import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parent.parent


def run(script, *arguments):
    subprocess.run(
        [sys.executable, str(REPOSITORY / "scripts" / script), *map(str, arguments)],
        cwd=REPOSITORY, check=True, capture_output=True, text=True,
    )


def construction(tmp_path, stream, edges):
    tmp_path.mkdir(parents=True, exist_ok=True)
    np.save(tmp_path / "stream.npy", np.asarray(stream, dtype=np.int16))
    root = tmp_path / "construction"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({
        "version": 2, "kind": "anchor_construction_stream",
        "stream_path": str(tmp_path / "stream.npy"), "stream_sha256": "test",
        "n": len(stream), "vocabulary_size": int(max(stream)) + 1,
        "edges": edges, "anchor_ids": list(range(len(edges))),
        "sampling_design": {},
    }))
    return root


def test_equal_endpoint_counts_are_exactly_legacy_counts(tmp_path):
    source = construction(tmp_path, [0, 1, 2, 3, 1, 0, 2, 2, 3], [9])
    old, new = tmp_path / "old", tmp_path / "new"
    run("prepare_checkpoint_count_delta.py", "--stream", source,
        "--checkpoint", 0, "--out", old)
    run("prepare_unequal_anchor_count_delta.py", "--stream", source,
        "--checkpoint", 0, "--m1", 4, "--m2", 4, "--out", new)
    assert np.array_equal(np.load(old / "unigram.npy"),
                          np.load(new / "unigram_y.npy"))
    for label in ("ya", "yb"):
        assert np.array_equal(np.load(old / f"keys_{label}.npy"),
                              np.load(new / f"keys_{label}.npy"))
        assert np.array_equal(np.load(old / f"counts_{label}.npy"),
                              np.load(new / f"counts_{label}.npy"))
    # The unequal path stores AB in estimator-native orientation
    # (row=context B, column=target A); the legacy support-only table used
    # (A,B).  Transposition must preserve the exact count dictionary.
    old_keys = np.load(old / "keys_ab.npy")
    old_counts = np.load(old / "counts_ab.npy")
    transposed = (old_keys % 4) * 4 + old_keys // 4
    expected = dict(zip(map(int, transposed), map(int, old_counts)))
    actual = dict(zip(map(int, np.load(new / "keys_ab.npy")),
                      map(int, np.load(new / "counts_ab.npy"))))
    assert actual == expected


def test_unequal_deltas_merge_to_direct_full_prefix(tmp_path):
    stream = [0, 1, 4, 3, 2, 4, 1, 0, 3, 4, 2, 1]
    source = construction(tmp_path, stream, [6, 12])
    d0, d1 = tmp_path / "d0", tmp_path / "d1"
    c0, c1 = tmp_path / "c0", tmp_path / "c1"
    direct_source = construction(tmp_path / "direct", stream, [12])
    direct = tmp_path / "direct_delta"
    for checkpoint, out in ((0, d0), (1, d1)):
        run("prepare_unequal_anchor_count_delta.py", "--stream", source,
            "--checkpoint", checkpoint, "--m1", 4, "--m2", 2, "--out", out)
    run("merge_unequal_anchor_counts.py", "--delta", d0, "--out", c0)
    run("merge_unequal_anchor_counts.py", "--delta", d1,
        "--previous", c0, "--out", c1)
    run("prepare_unequal_anchor_count_delta.py", "--stream", direct_source,
        "--checkpoint", 0, "--m1", 4, "--m2", 2, "--out", direct)
    for name in ("unigram_y", "unigram_a", "unigram_b"):
        assert np.array_equal(np.load(c1 / f"{name}.npy"),
                              np.load(direct / f"{name}.npy"))
    for label in ("ya", "yb", "ab"):
        assert np.array_equal(np.load(c1 / f"keys_{label}.npy"),
                              np.load(direct / f"keys_{label}.npy"))
        assert np.array_equal(np.load(c1 / f"counts_{label}.npy"),
                              np.load(direct / f"counts_{label}.npy"))


def test_natural_alphabet_marginals_and_pairs_smoke(tmp_path):
    stream = ([0, 1, 4, 3, 2, 4, 1, 0, 3, 4, 2, 1] * 3)
    source = construction(tmp_path, stream, [len(stream) - 2])
    counts = tmp_path / "counts"
    run("prepare_unequal_anchor_count_delta.py", "--stream", source,
        "--checkpoint", 0, "--m1", 4, "--m2", 2, "--out", counts)
    marginals = {}
    for symbol in "yab":
        path = tmp_path / f"marginal_{symbol}"
        run("estimate_unequal_checkpoint_unigram.py", "--counts", counts,
            "--symbol", symbol, "--jobs", 1, "--out", path)
        marginals[symbol] = path
    assert len(np.load(marginals["y"] / "marginal.npy")) == 5
    assert len(np.load(marginals["a"] / "marginal.npy")) == 5
    assert len(np.load(marginals["b"] / "marginal.npy")) == 3
    pairs = {}
    for pair, target, context, target_size, context_size in (
        ("ya", "y", "a", 5, 5), ("yb", "y", "b", 5, 3),
        ("ab", "a", "b", 5, 3),
    ):
        destination = tmp_path / f"pair_{pair}"
        pairs[pair] = destination
        run("estimate_unequal_checkpoint_pair.py", "--counts", counts,
            "--target-marginal", marginals[target],
            "--context-marginal", marginals[context], "--pair", pair,
            "--jobs", 1, "--out", destination)
        manifest = json.loads((destination / "manifest.json").read_text())
        assert manifest["target_alphabet_size"] == target_size
        assert manifest["context_alphabet_size"] == context_size
        assert len(np.load(destination / "right.npy")) == target_size
        assert len(np.load(destination / "context_marginal.npy")) == context_size
    problem, topology, fit = tmp_path / "problem.npz", tmp_path / "topology", tmp_path / "fit"
    run("assemble_unequal_checkpoint_problem.py", "--counts", counts,
        "--ya", pairs["ya"], "--yb", pairs["yb"], "--ab", pairs["ab"],
        "--out", problem)
    run("prepare_unequal_anchored_topology.py", "--problem", problem,
        "--out", topology)
    topology_manifest = json.loads((topology / "manifest.json").read_text())
    assert topology_manifest["version"] == 4
    assert topology_manifest["first_lag_alphabet_size"] == 5
    assert topology_manifest["second_lag_alphabet_size"] == 3
    run("fit_anchored_ya_checkpoint.py", "--problem", problem,
        "--topology", topology, "--steps", 4, "--batch-size", 4,
        "--exact-interval", 2, "--workers", 1, "--out", fit)
    score = tmp_path / "score.json"
    run("score_unequal_anchored_windows.py", "--problem", problem,
        "--topology", topology, "--fit", fit,
        "--stream", tmp_path / "stream.npy", "--windows", 1,
        "--out", score)
    payload = json.loads(score.read_text())
    assert (payload["V"], payload["M1"], payload["M2"]) == (5, 4, 2)
    assert payload["cumulative"][0]["tokens"] == 1
