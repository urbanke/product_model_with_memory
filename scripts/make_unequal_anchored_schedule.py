#!/usr/bin/env python3
"""Create the dependency-driven unequal-alphabet anchored production DAG."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from product_model_with_memory.anchored_state_maps import state_map_manifest
from product_model_with_memory.production_coding import PRODUCTION_SEQUENCE_ESTIMATOR


RESOURCE_PROFILES = {
    # Safe default for the 24-GiB laptop.  The private-memory budget leaves
    # room for the OS and shared mappings; the score contract is based on the
    # observed V=32K RSS rather than the former 2-GiB placeholder.
    "laptop": {
        "maximum_workers": 13, "maximum_private_memory_gib": 12.0,
        "marginal_min_memory_gib": 0.5, "marginal_max_memory_gib": 3.0,
        "pair_memory_gib": 3.5, "assembly_memory_gib": 3.0,
        "topology_memory_gib": 6.0, "fitting_memory_gib": 1.0,
        "scoring_memory_gib": 0.75, "pair_workers": 4,
        "fitting_workers": 1,
    },
    # M4 Pro Mac: 14 CPU cores (10 performance + 4 efficiency), 36 GiB RAM.
    # Start with every core available and 12 GiB left outside the scheduler
    # for macOS, shared mappings, and page cache; retune only from replay RSS.
    "m4pro": {
        "maximum_workers": 14, "maximum_private_memory_gib": 24.0,
        "marginal_min_memory_gib": 0.5, "marginal_max_memory_gib": 3.0,
        "pair_memory_gib": 3.5, "assembly_memory_gib": 3.0,
        "topology_memory_gib": 6.0, "fitting_memory_gib": 1.0,
        "scoring_memory_gib": 0.75, "pair_workers": 4,
        "fitting_workers": 1,
    },
    # Initial node-14 contract: use at most 192 of 256 GiB so the OS, shared
    # tables, mmap page cache, and unmodelled peaks retain substantial headroom.
    # This profile is independent of V; only explicitly measured job scale may
    # change an individual contract.
    "cpu64": {
        "maximum_workers": 64, "maximum_private_memory_gib": 192.0,
        "marginal_min_memory_gib": 0.5, "marginal_max_memory_gib": 3.0,
        "pair_memory_gib": 3.5, "assembly_memory_gib": 3.0,
        "topology_memory_gib": 6.0, "fitting_memory_gib": 1.0,
        "scoring_memory_gib": 0.75, "pair_workers": 4,
        "fitting_workers": 1,
    },
    # Conservative portable Slurm starting point.  Slurm wrappers should pass
    # the allocation's CPUs and usable memory explicitly after the first replay.
    "scitas": {
        "maximum_workers": 72, "maximum_private_memory_gib": 384.0,
        # Empirical private-RSS contract from the enwik9 V=65536 OOM gate:
        # the former 3-GiB ceiling admitted 40+ late MY jobs and exhausted a
        # 440-GiB allocation.  A full-V late marginal is therefore reserved
        # at 16 GiB.  Smaller natural alphabets scale proportionally, while
        # the 1-GiB floor covers process/runtime overhead at early prefixes.
        "marginal_min_memory_gib": 1.0, "marginal_max_memory_gib": 16.0,
        "pair_memory_gib": 3.5, "assembly_memory_gib": 3.0,
        "topology_memory_gib": 6.0, "fitting_memory_gib": 1.0,
        "scoring_memory_gib": 0.75, "pair_workers": 4,
        "fitting_workers": 1,
    },
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True); p.add_argument("--root", required=True)
    p.add_argument("--m1", type=int, required=True); p.add_argument("--m2", type=int, required=True)
    p.add_argument("--python", default=".venv/bin/python3"); p.add_argument("--out", required=True)
    p.add_argument("--resource-profile", choices=tuple(RESOURCE_PROFILES),
                   default="laptop")
    for option, kind in (
        ("maximum-workers", int), ("maximum-private-memory-gib", float),
        ("marginal-min-memory-gib", float),
        ("marginal-max-memory-gib", float), ("pair-memory-gib", float),
        ("assembly-memory-gib", float), ("topology-memory-gib", float),
        ("fitting-memory-gib", float), ("scoring-memory-gib", float),
        ("pair-workers", int), ("fitting-workers", int),
    ):
        p.add_argument(f"--{option}", type=kind, default=None)
    p.add_argument("--fitting-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--fit-seed", type=int, default=31081)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--exact-interval", type=int, default=25)
    p.add_argument("--slack-precision", type=float, default=1.0)
    p.add_argument("--bootstrap-seed", type=int, default=9173)
    a = p.parse_args()
    profile = RESOURCE_PROFILES[a.resource_profile]
    for name, value in profile.items():
        if getattr(a, name) is None:
            setattr(a, name, value)
    plan = json.loads(Path(a.plan).read_text())
    stream_manifest = json.loads(
        (Path(plan["stream"]).parent / "manifest.json").read_text()
    )
    v = int(stream_manifest["vocabulary_size"])
    mapping = state_map_manifest(v, a.m1, a.m2)
    if a.m1 == v and a.m2 == v:
        p.error("equal endpoint belongs to the established equal-alphabet schedule")
    memories = (a.maximum_private_memory_gib, a.marginal_min_memory_gib,
                a.marginal_max_memory_gib, a.pair_memory_gib,
                a.assembly_memory_gib, a.topology_memory_gib,
                a.fitting_memory_gib, a.scoring_memory_gib)
    if min(memories) <= 0 or a.marginal_min_memory_gib > a.marginal_max_memory_gib:
        p.error("invalid memory contracts")
    if min(a.maximum_workers, a.pair_workers, a.fitting_workers) < 1:
        p.error("workers must be positive")
    if max(a.pair_workers, a.fitting_workers) > a.maximum_workers:
        p.error("per-job workers exceed capacity")
    anchors = sorted(plan["anchors"], key=lambda row: int(row["prefix"]))
    root, causal = Path(a.root), Path(a.root) / "construction_stream"
    manifests = root / "job_manifests"
    jobs, waves, scores = [], [], []

    def add(jid, kind, checkpoint, command, deps, outputs, workers=1, memory=0.0):
        jobs.append({"id": jid, "type": kind, "checkpoint": checkpoint,
                     "command": command, "dependencies": deps,
                     "outputs": [str(x) for x in outputs],
                     "completion_manifest": str(manifests / f"{jid}.json"),
                     "workers": workers, "minimum_workers": workers,
                     "private_memory_bytes": int(memory * 1024**3)})

    add("R", "R", -1, [a.python, "-u", "scripts/prepare_anchor_construction_stream.py",
        "--plan", a.plan, "--out", str(causal)], [], [causal / "manifest.json"])
    waves.append(["R"]); previous = None
    windows = ",".join(map(str, plan["windows"]))
    for build, anchor in enumerate(anchors):
        anchor_id, prefix = int(anchor["anchor_id"]), int(anchor["prefix"])
        label = f"anchor_{anchor_id:03d}"
        delta, counts = root / "count_deltas" / f"build_{build:03d}", root / "counts" / f"build_{build:03d}"
        marg = {s: root / "marginals" / label / s for s in "yab"}
        pairs = {q: root / "pairs" / label / q for q in ("ya", "yb", "ab")}
        problem, topology = root / "problems" / f"{label}.npz", root / "topology_v4" / label
        fit, score = root / "fits_v4" / label, root / "scores_v4" / f"{label}.json"
        did, uid = f"D{build}", f"U{build}"
        mids = {s: f"M{s.upper()}{anchor_id}" for s in "yab"}
        pids = {q: f"P{q.upper()}{anchor_id}" for q in ("ya", "yb", "ab")}
        cid, tid, fid, sid = (f"C{anchor_id}", f"T{anchor_id}", f"F{anchor_id}", f"S{anchor_id}")
        add(did, "D", build, [a.python, "-u", "scripts/prepare_unequal_anchor_count_delta.py",
            "--stream", str(causal), "--checkpoint", str(build), "--m1", str(a.m1),
            "--m2", str(a.m2), "--out", str(delta)], ["R"], [delta / "manifest.json"])
        merge = [a.python, "-u", "scripts/merge_unequal_anchor_counts.py", "--delta", str(delta), "--out", str(counts)]
        if previous is not None: merge += ["--previous", str(previous)]
        add(uid, "U", build, merge, [did] + ([] if build == 0 else [f"U{build-1}"]), [counts / "manifest.json"])
        fraction = min(1.0, prefix / int(plan["n"]))
        for symbol in "yab":
            symbol_alphabet = {"y": v, "a": a.m1 + 1, "b": a.m2 + 1}[symbol]
            alphabet_fraction = symbol_alphabet / v
            marginal_memory = (
                a.marginal_min_memory_gib
                + (a.marginal_max_memory_gib-a.marginal_min_memory_gib)
                * alphabet_fraction * math.sqrt(fraction)
            )
            add(mids[symbol], "M", anchor_id, [a.python, "-u", "scripts/estimate_unequal_checkpoint_unigram.py",
                "--counts", str(counts), "--symbol", symbol, "--jobs", "1", "--out", str(marg[symbol])],
                [uid], [marg[symbol] / "manifest.json"], memory=marginal_memory)
        specs = {"ya": ("y", "a"), "yb": ("y", "b"), "ab": ("a", "b")}
        for pair, (target, context) in specs.items():
            # Reuse the scheduler's established multi-core estimator resource
            # classes so command_with_workers updates --jobs.  AB has the
            # same execution contract as YA and therefore uses class A.
            pair_kind = "B" if pair == "yb" else "A"
            add(pids[pair], pair_kind, anchor_id, [a.python, "-u", "scripts/estimate_unequal_checkpoint_pair.py",
                "--counts", str(counts), "--target-marginal", str(marg[target]),
                "--context-marginal", str(marg[context]), "--pair", pair,
                "--jobs", str(a.pair_workers), "--out", str(pairs[pair])],
                [mids[target], mids[context]], [pairs[pair] / "manifest.json"],
                a.pair_workers, a.pair_memory_gib)
        add(cid, "C", anchor_id, [a.python, "-u", "scripts/assemble_unequal_checkpoint_problem.py",
            "--counts", str(counts), "--ya", str(pairs["ya"]), "--yb", str(pairs["yb"]),
            "--ab", str(pairs["ab"]), "--out", str(problem)], list(pids.values()), [problem], memory=a.assembly_memory_gib)
        add(tid, "T", anchor_id, [a.python, "-u", "scripts/prepare_unequal_anchored_topology.py",
            "--problem", str(problem), "--out", str(topology)], [cid], [topology / "manifest.json"], memory=a.topology_memory_gib)
        add(fid, "F", anchor_id, [a.python, "-u", "scripts/fit_anchored_ya_checkpoint.py",
            "--problem", str(problem), "--topology", str(topology), "--out", str(fit),
            "--steps", str(a.fitting_steps), "--batch-size", str(a.batch_size),
            "--seed", str(a.fit_seed+anchor_id), "--workers", str(a.fitting_workers),
            "--learning-rate", str(a.learning_rate), "--exact-interval", str(a.exact_interval),
            "--slack-precision", str(a.slack_precision)], [tid], [fit / "manifest.json"], a.fitting_workers, a.fitting_memory_gib)
        add(sid, "S", anchor_id, [a.python, "-u", "scripts/score_unequal_anchored_windows.py",
            "--problem", str(problem), "--topology", str(topology), "--fit", str(fit),
            "--stream", plan["stream"], "--windows", windows, "--out", str(score)], [fid], [score], memory=a.scoring_memory_gib)
        # PERFORMANCE INVARIANT: anchor-causal priority prevents serial
        # marginals from starving ready four-core pair work. These are hints,
        # never barriers, under the dependency-driven scheduler.
        waves.extend([[did], [uid]] + [[mids[s]] for s in "yab"]
                     + [[pids[q]] for q in ("ya", "yb", "ab")]
                     + [[cid], [tid], [fid], [sid]])
        scores.append(sid); previous = counts
    aggregate = root / "potential.json"
    add("Z", "Z", -1, [a.python, "-u", "scripts/aggregate_anchored_potential.py",
        "--plan", a.plan, "--scores", str(root / "scores_v4"),
        "--bootstrap-seed", str(a.bootstrap_seed), "--out", str(aggregate)],
        scores, [aggregate]); waves.append(["Z"])
    payload = {"version": 2, "model": "anchored_ya_relaxed_pair_graph_v1",
               "sequence_estimator": PRODUCTION_SEQUENCE_ESTIMATOR,
               "resource_profile": a.resource_profile,
               "plan": str(Path(a.plan).resolve()), **mapping,
               "maximum_workers": a.maximum_workers,
               "maximum_private_memory_bytes": int(a.maximum_private_memory_gib*1024**3),
               "jobs": jobs, "waves": waves}
    destination = Path(a.out); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2)); print(destination)


if __name__ == "__main__": main()
