#!/usr/bin/env python3
"""Create the complete construction/topology/cold-fit/score potential schedule."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from product_model_with_memory.production_coding import (
    PRODUCTION_SEQUENCE_ESTIMATOR, require_production_anchor_plan,
)


RESOURCE_PROFILES = {
    "laptop": {},
    "m4pro": {
        "maximum_workers": 14, "maximum_private_memory_gib": 24.0,
        "fitting_private_memory_gib": 1.0,
        "scoring_private_memory_gib": 0.75,
    },
    "cpu64": {
        "maximum_workers": 64, "maximum_private_memory_gib": 192.0,
        "fitting_private_memory_gib": 1.0,
        "scoring_private_memory_gib": 0.75,
    },
    "scitas": {
        "maximum_workers": 72, "maximum_private_memory_gib": 384.0,
        "fitting_private_memory_gib": 1.0,
        "scoring_private_memory_gib": 0.75,
    },
}


def _explicit_option(name: str) -> bool:
    option = "--" + name.replace("_", "-")
    return option in sys.argv or any(value.startswith(option + "=") for value in sys.argv)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--python", default=".venv/bin/python3")
    p.add_argument("--resource-profile", choices=tuple(RESOURCE_PROFILES),
                   default="laptop")
    p.add_argument("--maximum-workers", type=int, default=13)
    p.add_argument(
        "--maximum-private-memory-gib", type=float, default=12.0,
        help="private-memory budget used by the live dependency scheduler",
    )
    p.add_argument(
        "--construction-workers", type=int,
        help="legacy override setting both unigram and pair workers",
    )
    p.add_argument("--unigram-workers", type=int, default=1)
    p.add_argument("--pair-workers", type=int, default=4)
    p.add_argument(
        "--construction-private-memory-gib", type=float, default=1.5,
        help="legacy reservation override for both M and A/B jobs",
    )
    p.add_argument("--unigram-private-memory-gib", type=float, default=1.5)
    p.add_argument(
        "--unigram-min-private-memory-gib", type=float, default=0.5,
        help="small-prefix M-job reservation; interpolated to the full-prefix ceiling",
    )
    p.add_argument("--pair-private-memory-gib", type=float, default=3.5)
    p.add_argument("--assembly-private-memory-gib", type=float, default=1.0)
    p.add_argument("--topology-private-memory-gib", type=float, default=2.0)
    p.add_argument("--fitting-private-memory-gib", type=float, default=2.0)
    p.add_argument("--scoring-private-memory-gib", type=float, default=1.0)
    p.add_argument("--fitting-workers", type=int, default=4)
    p.add_argument("--fitting-steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--fit-seed", type=int, default=31081)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--exact-interval", type=int, default=50)
    p.add_argument("--slack-precision", type=float, default=1.0)
    p.add_argument("--bootstrap-seed", type=int, default=9173)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    for name, value in RESOURCE_PROFILES[a.resource_profile].items():
        if not _explicit_option(name):
            setattr(a, name, value)
    if a.construction_workers is not None:
        a.unigram_workers = a.construction_workers
        a.pair_workers = a.construction_workers
        a.unigram_private_memory_gib = a.construction_private_memory_gib
        a.pair_private_memory_gib = a.construction_private_memory_gib
    if min(
        a.maximum_workers, a.unigram_workers, a.pair_workers,
        a.fitting_workers,
    ) < 1:
        p.error("workers must be positive")
    if min(
        a.maximum_private_memory_gib, a.construction_private_memory_gib,
        a.unigram_private_memory_gib, a.unigram_min_private_memory_gib,
        a.pair_private_memory_gib,
        a.assembly_private_memory_gib, a.topology_private_memory_gib,
        a.fitting_private_memory_gib, a.scoring_private_memory_gib,
    ) <= 0:
        p.error("memory reservations must be positive")
    if a.unigram_min_private_memory_gib > a.unigram_private_memory_gib:
        p.error("--unigram-min-private-memory-gib cannot exceed --unigram-private-memory-gib")
    if max(
        a.unigram_workers, a.pair_workers, a.fitting_workers
    ) > a.maximum_workers:
        p.error("per-job workers exceed schedule capacity")
    plan = json.loads(Path(a.plan).read_text())
    require_production_anchor_plan(plan, source=a.plan)
    anchors = sorted(plan["anchors"], key=lambda row: int(row["prefix"]))
    root = Path(a.root)
    causal = root / "construction_stream"
    jobs, waves = [], []
    manifests = root / "job_manifests"

    def add(
        job_id, kind, checkpoint, command, dependencies, outputs, workers=1,
        minimum_workers=None, private_memory_bytes=0,
    ):
        if minimum_workers is None:
            minimum_workers = workers
        jobs.append({
            "id": job_id, "type": kind, "checkpoint": checkpoint,
            "command": command, "dependencies": dependencies,
            "outputs": [str(value) for value in outputs],
            "completion_manifest": str(manifests / f"{job_id}.json"),
            "workers": workers, "minimum_workers": minimum_workers,
            "private_memory_bytes": private_memory_bytes,
        })

    add("R", "R", -1, [
        a.python, "-u", "scripts/prepare_anchor_construction_stream.py",
        "--plan", a.plan, "--out", str(causal),
    ], [], [causal / "manifest.json"])
    waves.append(["R"])
    families = {name: [] for name in ("D", "U", "M", "P", "C", "T", "F", "S")}
    previous_counts = None
    windows = ",".join(str(value) for value in plan["windows"])
    for build_index, anchor in enumerate(anchors):
        anchor_id = int(anchor["anchor_id"])
        # A full-prefix reservation here previously limited the scheduler to
        # four serial M jobs on a 12 GiB budget.  Actual M memory grows with
        # the decoded prefix, so interpolate conservatively between the
        # measured small-prefix floor and full-prefix ceiling.  Do not replace
        # this with one vocabulary-dependent constant: that silently destroys
        # CPU utilization even though the scheduler itself remains dynamic.
        prefix_fraction = min(1.0, int(anchor["prefix"]) / int(plan["n"]))
        unigram_memory_gib = (
            a.unigram_min_private_memory_gib
            + (a.unigram_private_memory_gib
               - a.unigram_min_private_memory_gib)
            * math.sqrt(prefix_fraction)
        )
        label = f"anchor_{anchor_id:03d}"
        delta = root / "count_deltas" / f"build_{build_index:03d}"
        counts = root / "counts" / f"build_{build_index:03d}"
        unigram = root / "pairs" / label / "unigram"
        ya, yb = root / "pairs" / label / "ya", root / "pairs" / label / "yb"
        problem = root / "problems" / f"{label}.npz"
        topology = root / "topology_v3" / label
        fit = root / "fits_v3" / label
        score = root / "scores_v3" / f"{label}.json"
        did, uid, mid = f"D{build_index}", f"U{build_index}", f"M{anchor_id}"
        aid, bid, cid = f"A{anchor_id}", f"B{anchor_id}", f"C{anchor_id}"
        tid, fid, sid = f"T{anchor_id}", f"F{anchor_id}", f"S{anchor_id}"
        add(did, "D", build_index, [a.python, "-u", "scripts/prepare_checkpoint_count_delta.py",
            "--stream", str(causal), "--checkpoint", str(build_index), "--out", str(delta)],
            ["R"], [delta / "manifest.json"])
        merge = [a.python, "-u", "scripts/merge_checkpoint_counts.py",
                 "--delta", str(delta), "--out", str(counts)]
        if previous_counts is not None:
            merge += ["--previous", str(previous_counts)]
        add(uid, "U", build_index, merge, [did] + ([] if build_index == 0 else [f"U{build_index-1}"]),
            [counts / "manifest.json"])
        add(mid, "M", anchor_id, [a.python, "-u", "scripts/estimate_checkpoint_unigram.py",
            "--counts", str(counts), "--jobs", str(a.unigram_workers), "--out", str(unigram)],
            [uid], [unigram / "manifest.json"], a.unigram_workers,
            private_memory_bytes=int(
                unigram_memory_gib * 1024 ** 3
            ))
        for jid, pair, output in ((aid, "ya", ya), (bid, "yb", yb)):
            add(jid, jid[0], anchor_id, [a.python, "-u", "scripts/estimate_checkpoint_pair.py",
                "--counts", str(counts), "--unigram", str(unigram), "--pair", pair,
                "--jobs", str(a.pair_workers), "--out", str(output)],
                [mid], [output / "manifest.json"], a.pair_workers,
                private_memory_bytes=int(
                    a.pair_private_memory_gib * 1024 ** 3
                ))
        add(cid, "C", anchor_id, [a.python, "-u", "scripts/assemble_checkpoint_problem.py",
            "--counts", str(counts), "--ya", str(ya), "--yb", str(yb), "--out", str(problem)],
            [aid, bid], [problem], private_memory_bytes=int(
                a.assembly_private_memory_gib * 1024 ** 3
            ))
        add(tid, "T", anchor_id, [a.python, "-u", "scripts/prepare_anchored_ya_topology.py",
            "--problem", str(problem), "--out", str(topology)], [cid],
            [topology / "manifest.json"], private_memory_bytes=int(
                a.topology_private_memory_gib * 1024 ** 3
            ))
        add(fid, "F", anchor_id, [a.python, "-u", "scripts/fit_anchored_ya_checkpoint.py",
            "--problem", str(problem), "--topology", str(topology), "--out", str(fit),
            "--steps", str(a.fitting_steps), "--batch-size", str(a.batch_size),
            "--seed", str(a.fit_seed + anchor_id), "--workers", str(a.fitting_workers),
            "--learning-rate", str(a.learning_rate), "--exact-interval", str(a.exact_interval),
            "--slack-precision", str(a.slack_precision)], [tid],
            [fit / "manifest.json"], a.fitting_workers,
            private_memory_bytes=int(a.fitting_private_memory_gib * 1024 ** 3))
        add(sid, "S", anchor_id, [a.python, "-u", "scripts/score_anchored_ya_windows.py",
            "--problem", str(problem), "--topology", str(topology), "--fit", str(fit),
            "--stream", plan["stream"], "--windows", windows, "--out", str(score)],
            [fid], [score], private_memory_bytes=int(
                a.scoring_private_memory_gib * 1024 ** 3
            ))
        for family, ids in (("D", [did]), ("U", [uid]), ("M", [mid]),
                            ("P", [aid, bid]), ("C", [cid]), ("T", [tid]),
                            ("F", [fid]), ("S", [sid])):
            families[family].extend(ids)
        # PERFORMANCE INVARIANT: these are priority bands, not synchronization
        # barriers in the live dependency scheduler.  Keep this anchor-causal
        # order unless a measured utilization replay and explicit user
        # agreement support changing it.  A corpus-wide M wave was tried and
        # left most cores idle because serial M jobs exhausted the memory
        # budget before ready four-core pair jobs could launch.
        #
        # Keep each anchor's causal chain ahead of later anchors so ready
        # multi-core pair work is not starved by a corpus-wide wave of serial
        # unigram jobs.  Spare resources can still advance later D/U/M work.
        waves.extend([[did], [uid], [mid], [aid], [bid], [cid], [tid], [fid], [sid]])
        previous_counts = counts
    aggregate = root / "potential.json"
    add("Z", "Z", -1, [a.python, "-u", "scripts/aggregate_anchored_potential.py",
        "--plan", a.plan, "--scores", str(root / "scores_v3"),
        "--bootstrap-seed", str(a.bootstrap_seed), "--out", str(aggregate)],
        families["S"], [aggregate])
    waves.append(["Z"])
    payload = {
        "version": 1, "model": "anchored_ya_relaxed_pair_graph_v1",
        "sequence_estimator": PRODUCTION_SEQUENCE_ESTIMATOR,
        "production_eligible": True,
        "tokenizer_provenance": plan["tokenizer_provenance"],
        "resource_profile": a.resource_profile,
        "plan": str(Path(a.plan).resolve()), "maximum_workers": a.maximum_workers,
        "maximum_private_memory_bytes": int(
            a.maximum_private_memory_gib * 1024 ** 3
        ),
        "jobs": jobs, "waves": waves,
    }
    destination = Path(a.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    print(destination)


if __name__ == "__main__":
    main()
