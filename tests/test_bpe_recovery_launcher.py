"""Static audit of every SCITAS BPE recovery array entry."""

import json
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]


def test_every_recovery_entry_matches_its_verified_stream_manifest():
    specification = json.loads(
        (REPOSITORY / "cluster/anchored_bpe_recovery_campaigns.json").read_text()
    )
    assert specification["tokenizer"] == {
        "representation": "bpe", "encoding": "cl100k_base"
    }
    rows = specification["campaigns"]
    assert [row["task"] for row in rows] == list(range(8))
    expected_streams = {
        "production_bpe_text8_v16384_stream": (19429294, 16384, "0ae5d896371d652ff05cfe37a13a972c00702dc3af34fc52f6c470f09ab6cade"),
        "production_bpe_text8_v32768_stream": (19429294, 32768, "b8fa7bd6e88b3e2c215780a6456acf5ba8583e9f93880db770fa7df493e433d9"),
        "production_bpe_enwik8_v32768_stream": (25793085, 32768, "7317d05ed3e8016ae3e28001899e08e0c1916c76da34b7bfd7aa0ebc6706ff43"),
        "production_bpe_enwik9_v65536_stream": (273662103, 65536, "3d97af5bb80994c9de4858c9a2adbdbde9a9c6e8cdcd8035d0bcfc1d01e748dc"),
    }
    for row in rows:
        n, v, digest = expected_streams[row["stream"]]
        assert (row["n"], row["v"], row["stream_sha256"]) == (n, v, digest)
        assert row["m1"] <= row["v"] and row["m2"] <= row["v"]
        assert row["corpus"] in row["stream"]


def test_launcher_checks_plan_schedule_and_result_provenance():
    source = (REPOSITORY / "cluster/job_anchored_bpe_recovery_jed.sbatch").read_text()
    assert "anchored_bpe_recovery_campaigns.json" in source
    assert 'p["stream_sha256"] == sys.argv[2]' in source
    assert 'schedule (V,M1,M2)' in source
    assert 'len(p["anchors"]) == 64' in source
