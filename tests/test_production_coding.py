"""Guards for the production-wide layered-estimator invariant."""

from types import SimpleNamespace

import numpy as np
import pytest

from product_model_with_memory import production_coding


def test_production_lmax_values_match_conditional_estimators():
    # These are the alphabets in the current scheduled campaign.  Keeping
    # the values explicit makes an accidental prefix/conditional convention
    # change visible in review rather than only through changed bpc results.
    assert production_coding.default_l_max(1024) == 33
    assert production_coding.default_l_max(4096) == 39
    assert production_coding.default_l_max(16384) == 46
    assert production_coding.default_l_max(100277) == 54


def test_layered_sequence_code_uses_depth_averaged_profile(monkeypatch, tmp_path):
    seen = {}
    store = tmp_path / "certified-store"
    store.mkdir()
    (store / "anchors.json").write_text("{}")
    monkeypatch.setenv("PMM_UNIVERSAL_TABLES", str(store))
    for key in ("PMM_PHI_LADDER_EVERY", "PMM_PHI_LADDER_DEGREE",
                "PMM_PHI_SADDLE_MIN_L"):
        monkeypatch.setenv(key, production_coding.PRODUCTION_TABLE_DEFAULTS[key])

    def fake(profiles, **kwargs):
        seen["profiles"] = profiles
        seen.update(kwargs)
        return {0: SimpleNamespace(log2_q_avg=-12.5)}

    monkeypatch.setattr(
        production_coding, "depth_averaged_codelength_profiles", fake
    )
    result = production_coding.layered_sequence_code(
        np.array([0, 3, 1, 3], dtype=np.int64), 4,
        jobs=2, universal_path=store,
    )
    assert result.bits == 12.5
    assert result.tokens == 7
    assert result.estimator == production_coding.PRODUCTION_SEQUENCE_ESTIMATOR
    assert result.l_max == production_coding.default_l_max(4)
    assert seen["profiles"] == {0: (3, 3, 1)}
    assert seen["d"] == 4
    # The memoryless sequence code and all conditional-row predictors use
    # this same depth range.  In particular, production does not silently
    # choose a different L_max for the prefix or escape payload.
    assert seen["l_max"] == production_coding.default_l_max(4)
    assert seen["jobs"] == 2
    assert seen["tables_source"] == "universal"
    assert seen["universal_path"] == store


def test_empty_layered_sequence_costs_zero_without_tables():
    result = production_coding.layered_sequence_code(
        np.zeros(0, dtype=np.int64), 0
    )
    assert result.bits == 0.0
    assert result.tokens == 0


def test_production_rejects_explicit_alternative_estimator():
    with pytest.raises(RuntimeError, match="scientific model change"):
        production_coding.require_production_sequence_estimator(
            "kt_jeffreys", source="test artifact"
        )


def test_numerical_identity_gate_accepts_roundoff_and_returns_error():
    error = production_coding.require_numerical_identity(
        1.25 + 2e-13, 1.25, gate="test"
    )
    assert error == pytest.approx(2e-13, abs=1e-15)


def test_numerical_identity_gate_rejects_material_difference():
    with pytest.raises(RuntimeError, match="D=1/first-order gate failed"):
        production_coding.require_numerical_identity(
            1.25, 1.24, gate="D=1/first-order"
        )


def test_production_tables_use_sealed_anchor_ladder(tmp_path, monkeypatch):
    store = tmp_path / "anchors"
    store.mkdir()
    (store / "anchors.json").write_text("{}")
    monkeypatch.setenv("PMM_UNIVERSAL_TABLES", str(store))
    for key in ("PMM_PHI_LADDER_EVERY", "PMM_PHI_LADDER_DEGREE",
                "PMM_PHI_SADDLE_MIN_L"):
        monkeypatch.setenv(key, production_coding.PRODUCTION_TABLE_DEFAULTS[key])

    selected = production_coding.configure_production_tables(store)

    assert selected == store
    assert production_coding.os.environ["PMM_UNIVERSAL_TABLES"] == str(store)
    assert production_coding.os.environ["PMM_PHI_LADDER_EVERY"] == "1"
    assert production_coding.os.environ["PMM_PHI_LADDER_DEGREE"] == "11"
    assert production_coding.os.environ["PMM_PHI_SADDLE_MIN_L"] == "54"


def test_production_tables_refuse_grow_on_demand_store(tmp_path, monkeypatch):
    monkeypatch.setenv("PMM_UNIVERSAL_TABLES", str(tmp_path / "exact"))
    for key in ("PMM_PHI_LADDER_EVERY", "PMM_PHI_LADDER_DEGREE",
                "PMM_PHI_SADDLE_MIN_L"):
        monkeypatch.setenv(key, production_coding.PRODUCTION_TABLE_DEFAULTS[key])
    with pytest.raises(RuntimeError, match="refusing to grow"):
        production_coding.configure_production_tables(tmp_path / "exact")
