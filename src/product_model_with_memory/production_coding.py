"""Non-negotiable estimator policy for production codelengths.

The scientific premise of this repository is the depth-averaged layered
product-simplex predictor. Every data-bearing symbol sequence in a production
code MUST use that predictor. Do not replace it with KT/Jeffreys,
add-one/Laplace-rule, a plug-in estimate, or another convenient smoother
without first
making that change an explicit scientific comparison and assigning it a new
estimator identifier. Metadata descriptions (for example, an enumerative
vocabulary-subset code) are descriptions rather than symbol predictors and
are outside this rule.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from product_model_with_memory.codelength import (
    default_l_max,
    depth_averaged_codelength_profiles,
)

PRODUCTION_SEQUENCE_ESTIMATOR = "layered_depth_averaged_product_simplex_v1"

PRODUCTION_TABLE_DEFAULTS = {
    "PMM_UNIVERSAL_TABLES": "tables/anchors_prod",
    "PMM_PHI_LADDER_EVERY": "1",
    "PMM_PHI_LADDER_DEGREE": "11",
    "PMM_PHI_SADDLE_MIN_L": "54",
}


def configure_production_tables(
    universal_path: str | Path | None = None,
) -> Path:
    """Select and validate the sealed production anchor store.

    Production entry points call this before the universal-table module is
    imported.  This is deliberately a hard guard: silently falling back to
    the grow-on-demand exact store changes both the evaluator and the runtime
    by orders of magnitude.
    """

    if universal_path is None:
        universal_path = os.environ.get(
            "PMM_UNIVERSAL_TABLES",
            PRODUCTION_TABLE_DEFAULTS["PMM_UNIVERSAL_TABLES"],
        )
    path = Path(universal_path)
    os.environ["PMM_UNIVERSAL_TABLES"] = str(path)
    for key in (
        "PMM_PHI_LADDER_EVERY",
        "PMM_PHI_LADDER_DEGREE",
        "PMM_PHI_SADDLE_MIN_L",
    ):
        required = PRODUCTION_TABLE_DEFAULTS[key]
        actual = os.environ.setdefault(key, required)
        if actual != required:
            raise RuntimeError(
                f"production requires {key}={required}, got {actual!r}"
            )
    if not (path / "anchors.json").is_file():
        raise RuntimeError(
            f"production table store {path} is not the sealed designed "
            "anchor store (anchors.json is missing); refusing to grow "
            "ad-hoc exact columns"
        )
    return path


@dataclass(frozen=True)
class LayeredSequenceCode:
    """Codelength and immutable estimator provenance for one sequence."""

    bits: float
    tokens: int
    alphabet_size: int
    l_max: int
    estimator: str = PRODUCTION_SEQUENCE_ESTIMATOR


def require_production_sequence_estimator(value: str, *, source: str) -> None:
    """Reject an artifact produced by a different, undeclared estimator."""

    if value != PRODUCTION_SEQUENCE_ESTIMATOR:
        raise RuntimeError(
            f"{source} uses sequence estimator {value!r}; production requires "
            f"{PRODUCTION_SEQUENCE_ESTIMATOR!r}. Changing the estimator is a "
            "scientific model change and must be explicitly discussed."
        )


def require_numerical_identity(
    actual: float,
    expected: float,
    *,
    gate: str,
    tolerance: float = 5e-12,
) -> float:
    """Enforce a production numerical gate and return the signed error."""

    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    error = float(actual) - float(expected)
    if not np.isfinite(error) or abs(error) > tolerance:
        raise RuntimeError(
            f"{gate} gate failed: actual={actual:.15g}, "
            f"expected={expected:.15g}, error={error:+.3e}, "
            f"tolerance={tolerance:.3e}"
        )
    return error


def layered_sequence_code(
    counts: np.ndarray,
    alphabet_size: int,
    *,
    jobs: int = 1,
    universal_path: str | Path | None = None,
) -> LayeredSequenceCode:
    """Code a complete finite-alphabet sequence with the layered predictor.

    The layered model is exchangeable, so its exact sequential codelength is a
    function of the final count profile. This is the production replacement
    for the former KT helper used on the initial prefix and escape payload.
    No estimator-selection argument is deliberately exposed here.
    """

    values = np.asarray(counts)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("sequence counts must be a one-dimensional integer array")
    if (values < 0).any():
        raise ValueError("sequence counts must be nonnegative")
    if alphabet_size < 0 or len(values) != alphabet_size:
        raise ValueError("counts must contain one entry per alphabet symbol")
    if jobs < 1:
        raise ValueError("jobs must be positive")
    total = int(values.sum())
    if total == 0:
        return LayeredSequenceCode(0.0, 0, alphabet_size, 0)
    if alphabet_size < 1:
        raise ValueError("a nonempty sequence requires a nonempty alphabet")

    profile = tuple(sorted(
        (int(value) for value in values if value > 0), reverse=True
    ))
    universal_path = configure_production_tables(universal_path)
    l_max = default_l_max(alphabet_size)
    result = depth_averaged_codelength_profiles(
        {0: profile},
        d=alphabet_size,
        l_max=l_max,
        jobs=jobs,
        tables_source="universal",
        universal_path=universal_path,
    )[0]
    return LayeredSequenceCode(
        bits=-float(result.log2_q_avg),
        tokens=total,
        alphabet_size=alphabet_size,
        l_max=l_max,
    )
