#!/usr/bin/env python3
"""Render the complete, parameter-sorted direct memory-two result tables."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ESTIMATOR = "layered_depth_averaged_product_simplex_v1"


def load_rows(relative_path: str) -> list[dict[str, int | float]]:
    payload = json.loads((ROOT / relative_path).read_text())
    if payload["sequence_estimator"] != ESTIMATOR:
        raise ValueError(f"wrong estimator in {relative_path}")
    return payload["rows"]


def add(
    rows: dict[tuple[int, int, int], tuple[int, float]],
    V: int,
    M1: int,
    M2: int,
    states: int,
    bpc: float,
) -> None:
    rows.setdefault((V, M1, M2), (states, bpc))


def add_accounting(
    rows: dict[tuple[int, int, int], tuple[int, float]], relative_path: str
) -> None:
    for row in load_rows(relative_path):
        add(
            rows,
            int(row["V"]),
            int(row["M1"]),
            int(row["M2"]),
            int(row["observed_states"]),
            float(row["honest_member_bits_per_character"]),
        )


def render_table(
    corpus: str, rows: dict[tuple[int, int, int], tuple[int, float]]
) -> str:
    label = f"tab:memory-two-all-{corpus}"
    lines = [
        r"\begin{longtable}{lrrrrr}",
        rf"\caption{{All distinct completed direct order-two experiments on \texttt{{{corpus}}}, ordered by $(V,M_1,M_2)$.}}\label{{{label}}}\\",
        r"\toprule",
        r"corpus & $V$ & $M_1$ & $M_2$ & observed states & honest bpc \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{6}{c}{\tablename\ \thetable\ -- continued} \\",
        r"\toprule",
        r"corpus & $V$ & $M_1$ & $M_2$ & observed states & honest bpc \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{6}{r}{continued on next page} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for (V, M1, M2), (states, bpc) in sorted(rows.items()):
        lines.append(
            rf"\texttt{{{corpus}}} & ${V:,}$ & ${M1:,}$ & ${M2:,}$ & ${states:,}$ & ${bpc:.9f}$ \\".replace(",", r"{,}")
        )
    lines.extend([r"\end{longtable}", ""])
    return "\n".join(lines)


def main() -> None:
    text8: dict[tuple[int, int, int], tuple[int, float]] = {}
    for row in [
        (8192, 4096, 0, 4097, 1.827971116884),
        (8192, 4096, 16, 51164, 1.845086234193),
        (8192, 8192, 0, 8192, 1.800937067383),
        (8192, 8192, 16, 91040, 1.827985729951),
        (16384, 16384, 0, 16384, 1.7645625088123693),
    ]:
        add(text8, *row)
    add_accounting(text8, "output/memory2_text8_wave1_20260810/accounting.json")
    add_accounting(text8, "output/memory2_text8_wave2_20260810/accounting.json")

    enwik8: dict[tuple[int, int, int], tuple[int, float]] = {}
    for row in [
        (16384, 8192, 0, 8193, 2.173788461120),
        (16384, 8192, 16, 69141, 2.182381940399),
        (16384, 16384, 0, 16384, 2.134131685271),
        (16384, 16384, 16, 125883, 2.156653920469),
        (32768, 32768, 0, 32768, 2.083819340000),
    ]:
        add(enwik8, *row)
    add_accounting(enwik8, "output/memory2_enwik8_wave1_20260810/accounting.json")

    enwik9: dict[tuple[int, int, int], tuple[int, float]] = {}
    for row in [
        (65536, 65536, 0, 65536, 1.828580231181),
        (65536, 65536, 16, 416184, 1.796406734070),
        (65536, 65536, 32, 566158, 1.779269471640),
        (65536, 65536, 64, 932299, 1.769144638956),
    ]:
        add(enwik9, *row)
    # Prefer the frozen 52-way campaign for duplicate triples, then add the
    # genuinely new power-of-two refinement points.
    add_accounting(enwik9, "output/memory2_triplet_campaign_20260809/accounting.json")
    add_accounting(enwik9, "output/memory2_power2_refinement_20260810/accounting.json")

    preamble = r"""\section{Complete direct memory-two experiments}
\label{app:memory-two-results}

The following tables contain every distinct completed experiment reported in
Section~\ref{sec:ordertwo}.  Duplicate reruns at the same triple are
consolidated.  Each rate retains all corpus descriptions and the declared
model-choice charge of the campaign from which the displayed row is taken;
consequently, reruns of one triple in differently sized frozen grids can
differ in the last few decimal places.  All data-bearing sequences use
\texttt{layered\_depth\_averaged\_product\_simplex\_v1}.

\begingroup
\footnotesize
\setlength{\tabcolsep}{5pt}
"""
    ending = "\\endgroup\n"
    output = preamble + "\n".join(
        render_table(corpus, rows)
        for corpus, rows in (("text8", text8), ("enwik8", enwik8), ("enwik9", enwik9))
    ) + ending
    target = ROOT / "paper/memory2_appendix_tables.tex"
    target.write_text(output)
    print(f"wrote {target}: text8={len(text8)}, enwik8={len(enwik8)}, enwik9={len(enwik9)}")


if __name__ == "__main__":
    main()
