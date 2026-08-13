"""Render Figure 12 — build-process flowchart of the four-phase pipeline.

One swim lane per phase, each naming the tool that runs in it, the parquet it
writes, and the plant count surviving at its boundary. Satisfies the required
build-process flowchart with per-tool swim lanes.

Counts are read from the parquets rather than hard-coded, so the figure cannot
drift from the baseline it is drawn from.

Run from the repository root:
    PYTHONPATH=. python3 thesis/figures/make_fig12_pipeline_dataflow.py
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "fig12_pipeline_dataflow.png"

INK = "#1f2933"
LANE_A = "#eef3f8"
LANE_B = "#e4ebf2"
BOX = "#ffffff"
EDGE = "#1f4e79"
ARROW = "#52636f"


def counts() -> dict[str, int]:
    p1 = pl.read_parquet(ROOT / "data/processed/phase1/ranked_candidates.parquet")
    p2 = pl.read_parquet(ROOT / "data/processed/phase2/energy_yield_estimates.parquet")
    p3h = pl.read_parquet(ROOT / "data/processed/phase3/head_estimates.parquet")
    p3t = pl.read_parquet(ROOT / "data/processed/phase3/turbine_sizing.parquet")
    p4 = pl.read_parquet(ROOT / "data/processed/phase4/financial_scorecards.parquet")
    return {
        "screened": p1.height,
        "flow_valid": int((~p2["excluded"]).sum()),
        "head_valid": int(p3h["head_valid"].sum()),
        "turbine_viable": int(p3t["turbine_viable"].sum()),
        "project_viable": int(p4["project_viable"].sum()),
    }


def main() -> None:
    c = counts()

    # lane label, tool modules, stage boxes, parquet written, surviving count
    lanes = [
        (
            "Phase 1\nfilter and rank",
            "ingest.py · filter_potw.py\ndmr_timeseries.py\nflow_features.py · ranking.py",
            ["ECHO / ICIS\ningest", "POTW filter\nactive permits", "Flow features\nand FDC", "Composite\nranking"],
            "ranked_candidates\n.parquet",
            f"{c['screened']:,}\nscreened",
        ),
        (
            "Phase 2\nMonte-Carlo energy",
            "head_assumptions.py\nenergy_physics.py\nmonte_carlo.py",
            ["Archetype head\ntriangulars", "Sample $\\eta$, $H$,\navailability", "Integrate\n$\\eta\\rho gQH$ over FDC", "P10 / P50 / P90\nper plant"],
            "energy_yield\n_estimates.parquet",
            f"{c['flow_valid']:,}\nflow-valid",
        ),
        (
            "Phase 3\nhead and machine",
            "outfall_coords.py\nelevation.py\nhead_estimation.py\nturbine_selection.py",
            ["Outfall coords\nvia NHD", "3DEP elevation\nquery", "Net head +\nplausibility gate", "Turbine match\nand rated flow"],
            "turbine_sizing\n.parquet",
            f"{c['turbine_viable']:,}\nturbine-viable",
        ),
        (
            "Phase 4\ncost and finance",
            "cost_models.py · revenue.py\nplant_consumption.py\nfinancials.py · sensitivity.py",
            ["CapEx bands\n+ BCM recalib.", "Revenue at\nEIA rates", "NPV / IRR /\npayback", "Gate and\ntier"],
            "financial\n_scorecards.parquet",
            f"{c['project_viable']:,}\nproject-viable",
        ),
    ]

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9})
    fig, ax = plt.subplots(figsize=(9.2, 6.4))

    lane_h = 1.0
    box_w, box_h = 1.42, 0.50
    x0 = 2.42
    gap = 0.30

    for i, (lane, tools, stages, parquet, surviving) in enumerate(lanes):
        y = (len(lanes) - 1 - i) * lane_h
        yc = y + lane_h / 2

        ax.add_patch(
            FancyBboxPatch((0.02, y + 0.06), 11.30, lane_h - 0.12,
                           boxstyle="round,pad=0.0,rounding_size=0.03",
                           facecolor=LANE_A if i % 2 == 0 else LANE_B,
                           edgecolor="none", zorder=0)
        )
        ax.text(0.16, yc + 0.17, lane, fontsize=9.5, fontweight="bold",
                color=INK, va="center", ha="left", zorder=3)
        ax.text(0.16, yc - 0.24, tools, fontsize=6.4, color="#55636e",
                va="center", ha="left", family="monospace", zorder=3)

        for j, stage in enumerate(stages):
            bx = x0 + j * (box_w + gap)
            ax.add_patch(
                FancyBboxPatch((bx, yc - box_h / 2), box_w, box_h,
                               boxstyle="round,pad=0.0,rounding_size=0.04",
                               facecolor=BOX, edgecolor=EDGE, linewidth=0.9, zorder=2)
            )
            ax.text(bx + box_w / 2, yc, stage, fontsize=7.2, color=INK,
                    ha="center", va="center", zorder=3)
            if j < len(stages) - 1:
                ax.add_patch(FancyArrowPatch(
                    (bx + box_w, yc), (bx + box_w + gap, yc),
                    arrowstyle="-|>", mutation_scale=8, color=ARROW,
                    linewidth=0.9, zorder=2, shrinkA=0, shrinkB=0))

        px = x0 + len(stages) * (box_w + gap) + 0.10
        ax.text(px, yc + 0.15, parquet, fontsize=6.6, color="#55636e",
                ha="left", va="center", family="monospace", zorder=3)
        ax.text(px, yc - 0.22, surviving, fontsize=8.2, color=EDGE,
                fontweight="bold", ha="left", va="center", zorder=3)

        if i < len(lanes) - 1:
            ax.add_patch(FancyArrowPatch(
                (px + 0.62, y + 0.10), (px + 0.62, y - 0.10),
                arrowstyle="-|>", mutation_scale=9, color=ARROW,
                linewidth=1.1, zorder=2, shrinkA=0, shrinkB=0))

    # export layer below the four phases
    y = -lane_h
    yc = y + lane_h / 2
    ax.add_patch(
        FancyBboxPatch((0.02, y + 0.06), 11.30, lane_h - 0.12,
                       boxstyle="round,pad=0.0,rounding_size=0.03",
                       facecolor="#dde7f0", edgecolor="none", zorder=0)
    )
    ax.text(0.16, yc + 0.17, "Export layer", fontsize=9.5, fontweight="bold",
            color=INK, va="center", ha="left", zorder=3)
    ax.text(0.16, yc - 0.24, "scripts/export_geojson.py", fontsize=6.4,
            color="#55636e", va="center", ha="left", family="monospace", zorder=3)
    for j, stage in enumerate(["Join phases\n1-4", "Round to\n58 properties",
                              "Sort by\nnpdes_id", "Write GeoJSON\n(byte-stable)"]):
        bx = x0 + j * (box_w + gap)
        ax.add_patch(
            FancyBboxPatch((bx, yc - box_h / 2), box_w, box_h,
                           boxstyle="round,pad=0.0,rounding_size=0.04",
                           facecolor=BOX, edgecolor=EDGE, linewidth=0.9, zorder=2)
        )
        ax.text(bx + box_w / 2, yc, stage, fontsize=7.2, color=INK,
                ha="center", va="center", zorder=3)
        if j < 3:
            ax.add_patch(FancyArrowPatch(
                (bx + box_w, yc), (bx + box_w + gap, yc),
                arrowstyle="-|>", mutation_scale=8, color=ARROW,
                linewidth=0.9, zorder=2, shrinkA=0, shrinkB=0))
    px = x0 + 4 * (box_w + gap) + 0.10
    ax.text(px, yc + 0.15, "viable_sites\n.geojson", fontsize=6.6, color="#55636e",
            ha="left", va="center", family="monospace", zorder=3)
    ax.text(px, yc - 0.22, f"{c['project_viable']:,}\nfeatures", fontsize=8.2,
            color=EDGE, fontweight="bold", ha="left", va="center", zorder=3)

    ax.set_xlim(0, 11.34)
    ax.set_ylim(-lane_h + 0.02, len(lanes) * lane_h - 0.02)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT, dpi=200)

    print(f"wrote {OUT}")
    for k, v in c.items():
        print(f"  {k:>16} {v:,}")


if __name__ == "__main__":
    main()
