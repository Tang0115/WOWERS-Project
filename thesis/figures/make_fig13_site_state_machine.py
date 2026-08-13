"""Render Figure 13 — state machine for one plant's passage through screening.

Named states with labeled transitions. Retention transitions run left to right
along the spine; every exclusion transition drops to a terminal state carrying
the pipeline's own reason string and the number of plants that ended there.

All transition counts are read from the parquets.

Run from the repository root:
    PYTHONPATH=. python3 thesis/figures/make_fig13_site_state_machine.py
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
OUT = Path(__file__).resolve().parent / "fig13_site_state_machine.png"

INK = "#1f2933"
LIVE_FC = "#dbe7f2"
LIVE_EC = "#1f4e79"
TERM_FC = "#f2e6e4"
TERM_EC = "#9c4a3c"
FINAL_FC = "#dceee0"
FINAL_EC = "#2f6b41"
ARROW = "#52636f"
DROP = "#9c4a3c"


def stats() -> dict[str, int]:
    p2 = pl.read_parquet(ROOT / "data/processed/phase2/energy_yield_estimates.parquet")
    p3h = pl.read_parquet(ROOT / "data/processed/phase3/head_estimates.parquet")
    p3t = pl.read_parquet(ROOT / "data/processed/phase3/turbine_sizing.parquet")
    p4 = pl.read_parquet(ROOT / "data/processed/phase4/financial_scorecards.parquet")

    reasons = dict(
        p2.filter(pl.col("excluded"))
        .group_by("exclusion_reason")
        .len()
        .iter_rows()
    )
    return {
        "screened": p2.height,
        "small_potw": reasons.get("small_potw", 0),
        "no_usable_flow": reasons.get("no_usable_flow", 0),
        "sparse_dmr": reasons.get("sparse_dmr_artifact", 0),
        "flow_valid": int((~p2["excluded"]).sum()),
        "head_unresolved": int((~p3h["head_valid"]).sum()),
        "head_valid": int(p3h["head_valid"].sum()),
        "below_floor": int((~p3t["turbine_viable"]).sum()) - int((~p3h["head_valid"]).sum()),
        "turbine_viable": int(p3t["turbine_viable"].sum()),
        "fails_gate": int((~p4["project_viable"]).sum()),
        "project_viable": int(p4["project_viable"].sum()),
    }


def main() -> None:
    s = stats()

    # spine states: (x, label, count, kind)
    spine = [
        (0.00, "ingested", s["screened"], "live"),
        (2.05, "flow-valid", s["flow_valid"], "live"),
        (4.10, "head-valid", s["head_valid"], "live"),
        (6.15, "turbine-viable", s["turbine_viable"], "live"),
        (8.20, "project-viable", s["project_viable"], "final"),
    ]

    # (from-index, transition label, terminal state, count, box centre x, y,
    #  where the transition label sits: "right" of the box or "above" it)
    drops = [
        (0, "mean flow < 0.5 MGD", "small_potw", s["small_potw"], 1.55, -1.05, "right"),
        (0, "no usable flow record", "no_usable_flow", s["no_usable_flow"], 1.55, -1.80, "right"),
        (0, "< 3 surviving months", "sparse_dmr_artifact", s["sparse_dmr"], 1.55, -2.55, "right"),
        (1, "no outfall elevation", "head_unresolved", s["head_unresolved"], 4.93, -1.05, "above"),
        (2, "rated power < 1 kW", "below_power_floor", s["below_floor"], 6.98, -1.05, "above"),
        (3, "fails NPV, payback, or IRR", "fails_econ_gate", s["fails_gate"], 9.03, -1.05, "above"),
    ]

    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(figsize=(9.2, 5.2))

    y_spine = 0.0
    bw, bh = 1.66, 0.56
    tw, th = 1.90, 0.52

    centers = {}
    for i, (x, label, n, kind) in enumerate(spine):
        fc, ec = (FINAL_FC, FINAL_EC) if kind == "final" else (LIVE_FC, LIVE_EC)
        ax.add_patch(FancyBboxPatch((x, y_spine - bh / 2), bw, bh,
                                    boxstyle="round,pad=0.0,rounding_size=0.06",
                                    facecolor=fc, edgecolor=ec, linewidth=1.3, zorder=3))
        ax.text(x + bw / 2, y_spine + 0.10, label, fontsize=9.2, fontweight="bold",
                color=INK, ha="center", va="center", zorder=4)
        ax.text(x + bw / 2, y_spine - 0.14, f"{n:,}", fontsize=8.4, color=ec,
                ha="center", va="center", zorder=4)
        centers[i] = (x + bw / 2, x, x + bw)

    for i in range(len(spine) - 1):
        ax.add_patch(FancyArrowPatch(
            (centers[i][2], y_spine), (centers[i + 1][1], y_spine),
            arrowstyle="-|>", mutation_scale=11, color=ARROW,
            linewidth=1.4, zorder=2, shrinkA=0, shrinkB=0))
        ax.text((centers[i][2] + centers[i + 1][1]) / 2, y_spine + 0.115,
                "retain", fontsize=6.8, color="#55636e", ha="center", va="bottom",
                zorder=4)

    y_min = min(d[5] for d in drops)
    for from_i, label, term, n, sx, sy, lab_pos in drops:
        ax.add_patch(FancyBboxPatch((sx - tw / 2, sy - th / 2), tw, th,
                                    boxstyle="round,pad=0.0,rounding_size=0.06",
                                    facecolor=TERM_FC, edgecolor=TERM_EC,
                                    linewidth=1.0, zorder=3))
        ax.text(sx, sy + 0.10, term, fontsize=6.9, color=INK, ha="center",
                va="center", family="monospace", zorder=4)
        ax.text(sx, sy - 0.14, f"{n:,}", fontsize=8.0, color=TERM_EC,
                ha="center", va="center", zorder=4)

        ax.add_patch(FancyArrowPatch(
            (centers[from_i][0], y_spine - bh / 2), (sx, sy + th / 2),
            arrowstyle="-|>", mutation_scale=9, color=DROP, linewidth=1.0,
            linestyle=(0, (4, 2)), zorder=2, shrinkA=2, shrinkB=2,
            connectionstyle="arc3,rad=0.08"))

        if lab_pos == "right":
            ax.text(sx + tw / 2 + 0.09, sy, label, fontsize=6.5, color=DROP,
                    ha="left", va="center", zorder=4)
        else:
            ax.text(sx, sy + th / 2 + 0.11, label, fontsize=6.5, color=DROP,
                    ha="center", va="bottom", zorder=4)

    ax.text(0.0, y_spine + 0.62, "Retention path", fontsize=8.2,
            fontweight="bold", color=LIVE_EC, ha="left", va="center")
    ax.text(0.0, y_min - 0.56, "Terminal states carry the reason string the pipeline records",
            fontsize=7.4, color=DROP, ha="left", va="center")

    ax.set_xlim(-0.60, 10.35)
    ax.set_ylim(y_min - 0.82, y_spine + 0.82)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT, dpi=200)

    print(f"wrote {OUT}")
    total_dropped = (s["small_potw"] + s["no_usable_flow"] + s["sparse_dmr"]
                     + s["head_unresolved"] + s["below_floor"] + s["fails_gate"])
    for k, v in s.items():
        print(f"  {k:>16} {v:,}")
    print(f"  {'total dropped':>16} {total_dropped:,}")
    print(f"  check: {s['screened']:,} - {total_dropped:,} = "
          f"{s['screened'] - total_dropped:,} (want {s['project_viable']:,})")


if __name__ == "__main__":
    main()
