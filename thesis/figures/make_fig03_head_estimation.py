"""Render Figure 3 — Phase 3 net-head estimation outcome.

Left panel:  net-head distributions for the two head sources among the 4,860
             head-valid facilities (USGS 3DEP elevation difference vs the
             Phase 2 literature archetype), with the 1.0 m validity floor.
Right panel: why 1,682 facilities fell back to the archetype head, and why 604
             3DEP readings were kept but marked invalid.

Run from the repository root:
    PYTHONPATH=. python3 thesis/figures/make_fig03_head_estimation.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "fig03_head_estimation.png"

LOSS_FRACTION = 0.15   # phase3.head_loss_fraction
MIN_NET_HEAD_M = 1.0   # phase3.min_net_head_m
MAX_DIVERGENCE_RATIO = 4.0  # head_estimation._MAX_DIVERGENCE_RATIO

INK = "#1f2933"
BAR = "#4a7fb5"
ALT = "#8fb8de"
ACCENT = "#c1553b"


def main() -> None:
    t = pl.read_parquet(ROOT / "data/processed/phase3/turbine_sizing.parquet")

    valid = t.filter(pl.col("head_valid"))
    dep = valid.filter(pl.col("head_source") == "usgs_3dep")["head_net_m"].to_numpy()
    lit = valid.filter(pl.col("head_source") == "phase2_literature")["head_net_m"].to_numpy()
    invalid = t.filter(~pl.col("head_valid"))["head_net_m"].to_numpy()

    # Fallback attribution for the literature-head cohort.
    litset = t.filter(pl.col("head_source") == "phase2_literature")
    both = litset.filter(
        pl.col("elevation_m").is_not_null() & pl.col("elev_outfall_m").is_not_null()
    ).with_columns(
        ((pl.col("elevation_m") - pl.col("elev_outfall_m")) * (1 - LOSS_FRACTION)).alias("cand")
    )
    n_neg = both.filter(pl.col("cand") <= 0).height
    n_div = (
        both.filter(pl.col("cand") > 0)
        .with_columns(
            ((pl.col("cand") - pl.col("head_m_p50")).abs() / pl.col("head_m_p50")).alias("r")
        )
        .filter(pl.col("r") > MAX_DIVERGENCE_RATIO)
        .height
    )
    n_nocoord = litset.filter(pl.col("elev_outfall_m").is_null()).height

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.7))

    # Log count axis: the archetype head takes only three discrete values, so on a
    # linear axis its spikes bury the 3DEP tail entirely.
    xmax = 25.0
    n_above = int((dep > xmax).sum() + (lit > xmax).sum())
    bins = np.linspace(0, xmax, 51)
    ax1.hist(dep, bins=bins, color=BAR, edgecolor="white",
             linewidth=0.3, label=f"USGS 3DEP ($n$ = {len(dep):,})")
    ax1.hist(lit, bins=bins, color=ALT, edgecolor="white",
             linewidth=0.3, alpha=0.85, label=f"Literature archetype ($n$ = {len(lit):,})")
    ax1.set_yscale("log")
    ax1.set_ylim(1, 3_000)
    ax1.axvline(MIN_NET_HEAD_M, color=ACCENT, linestyle=":", linewidth=1.6)
    ax1.text(MIN_NET_HEAD_M - 0.5, 1.6, "1.0 m validity floor",
             color=ACCENT, fontsize=8.5, va="bottom", ha="left", rotation=90)
    ax1.text(xmax, 1.6, f"{n_above:,} sites > 25 m  ",
             fontsize=8, color=INK, ha="right", va="bottom")
    ax1.set_xlabel("Net head (m)")
    ax1.set_ylabel("Facilities (log scale)")
    ax1.set_title("(a) Net head by source, 4,860 head-valid sites", fontsize=10, color=INK)
    ax1.legend(fontsize=8.5, frameon=False, loc="upper right")

    labels = [
        "Outfall at or above\nplant (negative head)",
        "No outfall coordinate\nor elevation",
        "3DEP–archetype\ndivergence > 4×",
        "3DEP head kept but\n< 1.0 m (invalid)",
    ]
    values = [n_neg, n_nocoord, n_div, len(invalid)]
    colors = [BAR, BAR, BAR, ACCENT]
    ypos = np.arange(len(values))[::-1]
    ax2.barh(ypos, values, color=colors, height=0.62)
    for y, val in zip(ypos, values):
        ax2.text(val + 18, y, f"{val:,}", va="center", fontsize=9, color=INK)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(labels, fontsize=8.5)
    ax2.set_xlim(0, max(values) * 1.22)
    ax2.set_xlabel("Facilities")
    ax2.set_title("(b) Why the elevation proxy was not used", fontsize=10, color=INK)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9, colors=INK)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200)

    print(f"wrote {OUT}")
    print(f"3DEP valid n={len(dep):,} p10={np.percentile(dep,10):.3f} "
          f"p50={np.percentile(dep,50):.3f} p90={np.percentile(dep,90):.3f} max={dep.max():.3f}")
    print(f"literature valid n={len(lit):,} p10={np.percentile(lit,10):.3f} "
          f"p50={np.percentile(lit,50):.3f} p90={np.percentile(lit,90):.3f}")
    print(f"fallback: negative={n_neg:,} no-coord={n_nocoord:,} divergence={n_div:,} "
          f"| sub-1 m invalid={len(invalid):,} (p50 {np.percentile(invalid,50):.4f} m)")


if __name__ == "__main__":
    main()
