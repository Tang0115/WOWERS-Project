"""Render Figure 15 — the head assumption made visible.

Section 4.3.4 names DEM-derived head as the largest methodological assumption in
the work. This figure shows what that assumption looks like across the fleet
rather than only asserting it in prose.

Left panel:  3DEP net head against the Phase 2 archetype head for the same plant,
             so the disagreement between the two estimates is visible directly.
Right panel: the net-head distribution split by the confidence label the pipeline
             assigns, which is what downstream filtering keys on.

Run from the repository root:
    PYTHONPATH=. python3 thesis/figures/make_fig15_head_confidence.py
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
OUT = Path(__file__).resolve().parent / "fig15_head_confidence.png"

INK = "#1f2933"
HIGH = "#1f4e79"
MEDIUM = "#c08a3e"
GUIDE = "#8a97a1"


def main() -> None:
    h = pl.read_parquet(ROOT / "data/processed/phase3/head_estimates.parquet")
    valid = h.filter(pl.col("head_valid"))

    dem = valid.filter(pl.col("head_source") == "usgs_3dep")
    arch = valid.filter(pl.col("head_source") == "phase2_literature")

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6),
                                   gridspec_kw={"width_ratios": [1.05, 1.0]})

    # (a) 3DEP net head against the archetype head the same plant would have got
    x = dem["head_m_p50"].to_numpy()
    y = dem["head_net_m"].to_numpy()
    ax1.scatter(x, y, s=5, alpha=0.22, color=HIGH, linewidths=0, zorder=2)

    lim_lo, lim_hi = 0.8, 60.0
    ax1.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=GUIDE, linewidth=1.0,
             linestyle="--", zorder=3, label="1:1, DEM agrees with archetype")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(lim_lo, lim_hi)
    ax1.set_ylim(lim_lo, lim_hi)
    ax1.set_xlabel("Phase 2 archetype head, P50 (m)")
    ax1.set_ylabel("Phase 3 net head from 3DEP (m)")
    ax1.legend(fontsize=7.6, frameon=False, loc="upper left")

    above = float((y > x).mean() * 100)
    ratio = np.median(y / x)
    ax1.text(0.97, 0.06,
             f"n = {len(x):,} plants on 3DEP\n"
             f"{above:.1f} % above the 1:1 line\n"
             f"median ratio {ratio:.2f}$\\times$",
             transform=ax1.transAxes, fontsize=7.6, color=INK,
             ha="right", va="bottom")
    ax1.set_title("(a) DEM head against archetype head", fontsize=10, color=INK)

    # (b) net-head distribution by confidence label
    bins = np.logspace(np.log10(0.8), np.log10(60), 46)
    ax2.hist(dem["head_net_m"].to_numpy(), bins=bins, color=HIGH, alpha=0.85,
             label=f"high, 3DEP (n = {dem.height:,})")
    ax2.hist(arch["head_net_m"].to_numpy(), bins=bins, color=MEDIUM, alpha=0.85,
             label=f"medium, archetype (n = {arch.height:,})")
    ax2.set_xscale("log")
    ax2.set_xlabel("Net head (m)")
    ax2.set_ylabel("Plants")
    ax2.legend(fontsize=7.8, frameon=False, loc="upper right")
    ax2.set_title("(b) Net head by confidence label", fontsize=10, color=INK)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9, colors=INK)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200)

    print(f"wrote {OUT}")
    print(f"  head-valid plants        {valid.height:,}")
    print(f"  3DEP  (high confidence)  {dem.height:,}")
    print(f"  archetype (medium)       {arch.height:,}")
    print(f"  3DEP above 1:1 line      {above:.2f} %")
    print(f"  median DEM:archetype     {ratio:.4f}")
    for name, col in (("3DEP", dem["head_net_m"]), ("archetype", arch["head_net_m"])):
        a = col.to_numpy()
        print(f"  {name:>10} net head p10/p50/p90 = "
              f"{np.percentile(a,10):.3f} / {np.median(a):.3f} / "
              f"{np.percentile(a,90):.3f} m")


if __name__ == "__main__":
    main()
