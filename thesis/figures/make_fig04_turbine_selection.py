"""Render Figure 4 — Phase 3 turbine selection envelope and machine mix.

Left panel:  the head-flow operating point of every one of the 3,778
             turbine-viable facilities, coloured by the selected machine type,
             with the selection-rule boundaries drawn on top.
Right panel: rated-power distribution per machine type (log scale) with the
             1 kW physics floor marked.

Run from the repository root:
    PYTHONPATH=. python3 thesis/figures/make_fig04_turbine_selection.py
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
OUT = Path(__file__).resolve().parent / "fig04_turbine_selection.png"

MIN_POWER_KW = 1.0   # turbine_selection.MIN_POWER_KW

INK = "#1f2933"
COLORS = {
    "Crossflow": "#4a7fb5",
    "Francis": "#c1553b",
    "Kaplan": "#3f7d58",
    "in_conduit_micro": "#8a6bbf",
}
ORDER = ["Crossflow", "Francis", "Kaplan", "in_conduit_micro"]


def main() -> None:
    t = pl.read_parquet(ROOT / "data/processed/phase3/turbine_sizing.parquet")
    v = t.filter(pl.col("turbine_viable"))

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.9))

    for name in ORDER:
        s = v.filter(pl.col("turbine_type") == name)
        ax1.scatter(
            s["q_design_m3s"].to_numpy(), s["head_net_m"].to_numpy(),
            s=4, alpha=0.45, linewidths=0, color=COLORS[name],
            label=f"{name.replace('_', ' ')} ({s.height:,})",
        )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    for h in (2.0, 10.0, 50.0):
        ax1.axhline(h, color=INK, linestyle=":", linewidth=0.9, alpha=0.55)
    ax1.axvline(0.5, color=INK, linestyle=":", linewidth=0.9, alpha=0.55)
    ax1.text(ax1.get_xlim()[0] * 1.15, 2.0, " H = 2 m", fontsize=8, color=INK, va="bottom")
    ax1.text(ax1.get_xlim()[0] * 1.15, 10.0, " H = 10 m", fontsize=8, color=INK, va="bottom")
    ax1.text(0.5, ax1.get_ylim()[0] * 1.1, " Q = 0.5 m$^3$/s", fontsize=8, color=INK,
             rotation=90, ha="left", va="bottom")
    ax1.set_xlabel("Design flow $Q$ (m$^3$/s, log scale)")
    ax1.set_ylabel("Net head $H$ (m, log scale)")
    ax1.set_title("(a) Selection envelope, 3,778 turbine-viable sites",
                  fontsize=10, color=INK)
    ax1.legend(fontsize=8, frameon=False, loc="lower right", handletextpad=0.3,
               markerscale=3.2, labelspacing=0.35, borderpad=0.2)

    p = v["rated_power_kw"].to_numpy()
    bins = np.logspace(np.log10(MIN_POWER_KW), np.log10(p.max()), 46)
    stacks = [v.filter(pl.col("turbine_type") == n)["rated_power_kw"].to_numpy() for n in ORDER]
    ax2.hist(stacks, bins=bins, stacked=True,
             color=[COLORS[n] for n in ORDER], edgecolor="white", linewidth=0.2,
             label=[n.replace("_", " ") for n in ORDER])
    ax2.set_xscale("log")
    med = float(np.median(p))
    ax2.axvline(med, color=INK, linestyle="-", linewidth=1.4)
    ax2.text(med, ax2.get_ylim()[1] * 0.97, f" median {med:.2f} kW",
             fontsize=9, color=INK, va="top", ha="left")
    ax2.set_xlabel("Rated power (kW, log scale)")
    ax2.set_ylabel("Facilities")
    ax2.set_title("(b) Rated power by machine type", fontsize=10, color=INK)
    ax2.legend(fontsize=8, frameon=False, loc="upper right", handletextpad=0.4)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9, colors=INK)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200)

    print(f"wrote {OUT}")
    summary = (
        v.group_by("turbine_type")
        .agg(
            pl.len().alias("n"),
            pl.col("rated_power_kw").median().alias("med_kw"),
            pl.col("head_net_m").median().alias("med_head"),
            pl.col("capacity_factor").median().alias("med_cf"),
            (pl.col("annual_energy_mwh").sum() / 1_000).alias("gwh"),
        )
        .sort("n", descending=True)
    )
    for r in summary.to_dicts():
        print(f"  {r['turbine_type']:18s} n={r['n']:5,} med {r['med_kw']:8.2f} kW "
              f"H {r['med_head']:6.2f} m CF {r['med_cf']:.3f} {r['gwh']:8.2f} GWh/yr")
    print(f"  total {v.height:,} sites, {v['annual_energy_mwh'].sum()/1000:.3f} GWh/yr")


if __name__ == "__main__":
    main()
