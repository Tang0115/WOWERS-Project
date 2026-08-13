"""Render Figure 11 — Monte-Carlo uncertainty bands across the retained fleet.

Left panel:  every one of the 5,464 retained plants, ordered by its median annual
             energy, with the P10-P90 band drawn around the P50 line on a log axis.
Right panel: the distribution of the per-site P90:P10 ratio, which is the width of
             that band expressed as a multiple.

Run from the repository root:
    PYTHONPATH=. python3 thesis/figures/make_fig11_mc_bands.py
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
OUT = Path(__file__).resolve().parent / "fig11_mc_bands.png"

INK = "#1f2933"
BAND = "#9fbcd8"
MED = "#1f4e79"
HIST = "#4a7fb5"


def main() -> None:
    p2 = pl.read_parquet(ROOT / "data/processed/phase2/energy_yield_estimates.parquet")
    ret = p2.filter(~pl.col("excluded")).sort("energy_p50_kwh_yr")

    p10 = ret["energy_p10_kwh_yr"].to_numpy()
    p50 = ret["energy_p50_kwh_yr"].to_numpy()
    p90 = ret["energy_p90_kwh_yr"].to_numpy()
    x = np.arange(len(p50))

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.5),
                                   gridspec_kw={"width_ratios": [1.5, 1.0]})

    ax1.fill_between(x, p10, p90, color=BAND, linewidth=0, label="P10--P90 band")
    ax1.plot(x, p50, color=MED, linewidth=1.2, label="P50")
    ax1.set_yscale("log")
    ax1.set_xlim(0, len(x))
    ax1.set_xlabel("Retained plants, ordered by median energy")
    ax1.set_ylabel("Annual energy (kWh/yr)")
    ax1.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax1.set_title(f"(a) Per-site energy band, {len(x):,} retained plants",
                  fontsize=10, color=INK)

    ratio = p90 / p10
    ax2.hist(ratio, bins=60, color=HIST, alpha=0.9)
    med = float(np.median(ratio))
    ax2.axvline(med, color=INK, linewidth=1.2, linestyle="--")
    ax2.text(med + 0.012, ax2.get_ylim()[1] * 0.92, f"median {med:.2f}$\\times$",
             fontsize=8.5, color=INK, va="top")
    ax2.set_xlabel("P90 : P10 ratio")
    ax2.set_ylabel("Plants")
    ax2.set_title("(b) Width of the band per site", fontsize=10, color=INK)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9, colors=INK)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200)

    print(f"wrote {OUT}")
    print(f"  fleet totals  P10 {p10.sum()/1e6:.2f}  P50 {p50.sum()/1e6:.2f}  "
          f"P90 {p90.sum()/1e6:.2f} GWh/yr")
    print(f"  median site   P10 {np.median(p10):,.0f}  P50 {np.median(p50):,.0f}  "
          f"P90 {np.median(p90):,.0f} kWh/yr")
    qs = np.percentile(ratio, [10, 50, 90])
    print(f"  P90:P10 ratio p10/p50/p90 = {qs[0]:.3f} / {qs[1]:.3f} / {qs[2]:.3f}, "
          f"max {ratio.max():.3f}")


if __name__ == "__main__":
    main()
