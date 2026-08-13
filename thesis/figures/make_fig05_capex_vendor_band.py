"""Render Figure 5 — equipment CapEx per kW against vendor-published bands.

One panel per machine type: the power-law equipment cost actually applied to
every scored site (points), the unclamped power law A*kW^B (dashed), and the
vendor-published $/kW envelope aggregated from
data/turbines/turbine_manufacturers.csv (shaded).  This is the F4-VENDORBAND
cross-check that reports 0 of 3,778 sites priced outside the vendor envelope.

Run from the repository root:
    PYTHONPATH=. python3 thesis/figures/make_fig05_capex_vendor_band.py
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
OUT = Path(__file__).resolve().parent / "fig05_capex_vendor_band.png"

# config/settings.yaml -> cost_model.types (A, B, min_per_kw, max_per_kw)
POWER_LAW = {
    "Crossflow":        (7_500.0, -0.28,   500.0,  6_000.0),
    "Francis":          (8_500.0, -0.32, 1_800.0,  9_000.0),
    "Kaplan":           (9_500.0, -0.35,   800.0, 10_000.0),
    "in_conduit_micro": (20_283.0, -0.181, 2_000.0, 15_000.0),
}
ORDER = ["Crossflow", "Francis", "Kaplan", "in_conduit_micro"]

INK = "#1f2933"
COLORS = {
    "Crossflow": "#4a7fb5",
    "Francis": "#c1553b",
    "Kaplan": "#3f7d58",
    "in_conduit_micro": "#8a6bbf",
}


def main() -> None:
    f = pl.read_parquet(ROOT / "data/processed/phase4/financial_scorecards.parquet")

    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "axes.titlesize": 12})
    fig, axes = plt.subplots(1, 4, figsize=(9.0, 3.3), sharey=True)

    for ax, name in zip(axes, ORDER):
        s = f.filter(pl.col("turbine_type") == name)
        kw = s["rated_power_kw"].to_numpy()
        per_kw = s["capex_per_kw"].to_numpy()
        lo = float(s["vendor_capex_per_kw_low"][0])
        hi = float(s["vendor_capex_per_kw_high"][0])
        a, b, clamp_lo, clamp_hi = POWER_LAW[name]

        ax.axhspan(lo, hi, color=COLORS[name], alpha=0.14, linewidth=0)
        grid = np.logspace(np.log10(max(kw.min(), 0.5)), np.log10(kw.max()), 200)
        ax.plot(grid, a * grid ** b, color=INK, linestyle="--", linewidth=1.0,
                alpha=0.8, zorder=2)
        ax.scatter(kw, per_kw, s=4, alpha=0.4, linewidths=0,
                   color=COLORS[name], zorder=3)
        for y in (clamp_lo, clamp_hi):
            ax.axhline(y, color=INK, linestyle=":", linewidth=0.8, alpha=0.6)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(300, 40_000)
        ax.set_title(f"{name.replace('_', ' ')}\n$n$ = {s.height:,}", fontsize=11.5, color=INK)
        ax.set_xlabel("Rated power (kW)", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10, colors=INK)

    axes[0].set_ylabel("Equipment cost (USD/kW)", fontsize=11.5)
    fig.text(0.5, 0.005,
             r"Shaded band: vendor-published \$/kW envelope   ·   dashed: unclamped "
             r"$A\,\mathrm{kW}^{B}$   ·   dotted: per-type clamps",
             ha="center", fontsize=10, color=INK)

    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(OUT, dpi=200)

    print(f"wrote {OUT}")
    print("outside vendor band:", int(f["capex_outside_vendor_band"].sum()), "/", f.height)
    for name in ORDER:
        s = f.filter(pl.col("turbine_type") == name)
        print(f"  {name:18s} n={s.height:5,} model $/kW min {s['capex_per_kw'].min():8,.0f} "
              f"med {s['capex_per_kw'].median():8,.0f} max {s['capex_per_kw'].max():8,.0f} "
              f"| vendor {s['vendor_capex_per_kw_low'][0]:,.0f}-{s['vendor_capex_per_kw_high'][0]:,.0f}")


if __name__ == "__main__":
    main()
