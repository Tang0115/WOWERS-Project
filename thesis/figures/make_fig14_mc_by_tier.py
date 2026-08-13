"""Render Figure 14 — Monte-Carlo energy spread by permitting tier (Appendix C).

The 1,138 project-viable sites split three ways by the permitting pathway their
head and capacity imply. This figure carries the Phase 2 P10/P50/P90 spread into
those three groups, so the appendix reports the distribution per tier rather than
only for the fleet as a whole.

Left panel:  per-tier P10-P50-P90 of annual energy, per site, on a log axis.
Right panel: how the tier's share of portfolio energy compares with its share of
             site count, which is where the small-count / large-energy tiers show up.

Run from the repository root:
    PYTHONPATH=. python3 thesis/figures/make_fig14_mc_by_tier.py
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
OUT = Path(__file__).resolve().parent / "fig14_mc_by_tier.png"

INK = "#1f2933"
BAND = "#9fbcd8"
MED = "#1f4e79"
COUNT_C = "#8fa8bd"
ENERGY_C = "#1f4e79"

TIER_LABEL = {
    "qualified_facility": "Qualified conduit\n(FERC notice of intent)",
    "small_ferc": "Small FERC\n(exemption or license)",
    "full_nepa": "Full NEPA\n(license + review)",
}
TIER_ORDER = ["qualified_facility", "small_ferc", "full_nepa"]


def main() -> None:
    p4 = pl.read_parquet(ROOT / "data/processed/phase4/financial_scorecards.parquet")
    p2 = pl.read_parquet(ROOT / "data/processed/phase2/energy_yield_estimates.parquet")

    viable = (
        p4.filter(pl.col("project_viable"))
        .select(["npdes_id", "permitting_tier", "annual_energy_kwh"])
        .join(
            p2.select(["npdes_id", "energy_p10_kwh_yr", "energy_p50_kwh_yr",
                       "energy_p90_kwh_yr"]),
            on="npdes_id", how="left",
        )
    )

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6),
                                   gridspec_kw={"width_ratios": [1.35, 1.0]})

    rows = []
    for k, tier in enumerate(TIER_ORDER):
        g = viable.filter(pl.col("permitting_tier") == tier)
        p10 = g["energy_p10_kwh_yr"].to_numpy()
        p50 = g["energy_p50_kwh_yr"].to_numpy()
        p90 = g["energy_p90_kwh_yr"].to_numpy()
        q = [np.percentile(p50, 10), np.median(p50), np.percentile(p50, 90)]
        rows.append((tier, g.height, p10, p50, p90, q,
                     float(g["annual_energy_kwh"].sum())))

        y = len(TIER_ORDER) - 1 - k
        ax1.hlines(y, np.median(p10), np.median(p90), color=BAND, linewidth=9,
                   zorder=1)
        ax1.plot([np.median(p50)], [y], marker="o", markersize=6.5, color=MED,
                 zorder=3)
        ax1.hlines(y, q[0], q[2], color=MED, linewidth=1.3, zorder=2)
        ax1.text(np.median(p90) * 1.35, y + 0.20,
                 f"median P50 {np.median(p50)/1000:,.0f} MWh/yr",
                 fontsize=7.6, color=INK, va="center")
        ax1.text(np.median(p90) * 1.35, y - 0.18, f"n = {g.height:,}",
                 fontsize=7.6, color="#55636e", va="center")

    ax1.set_yticks(range(len(TIER_ORDER)))
    ax1.set_yticklabels([TIER_LABEL[t] for t in reversed(TIER_ORDER)], fontsize=8)
    ax1.set_xscale("log")
    ax1.set_xlim(2e3, 6e6)
    ax1.set_ylim(-0.62, len(TIER_ORDER) - 0.38)
    ax1.set_xlabel("Annual energy per site (kWh/yr, log scale)")
    ax1.set_title("(a) Median P10-P90 band and P50 spread by tier",
                  fontsize=10, color=INK)

    tot_n = sum(r[1] for r in rows)
    tot_e = sum(r[6] for r in rows)
    x = np.arange(len(TIER_ORDER))
    w = 0.38
    share_n = [r[1] / tot_n * 100 for r in rows]
    share_e = [r[6] / tot_e * 100 for r in rows]
    ax2.bar(x - w / 2, share_n, w, color=COUNT_C, label="share of sites")
    ax2.bar(x + w / 2, share_e, w, color=ENERGY_C, label="share of energy")
    for xi, (sn, se) in enumerate(zip(share_n, share_e)):
        ax2.text(xi - w / 2, sn + 1.4, f"{sn:.1f}", fontsize=7.4, ha="center",
                 color=INK)
        ax2.text(xi + w / 2, se + 1.4, f"{se:.1f}", fontsize=7.4, ha="center",
                 color=INK)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Qualified\nconduit", "Small\nFERC", "Full\nNEPA"],
                        fontsize=8)
    ax2.set_ylabel("Share of viable portfolio (%)")
    ax2.set_ylim(0, max(max(share_n), max(share_e)) * 1.22)
    ax2.legend(fontsize=8, frameon=False, loc="upper right")
    ax2.set_title("(b) Sites against energy, per tier", fontsize=10, color=INK)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9, colors=INK)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200)

    print(f"wrote {OUT}")
    print(f"  viable sites {tot_n:,}   portfolio energy {tot_e/1e6:.2f} GWh/yr")
    for tier, n, p10, p50, p90, q, e in rows:
        print(f"  {tier:>19}  n={n:>5,}  "
              f"median P10/P50/P90 = {np.median(p10):>10,.0f} / "
              f"{np.median(p50):>10,.0f} / {np.median(p90):>10,.0f} kWh/yr  "
              f"energy {e/1e6:>6.2f} GWh/yr ({e/tot_e*100:>4.1f}%)")


if __name__ == "__main__":
    main()
