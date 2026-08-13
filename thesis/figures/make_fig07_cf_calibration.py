"""Render Figure 7 — Phase 2 implied capacity factor against real small-hydro CF.

Left panel:  density of the empirical annual capacity factor for EHA plants in the
             0.1-5 MW bucket (9,798 plant-years) against the Phase 2 implied
             capacity_factor_p50 of the 1,138 project-viable WWTP sites, with the
             LucidPipe Portland measured CF and the 0.60 central anchor marked.
Right panel: p10-p90 spread with the median marked, for the three EHA buckets and
             the WOWERS implied distribution.

Requires the EHA workbook on SANDISK (same default path as scripts/cf_calibration.py).

Run from the repository root:
    PYTHONPATH=. python3 thesis/figures/make_fig07_cf_calibration.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "fig07_cf_calibration.png"

# reuse the calibration script's own loaders so the figure cannot drift from it
sys.path.insert(0, str(ROOT / "scripts"))
from cf_calibration import (  # noqa: E402
    _DEFAULT_EHA_DIR,
    _load_eha_cf,
    WWTP_CENTRAL_CF,
)

INK = "#1f2933"
EHA_C = "#c1553b"
WOW_C = "#4a7fb5"
ANCHOR_C = "#3f7d4f"
LUCID_CF = 1_100_000 / (200 * 8_760)


def main() -> None:
    cf_df = _load_eha_cf(_DEFAULT_EHA_DIR)
    eha5 = cf_df.filter(pl.col("Capacity_MW") <= 5.0)["cf_calc"].to_numpy()
    eha1 = cf_df.filter(pl.col("Capacity_MW") <= 1.0)["cf_calc"].to_numpy()
    eha5r = cf_df.filter((pl.col("Capacity_MW") <= 5.0) & (pl.col("Year") >= 2013))["cf_calc"].to_numpy()

    p2 = pl.read_parquet(ROOT / "data/processed/phase2/energy_yield_estimates.parquet")
    p4 = pl.read_parquet(ROOT / "data/processed/phase4/financial_scorecards.parquet")
    viable = p4.filter(pl.col("project_viable")).select("npdes_id")
    wow = p2.join(viable, on="npdes_id", how="inner")["capacity_factor_p50"].drop_nulls().to_numpy()

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6),
                                   gridspec_kw={"width_ratios": [1.5, 1.0]})

    bins = np.linspace(0, 1.0, 51)
    ax1.hist(eha5, bins=bins, density=True, color=EHA_C, alpha=0.75,
             label=f"EHA 0.1--5 MW ({len(eha5):,} plant-years)")
    ax1.hist(wow, bins=bins, density=True, color=WOW_C, alpha=0.85,
             label=f"WOWERS implied ({len(wow):,} viable sites)")
    ax1.axvline(LUCID_CF, color=ANCHOR_C, linewidth=1.3, linestyle="--")
    ax1.axvline(WWTP_CENTRAL_CF, color=ANCHOR_C, linewidth=1.3, linestyle=":")
    ymax = ax1.get_ylim()[1]
    ax1.text(LUCID_CF + 0.012, ymax * 0.92, f"LucidPipe {LUCID_CF:.3f}",
             fontsize=8.5, color=ANCHOR_C, rotation=90, va="top")
    ax1.text(WWTP_CENTRAL_CF - 0.045, ymax * 0.92, f"anchor {WWTP_CENTRAL_CF:.2f}",
             fontsize=8.5, color=ANCHOR_C, rotation=90, va="top")
    ax1.set_xlim(0, 1.0)
    ax1.set_xlabel("Annual capacity factor")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=8, frameon=False, loc="upper left")
    ax1.set_title("(a) Measured river-scale CF vs the modelled CF", fontsize=10, color=INK)

    groups = [
        (f"EHA 0.1--1 MW\n({cf_df.filter(pl.col('Capacity_MW') <= 1.0)['EHA_PtID'].n_unique()} plants)", eha1),
        (f"EHA 0.1--5 MW\n2013--2022", eha5r),
        (f"EHA 0.1--5 MW\n({cf_df.filter(pl.col('Capacity_MW') <= 5.0)['EHA_PtID'].n_unique()} plants)", eha5),
        (f"WOWERS implied\n({len(wow):,} sites)", wow),
    ]
    ypos = np.arange(len(groups))
    for y, (label, vals) in zip(ypos, groups):
        p10, p50, p90 = np.percentile(vals, [10, 50, 90])
        colour = WOW_C if label.startswith("WOWERS") else EHA_C
        ax2.plot([p10, p90], [y, y], color=colour, linewidth=6, solid_capstyle="butt", alpha=0.55)
        ax2.plot([p50], [y], marker="|", color=INK, markersize=14, markeredgewidth=1.6)
        ax2.text(p90 + 0.02, y, f"{p50:.3f}", fontsize=8.5, color=INK, va="center")
    ax2.axvline(WWTP_CENTRAL_CF, color=ANCHOR_C, linewidth=1.2, linestyle=":")
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([g[0] for g in groups], fontsize=8)
    ax2.set_xlim(0, 1.18)
    ax2.set_xlabel("Annual capacity factor")
    ax2.set_title("(b) p10--p90 spread, median marked", fontsize=10, color=INK)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9, colors=INK)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200)

    print(f"wrote {OUT}")
    for label, vals in groups:
        p10, p25, p50, p75, p90 = np.percentile(vals, [10, 25, 50, 75, 90])
        flat = label.replace("\n", " ")
        print(f"  {flat:38s} n={len(vals):6,}  p10={p10:.4f} p25={p25:.4f} "
              f"p50={p50:.4f} p75={p75:.4f} p90={p90:.4f}")
    print(f"  LucidPipe CF = {LUCID_CF:.4f}; central anchor = {WWTP_CENTRAL_CF:.2f}")


if __name__ == "__main__":
    main()
