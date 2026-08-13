"""Render Figure 10 — national distribution of scored and project-viable sites.

Left panel:  every scored plant at its permitted location, non-viable in grey and
             project-viable coloured, with marker area scaled by rated power. The
             frame is the continental United States; the 52 scored plants in
             Alaska, Hawaii, Puerto Rico, Guam, the Northern Marianas, American
             Samoa and the Virgin Islands are counted in the caption rather than
             squeezed into the frame.
Right panel: the ten states holding the most project-viable energy.

No basemap is drawn: the site cloud is the map. Nothing in this figure is
interpolated or smoothed, so every dot is one NPDES-permitted facility.

Run from the repository root:
    PYTHONPATH=. python3 thesis/figures/make_fig10_national_map.py
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
OUT = Path(__file__).resolve().parent / "fig10_national_map.png"

INK = "#1f2933"
GREY = "#c3c9d1"
VIABLE = "#4a7fb5"
BAR = "#3f7d4f"

LON = (-125.0, -66.0)
LAT = (24.0, 50.0)


def main() -> None:
    p1 = pl.read_parquet(ROOT / "data/processed/phase1/ranked_candidates.parquet").select(
        ["npdes_id", "latitude", "longitude"])
    p4 = pl.read_parquet(ROOT / "data/processed/phase4/financial_scorecards.parquet").join(
        p1, on="npdes_id", how="left")

    inframe = p4.filter(
        pl.col("longitude").is_between(*LON) & pl.col("latitude").is_between(*LAT))
    off = p4.height - inframe.height
    off_viable = p4.filter(pl.col("project_viable")).height - inframe.filter(
        pl.col("project_viable")).height

    nv = inframe.filter(~pl.col("project_viable"))
    vi = inframe.filter(pl.col("project_viable"))

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6),
                                   gridspec_kw={"width_ratios": [1.75, 1.0]})

    ax1.scatter(nv["longitude"], nv["latitude"], s=3, c=GREY, linewidths=0,
                label=f"scored, not viable ({nv.height:,})")
    sizes = 3 + 40 * np.sqrt(vi["rated_power_kw"].to_numpy() / vi["rated_power_kw"].max())
    ax1.scatter(vi["longitude"], vi["latitude"], s=sizes, c=VIABLE, linewidths=0,
                alpha=0.75, label=f"project-viable ({vi.height:,})")
    ax1.set_xlim(*LON)
    ax1.set_ylim(*LAT)
    ax1.set_aspect(1 / np.cos(np.deg2rad(37)))
    ax1.set_xticks([])
    ax1.set_yticks([])
    for s in ax1.spines.values():
        s.set_visible(False)
    ax1.legend(fontsize=8, frameon=False, loc="lower left", scatterpoints=1)
    ax1.set_title("(a) Scored plants, marker area by rated power", fontsize=10, color=INK)

    st = (p4.filter(pl.col("project_viable"))
          .group_by("state_code")
          .agg([(pl.col("annual_energy_kwh").sum() / 1e6).alias("gwh"), pl.len().alias("n")])
          .sort("gwh", descending=True)
          .head(10))
    y = np.arange(st.height)[::-1]
    ax2.barh(y, st["gwh"], color=BAR, height=0.62, alpha=0.9)
    for yy, row in zip(y, st.to_dicts()):
        ax2.text(row["gwh"] + 1.2, yy, f"{row['gwh']:.1f} ({row['n']})",
                 fontsize=8, color=INK, va="center")
    ax2.set_yticks(y)
    ax2.set_yticklabels(st["state_code"], fontsize=9)
    ax2.set_xlim(0, float(st["gwh"].max()) * 1.42)
    ax2.set_xlabel("Project-viable energy (GWh/yr)")
    ax2.set_title("(b) Top ten states, site count in brackets", fontsize=10, color=INK)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(labelsize=9, colors=INK)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200)

    print(f"wrote {OUT}")
    print(f"  in frame: {inframe.height:,} of {p4.height:,} scored; "
          f"{vi.height:,} viable in frame, {off} scored and {off_viable} viable outside it")
    print("  top states:", st.to_dicts())


if __name__ == "__main__":
    main()
