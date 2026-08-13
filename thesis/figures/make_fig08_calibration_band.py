"""Render Figure 8 — the capacity-factor calibration band on portfolio energy.

Each bar is the 1,138-site portfolio energy after substituting a capacity factor
for the Phase 2 implied median of 0.8725. Multipliers are CF / 0.8725, so the
bottom bar is the unmodified physics estimate. Markers show the river-hydro tiers
recomputed from the narrower 0.1-1 MW EHA bucket.

P4-MEASURED-FLOOR (2026-08-06): the two *metered* conduit tiers are now drawn as
well, and the reported band floor is the lower of them. Point Loma is the only
metered treated-wastewater conduit plant in the country, so it carries the floor;
the median of all 115 metered conduit plants sits just above it. Both are shown in
a separate colour from the river-hydro tiers because they are the only rows in this
figure that rest on measured generation rather than on a different plant class or a
design projection. The CF 0.60 tier is labelled *optimistic*, not central, per the
2026-07-25 relabel.

Requires the EHA workbook on SANDISK (same default path as scripts/cf_calibration.py)
and data/raw/ground_truth/ferc_conduit_candidates.parquet for the metered tiers.

Run from the repository root:
    PYTHONPATH=. python3 thesis/figures/make_fig08_calibration_band.py
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
OUT = Path(__file__).resolve().parent / "fig08_calibration_band.png"

sys.path.insert(0, str(ROOT / "scripts"))
from cf_calibration import (  # noqa: E402
    _DEFAULT_CONDUIT_PATH,
    _DEFAULT_EHA_DIR,
    _load_conduit,
    _load_eha_cf,
    bucket_stats,
    calibration_band,
    conduit_cf_stats,
    phase2_viable_cf_stats,
    WWTP_CENTRAL_CF,
)

INK = "#1f2933"
MEASURED_C = "#8c3a5a"   # metered generation — the only measured rows here
FLOOR_C = "#c1553b"
CENTRAL_C = "#3f7d4f"
CEILING_C = "#4a7fb5"
SUB_C = "#6b7280"

# Top to bottom: harshest to ceiling. The two measured tiers lead because the
# lower of them is the reported band floor.
LABELS = [
    "Measured, Point Loma (WWTP)",
    "Measured, all conduit (n=115)",
    "River-hydro p25",
    "River-hydro p50",
    "River-hydro p75",
    "Optimistic, Conduit 3 proj.",
    "Physics ceiling, Phase 2",
]


def main() -> None:
    cf_df = _load_eha_cf(_DEFAULT_EHA_DIR)
    b5 = bucket_stats(cf_df, max_mw=5.0)
    b1 = bucket_stats(cf_df, max_mw=1.0)

    p2 = pl.read_parquet(ROOT / "data/processed/phase2/energy_yield_estimates.parquet")
    p4 = pl.read_parquet(ROOT / "data/processed/phase4/financial_scorecards.parquet")
    stats = phase2_viable_cf_stats(p2, p4)

    band5 = calibration_band(stats["headline_gwh"], stats["cf_p50"], b5, WWTP_CENTRAL_CF)
    band1 = calibration_band(stats["headline_gwh"], stats["cf_p50"], b1, WWTP_CENTRAL_CF)

    # ── Measured conduit tiers (P4-MEASURED-FLOOR) ────────────────────────────
    # Derived from metered EIA-923 generation, not from the EHA river fleet.
    conduit = _load_conduit(_DEFAULT_CONDUIT_PATH)
    cstats = conduit_cf_stats(conduit)
    cf_all = cstats["all"]["p50"]

    pl_row = conduit.filter(pl.col("site_name") == "Point Loma")
    if pl_row.height != 1:
        raise RuntimeError(
            f"expected exactly 1 Point Loma row in {_DEFAULT_CONDUIT_PATH}, "
            f"got {pl_row.height} — the band floor cannot be drawn from an "
            "ambiguous source"
        )
    cf_pl = float(pl_row["annual_energy_kwh"][0]) / (
        float(pl_row["capacity_kw"][0]) * 8_760.0)

    headline, cf_ref = stats["headline_gwh"], stats["cf_p50"]

    def _measured(tier: str, cf: float) -> dict:
        mult = cf / cf_ref
        return {"tier": tier, "cf": cf, "multiplier": mult, "gwh": headline * mult}

    measured = [
        _measured("Measured, Point Loma", cf_pl),
        _measured("Measured, all conduit", cf_all),
    ]
    rows = measured + band5           # harshest first, ceiling last

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11})
    fig, ax = plt.subplots(figsize=(9.0, 4.3))

    ypos = np.arange(len(rows))[::-1]
    colours = [MEASURED_C, MEASURED_C, FLOOR_C, FLOOR_C, FLOOR_C, CENTRAL_C, CEILING_C]
    for y, row, colour in zip(ypos, rows, colours):
        ax.barh(y, row["gwh"], color=colour, height=0.58, alpha=0.9)
        ax.text(row["gwh"] + 6, y, f"{row['gwh']:,.1f} GWh/yr", fontsize=9,
                color=INK, va="center")
        ax.text(6, y, f"CF {row['cf']:.3f}  ($\\times${row['multiplier']:.3f})",
                fontsize=8.5, color="white", va="center")

    # The 0.1-1 MW markers belong only to the three river-hydro rows.
    for y, row in zip(ypos[2:5], band1[:3]):
        ax.plot([row["gwh"]], [y - 0.36], marker="d", color=SUB_C, markersize=6)
    ax.plot([], [], marker="d", color=SUB_C, linestyle="none", markersize=6,
            label=f"river-hydro tier from the 0.1\u20131 MW bucket ({b1['n_plants']} plants)")

    # Mark the reported band explicitly so the figure cannot be read as endorsing
    # the ceiling.
    floor_gwh, top_gwh = rows[0]["gwh"], band5[3]["gwh"]
    ax.axvline(floor_gwh, color=MEASURED_C, linestyle=":", linewidth=1.2, alpha=0.8)
    ax.axvline(top_gwh, color=CENTRAL_C, linestyle=":", linewidth=1.2, alpha=0.8)
    ax.plot([], [], linestyle=":", color=INK, linewidth=1.2,
            label=f"reported band {floor_gwh:,.1f}\u2013{top_gwh:,.1f} GWh/yr")

    ax.set_yticks(ypos)
    ax.set_yticklabels(LABELS, fontsize=9)
    ax.set_xlim(0, 470)
    ax.set_xlabel("Portfolio energy across the 1,138 project-viable sites (GWh/yr)")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9, colors=INK)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200)

    print(f"wrote {OUT}")
    print(f"headline {stats['headline_gwh']} GWh/yr from {stats['n_viable_sites']} sites, "
          f"implied CF median {stats['cf_p50']}")
    print(f"0.1-5 MW bucket: {b5['n_plants']} plants / {b5['n_plant_years']:,} plant-years")
    print(f"0.1-1 MW bucket: {b1['n_plants']} plants / {b1['n_plant_years']:,} plant-years")
    print(f"metered conduit: n={cstats['all']['n']}, median CF {cf_all:.4f}")
    for row in measured:
        print(f"  {row['tier']:28s} cf={row['cf']:.4f} x{row['multiplier']:.3f} "
              f"= {row['gwh']:7.1f} GWh")
    print(f"  reported band: {rows[0]['gwh']:.1f}--{band5[3]['gwh']:.1f} GWh/yr")
    for row5, row1 in zip(band5, band1):
        print(f"  {row5['tier']:28s} 0.1-5MW cf={row5['cf']:.4f} x{row5['multiplier']:.3f} "
              f"= {row5['gwh']:7.1f} GWh | 0.1-1MW cf={row1['cf']:.4f} x{row1['multiplier']:.3f} "
              f"= {row1['gwh']:7.1f} GWh")


if __name__ == "__main__":
    main()
