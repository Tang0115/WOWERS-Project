"""Render Figure 9 — the site exclusion funnel and where the energy goes.

Left panel:  the five funnel stages by site count, with the drop at each stage
             labelled by its exclusion class (scale threshold, data gap, physics
             floor, economics). The Phase 2 drop is split by the reason string the
             phase wrote, because the 0.5 MGD floor is a scope decision rather
             than a missing record.
Right panel: the same funnel in energy terms, from the Phase 2 fleet estimate of
             699.18 GWh/yr down to the 409.17 GWh/yr carried by the viable cohort,
             separating losses caused by dropping sites from the loss caused by
             re-estimating energy with real head and a turbine efficiency curve.

Run from the repository root:
    PYTHONPATH=. python3 thesis/figures/make_fig09_exclusion_funnel.py
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
OUT = Path(__file__).resolve().parent / "fig09_exclusion_funnel.png"

INK = "#1f2933"
KEEP = "#4a7fb5"
GAP = "#c1553b"
PHYS = "#d8a13a"
ECON = "#3f7d4f"
MODEL = "#9aa5b1"


def main() -> None:
    p2 = pl.read_parquet(ROOT / "data/processed/phase2/energy_yield_estimates.parquet")
    p3 = pl.read_parquet(ROOT / "data/processed/phase3/turbine_sizing.parquet")
    p4 = pl.read_parquet(ROOT / "data/processed/phase4/financial_scorecards.parquet")

    n1 = p2.height
    ret = p2.filter(~pl.col("excluded"))
    n2 = ret.height
    hv = set(p3.filter(pl.col("head_valid"))["npdes_id"].to_list())
    tv = set(p3.filter(pl.col("turbine_viable"))["npdes_id"].to_list())
    vi = set(p4.filter(pl.col("project_viable"))["npdes_id"].to_list())
    n3, n4, n5 = len(hv), len(tv), len(vi)

    def p2_sum(ids: set[str] | None) -> float:
        sub = ret if ids is None else ret.filter(pl.col("npdes_id").is_in(list(ids)))
        return float(sub["energy_p50_kwh_yr"].sum()) / 1e6

    e_all = p2_sum(None)
    e_hv = p2_sum(hv)
    e_tv = p2_sum(tv)
    e_p3 = float(p3.filter(pl.col("turbine_viable"))["annual_energy_mwh"].sum()) / 1e3
    e_vi = float(p4.filter(pl.col("project_viable"))["annual_energy_kwh"].sum()) / 1e6

    # Phase 2 drops split by the reason string the phase itself wrote, because
    # "below the 0.5 MGD threshold" is a scope decision and not a data gap.
    reasons = dict(
        p2.filter(pl.col("excluded"))
        .group_by("exclusion_reason")
        .agg(pl.len().alias("n"))
        .select(["exclusion_reason", "n"])
        .iter_rows()
    )
    n_small = reasons["small_potw"]
    n_nodata = n1 - n2 - n_small

    stages = [
        ("Screened POTWs", n1, None, None),
        ("Flow-valid (Phase 2)", n2, [(n_small, "scale threshold"), (n_nodata, "data gap")], None),
        ("Head-valid (Phase 3)", n3, [(n2 - n3, "data gap")], None),
        ("Turbine-viable (Phase 3)", n4, [(n3 - n4, "physics floor")], None),
        ("Project-viable (Phase 4)", n5, [(n4 - n5, "economics")], None),
    ]
    dropcolour = {"data gap": GAP, "physics floor": PHYS, "economics": ECON,
                  "scale threshold": "#8a6fa8"}

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.8),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})

    ypos = np.arange(len(stages))[::-1]
    for y, (label, kept, drops, _unused) in zip(ypos, stages):
        ax1.barh(y, kept, color=KEEP, height=0.56)
        if not drops:
            ax1.text(kept + 250, y, f"{kept:,} screened", fontsize=8.5, color=INK,
                     va="center")
            continue
        left = kept
        for n, kind in drops:
            ax1.barh(y, n, left=left, color=dropcolour[kind], height=0.56, alpha=0.85)
            left += n
        ax1.text(left + 250, y + 0.16, f"{kept:,} kept", fontsize=8.5, color=INK,
                 va="center")
        offsets = [-0.20, -0.48]
        for (n, kind), dy in zip(drops, offsets):
            ax1.text(left + 250, y + dy, f"$-${n:,} {kind}", fontsize=8,
                     color=dropcolour[kind], va="center")
    ax1.set_yticks(ypos)
    ax1.set_yticklabels([s[0] for s in stages], fontsize=8.5)
    ax1.set_xlim(0, 21_500)
    ax1.set_xlabel("Facilities")
    ax1.set_title("(a) Sites retained and dropped per stage", fontsize=10, color=INK)

    bars = [
        ("Phase 2 fleet estimate", e_all, KEEP),
        ("less head-invalid sites", e_hv, GAP),
        ("less sub-1 kW sites", e_tv, PHYS),
        ("re-estimated with head\nand turbine curve", e_p3, MODEL),
        ("less uneconomic sites", e_vi, ECON),
    ]
    y2 = np.arange(len(bars))[::-1]
    for y, (label, val, colour) in zip(y2, bars):
        ax2.barh(y, val, color=colour, height=0.56, alpha=0.9)
        ax2.text(val + 8, y, f"{val:,.1f}", fontsize=9, color=INK, va="center")
    ax2.set_yticks(y2)
    ax2.set_yticklabels([b[0] for b in bars], fontsize=8)
    ax2.set_xlim(0, 830)
    ax2.set_xlabel("Energy (GWh/yr)")
    ax2.set_title("(b) Where the energy goes", fontsize=10, color=INK)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9, colors=INK)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200)

    print(f"wrote {OUT}")
    for label, kept, drops, _ in stages:
        detail = ", ".join(f"{n:,} {kind}" for n, kind in (drops or []))
        print(f"  {label:26s} kept={kept:6,}  dropped: {detail}")
    print(f"  energy: {e_all:.2f} -> {e_hv:.2f} -> {e_tv:.2f} -> {e_p3:.2f} -> {e_vi:.2f} GWh/yr")


if __name__ == "__main__":
    main()
