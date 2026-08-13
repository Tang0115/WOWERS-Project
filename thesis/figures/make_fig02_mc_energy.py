"""Render Figure 2 — Monte-Carlo energy distribution (Phase 2).

Left panel:  the 10,000-sample annual-energy distribution for one representative
             facility (PA0026280, Lewistown STP), re-drawn with the production
             site-keyed seed so the figure is byte-reproducible.
Right panel: the fleet distribution of per-site P50 annual energy for the 5,464
             facilities retained by Phase 2.

Run from the repository root:
    PYTHONPATH=. python3 thesis/figures/make_fig02_mc_energy.py
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

from src.phase2.energy_physics import run_monte_carlo  # noqa: E402
from src.phase2.head_assumptions import (  # noqa: E402
    classify_archetype,
    get_head_distribution,
)
from src.phase2.monte_carlo import _site_seed_sequence  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "fig02_mc_energy.png"

SITE = "PA0026280"          # Lewistown STP, PA — median-energy retained site
BASE_SEED = 42              # production default (src/phase2/run.py --seed)
N_ITER = 10_000

INK = "#1f2933"
BAR = "#4a7fb5"
ACCENT = "#c1553b"


def main() -> None:
    p1 = pl.read_parquet(ROOT / "data/processed/phase1/ranked_candidates.parquet")
    p2 = pl.read_parquet(ROOT / "data/processed/phase2/energy_yield_estimates.parquet")

    row = p1.filter(pl.col("npdes_id") == SITE).to_dicts()[0]
    fdc = np.asarray(row["flow_duration_curve"], dtype=np.float64)
    archetype = classify_archetype(row["design_flow_mgd"])
    head = get_head_distribution(archetype)

    rng = np.random.default_rng(_site_seed_sequence(BASE_SEED, SITE))
    h = rng.triangular(head.low_m, head.mode_m, head.high_m, N_ITER)
    eta = rng.triangular(0.70, 0.82, 0.90, N_ITER)
    avail = rng.triangular(0.90, 0.95, 0.98, N_ITER)

    from src.phase2.energy_physics import (
        GRAVITY,
        HOURS_PER_YEAR,
        MGD_TO_M3S,
        RHO,
        _FDC_EXCEEDANCES_ARR,
    )

    n = min(len(fdc), len(_FDC_EXCEEDANCES_ARR))
    q = fdc[:n] * MGD_TO_M3S
    exc = _FDC_EXCEEDANCES_ARR[:n]
    power = eta[:, None] * RHO * GRAVITY * q[None, :] * h[:, None] / 1_000.0
    energies = np.trapezoid(power, exc, axis=1) * HOURS_PER_YEAR * avail

    mc = run_monte_carlo(
        fdc_flows_mgd=fdc,
        head_low_m=head.low_m,
        head_mode_m=head.mode_m,
        head_high_m=head.high_m,
        n_iterations=N_ITER,
        rng=np.random.default_rng(_site_seed_sequence(BASE_SEED, SITE)),
    )

    retained = p2.filter(~pl.col("excluded"))
    fleet = retained["energy_p50_kwh_yr"].to_numpy()
    fleet = fleet[fleet > 0]

    # Sized for a 6.0 in text block (12pt, 1.5 in binding margin) so axis text
    # stays legible in print rather than shrinking with \includegraphics.
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.7))

    ax1.hist(energies / 1_000.0, bins=60, color=BAR, edgecolor="white", linewidth=0.3)
    for pct, style in ((10, ":"), (50, "-"), (90, ":")):
        v = np.percentile(energies, pct) / 1_000.0
        ax1.axvline(v, color=ACCENT, linestyle=style, linewidth=1.6)
        ax1.text(
            v, ax1.get_ylim()[1] * 0.96, f" P{pct}\n {v:,.1f}",
            color=ACCENT, fontsize=9, va="top", ha="left",
        )
    ax1.set_xlabel("Annual energy (MWh/yr)")
    ax1.set_ylabel("Monte-Carlo samples")
    ax1.set_title(
        f"(a) {SITE} — Lewistown STP, {archetype.replace('_', ' ')}\n"
        f"{N_ITER:,} samples, seed {BASE_SEED}",
        fontsize=10, color=INK,
    )

    bins = np.logspace(np.log10(max(fleet.min(), 1.0)), np.log10(fleet.max()), 60)
    ax2.hist(fleet / 1_000.0, bins=bins / 1_000.0, color=BAR, edgecolor="white", linewidth=0.3)
    ax2.set_xscale("log")
    med = np.median(fleet) / 1_000.0
    ax2.axvline(med, color=ACCENT, linestyle="-", linewidth=1.6)
    ax2.text(med, ax2.get_ylim()[1] * 0.96, f" median {med:,.1f} MWh/yr",
             color=ACCENT, fontsize=9, va="top", ha="left")
    ax2.set_xlabel("Per-site P50 annual energy (MWh/yr, log scale)")
    ax2.set_ylabel("Facilities")
    ax2.set_title(
        f"(b) Fleet distribution, {len(fleet):,} facilities retained by Phase 2",
        fontsize=10, color=INK,
    )

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9, colors=INK)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200)

    print(f"wrote {OUT}")
    print(
        "site P10/P50/P90 kWh/yr: "
        f"{mc['energy_p10_kwh_yr']:,.0f} / {mc['energy_p50_kwh_yr']:,.0f} / "
        f"{mc['energy_p90_kwh_yr']:,.0f}"
    )
    print(f"site head P50 {mc['head_m_p50']:.2f} m, power P50 {mc['power_p50_kw']:.3f} kW")
    print(f"fleet median {np.median(fleet):,.0f} kWh/yr, sum {fleet.sum()/1e6:,.2f} GWh/yr")


if __name__ == "__main__":
    main()
