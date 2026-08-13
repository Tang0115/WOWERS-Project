"""Render Figure 6 — Phase 4 sensitivity tornado on portfolio NPV.

Left panel:  portfolio NPV swing for the 848 project-viable sites that carry a
             DMR flow duration curve, so head and flow are perturbed through the
             physical model (re-optimised rating, re-integrated FDC) rather than
             by scaling energy.
Right panel: which input dominates NPV per site, across all 3,778 scored sites.

Run from the repository root:
    PYTHONPATH=. python3 thesis/figures/make_fig06_sensitivity_tornado.py
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
OUT = Path(__file__).resolve().parent / "fig06_sensitivity_tornado.png"

M = 1e6
INK = "#1f2933"
LOWC = "#c1553b"
HIGHC = "#4a7fb5"

SWEEPS = [
    ("Net head  ×0.50 / ×1.50", "sensitivity_head_npv_low", "sensitivity_head_npv_high"),
    ("Electricity rate  ×0.70 / ×1.30", "sensitivity_rate_npv_low", "sensitivity_rate_npv_high"),
    ("Flow  ×0.80 / ×1.20", "sensitivity_flow_npv_low", "sensitivity_flow_npv_high"),
]


def main() -> None:
    f = pl.read_parquet(ROOT / "data/processed/phase4/financial_scorecards.parquet")
    phys = f.filter(pl.col("project_viable") & (pl.col("dominant_sensitivity") != "energy_uncertain"))
    base = float(phys["npv_usd"].sum()) / M

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4),
                                   gridspec_kw={"width_ratios": [1.55, 1.0]})

    ypos = np.arange(len(SWEEPS))[::-1]
    for y, (label, lo_col, hi_col) in zip(ypos, SWEEPS):
        lo = float(phys[lo_col].sum()) / M
        hi = float(phys[hi_col].sum()) / M
        ax1.barh(y, lo - base, left=base, color=LOWC, height=0.5)
        ax1.barh(y, hi - base, left=base, color=HIGHC, height=0.5)
        ax1.text(lo - 4, y, f"{lo:,.1f}", ha="right", va="center", fontsize=8.5, color=LOWC)
        ax1.text(hi + 4, y, f"{hi:,.1f}", ha="left", va="center", fontsize=8.5, color=HIGHC)
    ax1.axvline(base, color=INK, linewidth=1.2)
    ax1.set_ylim(-0.55, len(SWEEPS) - 0.5)
    ax1.text(base + 6, len(SWEEPS) - 0.62, f"base \\${base:,.1f}M", fontsize=8.5,
             color=INK, ha="left", va="center")
    ax1.set_yticks(ypos)
    ax1.set_yticklabels([s[0] for s in SWEEPS], fontsize=9)
    ax1.set_xlim(-70, 300)
    ax1.set_xlabel("Portfolio NPV (USD millions)")
    ax1.set_title(f"(a) NPV swing, {phys.height:,} viable sites with a measured FDC",
                  fontsize=10, color=INK)

    counts = (
        f.group_by("dominant_sensitivity")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    names = {"head": "Net head", "rate": "Electricity rate", "flow": "Flow",
             "energy_uncertain": "No FDC\n(head/flow not\nseparable)"}
    labels = [names.get(r["dominant_sensitivity"], r["dominant_sensitivity"]) for r in counts.to_dicts()]
    vals = [r["n"] for r in counts.to_dicts()]
    y2 = np.arange(len(vals))[::-1]
    ax2.barh(y2, vals, color=[HIGHC if l != "No FDC\n(head/flow not\nseparable)" else "#9aa5b1"
                              for l in labels], height=0.6)
    for y, val in zip(y2, vals):
        ax2.text(val + 45, y, f"{val:,}", va="center", fontsize=9, color=INK)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(labels, fontsize=8.5)
    ax2.set_xlim(0, max(vals) * 1.28)
    ax2.set_xlabel("Facilities")
    ax2.set_title("(b) Dominant input per site", fontsize=10, color=INK)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9, colors=INK)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200)

    print(f"wrote {OUT}")
    print(f"physical-tornado cohort n={phys.height:,} base NPV ${base:,.2f}M")
    for label, lo_col, hi_col in SWEEPS:
        lo = float(phys[lo_col].sum()) / M
        hi = float(phys[hi_col].sum()) / M
        print(f"  {label:34s} ${lo:9,.2f}M .. ${hi:9,.2f}M  (swing ${hi-lo:,.2f}M)")
    print("dominant counts:", counts.to_dicts())


if __name__ == "__main__":
    main()
