"""P4-TIER-LADDER — read-only energy-tier x discount-rate what-if analysis.

Reads ``data/processed/phase4/financial_scorecards.parquet`` and re-scores every
turbine-viable row through the real Phase 4 financial functions
(``src.phase4.financials``) under a grid of:

  * six CF-calibration energy tiers (ceiling -> measured Point Loma), and
  * three discount rates (6 % commercial / 3.5 % municipal / 0 % CWSRF), and
  * two capital-subsidy levels (0 % / 50 % grant).

Purpose
-------
``thesis/business.md`` section 2.4 currently carries a *reconstruction* of the
Phase 4 NPV formula rather than pipeline output.  This script replaces that
reconstruction: it calls ``compute_scorecard`` — the same function Phase 4
itself calls — so the Ch3 business numbers rest on the pipeline's own model.

How it works
------------
For each energy tier multiplier ``m``, discount rate ``r`` and grant fraction
``g``, each row is re-scored as::

    energy'  = annual_energy_kwh x m
    revenue' = energy' x (state_elec_rate + rec_value)
    capex'   = total_capex_usd x (1 - g)
    opex'    = annual_opex_usd            (unchanged — O&M is CapEx-driven,
                                           not energy-driven)

``compute_scorecard`` then derives fresh NPV / IRR / payback / LCOE and
re-applies the unmodified viability gate (NPV > 0 AND payback <= 20 yr AND real
IRR).  Tier multipliers for the three published tiers are read from
``config/settings.yaml`` (``phase4.cf_calibration``); the two measured-conduit
tiers are derived from metered EIA-923 capacity factors (see ``_MEASURED_CFS``).

This script is STRICTLY read-only:
  - does NOT write any parquet file or checkpoint
  - does NOT modify settings.yaml or any source file
  - does NOT change any column or viability flag in the parquet

Usage:
    python scripts/tier_ladder_whatif.py
    python scripts/tier_ladder_whatif.py --out TIER_LADDER_REPORT.md
    python scripts/tier_ladder_whatif.py --parquet path/to/financial_scorecards.parquet
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import polars as pl

# ── Locate root and import financial helpers ──────────────────────────────────

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.common import config                      # noqa: E402
from src.phase4.financials import (                # noqa: E402
    DEGRADATION_RATE,
    PROJECT_YEARS,
    REC_PER_KWH,
    compute_scorecard,
)

_DEFAULT_PARQUET = _ROOT / "data" / "processed" / "phase4" / "financial_scorecards.parquet"

# ── Tier ladder ───────────────────────────────────────────────────────────────

# Phase 2's implied fleet-median capacity factor.  Every tier multiplier is
# tier_CF / this value, so the ceiling tier has multiplier 1.0 by construction.
# Source: CF_CALIBRATION_REPORT.md section 6 (P5-CF-CALIB, 2026-07-03).
_PHASE2_IMPLIED_CF: float = 0.872

# Metered capacity factors from the 115 EHA Canal/Conduit plants carrying
# Form EIA-923 generation (data/raw/ground_truth/ferc_conduit_candidates.parquet),
# routed into calibration on 2026-07-25.
#
# ``measured_point_loma`` became the reported band floor on 2026-08-06 and now
# lives in config/settings.yaml as ``phase4.cf_calibration.measured_point_loma``;
# the value below is the fallback and must agree with it. ``measured_all_conduit``
# is reported here only — it is not a pipeline column.
_MEASURED_CFS: dict[str, float] = {
    "measured_all_conduit": 0.2439,   # median CF across all 115 metered conduit plants
    "measured_point_loma":  0.1914,   # Point Loma WWTP, 2017 — only metered treated-
                                      # wastewater conduit plant in the U.S.
}

# Discount rates.  6 % is the pipeline default (config financials.discount_rate);
# the two lower rates model public-sector cost of capital.
_DISCOUNT_RATES: tuple[tuple[float, str], ...] = (
    (0.060, "commercial (pipeline default)"),
    (0.0475, "market municipal AA water/sewer revenue bond, 2026"),
    (0.035, "below-market municipal / partial SRF subsidy"),
    (0.000, "CWSRF 0 % loan with principal forgiveness"),
)

_GRANTS: tuple[float, ...] = (0.0, 0.5)

# P2-SEED baseline the ceiling / 6 % / no-grant scenario must reproduce.
_BASELINE = {"viable": 1138, "gwh": 409.17, "npv_musd": 310.13, "median_payback": 9.83}


def tier_ladder() -> list[dict]:
    """Build the energy-tier ladder, coarsest (ceiling) to harshest.

    Returns a list of dicts with keys ``key``, ``label``, ``multiplier``,
    ``cf`` and ``source``.  The three published tiers read their multiplier
    from ``config/settings.yaml``; the ceiling is 1.0 by definition and the two
    measured tiers derive from ``_MEASURED_CFS``.
    """
    calib = config.get("phase4.cf_calibration") or {}
    published = (
        ("ceiling", "Ceiling (Phase 2 as-modeled)", 1.0, "Phase 2 physics model"),
        ("optimistic", "Optimistic (Conduit 3 proj.)",
         float(calib.get("central", 0.688)), "settings.yaml phase4.cf_calibration.central"),
        ("floor_p50", "Floor p50 (EHA river-hydro median)",
         float(calib.get("floor_p50", 0.447)), "settings.yaml phase4.cf_calibration.floor_p50"),
        ("floor_p25", "Floor p25 (EHA river-hydro p25)",
         float(calib.get("floor_p25", 0.291)), "settings.yaml phase4.cf_calibration.floor_p25"),
    )
    rows = [
        {"key": k, "label": lab, "multiplier": m, "cf": m * _PHASE2_IMPLIED_CF, "source": src}
        for k, lab, m, src in published
    ]
    rows.append({
        "key": "measured_all_conduit",
        "label": "Measured all-conduit (EIA-923 median)",
        "multiplier": _MEASURED_CFS["measured_all_conduit"] / _PHASE2_IMPLIED_CF,
        "cf": _MEASURED_CFS["measured_all_conduit"],
        "source": "EIA-923 metered, 115 EHA canal/conduit plants",
    })
    # The band floor is a pipeline tier as of P4-MEASURED-FLOOR, so read its
    # multiplier from config rather than recomputing it here.
    pl_mult = float(calib.get("measured_point_loma",
                              _MEASURED_CFS["measured_point_loma"] / _PHASE2_IMPLIED_CF))
    rows.append({
        "key": "measured_point_loma",
        "label": "Band floor — measured Point Loma (only metered WWTP conduit)",
        "multiplier": pl_mult,
        "cf": pl_mult * _PHASE2_IMPLIED_CF,
        "source": "settings.yaml phase4.cf_calibration.measured_point_loma "
                  "(EIA-923 metered, Point Loma WWTP 2017)",
    })
    return rows


# ── Re-scoring ────────────────────────────────────────────────────────────────

def rescore_row(
    row:           dict,
    multiplier:    float,
    discount_rate: float,
    grant_pct:     float,
) -> dict:
    """Re-score one scorecard row at a given energy tier, rate and grant level.

    Revenue scales with the calibrated energy (same state rate + REC value the
    pipeline used, recovered from the stored columns).  OpEx is unchanged: it is
    a fraction of equipment CapEx, not of output.  The grant reduces the CapEx
    the owner finances, which moves NPV, IRR and payback together.

    Returns the ``compute_scorecard`` dict plus ``rated_power_kw`` and the
    net CapEx actually financed.
    """
    energy_kwh = float(row["annual_energy_kwh"]) * multiplier
    # Mirror phase4/run.py exactly: the state rate is passed WITHOUT the REC
    # (compute_scorecard adds REC_PER_KWH internally when it rebuilds the
    # revenue stream), while annual_revenue_usd is passed WITH it.
    elec_rate = float(row["elec_rate_per_kwh"])
    net_capex = float(row["total_capex_usd"]) * (1.0 - grant_pct)

    scorecard = compute_scorecard(
        annual_energy_kwh=energy_kwh,
        elec_rate_per_kwh=elec_rate,
        annual_opex_usd=float(row["annual_opex_usd"]),
        total_capex_usd=net_capex,
        annual_revenue_usd=energy_kwh * (elec_rate + REC_PER_KWH),
        discount_rate=discount_rate,
        project_years=PROJECT_YEARS,
        degradation_rate=DEGRADATION_RATE,
    )
    return {
        **scorecard,
        "npdes_id":         row["npdes_id"],
        "rated_power_kw":   float(row["rated_power_kw"]),
        "annual_energy_kwh": energy_kwh,
        "net_capex_usd":    net_capex,
    }


def summarize(scored: list[dict]) -> dict:
    """Portfolio metrics over the viable subset of a re-scored scenario."""
    viable = [s for s in scored if s["project_viable"]]
    paybacks = sorted(
        s["payback_years"] for s in viable
        if not math.isinf(s["payback_years"]) and s["payback_years"] < 1e6
    )
    median_payback = (
        paybacks[len(paybacks) // 2] if paybacks else float("nan")
    )
    return {
        "viable":        len(viable),
        "gwh":           sum(s["annual_energy_kwh"] for s in viable) / 1e6,
        "npv_musd":      sum(s["npv_usd"] for s in viable) / 1e6,
        "capex_musd":    sum(s["net_capex_usd"] for s in viable) / 1e6,
        "median_payback": median_payback,
        "ge_100kw":      sum(1 for s in viable if s["rated_power_kw"] >= 100.0),
    }


def run_grid(rows: list[dict]) -> list[dict]:
    """Re-score the full tier x discount-rate x grant grid.

    Returns one summary dict per scenario, in ladder order.
    """
    out: list[dict] = []
    for tier in tier_ladder():
        for rate, rate_label in _DISCOUNT_RATES:
            for grant in _GRANTS:
                t0 = time.time()
                scored = [rescore_row(r, tier["multiplier"], rate, grant) for r in rows]
                summary = summarize(scored)
                out.append({
                    "tier":       tier["label"],
                    "tier_key":   tier["key"],
                    "multiplier": tier["multiplier"],
                    "cf":         tier["cf"],
                    "rate":       rate,
                    "rate_label": rate_label,
                    "grant":      grant,
                    "elapsed_s":  time.time() - t0,
                    **summary,
                })
                print(
                    f"  {tier['key']:>21}  r={rate:.3f}  grant={grant:.0%}  "
                    f"-> {summary['viable']:>5,} viable  "
                    f"${summary['npv_musd']:>7.1f}M  "
                    f"{summary['median_payback']:>5.1f} yr  "
                    f"({time.time() - t0:.1f}s)",
                    file=sys.stderr,
                )
    return out


# ── Report ────────────────────────────────────────────────────────────────────

def _fmt_payback(v: float) -> str:
    return "—" if math.isnan(v) else f"{v:.1f}"


def render_report(grid: list[dict], n_scored: int) -> str:
    """Render the markdown report."""
    L: list[str] = []
    L.append("# WOWERS — Phase 4 Tier-Ladder Re-Run (P4-TIER-LADDER)")
    L.append("")
    L.append(
        f"Generated by `scripts/tier_ladder_whatif.py` over {n_scored:,} turbine-viable "
        "sites. Every row re-scored through `src.phase4.financials.compute_scorecard` — "
        "the same function Phase 4 calls — under the unmodified viability gate "
        "(NPV > 0 AND payback ≤ 20 yr AND real IRR)."
    )
    L.append("")
    L.append("**This is an internal development artifact, not a citable source.**")
    L.append("")

    # Ladder definition
    L.append("## 1. The tier ladder")
    L.append("")
    L.append(
        "Multiplier = tier capacity factor / "
        f"{_PHASE2_IMPLIED_CF} (Phase 2 implied fleet-median CF). Note that the "
        "multiplier and the capacity factor are different numbers — reporting a "
        "multiplier as if it were a CF is the labelling error this table exists to kill."
    )
    L.append("")
    L.append(
        "**Band energy** applies the multiplier to the fixed baseline cohort "
        f"({_BASELINE['gwh']:.2f} GWh/yr over the 1,138 baseline-viable sites) — this is the "
        "column that produces the published calibration band. **Re-scored energy** is the "
        "energy of the sites that remain viable *at that tier*, which is smaller because the "
        "harsher tiers eject sites from the portfolio. The two are different questions and "
        "must not be swapped."
    )
    L.append("")
    L.append(
        "| Tier | Capacity factor | Multiplier | Band energy (GWh/yr) | "
        "Re-scored viable energy (GWh/yr) | Source |"
    )
    L.append("|---|---:|---:|---:|---:|---|")
    for tier in tier_ladder():
        base = next(g for g in grid if g["tier_key"] == tier["key"]
                    and g["rate"] == 0.06 and g["grant"] == 0.0)
        L.append(
            f"| {tier['label']} | {tier['cf']:.4f} | {tier['multiplier']:.4f} | "
            f"{_BASELINE['gwh'] * tier['multiplier']:.1f} | "
            f"{base['gwh']:.1f} | {tier['source']} |"
        )
    L.append("")

    # Scenario blocks
    L.append("## 2. Scenario matrix")
    L.append("")
    for rate, rate_label in _DISCOUNT_RATES:
        for grant in _GRANTS:
            L.append(f"### r = {rate:.1%}, grant = {grant:.0%} — {rate_label}")
            L.append("")
            L.append("| Tier | Viable | Viable GWh/yr | Portfolio NPV | Owner CapEx | Median payback | ≥ 100 kW |")
            L.append("|---|---:|---:|---:|---:|---:|---:|")
            for tier in tier_ladder():
                g = next(x for x in grid if x["tier_key"] == tier["key"]
                         and x["rate"] == rate and x["grant"] == grant)
                L.append(
                    f"| {tier['label']} | {g['viable']:,} | {g['gwh']:.1f} | "
                    f"${g['npv_musd']:.1f}M | ${g['capex_musd']:.1f}M | "
                    f"{_fmt_payback(g['median_payback'])} yr | {g['ge_100kw']:,} |"
                )
            L.append("")

    # Viable-count pivot
    L.append("## 3. Viable-count pivot")
    L.append("")
    header = "| Tier | " + " | ".join(
        f"r={r:.1%} g={g:.0%}" for r, _ in _DISCOUNT_RATES for g in _GRANTS
    ) + " |"
    L.append(header)
    L.append("|---" * (1 + len(_DISCOUNT_RATES) * len(_GRANTS)) + "|")
    for tier in tier_ladder():
        cells = []
        for rate, _ in _DISCOUNT_RATES:
            for grant in _GRANTS:
                g = next(x for x in grid if x["tier_key"] == tier["key"]
                         and x["rate"] == rate and x["grant"] == grant)
                cells.append(f"{g['viable']:,}")
        L.append(f"| {tier['label']} | " + " | ".join(cells) + " |")
    L.append("")

    # Baseline reproduction check
    L.append("## 4. Baseline reproduction check")
    L.append("")
    base = next(g for g in grid if g["tier_key"] == "ceiling"
                and g["rate"] == 0.06 and g["grant"] == 0.0)
    checks = [
        ("Viable sites",   f"{base['viable']:,}",           f"{_BASELINE['viable']:,}"),
        ("Viable GWh/yr",  f"{base['gwh']:.2f}",            f"{_BASELINE['gwh']:.2f}"),
        ("Portfolio NPV",  f"${base['npv_musd']:.2f}M",     f"${_BASELINE['npv_musd']:.2f}M"),
        ("Median payback", f"{base['median_payback']:.2f} yr",
                           f"{_BASELINE['median_payback']:.2f} yr"),
    ]
    L.append("| Metric | This re-run | P2-SEED baseline | Match |")
    L.append("|---|---:|---:|:--:|")
    for name, got, want in checks:
        L.append(f"| {name} | {got} | {want} | {'✅' if got == want else '❌'} |")
    L.append("")
    L.append(
        "The ceiling / 6 % / no-grant scenario is the pipeline's own published "
        "baseline; it must reproduce exactly or the re-scoring harness is wrong."
    )
    L.append("")
    return "\n".join(L)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(parquet_path: Path, out_path: Path | None) -> None:
    if not parquet_path.exists():
        print(f"ERROR: parquet not found at {parquet_path}", file=sys.stderr)
        print("  Run Phase 4 first: python -m src.phase4.run", file=sys.stderr)
        sys.exit(1)

    df = pl.read_parquet(parquet_path)
    required = {"npdes_id", "annual_energy_kwh", "elec_rate_per_kwh", "annual_opex_usd",
                "total_capex_usd", "rated_power_kw", "project_viable"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: missing columns {sorted(missing)}", file=sys.stderr)
        sys.exit(1)

    rows = df.to_dicts()
    n_scenarios = len(tier_ladder()) * len(_DISCOUNT_RATES) * len(_GRANTS)
    print(
        f"Re-scoring {len(rows):,} sites x {n_scenarios} scenarios "
        f"({len(rows) * n_scenarios:,} scorecards) …",
        file=sys.stderr,
    )
    grid = run_grid(rows)
    report = render_report(grid, len(rows))

    if out_path is not None:
        out_path.write_text(report, encoding="utf-8")
        print(f"\nWrote {out_path}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 4 energy-tier x discount-rate what-if (read-only)"
    )
    parser.add_argument("--parquet", type=Path, default=_DEFAULT_PARQUET,
                        help="Path to financial_scorecards.parquet")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write the markdown report here instead of stdout")
    args = parser.parse_args()
    main(args.parquet, args.out)
