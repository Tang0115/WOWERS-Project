"""Phase 4 — Financial scorecard: NPV, IRR, payback, LCOE.

All financial parameters (discount rate, project lifetime, degradation rate)
are read from ``config/settings.yaml`` under ``financials``.

DCF model
---------
For year t = 1 … 30:
    energy_t   = annual_energy_kwh × (1 − degradation_rate)^(t−1)
    revenue_t  = energy_t × (elec_rate + rec_rate)
    opex_t     = annual_opex_usd  (constant in real terms)
    net_cf_t   = revenue_t − opex_t
    pv_t       = net_cf_t / (1 + discount_rate)^t

NPV = −total_capex + Σ pv_t

IRR is solved via scipy.optimize.brentq (more robust than numpy_financial.irr
for near-zero or negative edge cases).

LCOE = (CapEx + Σ PV(OpEx_t)) / Σ PV(Energy_t)   [$/kWh, real]

A secondary NPV with 50% grant coverage is also computed as a scenario column
(``npv_with_50pct_grant_usd``).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy.optimize import brentq

from src.common import config

# ── Financial parameters ──────────────────────────────────────────────────────

_FIN = config.get("financials", {})
DISCOUNT_RATE:        float = float(_FIN.get("discount_rate",            0.06))
PROJECT_YEARS:        int   = int(  _FIN.get("project_lifetime_years",    30))
DEGRADATION_RATE:     float = float(_FIN.get("degradation_rate",          0.002))
REC_PER_KWH:          float = float(_FIN.get("rec_value_per_kwh",         0.01))

# F4-MINREV revenue floor — kept as a reversible lever, currently disabled.
#
# 2026-05-20 (F4-MINREV-RAISE): raised from $5,000 → $20,000.
# 2026-06 (F4-MINREV-REMOVE): director decision — set to 0 in settings.yaml so
#   revenue_above_floor = (revenue >= 0) is always True (no-op).
#   project_viable is now NPV>0 AND payback≤20yr AND IRR real, no revenue gate.
#
# To re-enable: set ``min_annual_revenue_usd`` in config/settings.yaml to the
# desired USD/yr threshold (e.g. 20000).  Code path is preserved.
MIN_ANNUAL_REVENUE_USD: float = float(_FIN.get("min_annual_revenue_usd", 0.0))


# ── Cash-flow helpers ─────────────────────────────────────────────────────────

def _build_cashflows(
    annual_energy_kwh: float,
    elec_rate_per_kwh: float,
    annual_opex_usd:   float,
    total_capex_usd:   float,
    discount_rate:     float = DISCOUNT_RATE,
    project_years:     int   = PROJECT_YEARS,
    degradation_rate:  float = DEGRADATION_RATE,
    rec_per_kwh:       float = REC_PER_KWH,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute annual net cash flows and present values.

    Returns:
        (net_cfs, pv_cfs) — 1-D arrays of length project_years.
        Year 0 (−CapEx) is NOT included; caller adds it for NPV.
    """
    t = np.arange(1, project_years + 1, dtype=np.float64)
    energy_t  = annual_energy_kwh * (1 - degradation_rate) ** (t - 1)
    revenue_t = energy_t * (elec_rate_per_kwh + rec_per_kwh)
    net_cf_t  = revenue_t - annual_opex_usd
    pv_t      = net_cf_t / (1 + discount_rate) ** t
    return net_cf_t, pv_t


# ── Public financial functions ────────────────────────────────────────────────

def compute_npv(
    annual_energy_kwh: float,
    elec_rate_per_kwh: float,
    annual_opex_usd:   float,
    total_capex_usd:   float,
    discount_rate:     float = DISCOUNT_RATE,
    project_years:     int   = PROJECT_YEARS,
    degradation_rate:  float = DEGRADATION_RATE,
    capex_subsidy_pct: float = 0.0,
) -> float:
    """30-year NPV at specified discount rate (USD).

    Args:
        annual_energy_kwh:  Year-1 annual generation (kWh/yr).
        elec_rate_per_kwh:  State electricity rate ($/kWh).
        annual_opex_usd:    Annual O&M cost (USD/yr).
        total_capex_usd:    Total installed capital cost (USD).
        discount_rate:      Real discount rate (default 6%).
        project_years:      Asset lifetime (default 30 years).
        degradation_rate:   Annual output decline (default 0.2%).
        capex_subsidy_pct:  Fraction of CapEx covered by grant/subsidy (0–1).

    Returns:
        NPV in USD.  Positive = viable investment.
    """
    net_capex = total_capex_usd * (1.0 - capex_subsidy_pct)
    _, pv_t = _build_cashflows(
        annual_energy_kwh, elec_rate_per_kwh, annual_opex_usd,
        net_capex, discount_rate, project_years, degradation_rate,
    )
    return float(pv_t.sum() - net_capex)


def compute_irr(
    annual_energy_kwh: float,
    elec_rate_per_kwh: float,
    annual_opex_usd:   float,
    total_capex_usd:   float,
    project_years:     int   = PROJECT_YEARS,
    degradation_rate:  float = DEGRADATION_RATE,
) -> float:
    """Internal rate of return via Brent's method (decimal, e.g. 0.08 = 8%).

    Searches the interval [−99%, +300%] for a root of the NPV function.

    Returns:
        IRR as decimal.
        ``+3.0`` sentinel if NPV is positive at every rate in the range
        (i.e. IRR > 300% — trivially profitable at nano-scale CapEx).
        ``−0.99`` sentinel if NPV is always negative (project never pays back).
        ``math.nan`` only on unexpected solver exceptions.

    .. warning::
        Do **not** use ``math.isnan(irr)`` as the only check for invalid IRR.
        Sentinel values (−0.99 / +3.0) are not NaN. Filter them explicitly::

            viable = df.filter(
                (pl.col("irr") > 0) & (pl.col("irr") < 1.0)
            )
    """
    if total_capex_usd <= 0:
        return math.nan

    def _npv_at_rate(r: float) -> float:
        _, pv_t = _build_cashflows(
            annual_energy_kwh, elec_rate_per_kwh, annual_opex_usd,
            total_capex_usd, discount_rate=r, project_years=project_years,
            degradation_rate=degradation_rate,
        )
        return float(pv_t.sum() - total_capex_usd)

    lo, hi = -0.99, 3.00
    try:
        f_lo = _npv_at_rate(lo)
        f_hi = _npv_at_rate(hi)
        if f_lo * f_hi > 0:
            # No sign change — return boundary sentinel
            return hi if f_lo > 0 else lo
        return float(brentq(_npv_at_rate, lo, hi, xtol=1e-6, maxiter=200))
    except Exception:
        return math.nan


def compute_payback(
    annual_energy_kwh: float,
    elec_rate_per_kwh: float,
    annual_opex_usd:   float,
    total_capex_usd:   float,
    project_years:     int   = PROJECT_YEARS,
    degradation_rate:  float = DEGRADATION_RATE,
) -> float:
    """Simple (undiscounted) payback period in years.

    Returns ``math.inf`` if the project never pays back within project_years.
    """
    net_cfs, _ = _build_cashflows(
        annual_energy_kwh, elec_rate_per_kwh, annual_opex_usd,
        total_capex_usd, project_years=project_years,
        degradation_rate=degradation_rate,
    )
    cumulative = 0.0
    for t_idx, cf in enumerate(net_cfs, start=1):
        cumulative += float(cf)
        if cumulative >= total_capex_usd:
            overshoot = cumulative - total_capex_usd
            return float(t_idx) - overshoot / max(float(cf), 1e-9)
    return math.inf


def compute_lcoe(
    annual_energy_kwh: float,
    annual_opex_usd:   float,
    total_capex_usd:   float,
    discount_rate:     float = DISCOUNT_RATE,
    project_years:     int   = PROJECT_YEARS,
    degradation_rate:  float = DEGRADATION_RATE,
) -> float:
    """Levelized Cost of Energy ($/kWh, real-dollar basis).

    LCOE = (CapEx + PV(OpEx stream)) / PV(Energy stream)
    """
    if annual_energy_kwh <= 0:
        return math.inf

    t = np.arange(1, project_years + 1, dtype=np.float64)
    energy_t  = annual_energy_kwh * (1 - degradation_rate) ** (t - 1)
    discount  = (1 + discount_rate) ** t

    pv_opex   = float(np.sum(annual_opex_usd / discount))
    pv_energy = float(np.sum(energy_t        / discount))

    if pv_energy <= 0:
        return math.inf
    return (total_capex_usd + pv_opex) / pv_energy


# Sentinel value stored in parquet for "never" payback / infinite LCOE.
# Using 1e6 avoids collision with EPA ECHO's 999 sentinel (which appears in
# upstream flow data and could be misread as a financial value by Phase 5).
_INF_SENTINEL: float = 1e6


def compute_scorecard(
    annual_energy_kwh:      float,
    elec_rate_per_kwh:      float,
    annual_opex_usd:        float,
    total_capex_usd:        float,
    annual_revenue_usd:     float,
    discount_rate:          float = DISCOUNT_RATE,
    project_years:          int   = PROJECT_YEARS,
    degradation_rate:       float = DEGRADATION_RATE,
    min_annual_revenue_usd: float = MIN_ANNUAL_REVENUE_USD,
) -> dict[str, float]:
    """Compute the full financial scorecard for one facility.

    Args:
        min_annual_revenue_usd: F4-MINREV floor (currently 0 — disabled per
            director decision 2026-06).  With value 0 the gate is a no-op
            (revenue >= 0 is always True).  Pass an explicit positive value
            to re-enable without touching settings.yaml.

    Returns a flat dict suitable for building a Polars/pandas row.
    """
    npv = compute_npv(annual_energy_kwh, elec_rate_per_kwh,
                      annual_opex_usd, total_capex_usd,
                      discount_rate, project_years, degradation_rate)

    npv_grant = compute_npv(annual_energy_kwh, elec_rate_per_kwh,
                             annual_opex_usd, total_capex_usd,
                             discount_rate, project_years, degradation_rate,
                             capex_subsidy_pct=0.5)

    irr = compute_irr(annual_energy_kwh, elec_rate_per_kwh,
                      annual_opex_usd, total_capex_usd,
                      project_years, degradation_rate)

    payback = compute_payback(annual_energy_kwh, elec_rate_per_kwh,
                               annual_opex_usd, total_capex_usd,
                               project_years, degradation_rate)

    lcoe = compute_lcoe(annual_energy_kwh, annual_opex_usd, total_capex_usd,
                        discount_rate, project_years, degradation_rate)

    # annual_net_cf_usd uses the caller-supplied annual_revenue_usd (which already
    # includes REC value) for the headline row.  NPV internally recomputes revenue
    # from energy_kwh × (elec_rate + rec) for year-by-year degradation.  Callers
    # MUST pass annual_revenue_usd = energy_kwh × (elec_rate + rec) to keep these
    # consistent.  See phase4/run.py: rev_usd = annual_revenue(energy_kwh, state).
    annual_net_cf = annual_revenue_usd - annual_opex_usd

    # project_viable: NPV positive, payback within 20 years, AND IRR is a real
    # solver result (not a sentinel +3.0/−0.99 or NaN). Sentinel IRR signals
    # degenerate economics — e.g. CapEx so tiny IRR pegs out at >300%, or all
    # net cash flows negative (IRR locked at −99%).  Excluding those keeps
    # ``project_viable`` aligned with what a human investor would call viable.
    irr_real = (
        not math.isnan(irr)
        and irr > -0.99
        and irr < 3.0
    )
    # F4-MINREV: sites below the annual revenue floor are not bankable.
    revenue_above_floor = annual_revenue_usd >= min_annual_revenue_usd
    viable = bool(
        npv > 0
        and payback <= 20.0
        and irr_real
        and revenue_above_floor
    )

    return {
        "annual_net_cf_usd":          annual_net_cf,
        "npv_usd":                    npv,
        "npv_with_50pct_grant_usd":   npv_grant,
        "irr":                        irr,
        "payback_years":              payback if not math.isinf(payback) else _INF_SENTINEL,
        "lcoe_per_kwh":               lcoe   if not math.isinf(lcoe)    else _INF_SENTINEL,
        "project_viable":             viable,
    }


# ── P4-TIER: CF-calibrated energy columns (2026-07-06) ────────────────────────

# Default multipliers from CF_CALIBRATION_REPORT.md §6, primary 0.1–5 MW bucket.
# multiplier = empirical_CF / 0.872  (Phase 2 implied fleet median CF).
# Override via config/settings.yaml: phase4.cf_calibration.{floor_p25,floor_p50,central}
_CF_CALIB_DEFAULTS: dict[str, float] = {
    "floor_p25": 0.291,
    "floor_p50": 0.447,
    "central":   0.688,
    "measured_point_loma": 0.2195,   # P4-MEASURED-FLOOR (2026-08-06)
}


def add_calibrated_energy_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Append four CF-calibrated energy columns to the Phase 4 scorecard.

    New columns (additive — zero changes to existing columns):

    ``energy_kwh_calib_floor_p25``
        annual_energy_kwh × 0.291 — river-hydro p25 CF (river class, not ours)
    ``energy_kwh_calib_floor_p50``
        annual_energy_kwh × 0.447 — river-hydro p50 CF
    ``energy_kwh_calib_central``
        annual_energy_kwh × 0.688 — CF = 0.60. **Relabelled "optimistic" on
        2026-07-25**: it rests on the Conduit 3 (FERC P-14498) design
        projection, not a measurement. The config key keeps its original name
        so existing consumers do not break.
    ``energy_kwh_calib_measured_point_loma``
        annual_energy_kwh × 0.2195 — **the reported band floor** (P4-MEASURED-FLOOR,
        2026-08-06). Point Loma WWTP is the only metered treated-wastewater
        conduit plant in the U.S.: 1,500 kW / 2,515 MWh in 2017 per Form
        EIA-923 → CF 0.1914. Yields 89.8 GWh/yr across the viable fleet and
        replaces 118.9 as the floor of the reported calibration band.

    A note on why this tier was added rather than overwriting ``floor_p25``:
    0.2195 is not a 25th percentile of anything, so writing it into a column
    named ``floor_p25`` would make the column name false and invite the error
    back. Both tiers are therefore reported, and no previously published number
    changes value.

    The physics ceiling (Phase 2 assumed CF ≈ 0.872) is the pre-existing
    ``annual_energy_kwh`` column; no ceiling column is added.

    Design: **global fleet multipliers** are used, not per-site CF ratios.
    Rationale: matches CF_CALIBRATION_REPORT.md §6 method exactly (multiplier
    = tier_CF / 0.872 fleet median); per-site CF spread is tight (p10–p90
    = 0.856–0.883) so site-level ratios give near-identical results.
    Source: CF_CALIBRATION_REPORT.md §6 (session 2026-07-03, P5-CF-CALIB).

    Args:
        df: Phase 4 scorecard DataFrame (must contain ``annual_energy_kwh``).

    Returns:
        df with four new Float64 columns appended; all existing columns
        untouched and value-identical.
    """
    calib      = config.get("phase4.cf_calibration") or {}
    floor_p25  = float(calib.get("floor_p25", _CF_CALIB_DEFAULTS["floor_p25"]))
    floor_p50  = float(calib.get("floor_p50", _CF_CALIB_DEFAULTS["floor_p50"]))
    central    = float(calib.get("central",   _CF_CALIB_DEFAULTS["central"]))
    meas_floor = float(calib.get("measured_point_loma",
                                 _CF_CALIB_DEFAULTS["measured_point_loma"]))

    return df.with_columns([
        (pl.col("annual_energy_kwh") * floor_p25).alias("energy_kwh_calib_floor_p25"),
        (pl.col("annual_energy_kwh") * floor_p50).alias("energy_kwh_calib_floor_p50"),
        (pl.col("annual_energy_kwh") * central  ).alias("energy_kwh_calib_central"),
        (pl.col("annual_energy_kwh") * meas_floor).alias(
            "energy_kwh_calib_measured_point_loma"),
    ])
