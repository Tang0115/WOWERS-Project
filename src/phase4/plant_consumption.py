"""Phase 4 — Plant energy consumption estimator and turbine offset calculator.

Estimates the annual electricity consumption of a US wastewater treatment
plant (POTW) given its mean flow, and expresses the turbine's annual
output as a percentage of that consumption (the "offset %").

Point estimate uses EPRI 3002001433 Table 5-1 observed flow-band averages
(treatment-type-agnostic, validated against published utility actuals).

Sensitivity band uses Table 5-4 treatment-type curves:
  low  → trickling_filter         (least energy-intensive common process)
  high → advanced_with_nitrification  (most energy-intensive)

F4-OFFSET tag: all additions in this module are additive; no existing
scorecard columns, the MINREV floor, compute_scorecard, or derive_site_tier
are touched.

References
----------
EPRI 3002001433 (Pabi, Amarnath, Goldstein, Reekie), November 2013.
  Table 5-1 p.5-6  — observed flow-band intensity (point estimate basis).
  Table 5-4 p.5-15 — treatment-type × size grid (sensitivity band).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import yaml

# ── Load config once at import (mirrors cost_models.py loader pattern) ────────

_CFG_PATH: Path = Path(__file__).resolve().parents[2] / "config" / "energy_intensity.yaml"
_CFG: dict = yaml.safe_load(_CFG_PATH.read_text())

_BANDS: list[dict] = _CFG["observed_flow_band_intensity"]["bands"]
_TREATMENT_TYPES: dict = _CFG["treatment_types"]

_SENS_LOW: str  = _CFG["treatment_assignment"]["sensitivity_low"]   # trickling_filter
_SENS_HIGH: str = _CFG["treatment_assignment"]["sensitivity_high"]  # advanced_with_nitrification


# ── Public helpers (identical logic to scripts/validate_energy_intensity.py) ──

def observed_intensity(flow_mgd: float) -> float:
    """kWh/MG point estimate from EPRI Table 5-1 observed flow bands.

    Band lookup with strict ``flow < max_mgd`` test; the last band
    (max_mgd=null) is a catch-all for ≥ 100 MGD plants.

    Parameters
    ----------
    flow_mgd:
        Mean daily flow in million gallons per day. Must be > 0.
    """
    # F4-OFFSET: verbatim port from scripts/validate_energy_intensity.py
    for b in _BANDS:
        if b["max_mgd"] is None or flow_mgd < b["max_mgd"]:
            return float(b["kwh_per_mg"])
    return float(_BANDS[-1]["kwh_per_mg"])  # fallback (never reached)


def intensity(treatment_type: str, flow_mgd: float) -> float:
    """kWh/MG for a treatment type at a given flow.

    Log-linear interpolation between the Table 5-4 size points.
    Clamped at the table edges — does not extrapolate beyond bounds.

    Parameters
    ----------
    treatment_type:
        Must be a key in ``treatment_types`` in energy_intensity.yaml.
    flow_mgd:
        Mean daily flow in million gallons per day.
    """
    # F4-OFFSET: verbatim port from scripts/validate_energy_intensity.py
    t = _TREATMENT_TYPES[treatment_type]
    sizes = t["size_points_mgd"]
    vals  = t["kwh_per_mg"]

    if flow_mgd <= sizes[0]:
        return float(vals[0])
    if flow_mgd >= sizes[-1]:
        return float(vals[-1])

    for i in range(len(sizes) - 1):
        lo, hi = sizes[i], sizes[i + 1]
        if lo <= flow_mgd <= hi:
            vlo, vhi = vals[i], vals[i + 1]
            frac = math.log(flow_mgd / lo) / math.log(hi / lo)
            return float(vlo * (vhi / vlo) ** frac)

    return float(vals[-1])  # fallback (never reached)


def consumption_and_offset(
    mean_flow_mgd: Optional[float],
    annual_energy_kwh: float,
) -> Dict[str, Optional[float]]:
    """Return the 6 F4-OFFSET columns for a scored turbine site.

    Null guard: if ``mean_flow_mgd`` is None or ≤ 0, all 6 keys are None.

    Offset band (treatment-type sensitivity, independent of point estimate):
      ``energy_offset_pct_low``  = energy / HIGH_consumption * 100
                                   (bigger denominator → smaller %)
      ``energy_offset_pct_high`` = energy / LOW_consumption  * 100
                                   (smaller denominator → larger %)
    Guaranteed: ``offset_pct_low <= offset_pct_high`` (TF always < advanced+N).

    Note: the point estimate uses EPRI Table 5-1 observed averages, which
    exceed the Table 5-4 advanced+N (high) curve at every flow (the YAML
    documents this; validation confirmed it nationally).  ``offset_pct`` is
    therefore NOT bracketed by the low/high band — it sits below both.
    The low/high band represents treatment-type uncertainty, not error bars
    around the point estimate.

    Parameters
    ----------
    mean_flow_mgd:
        Mean daily flow from Phase 1/3 row (may be None for excluded sites).
    annual_energy_kwh:
        Turbine annual output in kWh/yr from Phase 3.
    """
    _null: Dict[str, Optional[float]] = {
        "est_plant_consumption_kwh_yr":      None,
        "est_plant_consumption_low_kwh_yr":  None,
        "est_plant_consumption_high_kwh_yr": None,
        "energy_offset_pct":                 None,
        "energy_offset_pct_low":             None,
        "energy_offset_pct_high":            None,
    }

    # F4-OFFSET null/zero guard — never divide by zero
    if mean_flow_mgd is None or mean_flow_mgd <= 0:
        return _null

    mgd = float(mean_flow_mgd)

    cons_point = mgd * 365.0 * observed_intensity(mgd)
    cons_low   = mgd * 365.0 * intensity(_SENS_LOW,  mgd)  # trickling_filter
    cons_high  = mgd * 365.0 * intensity(_SENS_HIGH, mgd)  # advanced_with_nitrification

    # Offset %: turbine output / plant consumption × 100
    # Inversion: offset_low uses HIGH consumption; offset_high uses LOW consumption.
    offset_pt   = annual_energy_kwh / cons_point * 100.0
    offset_low  = annual_energy_kwh / cons_high  * 100.0  # smaller % (bigger denominator)
    offset_high = annual_energy_kwh / cons_low   * 100.0  # larger  % (smaller denominator)

    return {
        "est_plant_consumption_kwh_yr":      cons_point,
        "est_plant_consumption_low_kwh_yr":  cons_low,
        "est_plant_consumption_high_kwh_yr": cons_high,
        "energy_offset_pct":                 offset_pt,
        "energy_offset_pct_low":             offset_low,
        "energy_offset_pct_high":            offset_high,
    }
