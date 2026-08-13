"""Phase 4 — Sensitivity (tornado) analysis.

Runs the NPV model at low/high values of three key uncertain inputs:

    head_factor        ×0.50 and ×1.50  (measurement / DEM error)
    flow_factor        ×0.80 and ×1.20  (seasonal / year-to-year variability)
    rate_factor        ×0.70 and ×1.30  (electricity price range)

Physical models (Option B)
--------------------------
Head perturbation:
    Re-runs ``optimize_rated_power`` with h_net_m × head_factor.  Because
    the optimal Q_rated changes with head (higher head → same power at lower
    flow → turbine resizes), this captures the nonlinear clipping effect on
    the FDC integral.  The result is a new P_rated and annual_energy_mwh
    that may differ non-linearly from the base case.

Flow perturbation:
    Scales every FDC flow bin by flow_factor (representing a wetter/drier
    year) and re-runs ``_compute_annual_energy`` with the original Q_rated
    and H_net.  Because η(q_i / Q_rated) is a nonlinear function of the
    Q_fraction, scaling FDC flows changes the part-load operating point and
    produces a non-linear energy response.

Rate perturbation:
    Scales elec_rate_per_kwh only — linear, no re-run of FDC.

If FDC data is not available (legacy call sites without new kwargs), the
function falls back to the original linear energy-scaling approach so that
existing callers (tests, CLI) continue to work unchanged.
"""

from __future__ import annotations

from src.phase4.financials import compute_npv, DISCOUNT_RATE, PROJECT_YEARS, DEGRADATION_RATE
from src.phase3.turbine_selection import (
    optimize_rated_power,
    _compute_annual_energy,
    MGD_TO_M3S,
    _PHASE1_FDC_EXCEEDANCES,
)

# Sensitivity sweep factors (hold two at 1.0, vary the third)
_HEAD_FACTORS:  tuple[float, float] = (0.50, 1.50)  # ±50%
_FLOW_FACTORS:  tuple[float, float] = (0.80, 1.20)  # ±20%
_RATE_FACTORS:  tuple[float, float] = (0.70, 1.30)  # ±30%

# Range widths used to normalise swings so dominant_sensitivity reflects
# NPV elasticity per unit of input change, not the width of the assumed range.
_HEAD_RANGE: float = _HEAD_FACTORS[1] - _HEAD_FACTORS[0]  # 1.00
_FLOW_RANGE: float = _FLOW_FACTORS[1] - _FLOW_FACTORS[0]  # 0.40
_RATE_RANGE: float = _RATE_FACTORS[1] - _RATE_FACTORS[0]  # 0.60


def run_tornado(
    annual_energy_kwh:  float,
    elec_rate_per_kwh:  float,
    annual_opex_usd:    float,
    total_capex_usd:    float,
    discount_rate:      float = DISCOUNT_RATE,
    project_years:      int   = PROJECT_YEARS,
    degradation_rate:   float = DEGRADATION_RATE,
    # ── Option B physical inputs (all optional for backward compat) ──────────
    h_net_m:            float | None = None,
    q_design_m3s:       float | None = None,
    fdc_flows_m3s:      list[float] | None = None,
    fdc_exceedances:    list[float] | None = None,
    turbine_type:       str | None = None,
    q_rated_m3s:        float | None = None,
) -> dict[str, float | str]:
    """Compute tornado sensitivity for one facility.

    When physical inputs (h_net_m, q_design_m3s, fdc_flows_m3s,
    fdc_exceedances, turbine_type) are all provided, uses physically distinct
    models for head and flow perturbations (Option B).  Otherwise falls back
    to linear energy scaling.

    Args:
        annual_energy_kwh:  Base case annual generation (kWh/yr).
        elec_rate_per_kwh:  Base case electricity rate ($/kWh).
        annual_opex_usd:    Annual O&M cost (USD/yr).
        total_capex_usd:    Total installed CapEx (USD).
        discount_rate:      Real discount rate.
        project_years:      Asset lifetime (years).
        degradation_rate:   Annual output decline.
        h_net_m:            Net head at base case (m).
        q_design_m3s:       Design flow (m³/s) — used to re-optimize Q_rated.
        fdc_flows_m3s:      FDC flows in m³/s (list, same grid as Phase 3).
        fdc_exceedances:    Exceedance probabilities matching fdc_flows_m3s.
        turbine_type:       Turbine type string (e.g. 'Kaplan').
        q_rated_m3s:        Base-case rated flow (m³/s) — used for flow sweep.

    Returns:
        Dict with 6 NPV columns + ``dominant_sensitivity`` string.
    """
    # Determine whether we have the full physical inputs for Option B
    _has_physical = (
        h_net_m is not None
        and q_design_m3s is not None
        and fdc_flows_m3s is not None
        and fdc_exceedances is not None
        and turbine_type is not None
        and len(fdc_flows_m3s) >= 2
        and len(fdc_exceedances) >= 2
    )

    base_kwh = annual_energy_kwh

    def _npv_from_energy(energy_kwh: float, rate_factor: float = 1.0) -> float:
        return compute_npv(
            energy_kwh,
            elec_rate_per_kwh * rate_factor,
            annual_opex_usd,
            total_capex_usd,
            discount_rate,
            project_years,
            degradation_rate,
        )

    if _has_physical:
        # ── Head sweep: re-optimize rated power at perturbed head ──────────
        def _npv_head(head_factor: float) -> float:
            h_perturbed = h_net_m * head_factor
            if h_perturbed <= 0:
                return _npv_from_energy(0.0)
            _, _, energy_mwh, _ = optimize_rated_power(
                turbine_type, q_design_m3s, h_perturbed,
                fdc_flows_m3s, fdc_exceedances,
            )
            return _npv_from_energy(energy_mwh * 1_000)

        # ── Flow sweep: scale FDC flows, keep Q_rated and H_net fixed ──────
        _q_rated = q_rated_m3s if q_rated_m3s is not None else q_design_m3s

        def _npv_flow(flow_factor: float) -> float:
            scaled_fdc = [q * flow_factor for q in fdc_flows_m3s]
            energy_mwh = _compute_annual_energy(
                turbine_type, _q_rated, h_net_m,
                scaled_fdc, fdc_exceedances,
            )
            return _npv_from_energy(energy_mwh * 1_000)

        npv_head_low  = _npv_head(_HEAD_FACTORS[0])
        npv_head_high = _npv_head(_HEAD_FACTORS[1])
        npv_flow_low  = _npv_flow(_FLOW_FACTORS[0])
        npv_flow_high = _npv_flow(_FLOW_FACTORS[1])

    else:
        # ── Fallback: linear energy scaling (legacy / test path) ───────────
        # NOTE: head and flow swings are algebraically identical here because
        # both scale annual_energy_kwh linearly.  dominant_sensitivity is set
        # to "energy_uncertain" for these sites so downstream consumers know
        # the head/flow distinction is not meaningful.
        npv_head_low  = _npv_from_energy(base_kwh * _HEAD_FACTORS[0])
        npv_head_high = _npv_from_energy(base_kwh * _HEAD_FACTORS[1])
        npv_flow_low  = _npv_from_energy(base_kwh * _FLOW_FACTORS[0])
        npv_flow_high = _npv_from_energy(base_kwh * _FLOW_FACTORS[1])

    npv_rate_low  = _npv_from_energy(base_kwh, _RATE_FACTORS[0])
    npv_rate_high = _npv_from_energy(base_kwh, _RATE_FACTORS[1])

    # Normalise raw NPV swing by the range width so dominant_sensitivity
    # reflects NPV elasticity per unit of input change, not range-width bias.
    swings = {
        "head": abs(npv_head_high - npv_head_low) / _HEAD_RANGE,
        "flow": abs(npv_flow_high - npv_flow_low) / _FLOW_RANGE,
        "rate": abs(npv_rate_high - npv_rate_low) / _RATE_RANGE,
    }
    if _has_physical:
        dominant = max(swings, key=swings.__getitem__)
    else:
        # Linear fallback: head_swing == flow_swing by construction — label
        # as uncertain rather than picking a meaningless winner.
        dominant = "energy_uncertain"

    return {
        "sensitivity_head_npv_low":  npv_head_low,
        "sensitivity_head_npv_high": npv_head_high,
        "sensitivity_flow_npv_low":  npv_flow_low,
        "sensitivity_flow_npv_high": npv_flow_high,
        "sensitivity_rate_npv_low":  npv_rate_low,
        "sensitivity_rate_npv_high": npv_rate_high,
        "dominant_sensitivity":      dominant,
    }
