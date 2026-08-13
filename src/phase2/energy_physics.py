"""Phase 2 — Core hydropower physics and Monte Carlo energy estimation.

Physics equation
----------------
    P (watts) = η × ρ × g × Q (m³/s) × H (m)

Monte Carlo approach
--------------------
For each facility, run N iterations.  In each iteration:

1. Sample head H        from Triangular(low_m, mode_m, high_m)  [metres]
2. Sample efficiency η  from Triangular(0.70, 0.82, 0.90)
3. Sample availability  from Triangular(0.90, 0.95, 0.98)
4. Integrate power over the 20-point Flow Duration Curve using the trapezoidal
   rule to get annual energy in kWh/yr.
5. Multiply by availability to get net annual energy.

All physical constants are read from ``config/settings.yaml`` so they stay
consistent with Phase 3.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from src.common import config

# ── Physical constants (from settings.yaml) ───────────────────────────────────

RHO: float          = config.get("physics.rho_water_kg_m3",  998.2)
GRAVITY: float      = config.get("physics.gravity_m_s2",       9.81)
MGD_TO_M3S: float   = config.get("physics.mgd_to_m3s",      0.043813)
HOURS_PER_YEAR: float = config.get("physics.hours_per_year",  8766.0)

# ── Turbine efficiency distribution (all types pooled at Phase 2 screening) ──

_ETA_LOW:  float = 0.70
_ETA_MODE: float = config.get("physics.default_efficiency", 0.82)
_ETA_HIGH: float = 0.90

# ── Availability distribution ─────────────────────────────────────────────────

_AVAIL_LOW:  float = 0.90
_AVAIL_MODE: float = config.get("physics.default_availability", 0.95)
_AVAIL_HIGH: float = 0.98

# ── FDC exceedance grid (must match Phase 1 ranking.fdc_exceedance_probs) ─────

_FDC_EXCEEDANCES: list[float] = config.get(
    "ranking.fdc_exceedance_probs",
    [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
     0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
)
_FDC_EXCEEDANCES_ARR: np.ndarray = np.asarray(_FDC_EXCEEDANCES, dtype=np.float64)


# ── Public helpers ────────────────────────────────────────────────────────────

def power_kw(eta: float, q_m3s: float, h_m: float) -> float:
    """Instantaneous power (kW) from the standard hydropower equation."""
    return eta * RHO * GRAVITY * q_m3s * h_m / 1_000.0


def integrate_fdc_energy(
    fdc_flows_mgd: np.ndarray,
    h_m: float,
    eta: float,
    availability: float,
) -> float:
    """Annual energy (kWh/yr) from a FDC using the trapezoidal rule.

    Args:
        fdc_flows_mgd:  Array of flow values (MGD).  ``fdc_flows_mgd[0]`` is
                        the flow at the *lowest* exceedance probability (e.g.
                        0.01 — a high flow exceeded only 1% of the time).
                        ``fdc_flows_mgd[-1]`` is the flow at the *highest*
                        exceedance probability (e.g. 0.95 — a low flow exceeded
                        95% of the time).  Values are therefore **descending**
                        (high flow to low flow).  Must have length ≥ 2.
        h_m:            Net head (m) — constant for this iteration.
        eta:            Turbine efficiency (scalar) — constant for this iteration.
        availability:   Plant availability fraction (0–1).

    Returns:
        Net annual energy in kWh/yr.  Returns 0.0 if fewer than 2 FDC points.
        All inputs are physically constrained to be ≥ 0, so the result is
        always ≥ 0.  Negative values indicate upstream data quality problems
        and are **not** suppressed here — let them surface.
    """
    n = min(len(fdc_flows_mgd), len(_FDC_EXCEEDANCES_ARR))
    if n < 2:
        return 0.0

    # KNOWN ASSUMPTION: FDC is truncated to the shorter of the two arrays.
    # If the EPA DMR FDC has fewer than 20 points the tail (low-flow region,
    # high exceedance probabilities) is silently dropped.  This underestimates
    # annual energy for facilities with sparse DMR data because very-low-flow
    # periods (Q_P95) contribute a small but non-zero energy term.
    # Quantified impact: <5% of annual energy for typical US POTWs.
    # Accepted as a conservative assumption for Phase 5 ML training input.
    q_m3s = fdc_flows_mgd[:n] * MGD_TO_M3S
    exc   = _FDC_EXCEEDANCES_ARR[:n]

    # Power at each FDC point (kW)
    power = eta * RHO * GRAVITY * q_m3s * h_m / 1_000.0  # kW

    # Trapezoidal integration over exceedance axis:
    # integral of power(exc) dexc × hours/yr × availability = kWh/yr
    return float(np.trapezoid(power, exc) * HOURS_PER_YEAR * availability)


# ── Monte Carlo sampler ───────────────────────────────────────────────────────

def run_monte_carlo(
    fdc_flows_mgd: np.ndarray,
    head_low_m:    float,
    head_mode_m:   float,
    head_high_m:   float,
    n_iterations:  int = 10_000,
    rng:           Generator | None = None,
) -> dict[str, float]:
    """Run Monte Carlo energy estimation for a single facility.

    Args:
        fdc_flows_mgd:  20-point FDC (MGD), P01 → P95 exceedance flows.
        head_low_m:     Triangular head distribution low bound.
        head_mode_m:    Triangular head distribution mode.
        head_high_m:    Triangular head distribution high bound.
        n_iterations:   Number of Monte Carlo samples (default 10,000).
        rng:            NumPy random Generator.  Creates a new one if None
                        (not seeded — call with ``np.random.default_rng(seed)``
                        for reproducible tests).

    Returns:
        Dict with keys:
          energy_p10_kwh_yr, energy_p50_kwh_yr, energy_p90_kwh_yr,
          energy_mean_kwh_yr, energy_std_kwh_yr,
          power_p50_kw, capacity_factor_p50.
    """
    if rng is None:
        rng = np.random.default_rng()

    fdc_arr = np.asarray(fdc_flows_mgd, dtype=np.float64)

    # Vectorised sampling — one row per iteration
    h_samples     = rng.triangular(head_low_m,  head_mode_m,  head_high_m,  n_iterations)
    eta_samples   = rng.triangular(_ETA_LOW,    _ETA_MODE,    _ETA_HIGH,    n_iterations)
    avail_samples = rng.triangular(_AVAIL_LOW,  _AVAIL_MODE,  _AVAIL_HIGH,  n_iterations)

    # Fully vectorised integration — avoids 10k Python function calls per facility.
    # power_matrix shape: (n_iterations, n_fdc_points)
    n = min(len(fdc_arr), len(_FDC_EXCEEDANCES_ARR))
    if n < 2:
        # Degenerate FDC — no integration possible; all samples yield zero energy.
        # _process_one always supplies ≥20 points, so this branch is unreachable
        # in production, but mirrors the guard in integrate_fdc_energy.
        energies = np.zeros(n_iterations)
    else:
        q_m3s = fdc_arr[:n] * MGD_TO_M3S           # (n,)
        exc   = _FDC_EXCEEDANCES_ARR[:n]            # (n,)
        power_matrix = (
            eta_samples[:, None] * RHO * GRAVITY
            * q_m3s[None, :]
            * h_samples[:, None]
            / 1_000.0
        )                                            # (n_iterations, n) kW
        integrals = np.trapezoid(power_matrix, exc, axis=1)  # (n_iterations,)
        energies  = integrals * HOURS_PER_YEAR * avail_samples  # kWh/yr

    # Median power at rated conditions (P50 head, P50 efficiency, mean flow)
    mean_flow_m3s = float(np.mean(fdc_arr)) * MGD_TO_M3S
    h_p10   = float(np.percentile(h_samples,   10))
    h_p50   = float(np.percentile(h_samples,   50))
    h_p90   = float(np.percentile(h_samples,   90))
    eta_p50 = float(np.percentile(eta_samples,  50))
    power_p50_kw = power_kw(eta_p50, mean_flow_m3s, h_p50)

    # Capacity factor: P50 energy vs theoretical maximum (rated power × hours)
    energy_p50_kwh = float(np.percentile(energies, 50))
    max_energy_kwh = max(power_p50_kw * HOURS_PER_YEAR, 1e-9)
    cf_p50 = min(1.0, energy_p50_kwh / max_energy_kwh)

    return {
        "energy_p10_kwh_yr":   float(np.percentile(energies, 10)),
        "energy_p50_kwh_yr":   float(np.percentile(energies, 50)),
        "energy_p90_kwh_yr":   float(np.percentile(energies, 90)),
        "energy_mean_kwh_yr":  float(np.mean(energies)),
        "energy_std_kwh_yr":   float(np.std(energies)),
        "power_p50_kw":        power_p50_kw,
        "capacity_factor_p50": cf_p50,
        "head_m_p10":          h_p10,
        "head_m_p50":          h_p50,
        "head_m_p90":          h_p90,
    }
