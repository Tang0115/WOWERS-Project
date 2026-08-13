"""Phase 3 — Turbine Type Selection and Capacity Sizing.

Given net head (H_net, metres) and mean flow (Q, m³/s), this module:

1. Selects the best turbine *type* using the standard H-Q operating envelope:

       Pelton        H > 50 m  AND  Q < 2 m³/s
       Francis       10 ≤ H ≤ 50 m  (or H > 50 m AND Q ≥ 2 m³/s)
       Kaplan        H < 10 m  AND  Q ≥ 0.5 m³/s
       In-conduit    H < 10 m  AND  Q < 0.5 m³/s   (micro / in-pipe)

2. Applies a part-load efficiency curve η(q) where q = Q_actual / Q_rated:

       Kaplan:       η(q) = min(0.93, -0.3q² + 0.6q + 0.63)   peaks ~0.93 at q=1.0
       Francis:      η(q) = min(0.91, -0.5q² + 0.9q + 0.45)
       Pelton:       η(q) = 0.88 for q ≥ 0.15, else 0.0   (nozzle minimum)
       In-conduit:   η(q) = 0.75 for q ≥ 0.10, else 0.0   (flat curve, gear-less)

3. Runs a rated-power optimizer: sweeps Q_rated from 30% to 100% of Q_design
   and picks the fraction that maximises annual energy output (MWh/yr) while
   maintaining a capacity factor CF ≥ settings.phase3.min_capacity_factor.

   Annual energy is computed by integrating the product:
       η(Q_fdc_i / Q_rated) × ρ × g × Q_fdc_i × H_net  ×  Δt_i
   over the Flow Duration Curve (FDC) loaded from Phase 1 output.  The FDC
   length is variable — Phase 1 stores 20-point FDCs using
   ranking.fdc_exceedance_probs; the exceedance grid is read from the same
   config key so that both arrays are always the same length.

4. Computes the rated power:
       P_rated_kw = η_peak × ρ × g × Q_rated × H_net / 1000

5. Looks up the compatible manufacturer(s) from turbine_manufacturers.csv
   and records the best-match manufacturer.

Output columns appended to the input DataFrame
----------------------------------------------
turbine_type        str     Kaplan | Francis | Pelton | in_conduit_micro
q_rated_m3s         float   design flow rate for the turbine (m³/s)
rated_power_kw      float   nameplate power (kW)
peak_efficiency_pct float   η at rated conditions (%)
annual_energy_mwh   float   estimated annual generation (MWh/yr)
capacity_factor     float   CF = actual / theoretical max
best_manufacturer   str     closest matching manufacturer name
turbine_viable      bool    True when P_rated ≥ 1 kW and head_valid
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import polars as pl

from src.common import config, logging_setup

log = logging_setup.get("wowers.phase3.turbine_selection")

# ── Physical constants ────────────────────────────────────────────────────────
RHO: float      = config.get("physics.rho_water_kg_m3", 998.2)   # kg/m³
GRAVITY: float  = config.get("physics.gravity_m_s2", 9.81)        # m/s²
MGD_TO_M3S: float = config.get("physics.mgd_to_m3s", 0.043813)
HOURS_PER_YEAR: float = config.get("physics.hours_per_year", 8766)

# ── Phase-3 config ────────────────────────────────────────────────────────────

def _cfg(key: str, default=None):
    return config.get(f"phase3.{key}", default)


MIN_CF: float   = _cfg("min_capacity_factor", 0.40)
Q_FRACTIONS: list[float] = _cfg(
    "sizing_q_fractions", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
TURBINE_DB_PATH: Path = config.project_root() / _cfg(
    "turbine_db_path", "data/turbines/turbine_manufacturers.csv"
)
MIN_POWER_KW: float = 1.0   # discard sites below this rated power

# Phase 1 stores FDCs using this exceedance grid (20 points).
# We read the *same* key so Phase 3's optimizer always receives matching arrays.
_PHASE1_FDC_EXCEEDANCES: list[float] = config.get(
    "ranking.fdc_exceedance_probs",
    [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
     0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
)


def select_turbine_type(h_net_m: float, q_m3s: float) -> str:
    """Rule-based turbine type selection from H-Q operating point.

    Decision tree matches the standard IEC/industry selection nomograph:
      1. Pelton          : H > 50 m  AND  Q < 2 m³/s
      2. Francis         : H ≥ 10 m  (any Q)
      3. Crossflow       : 2 ≤ H < 10 m  AND  Q < 0.5 m³/s
                           (Ossberger/CINK-style; good part-load at medium-low head)
      4. Kaplan          : 2 ≤ H < 10 m  AND  Q ≥ 0.5 m³/s  (high-flow low-head)
      5. in_conduit_micro: H < 2 m  (inline micro-turbine, very low head)
    """
    if h_net_m > 50 and q_m3s < 2.0:
        return "Pelton"
    if h_net_m >= 10.0:
        return "Francis"
    if h_net_m >= 2.0 and q_m3s < 0.5:
        return "Crossflow"  # medium-low head, modest flow → Ossberger/CINK
    if h_net_m >= 2.0 and q_m3s >= 0.5:
        return "Kaplan"     # low head, high flow — Kaplan needs adequate head clearance
    return "in_conduit_micro"  # H < 2 m or Q < 0.04 m³/s


# ── Efficiency curves η(q) ────────────────────────────────────────────────────

def efficiency_at_part_load(turbine_type: str, q_fraction: float) -> float:
    """Compute turbine efficiency at part-load ratio q = Q_actual / Q_rated.

    All curves are empirical fits to manufacturer published hill charts
    (see turbine_manufacturers.csv for peak_efficiency_pct references).
    q_fraction is clamped to [0, 1.0] before evaluation.

    Returns efficiency in range [0.0, peak_efficiency] where peak_efficiency
    is the type-specific ceiling.
    """
    q = max(0.0, min(1.0, q_fraction))

    if turbine_type == "Kaplan":
        # Variable-pitch blades → very flat efficiency curve
        eta = -0.30 * q ** 2 + 0.60 * q + 0.63
        return max(0.0, min(0.93, eta))

    if turbine_type == "Francis":
        # Symmetric bowl curve, best efficiency ~85% at q≈0.9
        eta = -0.50 * q ** 2 + 0.90 * q + 0.45
        return max(0.0, min(0.91, eta))

    if turbine_type == "Pelton":
        # Efficient over wide range; cutoff at minimum nozzle opening (~15%)
        return 0.88 if q >= 0.15 else 0.0

    if turbine_type == "Crossflow":
        # Ossberger/CINK-style cross-flow runner: good part-load response,
        # flat efficiency curve.  Empirical fit to manufacturer hill charts.
        eta = -0.10 * q ** 2 + 0.20 * q + 0.72
        return max(0.0, min(0.82, eta)) if q >= 0.10 else 0.0

    if turbine_type == "in_conduit_micro":
        # Simpler fixed geometry; minimum flow threshold for operation
        return 0.75 if q >= 0.10 else 0.0

    # Unknown type — use generic flat curve
    return 0.82 if q >= 0.20 else 0.0


def peak_efficiency(turbine_type: str) -> float:
    """Return the peak (rated-point) efficiency for the turbine type."""
    return efficiency_at_part_load(turbine_type, 1.0)


# ── Rated-power optimizer ─────────────────────────────────────────────────────

def _compute_annual_energy(
    turbine_type: str,
    q_rated_m3s: float,
    h_net_m: float,
    fdc_flows_m3s: list[float],
    fdc_exceedances: list[float],
) -> float:
    """Integrate η(q) × P_hydro over FDC to get annual energy in MWh/yr.

    Uses the trapezoidal rule over exceedance probability bands.
    Each FDC point (q_i, e_i) represents flow q_i exceeded fraction e_i of time.
    """
    n = min(len(fdc_flows_m3s), len(fdc_exceedances))
    if n < 2:
        return 0.0
    # Truncate both to the shorter array — guards against upstream length drift
    fdc_flows_m3s  = fdc_flows_m3s[:n]
    fdc_exceedances = fdc_exceedances[:n]

    total_kwh = 0.0
    for i in range(len(fdc_flows_m3s) - 1):
        q_i = min(fdc_flows_m3s[i], q_rated_m3s)
        q_next = min(fdc_flows_m3s[i + 1], q_rated_m3s)
        de = abs(fdc_exceedances[i] - fdc_exceedances[i + 1])

        eta_i    = efficiency_at_part_load(turbine_type, q_i    / q_rated_m3s if q_rated_m3s > 0 else 0)
        eta_next = efficiency_at_part_load(turbine_type, q_next / q_rated_m3s if q_rated_m3s > 0 else 0)

        # Average power in the band (kW)
        p_i    = eta_i    * RHO * GRAVITY * q_i    * h_net_m / 1000
        p_next = eta_next * RHO * GRAVITY * q_next * h_net_m / 1000
        p_avg  = (p_i + p_next) / 2.0

        total_kwh += p_avg * de * HOURS_PER_YEAR

    return total_kwh / 1_000  # kWh → MWh


def optimize_rated_power(
    turbine_type: str,
    q_design_m3s: float,
    h_net_m: float,
    fdc_flows_m3s: list[float],
    fdc_exceedances: list[float],
) -> tuple[float, float, float, float]:
    """Find Q_rated that maximises annual energy with CF ≥ MIN_CF.

    Returns (q_rated_m3s, rated_power_kw, annual_energy_mwh, capacity_factor).
    """
    best_energy = -1.0
    best_q = q_design_m3s  # fallback
    best_p = 0.0
    best_cf = 0.0

    for frac in Q_FRACTIONS:
        q_r = q_design_m3s * frac
        if q_r <= 0:
            continue

        eta_rated = peak_efficiency(turbine_type)
        p_rated = eta_rated * RHO * GRAVITY * q_r * h_net_m / 1000  # kW

        if p_rated < MIN_POWER_KW:
            continue

        # Capacity factor: ratio of actual to theoretical maximum
        annual_mwh = _compute_annual_energy(
            turbine_type, q_r, h_net_m, fdc_flows_m3s, fdc_exceedances
        )
        max_possible_mwh = p_rated * HOURS_PER_YEAR / 1000
        cf = annual_mwh / max_possible_mwh if max_possible_mwh > 0 else 0.0

        if cf < MIN_CF:
            continue  # undersized rating inflates CF requirement

        if annual_mwh > best_energy:
            best_energy = annual_mwh
            best_q = q_r
            best_p = p_rated
            best_cf = cf

    # If no candidate satisfied CF constraint, fall back to Q_design at 100%
    if best_energy < 0:
        q_r = q_design_m3s
        eta_rated = peak_efficiency(turbine_type)
        best_p = max(0.0, eta_rated * RHO * GRAVITY * q_r * h_net_m / 1000)
        best_energy = _compute_annual_energy(
            turbine_type, q_r, h_net_m, fdc_flows_m3s, fdc_exceedances
        )
        max_possible_mwh = best_p * HOURS_PER_YEAR / 1000
        best_cf = best_energy / max_possible_mwh if max_possible_mwh > 0 else 0.0
        best_q = q_r

    return best_q, best_p, best_energy, best_cf


# ── Manufacturer matching ─────────────────────────────────────────────────────

def _load_turbine_db() -> pl.DataFrame:
    if not TURBINE_DB_PATH.exists():
        log.warning(f"Turbine DB not found at {TURBINE_DB_PATH}; skipping manufacturer match")
        return pl.DataFrame(schema={
            "manufacturer": pl.Utf8,
            "turbine_type": pl.Utf8,
            "min_flow_m3s": pl.Float64,
            "max_flow_m3s": pl.Float64,
            "min_head_m":   pl.Float64,
            "max_head_m":   pl.Float64,
        })
    return pl.read_csv(TURBINE_DB_PATH)


_TURBINE_DB: Optional[pl.DataFrame] = None


def get_turbine_db() -> pl.DataFrame:
    global _TURBINE_DB
    if _TURBINE_DB is None:
        _TURBINE_DB = _load_turbine_db()
    return _TURBINE_DB


def find_best_manufacturer(
    turbine_type: str,
    q_rated_m3s: float,
    h_net_m: float,
) -> str:
    """Return the name of the best matching manufacturer.

    Matching criteria (in priority order):
      1. turbine_type matches
      2. Operating point (Q, H) within manufacturer's specified range
      3. Prefer wastewater_certified == 'yes'
      4. Among remaining ties, prefer lower capex_usd_per_kw_low
    """
    db = get_turbine_db()
    if len(db) == 0:
        return "Unknown"

    candidates = db.filter(
        (pl.col("turbine_type") == turbine_type)
        & (pl.col("min_flow_m3s") <= q_rated_m3s)
        & (pl.col("max_flow_m3s") >= q_rated_m3s)
        & (pl.col("min_head_m") <= h_net_m)
        & (pl.col("max_head_m") >= h_net_m)
    )

    if len(candidates) == 0:
        # Relax flow/head bounds — just match type
        candidates = db.filter(pl.col("turbine_type") == turbine_type)

    if len(candidates) == 0:
        return "Unknown"

    # Prefer WW-certified, then lowest capex
    if "wastewater_certified" in candidates.columns and "capex_usd_per_kw_low" in candidates.columns:
        candidates = candidates.sort(
            ["wastewater_certified", "capex_usd_per_kw_low"],
            descending=[True, False],
        )
    return candidates["manufacturer"][0]


# ── Row-level sizing ──────────────────────────────────────────────────────────

def _size_one(
    npdes_id: str,
    q_design_m3s: float | None,
    h_net_m: float | None,
    head_valid: bool,
    fdc_flows_m3s: list[float],
    fdc_exceedances: list[float],
) -> dict:
    """Size turbine for a single facility."""
    null_result = {
        "turbine_type":        "unknown",
        "q_rated_m3s":         None,
        "rated_power_kw":      None,
        "peak_efficiency_pct": None,
        "annual_energy_mwh":   None,
        "capacity_factor":     None,
        "best_manufacturer":   "Unknown",
        "turbine_viable":      False,
    }

    if not head_valid or h_net_m is None or h_net_m < _cfg("min_net_head_m", 1.0):
        return null_result
    if q_design_m3s is None or q_design_m3s <= _cfg("min_flow_for_sizing_m3s", 0.001):
        return null_result

    t_type = select_turbine_type(h_net_m, q_design_m3s)

    # Use FDC if available, otherwise single-point (flat FDC)
    if len(fdc_flows_m3s) >= 2 and len(fdc_exceedances) >= 2:
        fdc_q = fdc_flows_m3s
        fdc_e = fdc_exceedances
    else:
        fdc_q = [q_design_m3s, q_design_m3s * 0.5]
        fdc_e = [0.0, 1.0]

    q_rated, p_rated, energy_mwh, cf = optimize_rated_power(
        t_type, q_design_m3s, h_net_m, fdc_q, fdc_e
    )
    eta_pct = peak_efficiency(t_type) * 100.0
    manufacturer = find_best_manufacturer(t_type, q_rated, h_net_m)
    viable = p_rated >= MIN_POWER_KW and cf >= MIN_CF

    return {
        "turbine_type":        t_type,
        "q_rated_m3s":         q_rated,
        "rated_power_kw":      round(p_rated, 2),
        "peak_efficiency_pct": round(eta_pct, 1),
        "annual_energy_mwh":   round(energy_mwh, 1),
        "capacity_factor":     round(cf, 3),
        "best_manufacturer":   manufacturer,
        "turbine_viable":      viable,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def select_and_size_turbines(facilities: pl.DataFrame) -> pl.DataFrame:
    """Select turbine type and size rated power for each facility.

    Args:
        facilities: DataFrame from head_estimation.estimate_head() joined with
                    Phase 1/2 output.  Expected columns:
                      npdes_id, mean_flow_mgd (or q_design_m3s), h_net_m,
                      head_valid.  Optionally: flow_duration_curve (list).

    Returns:
        Input DataFrame with turbine sizing columns appended.
    """
    required = {"npdes_id", "head_net_m", "head_valid"}
    missing = required - set(facilities.columns)
    if missing:
        raise ValueError(f"select_and_size_turbines: missing columns {missing}")

    # Resolve design flow column  (prefer q_design_m3s, fall back to mean_flow_mgd)
    df = facilities
    if "q_design_m3s" not in df.columns:
        if "mean_flow_mgd" in df.columns:
            df = df.with_columns(
                (pl.col("mean_flow_mgd") * MGD_TO_M3S).alias("q_design_m3s")
            )
        else:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("q_design_m3s"))

    # FDC columns (list-of-floats stored as Polars List[Float64])
    has_fdc = "flow_duration_curve" in df.columns

    # Build O(1) lookup dict to avoid O(n²) DataFrame filter inside the row loop.
    # Phase 1 stores FDCs using _PHASE1_FDC_EXCEEDANCES (20-point); we always
    # pair the stored flows with that same exceedance grid for a length-safe match.
    fdc_lookup: dict[str, list[float]] = {}
    if has_fdc:
        for r in df.select(["npdes_id", "flow_duration_curve"]).to_dicts():
            raw = r["flow_duration_curve"]
            if raw is not None:
                fdc_lookup[r["npdes_id"]] = [float(v) * MGD_TO_M3S for v in raw]

    results: list[dict] = []
    for row in df.select([
        "npdes_id", "q_design_m3s", "head_net_m", "head_valid"
    ]).to_dicts():
        fdc_q: list[float] = fdc_lookup.get(row["npdes_id"], [])
        # Use Phase 1 exceedance grid (matches stored FDC length); fall back to
        # Phase 3 config probs for synthetic / test FDCs of different lengths.
        if fdc_q:
            fdc_e = _PHASE1_FDC_EXCEEDANCES[: len(fdc_q)]
        else:
            fdc_e = _cfg(
                "fdc_exceedance_probs",
                [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
            )

        results.append(_size_one(
            row["npdes_id"],
            row["q_design_m3s"],
            row["head_net_m"],
            bool(row["head_valid"]),
            fdc_q,
            fdc_e,
        ))

    sizing_df = pl.DataFrame({
        "turbine_type":        [r["turbine_type"]        for r in results],
        "q_rated_m3s":         [r["q_rated_m3s"]         for r in results],
        "rated_power_kw":      [r["rated_power_kw"]      for r in results],
        "peak_efficiency_pct": [r["peak_efficiency_pct"] for r in results],
        "annual_energy_mwh":   [r["annual_energy_mwh"]   for r in results],
        "capacity_factor":     [r["capacity_factor"]     for r in results],
        "best_manufacturer":   [r["best_manufacturer"]   for r in results],
        "turbine_viable":      [r["turbine_viable"]      for r in results],
    })

    result = df.hstack(sizing_df)

    # ── Summary logging ───────────────────────────────────────────────────────
    viable = result.filter(pl.col("turbine_viable"))
    if len(viable) > 0:
        by_type = (
            viable.group_by("turbine_type")
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
        )
        total_mwh = viable["annual_energy_mwh"].drop_nulls().sum()
        avg_kw    = viable["rated_power_kw"].drop_nulls().mean()
        log.info(
            f"Turbine sizing: {len(viable):,}/{len(result):,} viable sites | "
            f"total {total_mwh:,.0f} MWh/yr | avg {avg_kw:.0f} kW"
        )
        for row in by_type.to_dicts():
            log.info(f"  {row['turbine_type']:20s}: {row['n']:5,} sites")
    else:
        log.warning("Turbine sizing: no viable sites found")

    return result
