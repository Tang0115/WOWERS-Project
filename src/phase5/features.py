"""Phase 5 — WWTP scoring feature matrix (ARCH §5.3 step 2) + leakage lock.

D2 — build_feature_matrix
--------------------------
Joins Phase 1–4 parquets on ``npdes_id`` into a wide scoring matrix.

Spine: Phase 1 (17,148 POTWs post P1-COORD-GUARD). Phase 3 covers 5,464 of them;
Phase 4 covers 3,780.  The remaining rows have nulls on Phase 3/4 columns — this is expected
and intentional. Do NOT impute at this stage; leave nulls for LightGBM's
native missing-value handling.

OPEN TENSION (do NOT resolve here — human decision)
----------------------------------------------------
The labeled training rows are EIA/EHA dam plants. Each carries only:
  - ``actual_installed_kw`` (capacity)
  - ``latitude``, ``longitude``, ``state_code``
  - ``ground_truth_source``, ``source_plant_code``

The transferable feature intersection between the label set and the full
60-col scoring matrix is approximately::

    {latitude, longitude, state_code, actual_installed_kw}

The full 60-column matrix is a scoring-time artifact. Training on it would
require matching ground-truth plants to ECHO POTW records via spatial/fuzzy
matching (a separate future step), at which point POTW flow, head, and
physics-estimate features become available. Until that matching step is done,
the feature set for training is the small intersection above, which may be
insufficient for a useful ML model. See D3 ``select_model_features`` and
``allow_physics_estimate_feature`` toggle.

D3 — leakage lock
-----------------
Target = ``log(actual_annual_energy_kwh)``.

Any column that is a monotone function of the physics energy estimate is
leakage. Verified: ``annual_revenue_usd`` (Phase 4) correlates 0.915 with
``actual_annual_energy_kwh`` because revenue = energy × rate — it IS energy.

``select_model_features(df, allow_physics_estimate=False)`` drops the denylist
and, by default, also drops the physics estimate columns. Set
``allow_physics_estimate=True`` only when running the "physics-as-baseline"
comparison experiment; the flag is the unresolved methodology decision —
default is safe (no leakage).
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from src.common import config, io, logging_setup

logging_setup.setup_run_log("phase5")
log = logging_setup.get("wowers.phase5.features")

# ── Output path ───────────────────────────────────────────────────────────────

_FEATURE_MATRIX_PATH: Path = (
    config.processed_dir() / "phase5" / "feature_matrix.parquet"
)

# ── Phase 1–4 input paths ─────────────────────────────────────────────────────

_P1_PATH: Path = config.processed_dir() / "phase1" / "ranked_candidates.parquet"
_P2_PATH: Path = config.processed_dir() / "phase2" / "energy_yield_estimates.parquet"
_P3_PATH: Path = config.processed_dir() / "phase3" / "turbine_sizing.parquet"
_P4_PATH: Path = config.processed_dir() / "phase4" / "financial_scorecards.parquet"


# ══════════════════════════════════════════════════════════════════════════════
# D3 — Leakage denylist and guards
# ══════════════════════════════════════════════════════════════════════════════

# Columns that are monotone transforms of the target log(actual_annual_energy_kwh)
# or are derived FROM the physics energy estimate in a way that trivially encodes
# it (revenue = energy × rate → corr = 0.915 verified empirically).
#
# Verified correlations with actual_annual_energy_kwh on EIA ground truth:
#   annual_revenue_usd       : 0.915  (revenue = energy × rate — IS energy)
#   npv_usd                  : high   (NPV dominated by revenue stream)
#   capacity_factor          : direct (capacity_factor = energy / (installed_kw × 8760))
#   annual_energy_kwh        : 1.000  (Phase 4 physics estimate of the target)
#   annual_energy_mwh        : 1.000  (same variable, different units, Phase 3)
#   energy_p50_kwh_yr        : ~0.99  (Phase 2 Monte Carlo mean)
#   energy_p10_kwh_yr etc.   : ~0.98+
#
# "Physics estimate" columns (kept behind the allow_physics_estimate toggle):
PHYSICS_ESTIMATE_COLS: frozenset[str] = frozenset({
    "annual_energy_kwh",        # Phase 4 output (= Phase 3 estimate in kWh)
    "annual_energy_mwh",        # Phase 3 turbine selection output
    "energy_p10_kwh_yr",        # Phase 2 Monte Carlo
    "energy_p50_kwh_yr",
    "energy_p90_kwh_yr",
    "energy_mean_kwh_yr",
    "energy_std_kwh_yr",
    "power_p50_kw",             # Phase 2 Monte Carlo power
    "capacity_factor_p50",      # Phase 2 (= energy / (cap × 8760))
})

# Unconditionally denylisted — always dropped regardless of toggle.
# Includes anything whose correlation with the target is above ~0.7 by construction
# (revenue/financial columns) or that is a direct label-derived artifact.
LEAKAGE_DENYLIST: frozenset[str] = frozenset({
    # Phase 4 financial outputs derived from energy estimate
    "annual_revenue_usd",
    "annual_net_cf_usd",
    "npv_usd",
    "npv_with_50pct_grant_usd",
    "irr",
    "payback_years",
    "lcoe_per_kwh",
    # Viability/decision flags derived from the financial model
    "project_viable",
    "project_viable_high_confidence",
    "site_tier",
    "econ_cat_payback",
    "econ_cat_npv",
    "econ_cat_irr",
    # Capacity factor — encodes energy (CF = energy / (cap × 8760))
    "capacity_factor",
    # Energy consumption / offset — computed from energy estimate
    "energy_offset_pct",
    "energy_offset_pct_low",
    "energy_offset_pct_high",
    "est_plant_consumption_kwh_yr",
    "est_plant_consumption_low_kwh_yr",
    "est_plant_consumption_high_kwh_yr",
    # Sensitivity columns derived from NPV (and thus from energy)
    "sensitivity_head_npv_low",
    "sensitivity_head_npv_high",
    "sensitivity_flow_npv_low",
    "sensitivity_flow_npv_high",
    "sensitivity_rate_npv_low",
    "sensitivity_rate_npv_high",
    "dominant_sensitivity",
}) | PHYSICS_ESTIMATE_COLS  # physics estimates are always in the denylist by default


def assert_no_leakage(feature_names: list[str]) -> None:
    """Raise ``ValueError`` if any denylisted column is in ``feature_names``.

    Called at the top of ``nested_cv`` to refuse training on a leaky feature
    set.  Must be called after ``select_model_features`` to be safe.

    Args:
        feature_names: List of column names in the feature matrix passed to
            the model.

    Raises:
        ValueError: Lists every denylisted name found; fails loud so nothing
            trains silently with leakage.
    """
    bad = [f for f in feature_names if f in LEAKAGE_DENYLIST]
    if bad:
        raise ValueError(
            f"Leakage detected — the following feature(s) are in LEAKAGE_DENYLIST "
            f"and must be removed before training: {bad}"
        )


def select_model_features(
    df: pl.DataFrame,
    *,
    allow_physics_estimate: bool = False,
) -> list[str]:
    """Return the list of column names safe to use as model features.

    Drops ``LEAKAGE_DENYLIST`` unconditionally.  When ``allow_physics_estimate``
    is ``False`` (the default — no leakage), the physics energy/power estimate
    columns (``PHYSICS_ESTIMATE_COLS``) are also dropped.  When ``True``,
    retains them as inputs so the model can learn the residual between the
    physics estimate and the measured label — the "physics-as-baseline"
    comparison experiment.

    **This flag is the unresolved methodology decision (ARCH §5.3 step 3):**

    - ``False`` (default, safe): model learns entirely from flow/head/geography.
      Pros: no risk of leakage; valid if physics model is systematically biased.
      Cons: much smaller feature set; harder to beat physics baseline.
    - ``True`` (physics-as-feature): model learns correction on top of the
      physics estimate. Pros: can exploit physics structure; likely higher R².
      Cons: training labels (dams) have very different physics than WWTP
      outfalls; correction may not transfer across the dam→WWTP domain shift.

    The human must choose before any HP sweep or model comparison.

    Args:
        df:                    Feature matrix DataFrame (from ``build_feature_matrix``).
        allow_physics_estimate: See above.  Default ``False`` (no leakage).

    Returns:
        List of safe column names (strings), in their original order.
    """
    deny = LEAKAGE_DENYLIST.copy()
    if allow_physics_estimate:
        # Physics estimate columns are conditionally allowed — remove them from
        # the effective denylist for this call only.
        deny = deny - PHYSICS_ESTIMATE_COLS

    safe = [c for c in df.columns if c not in deny]
    n_dropped = len(df.columns) - len(safe)
    log.info(
        "select_model_features: %d/%d columns selected "
        "(dropped %d denylisted; allow_physics_estimate=%s)",
        len(safe), len(df.columns), n_dropped, allow_physics_estimate,
    )
    return safe


# ══════════════════════════════════════════════════════════════════════════════
# D2 — build_feature_matrix: pure transform + derived/interaction/geo features
# ══════════════════════════════════════════════════════════════════════════════

# Köppen–Geiger climate-zone approximation from latitude/longitude.
# Zones used here: tropical (<23.5°), subtropical (23.5–35°), temperate
# (35–50°), continental (50–66.5°), polar (>66.5°).  Coarse but deterministic
# and zero-leakage — captures seasonal flow regime differences.
def _climate_zone(lat: float | None) -> str | None:
    """Map latitude to a coarse climate zone label."""
    if lat is None or math.isnan(lat):
        return None
    a = abs(lat)
    if a < 23.5:
        return "tropical"
    if a < 35.0:
        return "subtropical"
    if a < 50.0:
        return "temperate"
    if a < 66.5:
        return "continental"
    return "polar"


def _add_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add interaction and geographic features that are not leakage.

    All derived features are constructed from inputs that are NOT monotone
    transforms of the energy target.  Energy-derived columns (annual_energy_*,
    revenue, NPV…) are excluded here and locked out by ``LEAKAGE_DENYLIST``.

    Returns the input DataFrame with additional columns appended.
    """
    exprs: list[pl.Expr] = []

    # flow × head interaction (proxy for power potential, not the physics calc)
    if "mean_flow_mgd" in df.columns and "head_net_m" in df.columns:
        exprs.append(
            (pl.col("mean_flow_mgd") * pl.col("head_net_m")).alias("flow_x_head")
        )

    # rated power density: kW per MGD of design flow
    if "rated_power_kw" in df.columns and "design_flow_mgd" in df.columns:
        exprs.append(
            (
                pl.col("rated_power_kw")
                / pl.col("design_flow_mgd").replace(0, None)
            ).alias("power_density_kw_per_mgd")
        )

    # revenue per kW (uses elec_rate which is a driver, not a derived output)
    # NOTE: annual_revenue_usd itself IS leakage; revenue_per_kw using
    # elec_rate × rated_power is safe because it doesn't encode observed energy.
    if "elec_rate_per_kwh" in df.columns and "rated_power_kw" in df.columns:
        exprs.append(
            (
                pl.col("elec_rate_per_kwh")
                * pl.col("rated_power_kw")
                * 8_760          # max annual kWh at rated power
            ).alias("revenue_potential_per_kw")
        )

    # climate zone from latitude (coarse Köppen proxy)
    if "latitude" in df.columns:
        exprs.append(
            pl.col("latitude")
            .map_elements(_climate_zone, return_dtype=pl.Utf8)
            .alias("climate_zone")
        )

    if exprs:
        df = df.with_columns(exprs)
    return df


# Columns to pull from each phase.  None = bring all non-npdes_id columns.
# P1 is the left-spine; later phases join on npdes_id (left join).
_P1_COLS: list[str] | None = None  # all P1 columns (spine)
_P2_COLS = [
    "npdes_id", "archetype", "head_m_p50",
    "energy_p10_kwh_yr", "energy_p50_kwh_yr", "energy_p90_kwh_yr",
    "energy_mean_kwh_yr", "energy_std_kwh_yr",
    "power_p50_kw", "capacity_factor_p50",
    "excluded", "exclusion_reason",
]
_P3_COLS = [
    "npdes_id", "head_gross_m", "head_net_m", "head_source", "head_confidence",
    "elevation_m", "elev_outfall_m",
    "turbine_type", "q_rated_m3s", "rated_power_kw",
    "peak_efficiency_pct", "annual_energy_mwh",
    "turbine_viable",
]
_P4_COLS = [
    "npdes_id", "capex_per_kw", "equipment_capex_usd",
    "installation_capex_usd", "interconnection_capex_usd", "permitting_capex_usd",
    "total_capex_usd", "annual_opex_usd", "elec_rate_per_kwh",
    "data_quality_tier", "permitting_tier",
    # leakage cols are explicitly excluded here (see LEAKAGE_DENYLIST)
    # annual_revenue_usd, npv_usd, irr, payback_years, lcoe_per_kwh,
    # annual_energy_kwh, capacity_factor, etc. → handled by select_model_features
]


def build_feature_matrix(
    phase1: pl.DataFrame,
    phase2: pl.DataFrame,
    phase3: pl.DataFrame,
    phase4: pl.DataFrame,
) -> pl.DataFrame:
    """Join Phase 1–4 frames into the WWTP scoring feature matrix.

    Spine = Phase 1 (17,148 POTWs post P1-COORD-GUARD).  Phase 2/3/4 are left-joined on ``npdes_id``.
    Rows absent from later phases have nulls on those columns — do NOT impute;
    LightGBM handles missingness natively.

    Derived interaction and geographic features are appended after the join.

    OPEN TENSION (do not resolve here):
        The labeled training set (EIA/EHA dams) has ~4 features that
        overlap with the scoring matrix (lat, lon, state, capacity).
        The full 60-col matrix requires spatial/fuzzy matching to ECHO plants
        to become available for training.  See module docstring.

    Args:
        phase1: Phase 1 ranked_candidates (17,148 POTWs spine post P1-COORD-GUARD).
        phase2: Phase 2 energy_yield_estimates.
        phase3: Phase 3 turbine_sizing.
        phase4: Phase 4 financial_scorecards.

    Returns:
        Wide feature matrix with ~60+ columns, one row per POTW.
        No imputations; nulls preserved.
    """
    # Restrict to declared column sets where specified
    p2 = phase2.select([c for c in _P2_COLS if c in phase2.columns])
    p3 = phase3.select([c for c in _P3_COLS if c in phase3.columns])
    p4 = phase4.select([c for c in _P4_COLS if c in phase4.columns])

    # Suffix strategy: later phases may carry duplicate cols (e.g. state_code,
    # data_quality) — rename with phase suffix before join to avoid collision.
    def _suffix(df: pl.DataFrame, tag: str) -> pl.DataFrame:
        return df.rename(
            {c: f"{c}_{tag}" for c in df.columns if c != "npdes_id" and c in phase1.columns}
        )

    p2 = _suffix(p2, "p2")
    p3 = _suffix(p3, "p3")
    p4 = _suffix(p4, "p4")

    mat = (
        phase1
        .join(p2, on="npdes_id", how="left")
        .join(p3, on="npdes_id", how="left")
        .join(p4, on="npdes_id", how="left")
    )

    mat = _add_derived_features(mat)
    log.info(
        "build_feature_matrix: %d rows × %d cols "
        "(P3 coverage %d, P4 coverage %d)",
        mat.height, len(mat.columns),
        mat["turbine_viable"].is_not_null().sum() if "turbine_viable" in mat.columns else 0,
        mat["total_capex_usd"].is_not_null().sum() if "total_capex_usd" in mat.columns else 0,
    )
    return mat


# ── Runner ────────────────────────────────────────────────────────────────────

def run_features(
    p1_path: Path | None = None,
    p2_path: Path | None = None,
    p3_path: Path | None = None,
    p4_path: Path | None = None,
) -> Path:
    """Build the feature matrix from Phase 1–4 parquets and write to disk.

    Args:
        p1_path … p4_path: Override default parquet paths (testing / CI only).

    Returns:
        Path to feature_matrix.parquet.
    """
    p1 = io.read_parquet(p1_path or _P1_PATH)
    p2 = io.read_parquet(p2_path or _P2_PATH)
    p3 = io.read_parquet(p3_path or _P3_PATH)
    p4 = io.read_parquet(p4_path or _P4_PATH)

    mat = build_feature_matrix(p1, p2, p3, p4)
    out = Path(config.get("phase5.feature_matrix_path") or str(_FEATURE_MATRIX_PATH))
    out.parent.mkdir(parents=True, exist_ok=True)
    io.write_parquet(mat, out)
    log.info("Feature matrix written: %s", out)
    return out
