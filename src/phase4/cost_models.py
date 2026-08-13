"""Phase 4 — CapEx and OpEx cost models for small/micro hydropower.

Equipment CapEx model: power-law scaling from DOE Hydropower Vision & ORNL
conduit hydropower studies.

    CapEx_per_kW = max(min_per_kw, min(max_per_kw, A × rated_kW ^ B))
    Equipment_CapEx = CapEx_per_kW × rated_kW

Balance-of-system adders (F4-INTERCON, F4-PERMIT — added 2026-05):
    Interconnection_CapEx  = tier lookup by rated_kw   ($50k–$200k)
    Permitting_CapEx       = fixed adder if rated_kw < small_site_threshold

Installation cost (F4-INSTALL): mechanical install / labor as a % of equipment
CapEx. Scope: mechanical labor ONLY — excludes civil works; does NOT overlap
interconnection or permitting (those are separate lines). One-time cost; not
O&M-bearing. Reversible: set ``installation_pct_of_equipment = 0.0`` to disable.

Project (all-in) CapEx for NPV/IRR:
    Project_CapEx = Equipment_CapEx + Installation_CapEx
                  + Interconnection_CapEx + Permitting_CapEx

OpEx: annual O&M as a fixed percentage of *equipment* CapEx (per turbine type).
Interconnection, installation, and permitting are one-time costs, not O&M-bearing.

All parameters are read from ``config/settings.yaml`` under ``cost_model`` so
they can be updated without touching source code.

References
----------
DOE Water Power Technologies Office, "Hydropower Vision" (2016), Chapter 3;
ORNL TM-2014/525, "New Stream-reach Development" appendix cost tables;
FERC small-hydro interconnection cost surveys (industry-typical $50k–$200k
range for sites ≤ 1 MW);
DOE/EERE permitting / environmental review cost benchmarks for small hydro
(fixed ~$150k overhead is non-trivial relative to micro-hydro CapEx).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.common import config

# ── Load cost model config ────────────────────────────────────────────────────

_CM = config.get("cost_model", {})

# Global fallback (unknown turbine types)
_CAPEX_A:    float = float(_CM.get("capex_A",          9_500))
_CAPEX_B:    float = float(_CM.get("capex_B",         -0.35))
_MIN_PER_KW: float = float(_CM.get("capex_min_per_kw",   800))
_MAX_PER_KW: float = float(_CM.get("capex_max_per_kw", 10_000))

# Per-type A/B/min/max — loaded from settings.yaml; hard-coded fallbacks
# match ARCHITECTURE.md §4.2 and are used only if YAML keys are absent.
_YAML_TYPES: dict = config.get("cost_model.types", {}) or {}

def _load_type(name: str, default_A: float, default_B: float,
               default_min: float, default_max: float) -> dict:
    y = _YAML_TYPES.get(name, {})
    return {
        "A":   float(y.get("A",          default_A)),
        "B":   float(y.get("B",          default_B)),
        "min": float(y.get("min_per_kw", default_min)),
        "max": float(y.get("max_per_kw", default_max)),
    }

_TYPE_PARAMS: dict[str, dict] = {
    "Kaplan":           _load_type("Kaplan",           9_500,  -0.35,   800, 10_000),
    # Francis min clamped to vendor floor ($1,800/kW, Canyon/Gilkes) — F4-VENDORBAND
    "Francis":          _load_type("Francis",           8_500,  -0.32, 1_800,  9_000),
    "Pelton":           _load_type("Pelton",            7_000,  -0.30,   600,  8_000),
    # in_conduit_micro A/B recalibrated Jun 2026 vs ORNL BCM canal/conduit (TM-2014/525); R²=0.89
    "in_conduit_micro": _load_type("in_conduit_micro", 20_283,  -0.181, 2_000, 15_000),
    # Crossflow (Ossberger/CINK): simpler runner → lower cost than Kaplan
    "Crossflow":        _load_type("Crossflow",         7_500,  -0.28,   500,  7_500),
}

# OpEx fraction of CapEx per type
_OPEX_PCT: dict[str, float] = {}
for t in ("Kaplan", "Francis", "Pelton", "in_conduit_micro", "Crossflow"):
    pct = config.get(f"cost_model.opex_pct_of_capex.{t}")
    if pct is None:
        pct = {"Kaplan": 0.025, "Francis": 0.020, "Pelton": 0.015,
               "in_conduit_micro": 0.030, "Crossflow": 0.020}[t]
    _OPEX_PCT[t] = float(pct)

# F4-INSTALL: mechanical installation / labor cost fraction (of equipment CapEx).
# Default 0.0 so that missing config key preserves prior behavior (backward-compat).
_INSTALL_PCT: float = float(_CM.get("installation_pct_of_equipment", 0.0))


# ── F4-INTERCON: grid interconnection cost tiers ──────────────────────────────
# Tier lookup by rated_power_kw upper bound (inclusive).  Tier costs cover the
# utility-side study fee, transformer / protection equipment, switchgear, and
# metering for a typical < 1 MW behind-the-meter or distribution-tie project.
# Industry-typical range: $50k–$200k for small / micro hydro (FERC small-hydro
# interconnection cost surveys; NREL distributed generation cost handbook).
_INTERCON_DEFAULT_TIERS: list[dict] = [
    {"max_kw":       10.0, "cost_usd":  50_000.0},
    {"max_kw":       50.0, "cost_usd": 100_000.0},
    {"max_kw":      250.0, "cost_usd": 150_000.0},
    {"max_kw": 1_000_000.0, "cost_usd": 200_000.0},  # catch-all
]

_INTERCON_TIERS_RAW = config.get("cost_model.interconnection.tiers")
if _INTERCON_TIERS_RAW:
    _INTERCON_TIERS = [
        {"max_kw": float(t["max_kw"]), "cost_usd": float(t["cost_usd"])}
        for t in _INTERCON_TIERS_RAW
    ]
else:
    _INTERCON_TIERS = _INTERCON_DEFAULT_TIERS

# Sort ascending by max_kw so lookup is deterministic.
_INTERCON_TIERS.sort(key=lambda t: t["max_kw"])

# F4-BTM (2026-06-01): behind-the-meter self-consumption branch.
# Micro hydro at a WWTP outfall offsets the plant's own load behind the facility
# meter (no grid export), so it avoids the utility distribution-tie cost the
# tiers above assume.  For sites at or below ``max_kw`` the interconnection cost
# collapses to a small non-export-relay + disconnect + utility-notification cost.
# Disabled (None) if the YAML block is absent, preserving prior tier-only behavior.
_INTERCON_BTM_RAW = config.get("cost_model.interconnection.behind_the_meter")
if _INTERCON_BTM_RAW:
    _INTERCON_BTM_MAX_KW: float | None = float(_INTERCON_BTM_RAW["max_kw"])
    _INTERCON_BTM_COST:   float | None = float(_INTERCON_BTM_RAW["cost_usd"])
else:
    _INTERCON_BTM_MAX_KW = None
    _INTERCON_BTM_COST   = None


# ── F4-PERMIT (tiered, F4-PERMIT-TIER 2026-05-20) ─────────────────────────────
# Permitting / environmental review cost as a 3-tier step function of rated
# power.  Replaces the original single-step $150k-or-$0 model, which created
# an unrealistic cliff at 50 kW and made 92 % of POTW sites economically
# unviable in the first Phase 4 calibration run.
#
# Tier semantics (matches FERC small-hydro practice):
#   - qualified_facility (≤ 25 kW): qualifying conduit hydropower facility —
#     FERC Notice of Intent only (18 CFR 4.401), no license/fee, no exhibits;
#     cost is NOI legal/prep + state water-quality coordination (F4-CONDUIT $5k).
#   - small_ferc (25 < kW ≤ 250):  abbreviated FERC review, state water-quality
#     cert, light environmental survey.  Typical $50k–$100k.
#   - full_nepa  (kW > 250):       full FERC licensing or major exemption with
#     formal NEPA EA/EIS.  Typical $100k–$200k.
_PERMIT_DEFAULT_TIERS: list[dict] = [
    {"max_kw":        25.0, "cost_usd":   5_000.0, "label": "qualified_facility"},  # F4-CONDUIT
    {"max_kw":       250.0, "cost_usd":  75_000.0, "label": "small_ferc"},
    {"max_kw": 1_000_000.0, "cost_usd": 150_000.0, "label": "full_nepa"},  # catch-all
]

_PERMIT_TIERS_RAW = config.get("cost_model.permitting.tiers")
if _PERMIT_TIERS_RAW:
    _PERMIT_TIERS = [
        {
            "max_kw":   float(t["max_kw"]),
            "cost_usd": float(t["cost_usd"]),
            "label":    str(t.get("label", f"tier_{i}")),
        }
        for i, t in enumerate(_PERMIT_TIERS_RAW)
    ]
else:
    _PERMIT_TIERS = _PERMIT_DEFAULT_TIERS

# Sort ascending by max_kw so lookup is deterministic.
_PERMIT_TIERS.sort(key=lambda t: t["max_kw"])


# ── Public API ────────────────────────────────────────────────────────────────

def capex_per_kw(turbine_type: str, rated_power_kw: float) -> float:
    """Installed cost per kW using the power-law model.

    Args:
        turbine_type:   One of Kaplan, Francis, Pelton, in_conduit_micro, Crossflow.
        rated_power_kw: Nameplate power (kW).  Must be > 0.

    Returns:
        Installed cost per kW (USD/kW), clamped to [min_per_kw, max_per_kw].
    """
    p = _TYPE_PARAMS.get(turbine_type, _TYPE_PARAMS["Kaplan"])
    if rated_power_kw <= 0:
        return float(p["max"])  # per-type max, not global fallback
    raw = p["A"] * (rated_power_kw ** p["B"])
    return max(p["min"], min(p["max"], raw))


def total_capex(turbine_type: str, rated_power_kw: float) -> float:
    """Total installed CapEx (USD).

    Args:
        turbine_type:   Turbine type string.
        rated_power_kw: Nameplate power (kW).

    Returns:
        Total capital expenditure in USD.
    """
    return capex_per_kw(turbine_type, rated_power_kw) * rated_power_kw


def annual_opex(turbine_type: str, capex_usd: float) -> float:
    """Annual O&M cost (USD/yr) as a percentage of installed CapEx.

    Args:
        turbine_type: Turbine type string.
        capex_usd:    Equipment CapEx (USD).  Do NOT pass project CapEx —
                      interconnection and permitting are one-time costs.

    Returns:
        Annual O&M cost (USD/yr).
    """
    pct = _OPEX_PCT.get(turbine_type, 0.025)
    return capex_usd * pct


# ── F4-INTERCON ───────────────────────────────────────────────────────────────

def interconnection_cost(rated_power_kw: float) -> float:
    """Grid interconnection cost (USD) — F4-INTERCON.

    Tier lookup by rated power.  Covers utility study fee, step-up transformer,
    protection relays, switchgear, and revenue-grade metering for a typical
    distribution-tie small / micro hydro project.

    F4-BTM: sites at or below ``behind_the_meter.max_kw`` (when configured) are
    treated as behind-the-meter self-consumption (no grid export) and return the
    small ``behind_the_meter.cost_usd`` instead of a distribution-tie tier cost.

    Args:
        rated_power_kw: Nameplate power (kW).  Must be > 0; non-positive
            inputs return the lowest tier cost as a safety floor.

    Returns:
        One-time interconnection CapEx (USD).
    """
    if rated_power_kw <= 0:
        return float(_INTERCON_TIERS[0]["cost_usd"])
    # F4-BTM: behind-the-meter micro sites skip the distribution-tie cost.
    if _INTERCON_BTM_MAX_KW is not None and rated_power_kw <= _INTERCON_BTM_MAX_KW:
        return float(_INTERCON_BTM_COST)
    for tier in _INTERCON_TIERS:
        if rated_power_kw <= tier["max_kw"]:
            return float(tier["cost_usd"])
    # Above every tier (shouldn't happen with catch-all) — return the top.
    return float(_INTERCON_TIERS[-1]["cost_usd"])


# ── F4-PERMIT (tiered) ────────────────────────────────────────────────────────

def _lookup_permit_tier(rated_power_kw: float) -> dict:
    """Return the matching permitting tier dict (or the lowest tier if input ≤ 0)."""
    if rated_power_kw <= 0:
        return _PERMIT_TIERS[0]
    for tier in _PERMIT_TIERS:
        if rated_power_kw <= tier["max_kw"]:
            return tier
    # Above every tier (shouldn't happen with catch-all) — return the top.
    return _PERMIT_TIERS[-1]


def permitting_cost(rated_power_kw: float) -> float:
    """Permitting / environmental review cost (USD) — F4-PERMIT (tiered).

    Tier lookup by rated power.  Replaces the original single-step model.
    Default tiers (qualified_facility recalibrated 2026-06-01, F4-CONDUIT):
        ≤  25 kW  →   $5,000  (qualifying conduit facility — FERC NOI, no fee)
        ≤ 250 kW  →  $75,000  (abbreviated FERC review)
        >  250 kW → $150,000  (full FERC licensing + NEPA review)

    Args:
        rated_power_kw: Nameplate power (kW).  Non-positive inputs return
            the lowest tier cost as a safety floor.

    Returns:
        One-time permitting CapEx (USD).
    """
    return float(_lookup_permit_tier(rated_power_kw)["cost_usd"])


def permitting_tier_label(rated_power_kw: float) -> str:
    """Return the permitting tier label string for ``rated_power_kw``.

    Used as a categorical column in the Phase 4 output parquet for cohort
    segmentation downstream.
    """
    return str(_lookup_permit_tier(rated_power_kw)["label"])


# ── F4-VENDORBAND: vendor $/kW sanity cross-check ─────────────────────────────
# The equipment power law (capex_per_kw) uses literature-anchored A/B
# coefficients that have never been fit to or validated against real installs.
# data/turbines/turbine_manufacturers.csv carries vendor-published
# capex_usd_per_kw_low/high ranges per turbine type (data_source =
# manufacturer_website) — a more trustable anchor.  This cross-check loads those
# ranges and flags any site whose power-law $/kW falls outside the vendor band
# for its turbine type, so we can quantify how often the model mis-prices using
# data we already own.  Read-only: does NOT change the CapEx fed to financials.

_TURBINE_DB_PATH: Path = config.project_root() / config.get(
    "phase3.turbine_db_path", "data/turbines/turbine_manufacturers.csv"
)

# Per-type vendor band: turbine_type -> (low_usd_per_kw, high_usd_per_kw).
# low = min of capex_usd_per_kw_low across manufacturers of that type;
# high = max of capex_usd_per_kw_high.  Widest defensible envelope.
_VENDOR_BAND: dict[str, tuple[float, float]] = {}


def _load_vendor_bands() -> dict[str, tuple[float, float]]:
    if not _TURBINE_DB_PATH.exists():
        return {}
    df = pl.read_csv(_TURBINE_DB_PATH)
    needed = {"turbine_type", "capex_usd_per_kw_low", "capex_usd_per_kw_high"}
    if not needed.issubset(df.columns):
        return {}
    agg = (
        df.group_by("turbine_type")
        .agg(
            pl.col("capex_usd_per_kw_low").min().alias("low"),
            pl.col("capex_usd_per_kw_high").max().alias("high"),
        )
    )
    bands: dict[str, tuple[float, float]] = {}
    for r in agg.to_dicts():
        lo, hi = r["low"], r["high"]
        if lo is not None and hi is not None:
            bands[str(r["turbine_type"])] = (float(lo), float(hi))
    return bands


_VENDOR_BAND = _load_vendor_bands()


def vendor_capex_band(turbine_type: str) -> tuple[float, float] | None:
    """Vendor-published equipment $/kW band for ``turbine_type``.

    Returns (low, high) in USD/kW aggregated across manufacturers, or None if
    the turbine type has no vendor data on file.
    """
    return _VENDOR_BAND.get(turbine_type)


def capex_vs_vendor_band(turbine_type: str, rated_power_kw: float) -> dict:
    """Cross-check power-law equipment $/kW against the vendor band — F4-VENDORBAND.

    Returns:
        Dict with keys:
            ``model_capex_per_kw``        — power-law equipment $/kW (the model)
            ``vendor_capex_per_kw_low``   — vendor band low (or None)
            ``vendor_capex_per_kw_high``  — vendor band high (or None)
            ``capex_outside_vendor_band`` — True if model $/kW outside [low, high].
                                            False when no vendor band exists
                                            (cannot judge), so the flag only ever
                                            marks confirmed divergence.
    """
    model_per_kw = capex_per_kw(turbine_type, rated_power_kw)
    band = vendor_capex_band(turbine_type)
    if band is None:
        return {
            "model_capex_per_kw":        model_per_kw,
            "vendor_capex_per_kw_low":   None,
            "vendor_capex_per_kw_high":  None,
            "capex_outside_vendor_band": False,
        }
    lo, hi = band
    return {
        "model_capex_per_kw":        model_per_kw,
        "vendor_capex_per_kw_low":   lo,
        "vendor_capex_per_kw_high":  hi,
        "capex_outside_vendor_band": bool(model_per_kw < lo or model_per_kw > hi),
    }


# ── Project CapEx aggregation ─────────────────────────────────────────────────

def project_capex(turbine_type: str, rated_power_kw: float) -> dict[str, float | str]:
    """Full project CapEx breakdown including all balance-of-system adders.

    The power-law equipment cost is treated as *equipment-only* (clamped to
    vendor equipment bands; see ``capex_vs_vendor_band``).  The installation
    line sits on top of it and is NOT included in the vendor-band check.

    Returns:
        Dict with keys:
            ``equipment_capex_usd``       — power-law turbine equipment-only cost
            ``installation_capex_usd``    — F4-INSTALL mechanical labor
                                            (``_INSTALL_PCT`` × equipment; scope:
                                            mechanical install/labor ONLY, excludes
                                            civil works, interconnection, permitting)
            ``interconnection_capex_usd`` — F4-INTERCON tier lookup
            ``permitting_capex_usd``      — F4-PERMIT-TIER tier lookup
            ``permitting_tier``           — F4-PERMIT-TIER label (categorical)
            ``total_project_capex_usd``   — equipment + installation
                                            + interconnection + permitting
    """
    eq      = total_capex(turbine_type, rated_power_kw)
    install = eq * _INSTALL_PCT
    intc    = interconnection_cost(rated_power_kw)
    perm    = permitting_cost(rated_power_kw)
    label   = permitting_tier_label(rated_power_kw)
    return {
        "equipment_capex_usd":       eq,
        "installation_capex_usd":    install,
        "interconnection_capex_usd": intc,
        "permitting_capex_usd":      perm,
        "permitting_tier":           label,
        "total_project_capex_usd":   eq + install + intc + perm,
    }
