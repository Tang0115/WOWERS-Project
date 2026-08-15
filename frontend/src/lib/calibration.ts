/**
 * Capacity-factor calibration band.
 *
 * The pipeline's own energy figure (`annual_energy_kwh`) implies a fleet
 * capacity factor of 0.872. Benchmarked against 629 metered small-hydro
 * plants over 9,798 plant-years, that sits above the measured distribution
 * at every percentile — the measured river-hydro median is 0.390, and even
 * its 75th percentile is below the WOWERS 10th. The export therefore carries
 * four *calibrated* energy columns alongside the as-modeled one, each site's
 * energy re-scaled by (tier capacity factor / 0.872).
 *
 * Two numbers per tier must never be swapped, so they live in separate
 * structures here:
 *
 *   • Band energy (`CALIB_TIERS` + the per-site columns) — the fixed baseline
 *     cohort re-scaled by the multiplier. Derived live from the GeoJSON.
 *   • Re-scored outcome (`RESCORE_SCENARIOS`) — what survives when the full
 *     cash flow is re-run at that tier and sites that no longer clear the
 *     viability gate drop out. NOT derivable from the exported file, which
 *     carries calibrated energy but not calibrated cash flows.
 */

export type CalibKey = "ceiling" | "central" | "floor_p50" | "floor_p25" | "measured";

/** Calibrated energy per tier, in whatever unit the producing function states. */
export type CalibEnergy = Record<CalibKey, number | null>;

export interface CalibTier {
  key: CalibKey;
  /** Full name, used in tables. */
  label: string;
  /** Short name, used in tight spaces. */
  short: string;
  /** Capacity factor this tier substitutes. */
  cf: number;
  /** cf / 0.872 — what the site's energy is multiplied by. Not a capacity factor. */
  multiplier: number;
  /** Where the capacity factor comes from. */
  basis: string;
  /** Evidence strength, not desirability: grey = unmeasured, green = metered. */
  color: string;
}

/** Ordered strongest-energy (weakest evidence) first, matching the band's own range. */
export const CALIB_TIERS: CalibTier[] = [
  {
    key: "ceiling",
    label: "Ceiling — as modeled",
    short: "Ceiling",
    cf: 0.872,
    multiplier: 1.0,
    basis: "The Phase 2 physics model itself. No measured support at any percentile.",
    color: "#9ca3af",
  },
  {
    key: "central",
    label: "Optimistic — Conduit 3 projection",
    short: "Optimistic",
    cf: 0.5999,
    multiplier: 0.688,
    basis: "A single in-conduit project's design projection, not metered output.",
    color: "#eab308",
  },
  {
    key: "floor_p50",
    label: "Floor — river-hydro median",
    short: "Floor p50",
    cf: 0.3898,
    multiplier: 0.447,
    basis: "Median of 629 metered small-hydro plants (EHA/EIA-923).",
    color: "#60a5fa",
  },
  {
    key: "floor_p25",
    label: "Floor — river-hydro 25th percentile",
    short: "Floor p25",
    cf: 0.2538,
    multiplier: 0.291,
    basis: "First quartile of the same metered small-hydro fleet.",
    color: "#2563eb",
  },
  {
    key: "measured",
    label: "Band floor — metered Point Loma",
    short: "Metered floor",
    cf: 0.1914,
    multiplier: 0.2195,
    basis: "The one metered treated-wastewater conduit plant in the US (EIA-923, 2017).",
    color: "#1e9e5a",
  },
];

export const TIER_BY_KEY: Record<CalibKey, CalibTier> = Object.fromEntries(
  CALIB_TIERS.map((t) => [t.key, t]),
) as Record<CalibKey, CalibTier>;

/** The two tiers that two independent metered datasets agree on. */
export const PLANNING_TIERS: CalibKey[] = ["floor_p25", "measured"];

/**
 * Portfolio outcome when the 30-year cash flow is re-run at each tier and the
 * unmodified viability gate (NPV > 0, payback ≤ 20 yr, real IRR) is re-applied.
 *
 * Source: `scripts/tier_ladder_whatif.py`, scenario r = 6.0 % / grant = 0 %
 * (the pipeline default, and the assumption behind every financial figure
 * shown elsewhere in this dashboard). The ceiling row reproduces the P2-SEED
 * baseline exactly, which is that harness's own correctness check.
 *
 * These are constants rather than derived values because the exported GeoJSON
 * carries calibrated energy only — re-scoring cash flows needs the Phase 4
 * scorecard function, which does not run in the browser.
 */
export interface RescoreScenario {
  key: CalibKey;
  viable: number;
  gwh: number;
  npvUsd: number;
  capexUsd: number;
  medianPayback: number;
}

export const RESCORE_SCENARIOS: RescoreScenario[] = [
  { key: "ceiling", viable: 1138, gwh: 409.2, npvUsd: 310.1e6, capexUsd: 211.3e6, medianPayback: 9.8 },
  { key: "central", viable: 439, gwh: 213.8, npvUsd: 152.6e6, capexUsd: 132.4e6, medianPayback: 10.4 },
  { key: "floor_p50", viable: 129, gwh: 96.0, npvUsd: 59.4e6, capexUsd: 74.3e6, medianPayback: 10.7 },
  { key: "floor_p25", viable: 27, gwh: 30.0, npvUsd: 19.0e6, capexUsd: 31.1e6, medianPayback: 10.6 },
  { key: "measured", viable: 12, gwh: 14.9, npvUsd: 7.1e6, capexUsd: 19.8e6, medianPayback: 11.6 },
];

export const RESCORE_SOURCE =
  "Re-scored through the Phase 4 cash-flow model at r = 6.0 %, no grant — the same assumptions as every financial figure on this dashboard.";
