/**
 * WOWERS data layer — single-file adapter (GEOJSON-UNIFY).
 *
 * Imports ``exports/scored_sites.geojson`` (git-tracked, all 3,778 scored
 * sites — viable and non-viable — 59 properties each) as a ?url import and
 * fetches it *once* via a module-level cached promise.  All four legacy
 * shapes — PlantCollection, National, Portfolio, PlantDetail — are derived
 * client-side from that one file.  The map shows every scored site (band
 * coloring includes the grey non-viable dots); national/portfolio KPI
 * aggregates filter to ``project_viable`` first, matching export_web_data.py.
 *
 * The view components (NationalMap, StatePortfolio, PlantDetail, MapView)
 * are unchanged: they call the same function names with the same signatures.
 */

import { useEffect, useState } from "react";
import type { CalibEnergy } from "./calibration";
import { CALIB_TIERS } from "./calibration";
import type {
  Band,
  Confidence,
  National,
  PlantCollection,
  PlantDetail,
  Portfolio,
  TableRow,
  TopOpportunity,
} from "./types";

// ── GeoJSON URL import (resolved by Vite at build/dev time) ──────────────────

import sitesUrl from "../../../exports/scored_sites.geojson?url";

// ── Internal shape for the raw geojson properties (59 fields) ────────────────
// P4-MEASURED-FLOOR (2026-08-06): energy_kwh_calib_measured_point_loma added.
// It carries the reported calibration-band floor (89.8 GWh/yr fleet-wide, from
// the only metered treated-wastewater conduit plant in the U.S.) and is the
// lowest of the four calibrated energy tiers.

interface SiteProps {
  npdes_id: string;
  facility_name: string | null;
  city: string | null;
  state_code: string | null;
  turbine_type: string | null;
  rated_power_kw: number | null;
  annual_energy_kwh: number | null;
  energy_kwh_calib_floor_p25: number | null;
  energy_kwh_calib_floor_p50: number | null;
  energy_kwh_calib_central: number | null;
  energy_kwh_calib_measured_point_loma: number | null;
  capacity_factor: number | null;
  total_capex_usd: number | null;
  npv_usd: number | null;
  irr: number | null;
  payback_years: number | null;
  lcoe_per_kwh: number | null;
  annual_revenue_usd: number | null;
  energy_offset_pct: number | null;
  site_tier: string | null;
  econ_cat_payback: string | null;
  econ_cat_npv: string | null;
  econ_cat_irr: string | null;
  data_quality_tier: number | null;
  project_viable: boolean;
  // P1 flow stats
  mean_flow_mgd: number | null;
  p10_flow_mgd: number | null;
  p90_flow_mgd: number | null;
  n_months_data: number | null;
  utilization_ratio: number | null;
  // P2 MC energy band
  energy_p10_kwh_yr: number | null;
  energy_p50_kwh_yr: number | null;
  energy_p90_kwh_yr: number | null;
  equivalent_homes_p50: number | null;
  // P3 elevation + turbine
  head_net_m: number | null;
  head_gross_m: number | null;
  elevation_m: number | null;
  elev_outfall_m: number | null;
  head_source: string | null;
  head_confidence: string | null;
  q_rated_m3s: number | null;
  peak_efficiency_pct: number | null;
  // P4 extended financials
  npv_with_50pct_grant_usd: number | null;
  annual_opex_usd: number | null;
  elec_rate_per_kwh: number | null;
  equipment_capex_usd: number | null;
  installation_capex_usd: number | null;
  interconnection_capex_usd: number | null;
  permitting_capex_usd: number | null;
  permitting_tier: string | null;
  sensitivity_head_npv_low: number | null;
  sensitivity_head_npv_high: number | null;
  sensitivity_flow_npv_low: number | null;
  sensitivity_flow_npv_high: number | null;
  sensitivity_rate_npv_low: number | null;
  sensitivity_rate_npv_high: number | null;
  dominant_sensitivity: string | null;
  data_quality: string | null;
  project_viable_high_confidence: boolean;
}

interface SiteFeature {
  type: "Feature";
  geometry: { type: "Point"; coordinates: [number, number] };
  properties: SiteProps;
}

interface SiteCollection {
  type: "FeatureCollection";
  features: SiteFeature[];
  meta: { plants_analyzed: number; scored_sites: number; baseline: string };
}

// ── Module-level cache ────────────────────────────────────────────────────────

let _cache: Promise<SiteCollection> | null = null;

function loadSites(): Promise<SiteCollection> {
  if (!_cache) {
    _cache = fetch(sitesUrl)
      .then((r) => {
        if (!r.ok) throw new Error(`GeoJSON fetch failed: ${r.status} ${r.url}`);
        return r.json() as Promise<SiteCollection>;
      })
      .catch((e) => {
        _cache = null; // don't cache the rejection — a retry can succeed
        throw e;
      });
  }
  return _cache;
}

// ── Classification helpers (mirrors export_web_data.py logic) ─────────────────

// Pipeline uses 1e6 as a "never pays back" sentinel; treat >= 1e5 as null
// and round to 2 d.p. — export_web_data.py did both BEFORE banding, so the
// band thresholds must see the rounded value (4.9996 → 5.0 → moderate).
function payback(v: number | null): number | null {
  return v != null && v < 1e5 ? Math.round(v * 100) / 100 : null;
}

// Payback alone is not viability: the gate is NPV > 0 AND payback <= 20 yr AND
// a real IRR, and payback is undiscounted while NPV is not. 865 scored sites
// clear the payback thresholds while failing the gate — 215 of them landing in
// the 5-15 yr "moderately viable" band with a negative NPV. Colouring those
// amber made the map claim a verdict the pipeline had refused, so a site that
// fails the gate is grey whatever its payback.
function viabilityBand(pb: number | null, viable: boolean): Band {
  if (!viable || pb === null) return "nonviable";
  if (pb < 5) return "high";
  if (pb <= 15) return "moderate";
  if (pb <= 20) return "marginal";
  return "nonviable";
}

// The four calibrated energy columns, plus the as-modeled ceiling, keyed by
// tier. Read straight from the export — the browser never re-scales anything
// itself, so a tier's value is whatever the pipeline wrote for it.
function calibOf(p: SiteProps): CalibEnergy {
  return {
    ceiling: p.annual_energy_kwh,
    central: p.energy_kwh_calib_central,
    floor_p50: p.energy_kwh_calib_floor_p50,
    floor_p25: p.energy_kwh_calib_floor_p25,
    measured: p.energy_kwh_calib_measured_point_loma,
  };
}

function siteConfidence(p: SiteProps): Confidence {
  if (p.project_viable_high_confidence || p.data_quality === "dmr") return "High";
  if (p.data_quality === "dmr_limited") return "Medium";
  return "Lower";
}

// ── Utility math ──────────────────────────────────────────────────────────────

function _median1dp(values: number[]): number | null {
  if (values.length === 0) return null;
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  const m = s.length % 2 === 0 ? (s[mid - 1] + s[mid]) / 2 : s[mid];
  return Math.round(m * 10) / 10;
}

function _sum(values: (number | null)[]): number {
  return values.reduce<number>((acc, v) => acc + (v ?? 0), 0);
}

// ── Public hook (unchanged API) ───────────────────────────────────────────────

export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    setData(null);
    setError(null);
    fn()
      .then((d) => alive && setData(d))
      .catch((e) => alive && setError(String(e)));
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return { data, error };
}

// ── fetchPlants ───────────────────────────────────────────────────────────────

export async function fetchPlants(): Promise<PlantCollection> {
  const sites = await loadSites();
  const features = sites.features.map((f) => {
    const p = f.properties;
    const pb = payback(p.payback_years);
    return {
      type: "Feature" as const,
      geometry: f.geometry,
      properties: {
        id: p.npdes_id,
        name: p.facility_name,
        city: p.city,
        state: p.state_code,
        turbine: p.turbine_type,
        rated_kw: p.rated_power_kw,
        energy_mwh:
          p.annual_energy_kwh != null
            ? Math.round(p.annual_energy_kwh / 1e3)
            : null,
        energy_kwh: p.annual_energy_kwh,
        energy_calib: calibOf(p),
        payback: pb,
        npv: p.npv_usd,
        tier: p.site_tier,
        viable: p.project_viable,
        band: viabilityBand(pb, p.project_viable),
        confidence: siteConfidence(p),
        flow_measured: p.data_quality === "dmr",
        head_measured: p.head_source === "usgs_3dep",
      },
    };
  });
  return { type: "FeatureCollection", features };
}

// ── fetchNational ─────────────────────────────────────────────────────────────

export async function fetchNational(): Promise<National> {
  const sites = await loadSites();
  const all = sites.features.map((f) => f.properties);
  // KPI aggregates are over viable sites only (mirrors export_web_data.py);
  // tier_a_sites is over all scored sites, matching the old national.json.
  const vs = all.filter((p) => p.project_viable);

  // by_state: count viable per state, sorted desc
  const stateMap = new Map<string, number>();
  for (const p of vs) {
    const s = p.state_code;
    if (s) stateMap.set(s, (stateMap.get(s) ?? 0) + 1);
  }
  const by_state = [...stateMap.entries()]
    .map(([state, viable]) => ({ state, viable }))
    .sort((a, b) => b.viable - a.viable);

  // top 5 viable by npv_usd
  const sorted = [...vs].sort(
    (a, b) => (b.npv_usd ?? -Infinity) - (a.npv_usd ?? -Infinity),
  );
  const top_opportunities: TopOpportunity[] = sorted.slice(0, 5).map((p) => ({
    id: p.npdes_id,
    name: p.facility_name,
    city: p.city,
    state: p.state_code,
    energy_mwh:
      p.annual_energy_kwh != null ? Math.round(p.annual_energy_kwh / 1e3) : null,
    payback: payback(p.payback_years),
    npv_usd: p.npv_usd,
  }));

  const paybacks = vs
    .map((p) => payback(p.payback_years))
    .filter((v): v is number => v != null);

  // Calibration band over the same viable cohort the KPIs use. Summed in kWh
  // and rounded once per tier, matching viable_energy_mwh's own rounding.
  const band_mwh = Object.fromEntries(
    CALIB_TIERS.map((t) => [
      t.key,
      Math.round(_sum(vs.map((p) => calibOf(p)[t.key])) / 1e3),
    ]),
  ) as CalibEnergy;

  return {
    plants_analyzed: sites.meta.plants_analyzed,
    scored_sites: sites.meta.scored_sites,
    viable_projects: vs.length,
    tier_a_sites: all.filter((p) => p.site_tier === "A").length,
    high_confidence_sites: vs.filter((p) => p.project_viable_high_confidence).length,
    portfolio_npv_usd: Math.round(_sum(vs.map((p) => p.npv_usd))),
    portfolio_capex_usd: Math.round(_sum(vs.map((p) => p.total_capex_usd))),
    annual_savings_usd: Math.round(_sum(vs.map((p) => p.annual_revenue_usd))),
    viable_energy_mwh: Math.round(_sum(vs.map((p) => p.annual_energy_kwh)) / 1e3),
    band_mwh,
    median_payback: _median1dp(paybacks),
    by_state,
    top_opportunities,
  };
}

// ── fetchPortfolio ────────────────────────────────────────────────────────────

export async function fetchPortfolio(state: string): Promise<Portfolio> {
  const sites = await loadSites();
  const fs = sites.features.filter((f) => f.properties.state_code === state);

  if (fs.length === 0) {
    throw new Error(`No portfolio data for state: ${state}`);
  }

  // Sort by npv_usd desc (mirrors export_web_data.py); table rows include
  // every scored site in the state, aggregates are viable-only.
  const sorted = [...fs].sort(
    (a, b) => (b.properties.npv_usd ?? -Infinity) - (a.properties.npv_usd ?? -Infinity),
  );

  const ps = sorted.map((f) => f.properties);
  const vs = ps.filter((p) => p.project_viable);
  const validPaybacks = vs
    .map((p) => payback(p.payback_years))
    .filter((v): v is number => v != null);

  const rows: TableRow[] = sorted.map((f, i) => {
    const p = f.properties;
    const pb = payback(p.payback_years);
    return {
      id: p.npdes_id,
      rank: i + 1,
      name: p.facility_name,
      city: p.city,
      flow_mgd: p.mean_flow_mgd,
      turbine: p.turbine_type,
      capex_usd: p.total_capex_usd,
      annual_savings_usd: p.annual_revenue_usd,
      payback: pb,
      npv_usd: p.npv_usd,
      energy_mwh:
        p.annual_energy_kwh != null ? Math.round(p.annual_energy_kwh / 1e3) : null,
      confidence: siteConfidence(p),
      band: viabilityBand(pb, p.project_viable),
      viable: p.project_viable,
    };
  });

  const avgPayback =
    validPaybacks.length > 0
      ? Math.round(
          (validPaybacks.reduce((a, b) => a + b, 0) / validPaybacks.length) * 10,
        ) / 10
      : null;

  return {
    state,
    n_scored: ps.length,
    n_viable: vs.length,
    combined_npv_usd: Math.round(_sum(vs.map((p) => p.npv_usd))),
    annual_savings_usd: Math.round(_sum(vs.map((p) => p.annual_revenue_usd))),
    total_capex_usd: Math.round(_sum(vs.map((p) => p.total_capex_usd))),
    avg_payback: avgPayback,
    high_confidence_sites: vs.filter((p) => p.project_viable_high_confidence).length,
    rows,
  };
}

// ── fetchPlant ────────────────────────────────────────────────────────────────

export async function fetchPlant(id: string): Promise<PlantDetail> {
  const sites = await loadSites();
  const feat = sites.features.find((f) => f.properties.npdes_id === id);

  if (!feat) throw new Error(`Plant "${id}" not found`);

  const p = feat.properties;
  // GeoJSON coordinates are [longitude, latitude]
  const [lon, lat] = feat.geometry.coordinates;

  const detail: PlantDetail = {
    id: p.npdes_id,
    name: p.facility_name,
    city: p.city,
    state: p.state_code,
    lat,
    lon,
    confidence: siteConfidence(p),
    dmr_backed: p.data_quality === "dmr" || p.data_quality === "dmr_limited",
    viable: p.project_viable,
    tier: p.site_tier,
    flow: {
      mean_mgd: p.mean_flow_mgd,
      p10_mgd: p.p10_flow_mgd,
      p90_mgd: p.p90_flow_mgd,
      n_months_dmr: p.n_months_data,
      utilization: p.utilization_ratio,
      data_quality: p.data_quality,
    },
    energy: {
      p10_mwh: p.energy_p10_kwh_yr != null ? Math.round(p.energy_p10_kwh_yr / 1e3) : null,
      p50_mwh: p.energy_p50_kwh_yr != null ? Math.round(p.energy_p50_kwh_yr / 1e3) : null,
      p90_mwh: p.energy_p90_kwh_yr != null ? Math.round(p.energy_p90_kwh_yr / 1e3) : null,
      equivalent_homes: p.equivalent_homes_p50,
      annual_kwh: p.annual_energy_kwh,
      offset_pct: p.energy_offset_pct,
      calib: calibOf(p),
    },
    elevation: {
      head_source: p.head_source,
      head_net_m: p.head_net_m,
      head_gross_m: p.head_gross_m,
      facility_elev_m: p.elevation_m,
      outfall_elev_m: p.elev_outfall_m,
      head_confidence: p.head_confidence,
    },
    turbine: {
      type: p.turbine_type,
      rated_power_kw: p.rated_power_kw,
      rated_flow_m3s: p.q_rated_m3s,
      peak_efficiency_pct: p.peak_efficiency_pct,
      capacity_factor: p.capacity_factor,
    },
    financial: {
      npv_usd: p.npv_usd,
      npv_with_grant_usd: p.npv_with_50pct_grant_usd,
      irr: p.irr,
      payback_years: payback(p.payback_years),
      lcoe_per_kwh: p.lcoe_per_kwh,
      annual_revenue_usd: p.annual_revenue_usd,
      annual_opex_usd: p.annual_opex_usd,
      elec_rate_per_kwh: p.elec_rate_per_kwh,
      capex: {
        total_usd: p.total_capex_usd,
        equipment_usd: p.equipment_capex_usd,
        installation_usd: p.installation_capex_usd,
        interconnection_usd: p.interconnection_capex_usd,
        permitting_usd: p.permitting_capex_usd,
        permitting_tier: p.permitting_tier,
      },
    },
    sensitivity: {
      head: [p.sensitivity_head_npv_low, p.sensitivity_head_npv_high],
      flow: [p.sensitivity_flow_npv_low, p.sensitivity_flow_npv_high],
      rate: [p.sensitivity_rate_npv_low, p.sensitivity_rate_npv_high],
      dominant: p.dominant_sensitivity,
    },
  };

  return detail;
}
