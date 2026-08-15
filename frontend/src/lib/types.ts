import type { CalibEnergy } from "./calibration";

export type Band = "high" | "moderate" | "marginal" | "nonviable";
export type Confidence = "High" | "Medium" | "Lower";

export interface MapProps {
  id: string;
  name: string | null;
  city: string | null;
  state: string | null;
  turbine: string | null;
  rated_kw: number | null;
  energy_mwh: number | null;
  // Unrounded annual energy, carried alongside energy_mwh so multi-site totals
  // sum in kWh and round once. Summing the per-site rounded energy_mwh instead
  // accumulates to +32 MWh/yr over the 1,138 viable sites, which disagreed with
  // the national KPI (409,170 vs 409,202 MWh/yr) for the same cohort.
  energy_kwh: number | null;
  // Calibrated energy per tier, kWh/yr. energy_calib.ceiling === energy_kwh.
  energy_calib: CalibEnergy;
  payback: number | null;
  npv: number | null;
  tier: string | null;
  viable: boolean;
  band: Band;
  confidence: Confidence;
  // Provenance of the two measured inputs, carried so a view can report how
  // much of a cohort rests on a fallback rather than on an observation.
  flow_measured: boolean;
  head_measured: boolean;
}

export interface PlantFeature {
  type: "Feature";
  geometry: { type: "Point"; coordinates: [number, number] };
  properties: MapProps;
}

export interface PlantCollection {
  type: "FeatureCollection";
  features: PlantFeature[];
}

export interface TopOpportunity {
  id: string;
  name: string | null;
  city: string | null;
  state: string | null;
  energy_mwh: number | null;
  payback: number | null;
  npv_usd: number | null;
}

export interface National {
  plants_analyzed: number;
  scored_sites: number;
  viable_projects: number;
  tier_a_sites: number;
  high_confidence_sites: number;
  portfolio_npv_usd: number | null;
  portfolio_capex_usd: number | null;
  annual_savings_usd: number | null;
  viable_energy_mwh: number | null;
  // Viable-cohort energy at each calibration tier, MWh/yr. The `ceiling` entry
  // equals viable_energy_mwh; the rest are the calibration band.
  band_mwh: CalibEnergy;
  median_payback: number | null;
  by_state: { state: string; viable: number }[];
  top_opportunities: TopOpportunity[];
}

export interface TableRow {
  id: string;
  rank: number;
  name: string | null;
  city: string | null;
  flow_mgd: number | null;
  turbine: string | null;
  capex_usd: number | null;
  annual_savings_usd: number | null;
  payback: number | null;
  npv_usd: number | null;
  energy_mwh: number | null;
  confidence: Confidence;
  band: Band;
  viable: boolean;
}

export interface Portfolio {
  state: string;
  n_scored: number;
  n_viable: number;
  combined_npv_usd: number | null;
  annual_savings_usd: number | null;
  total_capex_usd: number | null;
  avg_payback: number | null;
  high_confidence_sites: number;
  rows: TableRow[];
}

export interface PlantDetail {
  id: string;
  name: string | null;
  city: string | null;
  state: string | null;
  lat: number | null;
  lon: number | null;
  confidence: Confidence;
  dmr_backed: boolean;
  viable: boolean;
  tier: string | null;
  flow: {
    mean_mgd: number | null; p10_mgd: number | null; p90_mgd: number | null;
    n_months_dmr: number | null; utilization: number | null; data_quality: string | null;
  };
  energy: {
    p10_mwh: number | null; p50_mwh: number | null; p90_mwh: number | null;
    equivalent_homes: number | null; annual_kwh: number | null; offset_pct: number | null;
    // This site's energy at each calibration tier, kWh/yr.
    calib: CalibEnergy;
  };
  elevation: {
    head_source: string | null; head_net_m: number | null; head_gross_m: number | null;
    facility_elev_m: number | null; outfall_elev_m: number | null; head_confidence: string | null;
  };
  turbine: {
    type: string | null; rated_power_kw: number | null; rated_flow_m3s: number | null;
    peak_efficiency_pct: number | null; capacity_factor: number | null;
  };
  financial: {
    npv_usd: number | null; npv_with_grant_usd: number | null; irr: number | null;
    payback_years: number | null; lcoe_per_kwh: number | null; annual_revenue_usd: number | null;
    annual_opex_usd: number | null; elec_rate_per_kwh: number | null;
    capex: {
      total_usd: number | null; equipment_usd: number | null; installation_usd: number | null;
      interconnection_usd: number | null; permitting_usd: number | null; permitting_tier: string | null;
    };
  };
  sensitivity: {
    head: [number | null, number | null]; flow: [number | null, number | null];
    rate: [number | null, number | null]; dominant: string | null;
  };
}
