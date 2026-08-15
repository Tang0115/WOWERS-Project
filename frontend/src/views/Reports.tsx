import { useMemo } from "react";
import { Link } from "react-router-dom";
import { fetchNational, fetchPlants, useAsync } from "../lib/data";
import { downloadCsv } from "../lib/csv";
import { num, usd, years } from "../lib/format";
import Caveat from "../components/Caveat";
import { CalibrationBandCard, RescoreTable } from "../components/CalibrationBand";
import { MODELED_NOTE } from "../lib/notes";

/** Data export center: national summary + one-click CSV downloads. */
export default function Reports() {
  const { data: national, error: nErr } = useAsync(fetchNational, []);
  const { data: plants, error: pErr } = useAsync(fetchPlants, []);

  const all = useMemo(
    () => (plants ? plants.features.map((f) => f.properties) : []),
    [plants],
  );

  const siteHeaders = ["npdes_id", "name", "city", "state", "turbine", "rated_kw", "energy_mwh_yr", "payback_yr", "npv_usd", "tier", "viable", "confidence"];
  const siteRow = (p: (typeof all)[number]) =>
    [p.id, p.name, p.city, p.state, p.turbine, p.rated_kw, p.energy_mwh, p.payback, p.npv, p.tier, p.viable, p.confidence];

  const exportAll = () => downloadCsv("wowers_all_sites.csv", siteHeaders, all.map(siteRow));
  const exportViable = () =>
    downloadCsv("wowers_viable_sites.csv", siteHeaders, all.filter((p) => p.viable).map(siteRow));
  const exportStates = () => {
    if (!national) return;
    downloadCsv(
      "wowers_state_summary.csv",
      ["state", "viable_sites"],
      national.by_state.map((s) => [s.state, s.viable]),
    );
  };

  const error = nErr ?? pErr;
  if (error) return <div className="loading">Failed to load data: {error}</div>;
  if (!national || !plants) return <div className="loading">Loading reports…</div>;

  const summary: [string, string][] = [
    ["Plants analyzed", num(national.plants_analyzed)],
    ["Scored sites", num(national.scored_sites)],
    // "Tier A sites" was dropped here: site_tier A maps 1:1 onto project_viable
    // (1,138 / 1,138) and C onto non-viable, so the row restated the line above
    // it under a different name and read as independent corroboration.
    ["Viable projects", num(national.viable_projects)],
    ["High-confidence viable", num(national.high_confidence_sites)],
    ["Portfolio NPV", usd(national.portfolio_npv_usd)],
    ["Total CapEx", usd(national.portfolio_capex_usd)],
    ["Annual savings", `${usd(national.annual_savings_usd)}/yr`],
    ["Recoverable energy", `${num(national.viable_energy_mwh)} MWh/yr`],
    ["Median payback", years(national.median_payback)],
  ];

  // Provenance of the viable cohort, counted live rather than quoted, so the
  // method note cannot drift from the file it describes.
  const viable = all.filter((p) => p.viable);
  const archetypeHead = viable.filter((p) => !p.head_measured).length;
  const fallbackFlow = viable.filter((p) => !p.flow_measured).length;

  const exportBand = () =>
    downloadCsv(
      "wowers_calibration_band.csv",
      ["tier", "viable_energy_mwh_yr"],
      Object.entries(national.band_mwh).map(([tier, v]) => [tier, v]),
    );

  return (
    <div style={{ padding: 22, maxWidth: 900 }}>
      <h1 style={{ fontSize: 26, fontWeight: 800, color: "var(--navy)", margin: "0 0 2px" }}>Reports &amp; Exports</h1>
      <div className="muted" style={{ fontSize: 14, marginBottom: 18 }}>
        National summary and downloadable datasets
      </div>

      <Caveat tone="warn" style={{ marginBottom: 18 }}>
        {MODELED_NOTE} Every figure in the summary below is the uncalibrated ceiling tier;
        the calibration band and the re-scored portfolio follow it.
      </Caveat>

      <div className="card card-pad" style={{ marginBottom: 18 }}>
        <h3 className="card-title">National Summary</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: "4px 24px" }}>
          {summary.map(([l, v]) => (
            <div key={l} style={{ display: "flex", justifyContent: "space-between", padding: "8px 0", borderBottom: "1px solid #eef1f6", fontSize: 13 }}>
              <span className="muted">{l}</span>
              <span style={{ fontWeight: 700 }}>{v}</span>
            </div>
          ))}
        </div>
      </div>

      <div style={{ marginBottom: 18 }}>
        <CalibrationBandCard band={national.band_mwh} />
      </div>

      <div className="card card-pad" style={{ marginBottom: 18 }}>
        <div className="card-head">
          <h3 className="card-title">Portfolio Re-Scored at Each Tier</h3>
          <p className="card-sub">
            The band above re-scales energy for a fixed cohort. This table re-runs the full
            30-year cash flow at each tier and re-applies the viability gate, so sites that
            no longer clear it drop out. The count falls faster than the energy: fixed
            interconnection and permitting costs leave a small project no way to absorb a
            revenue cut.
          </p>
        </div>
        <RescoreTable />
      </div>

      <div className="card card-pad" style={{ marginBottom: 18 }}>
        <h3 className="card-title">CSV Downloads</h3>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <button className="btn btn-blue" onClick={exportAll}>⇩ All Scored Sites ({num(all.length)})</button>
          <button className="btn btn-green" onClick={exportViable}>⇩ Viable Sites ({num(national.viable_projects)})</button>
          <button className="btn btn-ghost" onClick={exportStates}>⇩ State Summary</button>
          <button className="btn btn-ghost" onClick={exportBand}>⇩ Calibration Band</button>
        </div>
        <div className="faint" style={{ fontSize: 11, marginTop: 10 }}>
          Per-state portfolio tables can be exported from each <Link to="/">state portfolio page</Link>.
        </div>
      </div>

      <div className="card card-pad">
        <h3 className="card-title">Data Sources &amp; Method</h3>
        <div className="muted" style={{ fontSize: 13, lineHeight: 1.6 }}>
          <p style={{ margin: "0 0 10px" }}>
            <strong>Measured inputs.</strong> Monthly discharge flows reported by permit holders
            to the EPA (NPDES / ICIS discharge monitoring reports) and point elevations from the
            USGS 3DEP query service. These two are observations.
          </p>
          <p style={{ margin: "0 0 10px" }}>
            <strong>Modeled outputs.</strong> Everything downstream — net head, turbine choice,
            efficiency, energy, capital cost, and cash flow — is modeled, with energy priors from
            EPRI. Sites are scored with Monte-Carlo energy bands (P10/P50/P90), turbine selection,
            and full financial modeling (NPV, IRR, payback, LCOE, ±20 % sensitivity).{" "}
            <strong>No site in this dataset has been built or metered</strong>, so this is a
            screen with stated bounds, not an underwriting document.
          </p>
          <p style={{ margin: "0 0 10px" }}>
            <strong>Calibration.</strong> The modeled fleet capacity factor of 0.872 was
            benchmarked against 629 metered small-hydro plants over 9,798 plant-years and sits
            above that distribution at every percentile. The band above substitutes measured
            capacity factors; the defensible planning figure is its floor, not its upper end.
            Closing the gap needs measured annual generation from an operating wastewater-outfall
            turbine, which does not currently exist outside a single plant.
          </p>
          <p style={{ margin: "0 0 10px" }}>
            <strong>What is not in the screen.</strong> Head at {num(archetypeHead)} of the{" "}
            {num(viable.length)} viable sites is a size-based archetype rather than a 3DEP
            elevation difference; {num(fallbackFlow)} rest on design or annual-average flow
            rather than a measured duration curve, and those carry a disproportionate share of
            portfolio NPV. Both are flagged per site by the confidence badge. Supplier links on
            plant pages are representative, not endorsements or quotes.
          </p>
          <p style={{ margin: 0 }}>
            Baseline: {plants.features.length ? "scored_sites.geojson" : "—"} · 59 properties per
            site · {num(national.scored_sites)} scored of {num(national.plants_analyzed)} analyzed.
          </p>
        </div>
      </div>
    </div>
  );
}
