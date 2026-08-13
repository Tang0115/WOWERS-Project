import { useMemo } from "react";
import { Link } from "react-router-dom";
import { fetchNational, fetchPlants, useAsync } from "../lib/data";
import { downloadCsv } from "../lib/csv";
import { num, usd, years } from "../lib/format";

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
    ["Viable projects", num(national.viable_projects)],
    ["Tier A sites", num(national.tier_a_sites)],
    ["High-confidence viable", num(national.high_confidence_sites)],
    ["Portfolio NPV", usd(national.portfolio_npv_usd)],
    ["Total CapEx", usd(national.portfolio_capex_usd)],
    ["Annual savings", `${usd(national.annual_savings_usd)}/yr`],
    ["Recoverable energy", `${num(national.viable_energy_mwh)} MWh/yr`],
    ["Median payback", years(national.median_payback)],
  ];

  return (
    <div style={{ padding: 22, maxWidth: 900 }}>
      <h1 style={{ fontSize: 26, fontWeight: 800, color: "var(--navy)", margin: "0 0 2px" }}>Reports &amp; Exports</h1>
      <div className="muted" style={{ fontSize: 14, marginBottom: 18 }}>
        National summary and downloadable datasets
      </div>

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

      <div className="card card-pad" style={{ marginBottom: 18 }}>
        <h3 className="card-title">CSV Downloads</h3>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <button className="btn btn-blue" onClick={exportAll}>⇩ All Scored Sites ({num(all.length)})</button>
          <button className="btn btn-green" onClick={exportViable}>⇩ Viable Sites ({num(national.viable_projects)})</button>
          <button className="btn btn-ghost" onClick={exportStates}>⇩ State Summary</button>
        </div>
        <div className="faint" style={{ fontSize: 11, marginTop: 10 }}>
          Per-state portfolio tables can be exported from each <Link to="/">state portfolio page</Link>.
        </div>
      </div>

      <div className="card card-pad">
        <h3 className="card-title">Data Sources &amp; Method</h3>
        <div className="muted" style={{ fontSize: 13, lineHeight: 1.6 }}>
          Flow data from EPA NPDES/DMR discharge monitoring reports; energy priors from EPRI;
          elevation and hydraulic head from USGS 3DEP. Sites are scored with Monte-Carlo energy
          bands (P10/P50/P90), turbine selection, and full financial modeling (NPV, IRR, payback,
          LCOE, ±20% sensitivity). Baseline: {plants.features.length ? "scored_sites.geojson" : "—"}.
        </div>
      </div>
    </div>
  );
}
