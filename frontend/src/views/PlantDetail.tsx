import { Link, useParams } from "react-router-dom";
import { fetchPlant, useAsync } from "../lib/data";
import { CONF_PILL, DOE_FUNDING_URL, TURBINE_LABEL, TURBINE_VENDOR } from "../lib/colors";
import { num, pct, pctRaw, usd, years } from "../lib/format";
import Gauge from "../components/charts/Gauge";
import { MiniMap } from "../components/MapView";
import { EfficiencyCurve, Tornado } from "../components/charts/Charts";

function Row({ l, v, hint }: { l: string; v: string; hint?: string }) {
  return (
    <div className="drow">
      <span className={`muted${hint ? " hint" : ""}`} title={hint}>{l}</span>
      <span className="val">{v}</span>
    </div>
  );
}

function CardHead({ title }: { title: string }) {
  return (
    <div className="card-head">
      <h3 className="card-title">{title}</h3>
    </div>
  );
}

export default function PlantDetail() {
  const { id = "" } = useParams();
  const { data: p, error } = useAsync(() => fetchPlant(id), [id]);

  if (error) return <div className="loading">Plant “{id}” not found. <Link to="/">← Back to map</Link></div>;
  if (!p) return <div className="loading">Loading plant…</div>;

  const f = p.financial;
  const vendor = p.turbine.type ? TURBINE_VENDOR[p.turbine.type] : undefined;
  const similarParams = new URLSearchParams();
  if (p.state) similarParams.set("state", p.state);
  if (p.turbine.type) similarParams.set("turbine", p.turbine.type);
  const npvColor = f.npv_usd == null ? "var(--text)" : f.npv_usd >= 0 ? "var(--green)" : "#dc2626";
  const irrColor = f.irr == null ? "var(--text)" : f.irr >= 0 ? "var(--green)" : "#dc2626";

  return (
    <div style={{ padding: 22 }}>
      <div className="muted" style={{ fontSize: 12, marginBottom: 8 }}>
        <Link to="/">Dashboard</Link> ›{" "}
        {p.state && <><Link to={`/state/${p.state}`}>{p.state}</Link> › </>}
        {p.name}
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
        <h1 style={{ fontSize: 26, fontWeight: 800, color: "var(--navy)", margin: 0 }}>{p.name}</h1>
        <span className={`pill ${CONF_PILL[p.confidence]}`}>{p.confidence} confidence</span>
        {p.dmr_backed && <span className="pill pill-high">✓ DMR-Backed</span>}
        {p.tier && <span className="pill pill-lower">Tier {p.tier}</span>}
      </div>
      <div className="muted" style={{ fontSize: 13, margin: "4px 0 18px" }}>
        {p.city}, {p.state} &nbsp;|&nbsp; NPDES: {p.id} &nbsp;|&nbsp; {num(p.flow.mean_mgd)} MGD mean flow
      </div>

      <div className="pd-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1.2fr 1fr", gap: 18, alignItems: "start" }}>
        {/* left column */}
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <div className="card card-pad">
            <CardHead title="≈ Flow Data" />
            <Row l="Mean Flow" v={`${num(p.flow.mean_mgd)} MGD`} />
            <Row l="P10 Flow" v={p.flow.p10_mgd != null ? `${num(p.flow.p10_mgd)} MGD` : "—"} />
            <Row l="P90 Flow" v={p.flow.p90_mgd != null ? `${num(p.flow.p90_mgd)} MGD` : "—"} />
            <Row l="Data Quality" v={p.flow.data_quality ?? "—"} />
            <Row l="Months DMR" v={num(p.flow.n_months_dmr)} />
            <Row l="Utilization" v={pct(p.flow.utilization)} />
          </div>
          <div className="card card-pad">
            <CardHead title="⛰ Elevation Data" />
            <Row l="Head Source" v={p.elevation.head_source === "usgs_3dep" ? "USGS 3DEP ✓" : "Literature"} hint="USGS 3DEP is a measured elevation model; Literature is a size-based statistical estimate used when 3DEP coordinates aren't available." />
            <Row l="Net Head" v={p.elevation.head_net_m != null ? `${p.elevation.head_net_m} m` : "—"} />
            <Row l="Gross Head" v={p.elevation.head_gross_m != null ? `${p.elevation.head_gross_m} m` : "—"} />
            <Row l="Facility Elevation" v={p.elevation.facility_elev_m != null ? `${num(p.elevation.facility_elev_m)} m` : "—"} />
            <Row l="Outfall Elevation" v={p.elevation.outfall_elev_m != null ? `${num(p.elevation.outfall_elev_m)} m` : "—"} />
          </div>
          {p.lat != null && p.lon != null && (
            <div className="card card-pad">
              <CardHead title="⌖ Site Location" />
              <MiniMap lat={p.lat} lon={p.lon} />
              <div className="faint" style={{ fontSize: 11, marginTop: 8, textAlign: "center" }}>
                {p.lat.toFixed(4)}, {p.lon.toFixed(4)}
              </div>
            </div>
          )}
        </div>

        {/* center column */}
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <div className="card card-pad card-col">
            <CardHead title="Energy Recovery Estimate" />
            {p.energy.p50_mwh != null && p.energy.p10_mwh != null && p.energy.p90_mwh != null ? (
              <Gauge p10={p.energy.p10_mwh} p50={p.energy.p50_mwh} p90={p.energy.p90_mwh} capacityFactor={p.turbine.capacity_factor} />
            ) : <div className="muted">No energy estimate.</div>}
            <div className="card-foot faint" style={{ fontSize: 11, textAlign: "center", paddingTop: 10 }}>
              Offsets {pctRaw(p.energy.offset_pct)} of plant electricity · ≈{num(p.energy.equivalent_homes)} homes
            </div>
          </div>
          <div className="card card-pad card-col">
            <CardHead title="Recommended Turbine" />
            <div className="headline-box" style={{ marginBottom: 6 }}>
              <div style={{ fontSize: 18, fontWeight: 700, color: "var(--navy)" }}>
                {TURBINE_LABEL[p.turbine.type ?? ""] ?? p.turbine.type} Turbine
              </div>
            </div>
            <Row l="Rated Power" v={`${num(p.turbine.rated_power_kw)} kW`} />
            <Row l="Rated Flow" v={p.turbine.rated_flow_m3s != null ? `${p.turbine.rated_flow_m3s} m³/s` : "—"} />
            <Row l="Peak Efficiency" v={p.turbine.peak_efficiency_pct != null ? `${p.turbine.peak_efficiency_pct}%` : "—"} />
            <Row l="Capacity Factor" v={pct(p.turbine.capacity_factor)} hint="Actual annual output vs. running at full rated power 24/7 — this also drives the gauge needle above." />
            {p.turbine.rated_flow_m3s != null && p.turbine.peak_efficiency_pct != null && (
              <div className="card-foot" style={{ paddingTop: 10 }}>
                <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>Efficiency Curve</div>
                <EfficiencyCurve qRated={p.turbine.rated_flow_m3s} peakPct={p.turbine.peak_efficiency_pct} />
              </div>
            )}
          </div>
        </div>

        {/* right column */}
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <div className="card card-pad">
            <CardHead title="Financial Scorecard" />
            <div className="headline-box" style={{ marginBottom: 6 }}>
              <div>
                <div className="muted" style={{ fontSize: 12 }}>NPV</div>
                <div style={{ fontSize: 30, fontWeight: 800, color: npvColor }}>{usd(f.npv_usd)}</div>
              </div>
              <div>
                <div className="muted" style={{ fontSize: 12 }}>IRR</div>
                <div style={{ fontSize: 22, fontWeight: 800, color: irrColor }}>{pct(f.irr)}</div>
              </div>
            </div>
            <Row l="Payback" v={years(f.payback_years)} />
            <Row l="Installation Cost" v={usd(f.capex.installation_usd)} />
            <Row l="Total CapEx" v={usd(f.capex.total_usd)} />
            <Row l="Annual Savings" v={`${usd(f.annual_revenue_usd)}/yr`} />
            <Row l="LCOE" v={f.lcoe_per_kwh != null ? `$${f.lcoe_per_kwh.toFixed(3)}/kWh` : "—"} />
            <Row l="NPV w/ 50% grant" v={usd(f.npv_with_grant_usd)} />
          </div>
          <div className="card card-pad">
            <CardHead title="Sensitivity Analysis" />
            <div className="faint" style={{ fontSize: 11, marginBottom: 6 }}>Impact on NPV (±20%). Dominant driver: {p.sensitivity.dominant}</div>
            {f.npv_usd != null && (
              <Tornado
                base={f.npv_usd}
                rows={[
                  { label: "Head ±20%", low: p.sensitivity.head[0], high: p.sensitivity.head[1] },
                  { label: "Flow ±20%", low: p.sensitivity.flow[0], high: p.sensitivity.flow[1] },
                  { label: "Elec. rate ±20%", low: p.sensitivity.rate[0], high: p.sensitivity.rate[1] },
                ]}
              />
            )}
          </div>
          <div className="card card-pad">
            <CardHead title="Next Steps" />
            {vendor && (
              <a href={vendor.url} target="_blank" rel="noopener noreferrer" className="btn btn-blue" style={{ width: "100%", justifyContent: "center", marginBottom: 8 }}>
                Get Turbine Quotes ↗
              </a>
            )}
            <a href={DOE_FUNDING_URL} target="_blank" rel="noopener noreferrer" className="btn btn-green" style={{ width: "100%", justifyContent: "center", marginBottom: 8 }}>
              DOE Water Power Funding ↗
            </a>
            <Link to={`/plants?${similarParams.toString()}`} className="btn btn-ghost" style={{ width: "100%", justifyContent: "center" }}>
              View Similar Sites
            </Link>
            {vendor && (
              <div className="faint" style={{ fontSize: 11, marginTop: 10, textAlign: "center" }}>
                Supplier: {vendor.vendor}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
