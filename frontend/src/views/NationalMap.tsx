import { useMemo } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { fetchNational, fetchPlants, useAsync } from "../lib/data";
import type { PlantCollection } from "../lib/types";
import { BAND_COLOR, BAND_LABEL, TURBINE_LABEL } from "../lib/colors";
import { mwh, num, usd, years } from "../lib/format";
import KpiTile from "../components/KpiTile";
import MapView from "../components/MapView";
import type { LngLatBounds } from "../components/MapView";
import { StateBar } from "../components/charts/Charts";

const NONE = "__none__";
const TURBINES = ["Kaplan", "Francis", "Crossflow", "in_conduit_micro", NONE];

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  const m = s.length % 2 === 0 ? (s[mid - 1] + s[mid]) / 2 : s[mid];
  return Math.round(m * 10) / 10;
}

export default function NationalMap() {
  const { data: national, error: nErr } = useAsync(fetchNational, []);
  const { data: plants, error: pErr } = useAsync(fetchPlants, []);

  // Filters live in the URL so views are shareable and survive refresh.
  const [params, setParams] = useSearchParams();
  const state = params.get("state") ?? "";
  const turbines = useMemo(() => {
    const t = params.get("turb");
    return t == null ? [...TURBINES] : t.split(",").filter(Boolean);
  }, [params]);
  const maxPayback = Number(params.get("pb") ?? 20) || 20;
  const highOnly = params.get("hc") === "1";

  const update = (patch: Record<string, string | null>) => {
    const next = new URLSearchParams(params);
    for (const [k, v] of Object.entries(patch)) {
      if (v == null || v === "") next.delete(k);
      else next.set(k, v);
    }
    setParams(next, { replace: true });
  };

  const setState = (s: string) => update({ state: s || null });
  const setMaxPayback = (v: number) => update({ pb: v >= 20 ? null : String(v) });
  const setHighOnly = (v: boolean) => update({ hc: v ? "1" : null });
  const toggleTurbine = (t: string) => {
    const next = turbines.includes(t) ? turbines.filter((x) => x !== t) : [...turbines, t];
    update({ turb: next.length === TURBINES.length ? null : next.join(",") });
  };
  const resetFilters = () => setParams(new URLSearchParams(), { replace: true });

  const filtered: PlantCollection | null = useMemo(() => {
    if (!plants) return null;
    const feats = plants.features.filter((f) => {
      const p = f.properties;
      if (state && p.state !== state) return false;
      if (!turbines.includes(p.turbine ?? NONE)) return false;
      if (highOnly && p.confidence !== "High") return false;
      // Slider at its 20-yr max = no payback filter: all scored sites,
      // including finite >20 yr and "never pays back" (null) sentinels.
      if (maxPayback >= 20) return true;
      if (p.payback == null) return false;
      return p.payback <= maxPayback;
    });
    return { type: "FeatureCollection", features: feats };
  }, [plants, state, turbines, maxPayback, highOnly]);

  // Live KPIs reflect the current filter set (viable sites within it).
  const kpis = useMemo(() => {
    if (!filtered) return null;
    const viable = filtered.features.map((f) => f.properties).filter((p) => p.viable);
    return {
      shown: filtered.features.length,
      viable: viable.length,
      npv: viable.reduce((a, p) => a + (p.npv ?? 0), 0),
      medianPayback: median(viable.map((p) => p.payback).filter((v): v is number => v != null)),
    };
  }, [filtered]);

  // When a state is selected, fit the map to that state's sites.
  const bounds: LngLatBounds | null = useMemo(() => {
    if (!state || !plants) return null;
    let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
    for (const f of plants.features) {
      if (f.properties.state !== state) continue;
      const [lng, lat] = f.geometry.coordinates;
      if (lng < w) w = lng;
      if (lng > e) e = lng;
      if (lat < s) s = lat;
      if (lat > n) n = lat;
    }
    return e >= w ? [[w, s], [e, n]] : null;
  }, [state, plants]);

  // Derive from all scored sites, not national.by_state — that list only has
  // states with viable sites, dropping territories like AS/MP/VI entirely.
  const states = useMemo(
    () =>
      plants
        ? [...new Set(plants.features.map((f) => f.properties.state).filter((s): s is string => !!s))].sort()
        : [],
    [plants],
  );

  const error = nErr ?? pErr;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      {/* header */}
      <div className="nm-header" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap", padding: "14px 22px", background: "#fff", borderBottom: "1px solid var(--border)" }}>
        <div>
          <div style={{ fontSize: 20, fontWeight: 800, color: "var(--navy)" }}>National Opportunity Map</div>
          <div className="muted" style={{ fontSize: 12 }}>Micro-hydro energy recovery across US wastewater plants</div>
        </div>
        {kpis && (
          <div className="kpi-row">
            <KpiTile value={num(kpis.shown)} label="Sites Shown" />
            <KpiTile value={num(kpis.viable)} label="Viable in Filter" accent="var(--blue)" />
            <KpiTile value={usd(kpis.npv)} label="NPV (viable)" accent="var(--green)" />
            <KpiTile value={years(kpis.medianPayback)} label="Median Payback" />
          </div>
        )}
      </div>

      {/* body */}
      <div className="nm-body" style={{ display: "flex", flex: 1, minHeight: 0 }}>
        {/* filters */}
        <div className="nm-filters" style={{ width: 230, flexShrink: 0, background: "#fff", borderRight: "1px solid var(--border)", padding: 18, overflowY: "auto" }}>
          <h3 className="card-title">Filters</h3>

          <label className="muted" style={{ fontSize: 12, fontWeight: 600 }}>State</label>
          <select value={state} onChange={(e) => setState(e.target.value)} style={{ width: "100%", margin: "6px 0 10px" }}>
            <option value="">All States</option>
            {states.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
          {state && (
            <Link to={`/state/${state}`} className="btn btn-blue" style={{ width: "100%", justifyContent: "center", marginBottom: 18 }}>
              View {state} Portfolio →
            </Link>
          )}

          <div className="muted" style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>Turbine Type</div>
          {TURBINES.map((t) => (
            <label key={t} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, padding: "3px 0" }}>
              <input type="checkbox" checked={turbines.includes(t)} onChange={() => toggleTurbine(t)} />
              {t === NONE ? "Unassigned" : TURBINE_LABEL[t] ?? t}
            </label>
          ))}

          <div className="muted" style={{ fontSize: 12, fontWeight: 600, margin: "18px 0 6px" }}>
            Max Payback: {maxPayback >= 20 ? "no limit" : `${maxPayback} yr`}
          </div>
          <input type="range" min={1} max={20} value={maxPayback} onChange={(e) => setMaxPayback(+e.target.value)} style={{ width: "100%" }} />

          <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, marginTop: 18 }}>
            <input type="checkbox" checked={highOnly} onChange={(e) => setHighOnly(e.target.checked)} />
            High confidence only
          </label>

          <div style={{ marginTop: 22, borderTop: "1px solid var(--border)", paddingTop: 14 }}>
            {(Object.keys(BAND_LABEL) as (keyof typeof BAND_LABEL)[]).map((b) => (
              <div key={b} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11, padding: "3px 0" }} className="muted">
                <span className="dot" style={{ background: BAND_COLOR[b] }} />
                {BAND_LABEL[b]}
              </div>
            ))}
          </div>
          {filtered && <div className="faint" style={{ fontSize: 11, marginTop: 14 }}>{num(filtered.features.length)} sites shown</div>}
        </div>

        {/* map */}
        <div className="nm-map" style={{ flex: 1, minWidth: 0, position: "relative" }}>
          {error ? (
            <div className="loading">
              Failed to load site data.<br />
              <span className="faint" style={{ fontSize: 12 }}>{error}</span><br />
              <button className="btn btn-blue" style={{ marginTop: 12 }} onClick={() => window.location.reload()}>Retry</button>
            </div>
          ) : filtered ? (
            <MapView data={filtered} bounds={bounds} />
          ) : (
            <div className="loading">Loading map…</div>
          )}
          {filtered && filtered.features.length === 0 && (
            <div
              style={{
                position: "absolute", inset: 0, zIndex: 2, display: "flex",
                alignItems: "center", justifyContent: "center", pointerEvents: "none",
              }}
            >
              <div className="card card-pad" style={{ textAlign: "center", pointerEvents: "auto", maxWidth: 260 }}>
                <div style={{ fontWeight: 700, marginBottom: 4 }}>No sites match these filters</div>
                <div className="muted" style={{ fontSize: 13, marginBottom: 12 }}>
                  Try widening the payback range or re-enabling a turbine type.
                </div>
                <button className="btn btn-blue" onClick={resetFilters}>Reset filters</button>
              </div>
            </div>
          )}
        </div>

        {/* top opportunities */}
        <div className="nm-side" style={{ width: 300, flexShrink: 0, background: "#fff", borderLeft: "1px solid var(--border)", padding: 18, overflowY: "auto" }}>
          <h3 className="card-title">Top 5 Opportunities</h3>
          {national?.top_opportunities.map((o, i) => (
            <Link key={o.id} to={`/plant/${o.id}`} style={{ display: "block", color: "inherit" }}>
              <div style={{ display: "flex", gap: 10, padding: "10px 0", borderBottom: "1px solid #eef1f6" }}>
                <div style={{ width: 22, height: 22, borderRadius: 6, background: "var(--navy)", color: "#fff", fontSize: 12, fontWeight: 700, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>{i + 1}</div>
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontWeight: 600, fontSize: 13 }}>{o.name}</div>
                  <div className="muted" style={{ fontSize: 11 }}>{o.city}, {o.state}</div>
                  <div style={{ fontSize: 11, marginTop: 3 }}>
                    <span className="muted">Energy</span> {mwh(o.energy_mwh)} · <span className="muted">Payback</span> {years(o.payback)}
                  </div>
                </div>
              </div>
            </Link>
          ))}
          <Link to="/opportunities" className="btn btn-ghost" style={{ width: "100%", justifyContent: "center", marginTop: 12 }}>
            All opportunities →
          </Link>
          {national && (
            <>
              <h3 className="card-title" style={{ marginTop: 22 }}>Viable Sites by State</h3>
              <StateBar data={national.by_state.slice(0, 18)} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
