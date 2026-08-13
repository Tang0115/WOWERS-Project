import { useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import { fetchPlants, useAsync } from "../lib/data";
import { TURBINE_LABEL } from "../lib/colors";
import { num } from "../lib/format";
import SiteTable from "../components/SiteTable";

/** Full plant directory — every scored site, filterable via URL params
    (?state=MN&turbine=Kaplan&viable=1) so other pages can deep-link. */
export default function Plants() {
  const { data: plants, error } = useAsync(fetchPlants, []);
  const [params, setParams] = useSearchParams();

  const state = params.get("state") ?? "";
  const turbine = params.get("turbine") ?? "";
  const viableOnly = params.get("viable") === "1";

  const setParam = (key: string, value: string) => {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    setParams(next, { replace: true });
  };

  const all = useMemo(
    () => (plants ? plants.features.map((f) => f.properties) : []),
    [plants],
  );

  const states = useMemo(
    () => [...new Set(all.map((p) => p.state).filter((s): s is string => !!s))].sort(),
    [all],
  );
  const turbines = useMemo(
    () => [...new Set(all.map((p) => p.turbine).filter((t): t is string => !!t))].sort(),
    [all],
  );

  const sites = useMemo(
    () =>
      all.filter((p) => {
        if (state && p.state !== state) return false;
        if (turbine && p.turbine !== turbine) return false;
        if (viableOnly && !p.viable) return false;
        return true;
      }),
    [all, state, turbine, viableOnly],
  );

  if (error) return <div className="loading">Failed to load plant data: {error}</div>;
  if (!plants) return <div className="loading">Loading plants…</div>;

  return (
    <div style={{ padding: 22 }}>
      <h1 style={{ fontSize: 26, fontWeight: 800, color: "var(--navy)", margin: "0 0 2px" }}>Plant Directory</h1>
      <div className="muted" style={{ fontSize: 14, marginBottom: 18 }}>
        All {num(all.length)} scored wastewater sites — search, filter, export
      </div>

      <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap", marginBottom: 14 }}>
        <select value={state} onChange={(e) => setParam("state", e.target.value)}>
          <option value="">All States</option>
          {states.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
        <select value={turbine} onChange={(e) => setParam("turbine", e.target.value)}>
          <option value="">All Turbines</option>
          {turbines.map((t) => <option key={t} value={t}>{TURBINE_LABEL[t] ?? t}</option>)}
        </select>
        <label style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 13 }}>
          <input type="checkbox" checked={viableOnly} onChange={(e) => setParam("viable", e.target.checked ? "1" : "")} />
          Viable only
        </label>
      </div>

      <SiteTable sites={sites} csvName={`wowers_plants${state ? `_${state}` : ""}.csv`} />
    </div>
  );
}
