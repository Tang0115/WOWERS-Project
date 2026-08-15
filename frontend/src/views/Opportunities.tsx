import { useMemo, useState } from "react";
import { fetchPlants, useAsync } from "../lib/data";
import type { Band } from "../lib/types";
import { BAND_COLOR, BAND_LABEL } from "../lib/colors";
import { num, usd } from "../lib/format";
import KpiTile from "../components/KpiTile";
import Caveat from "../components/Caveat";
import SiteTable from "../components/SiteTable";
import { CEILING_NOTE, MODELED_NOTE, TIER_HINT, TIER_SUB } from "../lib/notes";

const BANDS: Band[] = ["high", "moderate", "marginal"];

/** Ranked list of viable investment opportunities, filterable by payback band. */
export default function Opportunities() {
  const { data: plants, error } = useAsync(fetchPlants, []);
  const [bands, setBands] = useState<Band[]>([...BANDS]);

  const viable = useMemo(
    () => (plants ? plants.features.map((f) => f.properties).filter((p) => p.viable) : []),
    [plants],
  );

  const sites = useMemo(
    () => viable.filter((p) => bands.includes(p.band)),
    [viable, bands],
  );

  const totalNpv = useMemo(() => sites.reduce((a, p) => a + (p.npv ?? 0), 0), [sites]);
  // Sum in kWh and round once, matching fetchNational's viable_energy_mwh.
  // Summing the per-site rounded energy_mwh drifts +32 MWh/yr at full selection.
  const totalEnergy = useMemo(
    () => Math.round(sites.reduce((a, p) => a + (p.energy_kwh ?? 0), 0) / 1e3),
    [sites],
  );
  // Same selection, re-read at the two ends of the calibration band, so the
  // headline energy is never shown without the range it actually sits in.
  const bandGwh = useMemo(() => {
    const sumGwh = (key: "central" | "measured") =>
      sites.reduce((a, p) => a + (p.energy_calib[key] ?? 0), 0) / 1e6;
    return { low: sumGwh("measured"), high: sumGwh("central") };
  }, [sites]);

  const toggle = (b: Band) =>
    setBands((cur) => (cur.includes(b) ? cur.filter((x) => x !== b) : [...cur, b]));

  if (error) return <div className="loading">Failed to load data: {error}</div>;
  if (!plants) return <div className="loading">Loading opportunities…</div>;

  return (
    <div style={{ padding: 22 }}>
      <h1 style={{ fontSize: 26, fontWeight: 800, color: "var(--navy)", margin: "0 0 2px" }}>Investment Opportunities</h1>
      <div className="muted" style={{ fontSize: 14, marginBottom: 18 }}>
        {num(viable.length)} economically viable micro-hydro projects nationwide
      </div>

      <div className="kpi-row" style={{ marginBottom: 14 }}>
        <KpiTile value={num(sites.length)} label="Opportunities Shown" sub={TIER_SUB} hint={TIER_HINT} />
        <KpiTile value={usd(totalNpv)} label="Combined NPV" accent="var(--green)" sub={TIER_SUB} hint={TIER_HINT} />
        <KpiTile
          value={`${num(totalEnergy)} MWh/yr`}
          label="Combined Energy"
          accent="var(--blue)"
          sub={`calibrated ${bandGwh.low.toFixed(1)}–${bandGwh.high.toFixed(1)} GWh/yr`}
          hint={TIER_HINT}
        />
      </div>

      <Caveat tone="warn" style={{ marginBottom: 18 }}>
        {CEILING_NOTE} {MODELED_NOTE}
      </Caveat>

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
        {BANDS.map((b) => (
          <button
            key={b}
            onClick={() => toggle(b)}
            className="pill"
            style={{
              cursor: "pointer",
              border: `1px solid ${bands.includes(b) ? BAND_COLOR[b] : "var(--border)"}`,
              background: bands.includes(b) ? `${BAND_COLOR[b]}22` : "#fff",
              color: bands.includes(b) ? "var(--text)" : "var(--text-faint)",
            }}
          >
            <span className="dot" style={{ background: BAND_COLOR[b] }} />
            {BAND_LABEL[b]}
          </button>
        ))}
      </div>

      <SiteTable sites={sites} csvName="wowers_opportunities.csv" />
    </div>
  );
}
