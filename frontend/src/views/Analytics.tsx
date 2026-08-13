import { useMemo } from "react";
import { fetchNational, fetchPlants, useAsync } from "../lib/data";
import { BAND_COLOR, CONF_COLOR, TURBINE_LABEL } from "../lib/colors";
import type { Band, Confidence } from "../lib/types";
import { num, usd, years } from "../lib/format";
import KpiTile from "../components/KpiTile";
import { Histogram, MixBar, StateBar } from "../components/charts/Charts";

const PB_BUCKETS: { label: string; min: number; max: number; band: Band }[] = [
  { label: "<3 yr", min: 0, max: 3, band: "high" },
  { label: "3–5", min: 3, max: 5, band: "high" },
  { label: "5–8", min: 5, max: 8, band: "moderate" },
  { label: "8–11", min: 8, max: 11, band: "moderate" },
  { label: "11–15", min: 11, max: 15, band: "moderate" },
];

/** Portfolio-level analytics across all scored sites. */
export default function Analytics() {
  const { data: national, error: nErr } = useAsync(fetchNational, []);
  const { data: plants, error: pErr } = useAsync(fetchPlants, []);

  const all = useMemo(
    () => (plants ? plants.features.map((f) => f.properties) : []),
    [plants],
  );
  const viable = useMemo(() => all.filter((p) => p.viable), [all]);

  const paybackHist = useMemo(() => {
    const buckets = PB_BUCKETS.map((b) => ({
      bucket: b.label,
      count: viable.filter((p) => p.payback != null && p.payback >= b.min && p.payback < b.max).length,
      color: BAND_COLOR[b.band],
    }));
    // Viable sites never exceed ~14 yr payback (the NPV>0 gate cuts them off
    // well before the 20-yr limit), so everything past 15 yr — including the
    // ">20 yr" band and never-pays-back sites — is the non-viable tail.
    buckets.push({
      bucket: "> 15 yr\n(Non-viable)",
      count: all.filter((p) => p.payback == null || p.payback > 15).length,
      color: BAND_COLOR.nonviable,
    });
    return buckets;
  }, [all, viable]);

  const turbineMix = useMemo(() => {
    const m = new Map<string, number>();
    for (const p of viable) {
      const t = p.turbine ?? "Unassigned";
      m.set(t, (m.get(t) ?? 0) + 1);
    }
    return [...m.entries()]
      .map(([t, count]) => ({ label: TURBINE_LABEL[t] ?? t, count }))
      .sort((a, b) => b.count - a.count);
  }, [viable]);

  const confMix = useMemo(() => {
    const order: Confidence[] = ["High", "Medium", "Lower"];
    return order.map((c) => ({
      label: c,
      count: viable.filter((p) => p.confidence === c).length,
      color: CONF_COLOR[c],
    }));
  }, [viable]);

  const bandMix = useMemo(() => {
    const order: { b: Band; label: string }[] = [
      { b: "high", label: "Highly viable" },
      { b: "moderate", label: "Moderate" },
      { b: "marginal", label: "Marginal" },
    ];
    return order.map(({ b, label }) => ({
      label,
      count: viable.filter((p) => p.band === b).length,
      color: BAND_COLOR[b],
    }));
  }, [viable]);

  const error = nErr ?? pErr;
  if (error) return <div className="loading">Failed to load data: {error}</div>;
  if (!national || !plants) return <div className="loading">Loading analytics…</div>;

  return (
    <div style={{ padding: 22 }}>
      <h1 style={{ fontSize: 26, fontWeight: 800, color: "var(--navy)", margin: "0 0 2px" }}>Portfolio Analytics</h1>
      <div className="muted" style={{ fontSize: 14, marginBottom: 18 }}>
        Distribution views across {num(national.scored_sites)} scored sites, {num(national.viable_projects)} viable
      </div>

      <div className="kpi-row" style={{ marginBottom: 18 }}>
        <KpiTile value={usd(national.portfolio_npv_usd)} label="Portfolio NPV" accent="var(--green)" />
        <KpiTile value={usd(national.portfolio_capex_usd)} label="Total CapEx" />
        <KpiTile value={`${usd(national.annual_savings_usd)}/yr`} label="Annual Savings" accent="var(--blue)" />
        <KpiTile value={`${num(national.viable_energy_mwh)} MWh/yr`} label="Recoverable Energy" />
        <KpiTile value={years(national.median_payback)} label="Median Payback" />
        <KpiTile value={num(national.high_confidence_sites)} label="High-Confidence Sites" />
      </div>

      <div className="an-grid">
        <div className="card card-pad">
          <h3 className="card-title">Payback Distribution</h3>
          <Histogram data={paybackHist} />
          <div className="faint" style={{ fontSize: 11, marginTop: 6 }}>
            Colored bars: viable sites by payback. Grey bar: non-viable sites — payback over 15 yr or never pays back.
          </div>
        </div>
        <div className="card card-pad">
          <h3 className="card-title">Viable Sites by State</h3>
          <StateBar data={national.by_state} height={240} />
        </div>
        <div className="card card-pad">
          <h3 className="card-title">Turbine Mix (viable sites)</h3>
          <MixBar data={turbineMix} />
        </div>
        <div className="card card-pad">
          <h3 className="card-title">Viability &amp; Confidence Breakdown</h3>
          <MixBar data={[...bandMix, ...confMix]} />
        </div>
      </div>
    </div>
  );
}
