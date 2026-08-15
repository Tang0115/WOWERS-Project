import type { CalibEnergy } from "../lib/calibration";
import {
  CALIB_TIERS,
  PLANNING_TIERS,
  RESCORE_SCENARIOS,
  RESCORE_SOURCE,
  TIER_BY_KEY,
} from "../lib/calibration";
import { gwh, num, usd, years } from "../lib/format";

/** Widest tier value, used as the 100 % reference for every bar. */
function ceilingOf(band: CalibEnergy): number {
  return Math.max(1, ...CALIB_TIERS.map((t) => band[t.key] ?? 0));
}

/**
 * Portfolio-scale calibration band: one bar per tier over the same viable
 * cohort the KPI tiles report, every value summed from the exported file.
 */
export function CalibrationBandCard({ band }: { band: CalibEnergy }) {
  const max = ceilingOf(band);
  const planning = PLANNING_TIERS.map((k) => band[k]).filter(
    (v): v is number => v != null,
  );
  const low = planning.length ? Math.min(...planning) : null;
  const high = planning.length ? Math.max(...planning) : null;

  return (
    <div className="card card-pad">
      <div className="card-head">
        <h3 className="card-title">Calibration Band — Annual Energy</h3>
        <p className="card-sub">
          The pipeline's own output implies a fleet capacity factor of 0.872, above every
          percentile of 629 metered small-hydro plants. Each row below substitutes a
          measured capacity factor for that assumption and re-scales the same viable cohort.
        </p>
      </div>

      {CALIB_TIERS.map((t) => {
        const v = band[t.key];
        const isPlanning = PLANNING_TIERS.includes(t.key);
        return (
          <div key={t.key} className="cal-row">
            <div className="cal-head">
              <span className="cal-label">
                {t.label}
                {isPlanning && <span className="cal-tag">planning range</span>}
              </span>
              <span className="cal-val">{gwh(v)}</span>
            </div>
            <div className="cal-track">
              <div
                className="cal-fill"
                style={{ width: `${((v ?? 0) / max) * 100}%`, background: t.color }}
              />
            </div>
            <div className="cal-basis">
              ×{t.multiplier.toFixed(3)} · CF {t.cf.toFixed(3)} · {t.basis}
            </div>
          </div>
        );
      })}

      <div className="cal-note">
        {low != null && high != null && (
          <>
            <strong>
              Defensible planning range: {(low / 1e3).toFixed(1)} – {(high / 1e3).toFixed(1)} GWh/yr.
            </strong>{" "}
          </>
        )}
        It is the only part of the band that two independent metered datasets agree on. The
        optimistic tier rests on a single project's design projection, so treat{" "}
        {gwh(band.central)} as the ceiling of plausibility rather than the expectation, and
        the as-modeled figure as the model's own upper bound.
      </div>
    </div>
  );
}

/**
 * Site-scale version of the same band, for a single plant's energy card.
 * Values arrive in kWh/yr and are shown in MWh/yr to match the gauge above it.
 */
export function SiteCalibrationBand({ calib }: { calib: CalibEnergy }) {
  const mwhOf = (v: number | null) => (v == null ? null : Math.round(v / 1e3));
  const max = ceilingOf(calib);

  return (
    <div className="cal-site">
      <div className="muted" style={{ fontSize: 12, marginBottom: 6 }}>
        Calibration band
      </div>
      {CALIB_TIERS.map((t) => {
        const v = calib[t.key];
        return (
          <div key={t.key} className="cal-site-row">
            <span className="cal-site-label" title={t.basis}>
              <span className="dot" style={{ background: t.color }} /> {t.short}
            </span>
            <span className="cal-site-bar">
              <span
                style={{
                  display: "block",
                  height: 6,
                  borderRadius: 3,
                  width: `${((v ?? 0) / max) * 100}%`,
                  background: t.color,
                }}
              />
            </span>
            <span className="cal-site-val">{num(mwhOf(v))} MWh</span>
          </div>
        );
      })}
      <div className="faint" style={{ fontSize: 11, marginTop: 8, lineHeight: 1.45 }}>
        The ceiling row is the headline figure on the gauge above, and every financial figure on
        this page is computed from it. The lower tiers substitute capacity factors measured at
        real plants; the metered floor comes from the one metered treated-wastewater conduit
        plant in the country. The screening-stage spread under the gauge is a separate, earlier
        estimate and sits higher at most sites.
      </div>
    </div>
  );
}

/**
 * What survives a full cash-flow re-run at each tier. These are published
 * results of the re-scoring harness, not browser-side arithmetic — the export
 * carries calibrated energy but no calibrated cash flows.
 */
export function RescoreTable() {
  return (
    <div style={{ overflowX: "auto" }}>
      <table className="tbl">
        <thead>
          <tr>
            <th style={{ cursor: "default" }}>Tier</th>
            <th className="num" style={{ cursor: "default" }}>Viable sites</th>
            <th className="num" style={{ cursor: "default" }}>Energy</th>
            <th className="num" style={{ cursor: "default" }}>Portfolio NPV</th>
            <th className="num" style={{ cursor: "default" }}>CapEx</th>
            <th className="num" style={{ cursor: "default" }}>Median payback</th>
          </tr>
        </thead>
        <tbody>
          {RESCORE_SCENARIOS.map((s) => {
            const t = TIER_BY_KEY[s.key];
            return (
              <tr key={s.key}>
                <td>
                  <span className="dot" style={{ background: t.color, marginRight: 7 }} />
                  {t.label}
                </td>
                <td className="num">{num(s.viable)}</td>
                <td className="num">{s.gwh.toFixed(1)} GWh/yr</td>
                <td className="num">{usd(s.npvUsd)}</td>
                <td className="num">{usd(s.capexUsd)}</td>
                <td className="num">{years(s.medianPayback)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="faint" style={{ fontSize: 11, marginTop: 10, lineHeight: 1.5 }}>
        {RESCORE_SOURCE} Energy here is the energy of the sites that <em>remain viable</em> at
        that tier, which is smaller than the band figure above — the harsher tiers eject sites
        from the portfolio rather than only shrinking them. The two columns answer different
        questions and should not be swapped.
      </div>
    </div>
  );
}
