import { num } from "../../lib/format";

/*
 * Energy-recovery gauge.
 *
 * The needle position is driven by capacity factor — actual output vs. running
 * the turbine flat-out 24/7/365 — on a fixed 0–100% scale. This is the number
 * that genuinely differs site to site (roughly 40–90% across the portfolio).
 *
 * We deliberately do NOT drive the needle off where P50 falls between P10 and
 * P90: that ratio is set by the Monte Carlo model's *relative* uncertainty on
 * head/efficiency/availability, which is applied uniformly to every site, so
 * it always lands in a narrow ~43–50% band and the needle would look
 * identical everywhere — informative-looking but not actually informative.
 */
export default function Gauge({
  p10,
  p50,
  p90,
  capacityFactor,
  unit = "MWh/yr",
}: {
  p10: number;
  p50: number;
  p90: number;
  capacityFactor: number | null;
  unit?: string;
}) {
  const W = 340;
  const H = 210;
  const cx = W / 2;
  const cy = 170;
  const r = 128;

  const frac = Math.min(1, Math.max(0, capacityFactor ?? 0));

  const polar = (f: number, rr = r) => {
    const a = Math.PI * (1 - f); // f=0 → left (π), f=1 → right (0)
    return [cx + rr * Math.cos(a), cy - rr * Math.sin(a)] as const;
  };
  const arcPath = (f0: number, f1: number, rr = r) => {
    const [x0, y0] = polar(f0, rr);
    const [x1, y1] = polar(f1, rr);
    const large = f1 - f0 > 0.5 ? 1 : 0;
    return `M ${x0} ${y0} A ${rr} ${rr} 0 ${large} 1 ${x1} ${y1}`;
  };

  const [nx, ny] = polar(frac, r - 12);
  const [dotx, doty] = polar(frac);
  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div style={{ textAlign: "center" }}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 360 }}>
        <defs>
          <linearGradient id="gaugeGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#f97316" />
            <stop offset="55%" stopColor="#eab308" />
            <stop offset="100%" stopColor="#2bb673" />
          </linearGradient>
        </defs>

        {/* track + colored capacity-factor scale (fixed 0-100%, not data-dependent) */}
        <path d={arcPath(0, 1)} stroke="#eef1f6" strokeWidth={18} fill="none" strokeLinecap="round" />
        <path d={arcPath(0, 1)} stroke="url(#gaugeGrad)" strokeWidth={18} fill="none" strokeLinecap="round" />

        {/* minor ticks at 0/25/50/75/100% */}
        {ticks.map((t) => {
          const [ix, iy] = polar(t, r - 13);
          const [ox, oy] = polar(t, r + 4);
          return <line key={t} x1={ix} y1={iy} x2={ox} y2={oy} stroke="#c2ccda" strokeWidth={1.5} />;
        })}

        {/* needle → capacity factor */}
        <line x1={cx} y1={cy} x2={nx} y2={ny} stroke="#0f1b2d" strokeWidth={4} strokeLinecap="round" />
        <circle cx={dotx} cy={doty} r={5.5} fill="#0f1b2d" />
        <circle cx={cx} cy={cy} r={7} fill="#0f1b2d" />

        {/* fixed scale end labels */}
        <text x={polar(0)[0] - 4} y={cy + 20} fontSize={11} fontWeight={600} fill="#5b6b80" textAnchor="middle">Underused</text>
        <text x={polar(0)[0] - 4} y={cy + 34} fontSize={11} fill="#8c99ab" textAnchor="middle">0%</text>
        <text x={polar(1)[0] + 4} y={cy + 20} fontSize={11} fontWeight={600} fill="#5b6b80" textAnchor="middle">Fully utilized</text>
        <text x={polar(1)[0] + 4} y={cy + 34} fontSize={11} fill="#8c99ab" textAnchor="middle">100%</text>
      </svg>

      <div style={{ marginTop: -6 }}>
        <div style={{ fontSize: 32, fontWeight: 800, color: "#0f1b2d", lineHeight: 1 }}>
          {Math.round(p50).toLocaleString()}
        </div>
        <div className="muted" style={{ fontSize: 12, marginTop: 2 }}>
          {unit} · most-likely estimate
        </div>
        <div className="hint" style={{ fontSize: 12, marginTop: 6, fontWeight: 600, color: "#0f1b2d" }} title="Actual energy captured vs. running the turbine at full rated power every hour of the year. Reflects how well flow matches the turbine's design point.">
          {capacityFactor != null ? `${Math.round(capacityFactor * 100)}% capacity factor` : "Capacity factor unavailable"}
        </div>
      </div>

      <div style={{ display: "flex", justifyContent: "space-around", marginTop: 10, fontSize: 12 }}>
        <div style={{ textAlign: "center" }} title="Conservative estimate (P10): actual output should beat this ~90% of the time.">
          <div className="muted hint">Conservative</div><b>{num(p10)}</b>
        </div>
        <div style={{ textAlign: "center" }} title="Most-likely estimate (P50 / median): the central planning number.">
          <div className="muted hint">Most likely</div><b>{num(p50)}</b>
        </div>
        <div style={{ textAlign: "center" }} title="Optimistic estimate (P90): output tops this only ~10% of the time.">
          <div className="muted hint">Optimistic</div><b>{num(p90)}</b>
        </div>
      </div>
    </div>
  );
}
