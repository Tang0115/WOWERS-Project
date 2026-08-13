import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceArea,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { BAND_COLOR } from "../../lib/colors";
import type { Band } from "../../lib/types";
import { usd } from "../../lib/format";

const AXIS = { fontSize: 11, fill: "#5b6b80" };

const kFmt = (v: number) =>
  Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(v % 1000 === 0 ? 0 : 1)}K` : `${v}`;

/* Horizontal bar: Annual energy by site (top N) */
export function EnergyBar({ data }: { data: { name: string; mwh: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={Math.max(220, data.length * 30)}>
      <BarChart data={data} layout="vertical" margin={{ left: 10, right: 30, top: 4, bottom: 4 }}>
        <CartesianGrid horizontal={false} stroke="#eef1f6" />
        <XAxis type="number" tick={AXIS} tickFormatter={kFmt} />
        <YAxis type="category" dataKey="name" width={130} tick={{ fontSize: 11, fill: "#0f1b2d" }} />
        <Tooltip formatter={(v) => [`${Number(v).toLocaleString()} MWh/yr`, "Energy"]} />
        <Bar dataKey="mwh" radius={[0, 4, 4, 0]}>
          {data.map((_, i) => (
            <Cell key={i} fill={i === 0 ? "#16356a" : i < 3 ? "#2563eb" : "#7aa7f0"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/* Vertical bar: viable sites by state */
export function StateBar({ data, height = 230 }: { data: { state: string; viable: number }[]; height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ left: 0, right: 8, top: 8, bottom: 4 }}>
        <CartesianGrid vertical={false} stroke="#eef1f6" />
        <XAxis dataKey="state" tick={{ fontSize: 9, fill: "#5b6b80" }} interval={0} />
        <YAxis tick={AXIS} width={32} />
        <Tooltip formatter={(v) => [`${Number(v)} viable`, "Sites"]} />
        <Bar dataKey="viable" fill="#16356a" radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/* Scatter: risk vs return (payback x, NPV y), colored by band.
   Points with payback > 20 yr are excluded by the caller (domain is 0–20). */
export function RiskReturn({
  data,
}: {
  data: { payback: number; npv: number; band: Band; name: string }[];
}) {
  return (
    <ResponsiveContainer width="100%" height={320}>
      <ScatterChart margin={{ left: 8, right: 16, top: 12, bottom: 16 }}>
        <CartesianGrid stroke="#eef1f6" />
        <ReferenceArea x1={0} x2={7} fill="#22c55e" fillOpacity={0.07} />
        <XAxis
          type="number"
          dataKey="payback"
          name="Payback"
          unit="yr"
          domain={[0, 20]}
          tick={AXIS}
          label={{ value: "Payback Period (years)", position: "insideBottom", offset: -8, fontSize: 11, fill: "#5b6b80" }}
        />
        <YAxis
          type="number"
          dataKey="npv"
          name="NPV"
          tick={AXIS}
          tickFormatter={(v) => usd(Number(v))}
          width={56}
          label={{ value: "Net Present Value", angle: -90, position: "insideLeft", fontSize: 11, fill: "#5b6b80" }}
        />
        <ZAxis range={[35, 35]} />
        <Tooltip
          cursor={{ strokeDasharray: "3 3" }}
          formatter={(v, n) => (n === "NPV" ? usd(Number(v)) : `${Number(v).toFixed(1)} yr`)}
          labelFormatter={() => ""}
        />
        <Scatter data={data} fillOpacity={0.75}>
          {data.map((d, i) => (
            <Cell key={i} fill={BAND_COLOR[d.band]} />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}

/* Horizontal tornado: sensitivity impact on NPV */
export function Tornado({
  base,
  rows,
}: {
  base: number;
  rows: { label: string; low: number | null; high: number | null }[];
}) {
  const denom = Math.abs(base);
  const data = rows
    .filter((r) => r.low != null && r.high != null)
    .map((r) => {
      const lowPct = denom > 0 ? ((r.low! - base) / denom) * 100 : 0;
      const highPct = denom > 0 ? ((r.high! - base) / denom) * 100 : 0;
      return { label: r.label, low: Math.min(lowPct, highPct), high: Math.max(lowPct, highPct) };
    });
  return (
    <ResponsiveContainer width="100%" height={150}>
      <BarChart data={data} layout="vertical" stackOffset="sign" margin={{ left: 10, right: 20, top: 4, bottom: 4 }}>
        <CartesianGrid horizontal={false} stroke="#eef1f6" />
        <XAxis type="number" tick={AXIS} unit="%" />
        <YAxis type="category" dataKey="label" width={120} tick={{ fontSize: 11, fill: "#0f1b2d" }} />
        <Tooltip formatter={(v) => `${Number(v).toFixed(0)}%`} />
        <Bar dataKey="low" fill="#7aa7f0" stackId="s" />
        <Bar dataKey="high" fill="#16356a" stackId="s" />
      </BarChart>
    </ResponsiveContainer>
  );
}

/* Area: turbine efficiency curve (synthesized around rated point) */
export function EfficiencyCurve({ qRated, peakPct }: { qRated: number; peakPct: number }) {
  const pts = [];
  for (let i = 0; i <= 40; i++) {
    const q = (i / 40) * qRated * 1.4;
    const x = q / qRated;
    // simple unimodal efficiency shape peaking near rated flow
    const eff = peakPct * Math.max(0, 1 - 1.1 * Math.pow(x - 1, 2)) * (x < 0.15 ? x / 0.15 : 1);
    pts.push({ q: +q.toFixed(2), eff: +Math.max(0, eff).toFixed(1) });
  }
  return (
    <ResponsiveContainer width="100%" height={170}>
      <AreaChart data={pts} margin={{ left: 0, right: 10, top: 8, bottom: 4 }}>
        <CartesianGrid stroke="#eef1f6" />
        <XAxis dataKey="q" tick={AXIS} label={{ value: "Flow (m³/s)", position: "insideBottom", offset: -2, fontSize: 10, fill: "#5b6b80" }} />
        <YAxis tick={AXIS} domain={[0, 100]} width={42} unit="%" />
        <Tooltip formatter={(v) => [`${Number(v)}%`, "Efficiency"]} labelFormatter={(l) => `${l} m³/s`} />
        <Area dataKey="eff" stroke="#2563eb" strokeWidth={2} fill="#2563eb" fillOpacity={0.12} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

/* Two-line category tick: splits the label on "\n" so long bucket names
   (e.g. "> 15 yr\n(Non-viable)") don't collide with their neighbors. */
function BucketTick({ x, y, payload }: { x?: number; y?: number; payload?: { value?: string } }) {
  const [l1, l2] = String(payload?.value ?? "").split("\n");
  return (
    <text x={x} y={(y ?? 0) + 10} textAnchor="middle" fontSize={10} fill="#5b6b80">
      <tspan x={x}>{l1}</tspan>
      {l2 && <tspan x={x} dy={11}>{l2}</tspan>}
    </text>
  );
}

/* Vertical bar histogram: distribution over labelled buckets.
   Per-bucket `color` overrides the default fill. */
export function Histogram({
  data,
  color = "#2563eb",
  unit = "sites",
}: {
  data: { bucket: string; count: number; color?: string }[];
  color?: string;
  unit?: string;
}) {
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={data} margin={{ left: 0, right: 8, top: 8, bottom: 14 }}>
        <CartesianGrid vertical={false} stroke="#eef1f6" />
        <XAxis dataKey="bucket" tick={<BucketTick />} interval={0} />
        <YAxis tick={AXIS} width={40} tickFormatter={kFmt} />
        <Tooltip
          formatter={(v) => [`${Number(v).toLocaleString()} ${unit}`, "Count"]}
          labelFormatter={(l) => String(l).replace("\n", " ")}
        />
        <Bar dataKey="count" radius={[3, 3, 0, 0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.color ?? color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/* Horizontal bar: category mix (e.g. turbine types) with per-row colors */
export function MixBar({
  data,
}: {
  data: { label: string; count: number; color?: string }[];
}) {
  return (
    <ResponsiveContainer width="100%" height={Math.max(160, data.length * 38)}>
      <BarChart data={data} layout="vertical" margin={{ left: 10, right: 30, top: 4, bottom: 4 }}>
        <CartesianGrid horizontal={false} stroke="#eef1f6" />
        <XAxis type="number" tick={AXIS} tickFormatter={kFmt} />
        <YAxis type="category" dataKey="label" width={110} tick={{ fontSize: 11, fill: "#0f1b2d" }} />
        <Tooltip formatter={(v) => [Number(v).toLocaleString(), "Sites"]} />
        <Bar dataKey="count" radius={[0, 4, 4, 0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.color ?? "#16356a"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
