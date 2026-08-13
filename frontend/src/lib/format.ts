export const usd = (v: number | null | undefined, full = false): string => {
  if (v == null) return "—";
  const a = Math.abs(v);
  if (full) return "$" + Math.round(v).toLocaleString();
  if (a >= 1e9) return `$${(v / 1e9).toFixed(1)}B`;
  if (a >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
  if (a >= 1e3) return `$${(v / 1e3).toFixed(0)}K`;
  return `$${v.toFixed(0)}`;
};

export const num = (v: number | null | undefined, d = 0): string =>
  v == null ? "—" : v.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });

export const mwh = (v: number | null | undefined): string =>
  v == null ? "—" : `${num(v)} MWh/yr`;

export const years = (v: number | null | undefined): string =>
  v == null ? "—" : `${v.toFixed(1)} yr`;

export const pct = (v: number | null | undefined, d = 0): string =>
  v == null ? "—" : `${(v * 100).toFixed(d)}%`;

export const pctRaw = (v: number | null | undefined, d = 1): string =>
  v == null ? "—" : `${v.toFixed(d)}%`;
