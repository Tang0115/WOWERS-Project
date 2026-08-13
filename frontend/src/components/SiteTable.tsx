import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import type { MapProps, Confidence } from "../lib/types";
import { BAND_COLOR, CONF_PILL, TURBINE_LABEL } from "../lib/colors";
import { downloadCsv } from "../lib/csv";
import { num, usd, years } from "../lib/format";

type Key = keyof MapProps;

const COLS: { key: Key; label: string; n?: boolean }[] = [
  { key: "name", label: "Plant Name" },
  { key: "city", label: "City" },
  { key: "state", label: "State" },
  { key: "turbine", label: "Turbine" },
  { key: "rated_kw", label: "Rated (kW)", n: true },
  { key: "energy_mwh", label: "Energy (MWh/yr)", n: true },
  { key: "payback", label: "Payback", n: true },
  { key: "npv", label: "NPV", n: true },
  { key: "confidence", label: "Confidence" },
];

const CONF_RANK: Record<Confidence, number> = { High: 3, Medium: 2, Lower: 1 };

export const PAGE_SIZE = 50;

/** Sortable, searchable, paginated site table with CSV export. */
export default function SiteTable({
  sites,
  csvName,
}: {
  sites: MapProps[];
  csvName: string;
}) {
  const [sort, setSort] = useState<{ key: Key; dir: 1 | -1 }>({ key: "npv", dir: -1 });
  const [q, setQ] = useState("");
  const [page, setPage] = useState(0);

  const rows = useMemo(() => {
    let r = sites;
    if (q) {
      const s = q.toLowerCase();
      r = r.filter(
        (x) =>
          (x.name ?? "").toLowerCase().includes(s) ||
          (x.city ?? "").toLowerCase().includes(s) ||
          (x.state ?? "").toLowerCase().includes(s) ||
          x.id.toLowerCase().includes(s),
      );
    }
    return [...r].sort((a, b) => {
      let av: unknown = a[sort.key], bv: unknown = b[sort.key];
      if (sort.key === "confidence") {
        av = CONF_RANK[a.confidence];
        bv = CONF_RANK[b.confidence];
      }
      if (av == null) return 1;
      if (bv == null) return -1;
      return (av as number | string) > (bv as number | string) ? sort.dir
        : (av as number | string) < (bv as number | string) ? -sort.dir : 0;
    });
  }, [sites, sort, q]);

  const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const cur = Math.min(page, pages - 1);
  const slice = rows.slice(cur * PAGE_SIZE, (cur + 1) * PAGE_SIZE);

  const setKey = (key: Key) => {
    setPage(0);
    setSort((s) => (s.key === key ? { key, dir: (-s.dir) as 1 | -1 } : { key, dir: -1 }));
  };

  const exportCsv = () =>
    downloadCsv(
      csvName,
      ["npdes_id", "name", "city", "state", "turbine", "rated_kw", "energy_mwh_yr", "payback_yr", "npv_usd", "tier", "viable", "confidence"],
      rows.map((r) => [r.id, r.name, r.city, r.state, r.turbine, r.rated_kw, r.energy_mwh, r.payback, r.npv, r.tier, r.viable, r.confidence]),
    );

  return (
    <div className="card card-pad">
      <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap", marginBottom: 12 }}>
        <input
          type="text"
          placeholder="Search name, city, state, NPDES…"
          value={q}
          onChange={(e) => { setQ(e.target.value); setPage(0); }}
          style={{ width: 260 }}
        />
        <span className="faint" style={{ fontSize: 12 }}>{num(rows.length)} sites</span>
        <button className="btn btn-ghost" style={{ marginLeft: "auto" }} onClick={exportCsv}>
          ⇩ Export CSV
        </button>
      </div>
      <div style={{ overflowX: "auto" }}>
        <table className="tbl">
          <thead>
            <tr>
              {COLS.map((c) => (
                <th
                  key={c.key}
                  className={c.n ? "num" : ""}
                  onClick={() => setKey(c.key)}
                  aria-sort={sort.key === c.key ? (sort.dir === -1 ? "descending" : "ascending") : "none"}
                >
                  {c.label}{sort.key === c.key ? (sort.dir === -1 ? " ▾" : " ▴") : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {slice.map((r) => (
              <tr key={r.id}>
                <td><Link to={`/plant/${r.id}`}>{r.name ?? r.id}</Link></td>
                <td>{r.city}</td>
                <td>{r.state}</td>
                <td>{r.turbine ? TURBINE_LABEL[r.turbine] ?? r.turbine : "—"}</td>
                <td className="num">{num(r.rated_kw)}</td>
                <td className="num">{num(r.energy_mwh)}</td>
                <td className="num">{years(r.payback)}</td>
                <td className="num">{usd(r.npv)}</td>
                <td><span className={`pill ${CONF_PILL[r.confidence]}`}><span className="dot" style={{ background: BAND_COLOR[r.band] }} />{r.confidence}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {pages > 1 && (
        <div className="pager">
          <button className="btn btn-ghost" disabled={cur === 0} onClick={() => setPage(cur - 1)}>← Prev</button>
          <span className="muted" style={{ fontSize: 12 }}>Page {cur + 1} of {num(pages)}</span>
          <button className="btn btn-ghost" disabled={cur >= pages - 1} onClick={() => setPage(cur + 1)}>Next →</button>
        </div>
      )}
    </div>
  );
}
