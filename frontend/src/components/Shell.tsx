import { NavLink } from "react-router-dom";
import type { ReactNode } from "react";

const NAV = [
  { to: "/", label: "Dashboard", ico: "▦", end: true },
  { to: "/opportunities", label: "Opportunities", ico: "◎" },
  { to: "/plants", label: "Plants", ico: "🏭" },
  { to: "/analytics", label: "Analytics", ico: "📊" },
  { to: "/reports", label: "Reports", ico: "🗎" },
];

export default function Shell({ children }: { children: ReactNode }) {
  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <h1>WOWERS</h1>
          <p>Wastewater Energy Recovery Intelligence Platform</p>
        </div>
        <nav className="nav">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end}>
              <span className="nav-ico">{n.ico}</span>
              {n.label}
            </NavLink>
          ))}
        </nav>
        <div style={{ marginTop: "auto", padding: 16, fontSize: 11 }} className="faint">
          Data: EPA NPDES/DMR · EPRI · USGS 3DEP
        </div>
      </aside>
      <div className="main">{children}</div>
    </div>
  );
}
