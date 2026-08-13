import type { Band, Confidence } from "./types";

export const BAND_COLOR: Record<Band, string> = {
  high: "#22c55e",
  moderate: "#eab308",
  marginal: "#f97316",
  nonviable: "#9ca3af",
};

export const BAND_LABEL: Record<Band, string> = {
  high: "Highly viable (< 5 yr)",
  moderate: "Moderately viable (5–15 yr)",
  marginal: "Marginal (15–20 yr)",
  nonviable: "Non-viable (> 20 yr)",
};

export const CONF_COLOR: Record<Confidence, string> = {
  High: "#1e9e5a",
  Medium: "#eab308",
  Lower: "#60a5fa",
};

export const CONF_PILL: Record<Confidence, string> = {
  High: "pill-high",
  Medium: "pill-medium",
  Lower: "pill-lower",
};

export const TURBINE_LABEL: Record<string, string> = {
  Kaplan: "Kaplan",
  Francis: "Francis",
  Crossflow: "Crossflow",
  in_conduit_micro: "In-conduit",
};

/** Manufacturer / supplier pages for each recommended turbine type. */
export const TURBINE_VENDOR: Record<string, { vendor: string; url: string }> = {
  Kaplan: { vendor: "Mavel (Kaplan micro-hydro)", url: "https://www.mavel.cz" },
  Francis: { vendor: "Canyon Hydro (Francis)", url: "https://www.canyonhydro.com" },
  Crossflow: { vendor: "Ossberger (Crossflow)", url: "https://ossberger.de/en/" },
  in_conduit_micro: { vendor: "InPipe Energy (in-conduit)", url: "https://www.inpipeenergy.com" },
};

export const DOE_FUNDING_URL =
  "https://www.energy.gov/eere/water/water-power-technologies-office";
