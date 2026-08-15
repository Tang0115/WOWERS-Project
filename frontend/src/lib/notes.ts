/**
 * Standing caveats.
 *
 * Every one of these is a bound the analysis states about itself, so the
 * wording lives in one place and is quoted verbatim wherever it appears
 * rather than paraphrased per view.
 */

/** What is measured and what is modeled. Applies to every number shown. */
export const MODELED_NOTE =
  "Screening estimates. Discharge flows and elevations are measured; head, machine choice, efficiency, energy, cost, and cash flow are all modeled. No site in this dataset has been built or metered.";

/** Attached wherever an energy figure is shown at the uncalibrated ceiling. */
export const CEILING_NOTE =
  "Energy and every financial figure here are the ceiling tier — the model's own output, uncalibrated. Measured small-hydro capacity factors put the defensible planning range far below it.";

/** Attached to financial figures specifically, which have no calibrated variant in the export. */
export const FINANCIAL_CEILING_NOTE =
  "NPV, IRR, payback, and CapEx are computed at the ceiling tier only. Re-scoring the cash flow at a calibrated tier ejects most small sites from the viable set.";

/** Qualifier printed under a KPI tile whose figure is uncalibrated. */
export const TIER_SUB = "ceiling tier";

/** Hover text for that qualifier. */
export const TIER_HINT =
  "Computed at the as-modeled capacity factor (0.872), which sits above every percentile of 629 metered small-hydro plants. See the calibration band for what measured capacity factors do to this figure.";

/** Plant detail: the supplier links. */
export const VENDOR_NOTE =
  "Representative supplier for this turbine type, chosen for demonstration. Not an endorsement, not a quote, and not a confirmed availability check — re-verify before any external use.";

/** Plant detail: the efficiency curve drawn under the recommended turbine. */
export const EFFICIENCY_CURVE_NOTE =
  "Synthesized unimodal approximation from rated flow and peak efficiency, not a measured or vendor-published curve. It shows the operating envelope, not an engineering specification.";
