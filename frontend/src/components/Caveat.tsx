import type { CSSProperties, ReactNode } from "react";

/**
 * A standing bound on the numbers around it. Deliberately plain and always
 * visible rather than tucked behind a tooltip: the figures on this dashboard
 * are modeled, and a reader should not have to hover to learn that.
 */
export default function Caveat({
  children,
  tone = "note",
  style,
}: {
  children: ReactNode;
  /** "note" for a standing bound, "warn" for one that changes how a figure should be read. */
  tone?: "note" | "warn";
  style?: CSSProperties;
}) {
  return (
    <div className={`caveat caveat-${tone}`} style={style}>
      <span className="caveat-ico" aria-hidden="true">
        {tone === "warn" ? "!" : "i"}
      </span>
      <span>{children}</span>
    </div>
  );
}
