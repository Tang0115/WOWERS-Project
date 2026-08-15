export default function KpiTile({
  value,
  label,
  accent,
  sub,
  hint,
}: {
  value: string;
  label: string;
  accent?: string;
  /** Small qualifier under the label — used to name the calibration tier a figure is on. */
  sub?: string;
  /** Hover explanation for the qualifier. */
  hint?: string;
}) {
  return (
    <div className="kpi">
      <div className="v" style={accent ? { color: accent } : undefined}>
        {value}
      </div>
      <div className="l">{label}</div>
      {sub && (
        <div className={`kpi-sub${hint ? " hint" : ""}`} title={hint}>
          {sub}
        </div>
      )}
    </div>
  );
}
