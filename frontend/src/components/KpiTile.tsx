export default function KpiTile({
  value,
  label,
  accent,
}: {
  value: string;
  label: string;
  accent?: string;
}) {
  return (
    <div className="kpi">
      <div className="v" style={accent ? { color: accent } : undefined}>
        {value}
      </div>
      <div className="l">{label}</div>
    </div>
  );
}
