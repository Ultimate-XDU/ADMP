export type DonutChartDatum = {
  label: string;
  value: number;
  color: string;
};

export function DonutChart({ data, size = 140, strokeWidth = 18 }: { data: DonutChartDatum[]; size?: number; strokeWidth?: number }) {
  const total = data.reduce((sum, item) => sum + item.value, 0);
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  let accumulated = 0;

  return (
    <div className="flex flex-col items-center gap-3">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="drop-shadow-[0_0_12px_rgba(255,255,255,0.06)]">
        <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={strokeWidth} />
        {data.map((item) => {
          const ratio = total > 0 ? item.value / total : 0;
          const dashLength = circumference * ratio;
          const gap = circumference - dashLength;
          const offset = -circumference * accumulated + circumference * 0.25;
          accumulated += ratio;
          return (
            <circle
              key={item.label}
              cx={size / 2}
              cy={size / 2}
              r={radius}
              fill="none"
              stroke={item.color}
              strokeWidth={strokeWidth}
              strokeDasharray={`${dashLength} ${gap}`}
              strokeDashoffset={offset}
              strokeLinecap="butt"
              className="transition-all duration-700"
            />
          );
        })}
        <text x={size / 2} y={size / 2 - 6} textAnchor="middle" fill="white" fontSize="22" fontWeight="600" fontFamily="inherit">
          {total}
        </text>
        <text x={size / 2} y={size / 2 + 14} textAnchor="middle" fill="rgba(255,255,255,0.45)" fontSize="10" letterSpacing="0.12em" fontFamily="inherit">
          TOTAL
        </text>
      </svg>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1">
        {data.map((item) => (
          <div key={item.label} className="flex items-center gap-2 text-[11px] text-white/60">
            <span className="inline-block h-2 w-2 shrink-0 rounded-full" style={{ background: item.color }} />
            <span className="truncate">{item.label}</span>
            <span className="ml-auto tabular-nums text-white/40">{item.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
