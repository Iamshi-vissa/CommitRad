import { useEffect, useMemo, useRef, useState } from "react";
import { COMPONENT_LABELS, COMPONENTS, fixed, heatColor, heatmap, runs, weekMs } from "@/lib/data";
import type { Component, Push, StrainWeek } from "@/types";

const BLUE = "#58a6ff";
const WARM = "#f0883e";

/** One square per author-week; hover isolates a cell and shows its components. */
export function Heatmap({ strain, weeks, top }: { strain: StrainWeek[]; weeks: number; top: number }) {
  const heat = useMemo(() => heatmap(strain, weeks, top), [strain, weeks, top]);
  const [hover, setHover] = useState<{ key: string; x: number; y: number } | null>(null);
  if (!heat.authors.length) return <p className="text-muted">No commits in this window.</p>;

  const labelW = 190, cell = Math.max(10, Math.min(26, Math.floor(900 / heat.weeks.length))), gap = 3, rowH = 26;
  const width = labelW + heat.weeks.length * cell + 10;
  const height = heat.authors.length * rowH + 34;
  const ticks = heat.weeks.filter((_, i) => i % Math.max(1, Math.round(heat.weeks.length / 8)) === 0);
  const hovered = hover ? heat.cells.get(hover.key) : undefined;

  return (
    <div className="relative">
      <div className="mb-4 flex items-center gap-3 font-mono text-xs text-muted">
        strain
        <span className="h-2.5 w-40 rounded-sm" style={{ background: `linear-gradient(90deg, ${heatColor(0)}, ${heatColor(0.5)}, ${heatColor(1)})` }} />
        <span>0.0 → 1.0</span>
        <span className="text-faint">· empty = no commits that week</span>
      </div>
      <div className="overflow-x-auto">
        <svg width={width} height={height} className="block" onMouseLeave={() => setHover(null)} role="img" aria-label="Weekly strain heatmap">
          {heat.authors.map((a, r) => (
            <g key={a.email} transform={`translate(0, ${r * rowH})`}>
              <text x={labelW - 12} y={rowH / 2 + 4} textAnchor="end" className="fill-muted font-mono text-[12px]">
                {a.name.length > 24 ? `${a.name.slice(0, 23)}…` : a.name}
              </text>
              {heat.weeks.map((w, c) => {
                const key = `${a.email}|${w}`;
                const s = heat.cells.get(key);
                if (!s) return null;
                const dim = hover && hover.key !== key;
                return (
                  <rect
                    key={w}
                    x={labelW + c * cell}
                    y={(rowH - (cell - gap)) / 2}
                    width={cell - gap}
                    height={cell - gap}
                    rx={3}
                    fill={heatColor(s.strain)}
                    opacity={dim ? 0.25 : 1}
                    className="cursor-pointer transition-opacity duration-200"
                    onMouseEnter={(e) => setHover({ key, x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY })}
                  />
                );
              })}
            </g>
          ))}
          {ticks.map((w) => (
            <text key={w} x={labelW + heat.weeks.indexOf(w) * cell} y={height - 8} className="fill-faint font-mono text-[11px]">
              {w.slice(0, 7)}
            </text>
          ))}
        </svg>
      </div>
      {hover && hovered && (
        <div
          className="pointer-events-none absolute z-10 w-56 rounded-lg border border-line-strong bg-raised p-3 font-mono text-xs shadow-xl"
          style={{ left: Math.min(hover.x + 16, width - 230), top: hover.y + 40 }}
        >
          <div className="text-ink">{hovered.author}</div>
          <div className="text-faint">week of {hovered.week} · {hovered.n_commits} commits</div>
          <div className="mt-2 flex justify-between text-accent"><span>strain</span><span>{fixed(hovered.strain)}</span></div>
          {COMPONENTS.map((c) => (
            <div key={c} className="flex justify-between text-muted">
              <span>{COMPONENT_LABELS[c].toLowerCase()}</span>
              <span>{fixed(hovered.components[c])}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/** Four-axis radar: author (blue) against the repository average (orange). */
export function Radar({ author, repo, name }: { author: Record<Component, number>; repo: Record<Component, number>; name: string }) {
  const size = 400, c = size / 2, R = 118;
  const extent = Math.max(0.05, ...COMPONENTS.map((k) => Math.max(author[k], repo[k])));
  const angle = (i: number) => -Math.PI / 2 + (i * 2 * Math.PI) / COMPONENTS.length;
  const pt = (i: number, v: number) => [c + Math.cos(angle(i)) * (v / extent) * R, c + Math.sin(angle(i)) * (v / extent) * R];
  const poly = (vals: Record<Component, number>) => COMPONENTS.map((k, i) => pt(i, vals[k]).join(",")).join(" ");
  const labelPos = (i: number) => [c + Math.cos(angle(i)) * (R + 30), c + Math.sin(angle(i)) * (R + 22)];

  return (
    <div>
      <div className="flex gap-5 font-mono text-xs text-muted">
        <span className="flex items-center gap-2"><span className="h-2 w-2 rounded-full" style={{ background: BLUE }} />{name}</span>
        <span className="flex items-center gap-2"><span className="h-2 w-2 rounded-full" style={{ background: WARM }} />repo average</span>
      </div>
      <svg viewBox={`-70 0 ${size + 140} ${size}`} className="mx-auto mt-2 block w-full max-w-[460px]" role="img" aria-label="Strain components radar">
        {[0.25, 0.5, 0.75, 1].map((f) => (
          <polygon key={f} points={COMPONENTS.map((_, i) => pt(i, f * extent).join(",")).join(" ")} fill="none" stroke="#262e39" />
        ))}
        {COMPONENTS.map((_, i) => {
          const [x, y] = pt(i, extent);
          return <line key={i} x1={c} y1={c} x2={x} y2={y} stroke="#262e39" />;
        })}
        <polygon points={poly(repo)} fill={WARM} fillOpacity={0.12} stroke={WARM} strokeWidth={2} />
        <polygon points={poly(author)} fill={BLUE} fillOpacity={0.18} stroke={BLUE} strokeWidth={2} />
        {COMPONENTS.map((k, i) => {
          const [x, y] = pt(i, author[k]);
          const [lx, ly] = labelPos(i);
          return (
            <g key={k}>
              <circle cx={x} cy={y} r={4} fill={BLUE} />
              <text
                x={Math.abs(Math.cos(angle(i))) > 0.5 ? c + Math.cos(angle(i)) * (R + 14) : lx}
                y={ly + 4}
                textAnchor={Math.cos(angle(i)) > 0.5 ? "start" : Math.cos(angle(i)) < -0.5 ? "end" : "middle"}
                className="fill-ink font-mono text-[11px] tracking-wider uppercase"
              >
                {COMPONENT_LABELS[k]}
              </text>
            </g>
          );
        })}
      </svg>
      <p className="text-center font-mono text-[11px] text-faint">outer ring = {fixed(extent)}</p>
    </div>
  );
}

/** Weekly strain over time; breaks after pauses longer than four weeks. */
export function WeeklyLine({ strain, email, reference, label }: { strain: StrainWeek[]; email?: string; reference?: number; label: string }) {
  const parts = useMemo(() => runs(strain, email), [strain, email]);
  const all = parts.flat();
  const [hover, setHover] = useState<StrainWeek | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const [W, setW] = useState(560);
  useEffect(() => {
    if (!box.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, Math.round(e.contentRect.width))));
    ro.observe(box.current);
    return () => ro.disconnect();
  }, []);
  if (!all.length) return <p className="text-muted">No weekly data.</p>;

  const H = 300, L = 40, B = 28, T = 12, Rr = 12;
  const t0 = weekMs(all[0].week), t1 = Math.max(weekMs(all.at(-1)!.week), t0 + 1);
  const x = (s: StrainWeek) => L + ((weekMs(s.week) - t0) / (t1 - t0)) * (W - L - Rr);
  const y = (v: number) => T + (1 - v) * (H - T - B);
  const years = new Set<number>();
  for (let yr = new Date(t0).getUTCFullYear() + 1; yr <= new Date(t1).getUTCFullYear(); yr++) years.add(yr);
  const yearTicks = [...years].filter((_, i, a) => a.length < 9 || i % Math.ceil(a.length / 8) === 0);

  return (
    <div ref={box} className="min-w-0">
      <div className="flex flex-wrap gap-5 font-mono text-xs text-muted">
        <span className="flex items-center gap-2"><span className="h-2 w-2 rounded-full" style={{ background: BLUE }} />{label}</span>
        {reference != null && (
          <span className="flex items-center gap-2"><span className="h-0.5 w-4" style={{ background: WARM }} />repo median week {fixed(reference)}</span>
        )}
        {hover && <span className="text-ink">{hover.week} · strain {fixed(hover.strain)} · {hover.n_commits} commits</span>}
      </div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="mt-3 block max-w-full" role="img" aria-label="Weekly strain over time" onMouseLeave={() => setHover(null)}>
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line x1={L} x2={W - Rr} y1={y(v)} y2={y(v)} stroke="#1c232c" />
            <text x={L - 8} y={y(v) + 4} textAnchor="end" className="fill-faint font-mono text-[10px]">{v.toFixed(2)}</text>
          </g>
        ))}
        {yearTicks.map((yr) => {
          const xx = L + ((Date.UTC(yr, 0, 1) - t0) / (t1 - t0)) * (W - L - Rr);
          return <text key={yr} x={xx} y={H - 8} textAnchor="middle" className="fill-faint font-mono text-[10px]">{yr}</text>;
        })}
        {reference != null && <line x1={L} x2={W - Rr} y1={y(reference)} y2={y(reference)} stroke={WARM} strokeDasharray="5 4" />}
        {parts.map((run, i) => (
          <polyline key={i} points={run.map((s) => `${x(s)},${y(s.strain)}`).join(" ")} fill="none" stroke={BLUE} strokeWidth={1.6} strokeOpacity={0.8} />
        ))}
        {all.map((s) => (
          <circle
            key={s.week}
            cx={x(s)}
            cy={y(s.strain)}
            r={hover === s ? 5.5 : 3.5}
            fill={BLUE}
            className="cursor-pointer transition-all duration-150"
            onMouseEnter={() => setHover(s)}
          />
        ))}
      </svg>
    </div>
  );
}

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

/** Pushes by weekday and hour (UTC). */
export function PushRhythm({ pushes }: { pushes: Push[] }) {
  const grid = Array.from({ length: 7 }, () => Array(24).fill(0) as number[]);
  for (const p of pushes) grid[p.weekday][p.hour] += 1;
  const max = Math.max(1, ...grid.flat());
  return (
    <div className="overflow-x-auto">
      <div className="inline-grid gap-[3px]" style={{ gridTemplateColumns: `44px repeat(24, 22px)` }}>
        <span />
        {Array.from({ length: 24 }, (_, h) => (
          <span key={h} className="text-center font-mono text-[10px] text-faint">{h % 3 === 0 ? h : ""}</span>
        ))}
        {grid.map((row, d) => (
          <div key={d} className="contents">
            <span className="font-mono text-[11px] leading-[22px] text-muted">{DAYS[d]}</span>
            {row.map((n, h) => (
              <span
                key={h}
                title={`${DAYS[d]} ${h}:00 UTC · ${n} pushes`}
                className="h-[22px] rounded-[4px] transition-transform duration-150 hover:scale-110"
                style={{ background: n ? heatColor(0.15 + 0.85 * (n / max)) : "#161c24" }}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
