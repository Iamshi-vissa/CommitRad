import type { ReactNode } from "react";

export function Section({ index, title, intro, children, right }: {
  index: string;
  title: string;
  intro?: ReactNode;
  children: ReactNode;
  right?: ReactNode;
}) {
  return (
    <section className="mt-16 scroll-mt-20">
      <div className="flex items-center gap-4">
        <h2 className="font-mono text-2xl font-bold tracking-[0.12em] uppercase sm:text-[26px]">
          <span className="text-accent">{index}</span> <span className="text-ink">{title}</span>
        </h2>
        <div className="h-px flex-1 bg-line" />
        {right}
      </div>
      {intro && <p className="mt-3 max-w-3xl text-[15px] leading-relaxed text-muted">{intro}</p>}
      <div className="mt-6">{children}</div>
    </section>
  );
}

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div className={`rounded-xl border border-line bg-surface transition-colors duration-200 hover:border-line-strong ${className}`}>
      {children}
    </div>
  );
}

export function Tile({ label, value, sub, tone }: { label: string; value: ReactNode; sub?: ReactNode; tone?: "warm" }) {
  return (
    <Card className="p-5">
      <div className="font-mono text-[11px] tracking-[0.18em] text-muted uppercase">{label}</div>
      <div className={`mt-3 text-[34px] leading-none font-semibold tabular-nums ${tone === "warm" ? "text-warm" : "text-ink"}`}>
        {value}
      </div>
      {sub && <div className="mt-3 font-mono text-xs text-faint">{sub}</div>}
    </Card>
  );
}

export function Chip({ children, dot }: { children: ReactNode; dot?: "green" | "amber" }) {
  return (
    <span className="inline-flex items-center gap-2 rounded-md border border-line px-2.5 py-1 font-mono text-xs text-muted">
      {dot && <span className={`h-1.5 w-1.5 rounded-full ${dot === "green" ? "bg-down" : "bg-warm"}`} />}
      {children}
    </span>
  );
}

export function Label({ children }: { children: ReactNode }) {
  return <div className="font-mono text-[11px] tracking-[0.18em] text-muted uppercase">{children}</div>;
}

export function Slider({ label, value, min, max, onChange, hint }: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
  hint: string;
}) {
  return (
    <label className="block" title={hint}>
      <div className="flex items-baseline justify-between">
        <Label>{label} · view only</Label>
        <span className="font-mono text-sm text-accent tabular-nums">{value}</span>
      </div>
      <input
        type="range"
        className="mt-2 w-full cursor-pointer"
        value={value}
        min={min}
        max={max}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

export function Spinner({ text }: { text: string }) {
  return (
    <div className="mt-16 flex items-center gap-3 font-mono text-sm text-muted">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-line border-t-accent" />
      {text}
    </div>
  );
}

export function ErrorBox({ children }: { children: ReactNode }) {
  return (
    <div className="mt-10 rounded-lg border border-up/40 bg-up/10 px-4 py-3 font-mono text-sm text-up">{children}</div>
  );
}
