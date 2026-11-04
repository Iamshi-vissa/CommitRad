import type { Commit, Component, StrainWeek } from "@/types";

export const COMPONENTS: Component[] = ["off_hours", "risk", "burstiness", "negativity"];
export const COMPONENT_LABELS: Record<Component, string> = {
  off_hours: "Off-hours",
  risk: "Commit risk",
  burstiness: "Burstiness",
  negativity: "Negativity",
};

const DAY = 86_400_000;
const WEEK = 7 * DAY;

export const pct = (x: number | null | undefined, digits = 0) =>
  x == null ? "–" : `${(x * 100).toFixed(digits)}%`;
export const num = (x: number) => x.toLocaleString("en-US");
export const fixed = (x: number, d = 2) => x.toFixed(d);
export const signed = (x: number, d = 2) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`;
export const percentile = (p: number) => (p * 100).toFixed(1);

/** Plain-English position of a commit against the model's reference commits. */
export const riskBand = (p: number) => `in the top ${Math.max(1, Math.ceil((1 - p) * 100))}% of reference commits`;

const t = (iso: string) => new Date(`${iso}T00:00:00Z`).getTime();
const iso = (ms: number) => new Date(ms).toISOString().slice(0, 10);

export interface Heat {
  authors: { email: string; name: string; commits: number }[];
  weeks: string[];
  cells: Map<string, StrainWeek>; // key `${email}|${week}`
}

/** Last `weeks` weeks for the `top` most active authors. Missing weeks mean no commits. */
export function heatmap(strain: StrainWeek[], weeks = 52, top = 12): Heat {
  if (!strain.length) return { authors: [], weeks: [], cells: new Map() };
  let end = Math.max(...strain.map((s) => t(s.week)));
  end -= ((new Date(end).getUTCDay() + 6) % 7) * DAY; // Monday
  const start = end - (weeks - 1) * WEEK;
  const window = strain.filter((s) => t(s.week) >= start - 6 * DAY);
  const totals = new Map<string, { name: string; commits: number }>();
  for (const s of window) {
    const k = s.author_email ?? "";
    const cur = totals.get(k) ?? { name: s.author ?? k, commits: 0 };
    cur.commits += s.n_commits;
    totals.set(k, cur);
  }
  const authors = [...totals.entries()]
    .sort((a, b) => b[1].commits - a[1].commits)
    .slice(0, top)
    .map(([email, v]) => ({ email, ...v }));
  const named = disambiguate(authors);
  const keep = new Set(authors.map((a) => a.email));
  const cells = new Map<string, StrainWeek>();
  for (const s of window) {
    if (!keep.has(s.author_email ?? "")) continue;
    const monday = t(s.week) - ((new Date(t(s.week)).getUTCDay() + 6) % 7) * DAY;
    cells.set(`${s.author_email}|${iso(monday)}`, s);
  }
  const axis = Array.from({ length: weeks }, (_, i) => iso(start + i * WEEK));
  return { authors: named, weeks: axis, cells };
}

/** Commit-weighted mean of each component, for one author or the whole repo. */
export function profile(strain: StrainWeek[], email?: string): Record<Component | "strain", number> {
  const rows = email ? strain.filter((s) => s.author_email === email) : strain;
  const w = rows.reduce((a, s) => a + s.n_commits, 0) || 1;
  const out = { strain: 0, off_hours: 0, risk: 0, burstiness: 0, negativity: 0 };
  for (const s of rows) {
    out.strain += (s.strain * s.n_commits) / w;
    for (const c of COMPONENTS) out[c] += (s.components[c] * s.n_commits) / w;
  }
  return out;
}

export function median(xs: number[]): number {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Authors by commit count, most active first. */
export function authorsByActivity(strain: StrainWeek[]) {
  const m = new Map<string, { name: string; commits: number }>();
  for (const s of strain) {
    const k = s.author_email ?? "";
    const cur = m.get(k) ?? { name: s.author ?? k, commits: 0 };
    cur.commits += s.n_commits;
    m.set(k, cur);
  }
  return disambiguate([...m.entries()].sort((a, b) => b[1].commits - a[1].commits).map(([email, v]) => ({ email, ...v })));
}

/** Same display name, different emails: add the email's user part so rows can be told apart. */
export function disambiguate<T extends { email: string; name: string }>(rows: T[]): T[] {
  const count = new Map<string, number>();
  for (const r of rows) count.set(r.name, (count.get(r.name) ?? 0) + 1);
  return rows.map((r) => ((count.get(r.name) ?? 0) > 1 ? { ...r, name: `${r.name} (${r.email.split("@")[0]})` } : r));
}

/** An author's weeks split into runs; a pause longer than 4 weeks starts a new run. */
export function runs(strain: StrainWeek[], email?: string): StrainWeek[][] {
  const rows = strain.filter((s) => !email || s.author_email === email).sort((a, b) => t(a.week) - t(b.week));
  const out: StrainWeek[][] = [];
  rows.forEach((r, i) => {
    if (i === 0 || t(r.week) - t(rows[i - 1].week) > 4 * WEEK) out.push([]);
    out[out.length - 1].push(r);
  });
  return out;
}

export const weekMs = t;

export function commitUrl(repoUrl: string, sha: string): string | undefined {
  if (repoUrl.startsWith("https://gitlab.com/")) return `${repoUrl}/-/commit/${sha}`;
  if (repoUrl.startsWith("https://github.com/") || repoUrl.startsWith("https://codeberg.org/"))
    return `${repoUrl}/commit/${sha}`;
  return undefined;
}

export function repoParts(url: string): [string, string] {
  const parts = url.replace(/\/+$/, "").split("/").filter(Boolean);
  return parts.length >= 4 ? [parts[parts.length - 2], parts[parts.length - 1]] : ["", parts.at(-1) ?? url];
}

export const topRisky = (commits: Commit[], n = 5) => [...commits].sort((a, b) => b.risk - a.risk).slice(0, n);

/** Heatmap colour: dark navy (0) -> GitHub blue (0.5) -> pale blue (1). */
export function heatColor(v: number): string {
  const stops: [number, [number, number, number]][] = [
    [0, [12, 35, 82]],
    [0.5, [31, 111, 235]],
    [1, [165, 214, 255]],
  ];
  const x = Math.min(1, Math.max(0, v));
  const i = x <= 0.5 ? 0 : 1;
  const [a, ca] = stops[i];
  const [b, cb] = stops[i + 1];
  const f = (x - a) / (b - a);
  const c = ca.map((v0, k) => Math.round(v0 + (cb[k] - v0) * f));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}
