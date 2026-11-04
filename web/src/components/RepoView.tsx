import { useMemo, useState } from "react";
import { Heatmap, Radar, WeeklyLine } from "@/components/charts";
import { RiskCarousel } from "@/components/RiskCarousel";
import { Card, Chip, Label, Section, Slider, Tile } from "@/components/ui";
import { authorsByActivity, COMPONENT_LABELS, COMPONENTS, fixed, median, num, pct, profile, repoParts, signed, topRisky } from "@/lib/data";
import type { RepoResult } from "@/types";

export function RepoView({ data }: { data: RepoResult }) {
  const [owner, name] = repoParts(data.repo);
  const authors = useMemo(() => authorsByActivity(data.strain), [data]);
  const [weeks, setWeeks] = useState(52);
  const [top, setTop] = useState(12);
  const [email, setEmail] = useState(authors[0]?.email ?? "");
  const current = authors.find((a) => a.email === email) ?? authors[0];
  const repoAvg = useMemo(() => profile(data.strain), [data]);
  const mine = useMemo(() => profile(data.strain, current?.email), [data, current]);
  const ref = useMemo(() => median(data.strain.map((s) => s.strain)), [data]);
  const risky = useMemo(() => topRisky(data.commits), [data]);

  return (
    <div>
      <header className="flex flex-wrap items-center gap-3">
        <a href={data.repo} target="_blank" rel="noreferrer" className="group font-mono text-3xl font-bold transition-colors">
          <span className="text-accent">{owner}/</span>
          <span className="text-ink group-hover:text-accent">{name}</span>
          <span className="ml-2 text-xl text-accent">↗</span>
        </a>
        <Chip>{num(data.n_commits)} commits</Chip>
        <Chip dot={data.model.in_training_data ? "amber" : "green"}>
          {data.model.in_training_data ? "in training data" : "out-of-sample"}
        </Chip>
        {data.snapshot_at && <Chip>snapshot {data.snapshot_at.slice(0, 10)}</Chip>}
      </header>

      <div className="mt-6 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Tile label="Commits analyzed" value={num(data.n_commits)} sub="non-merge" />
        <Tile
          label="Bug-introducing"
          value={num(data.n_bug_introducing_szz)}
          sub={`${pct(data.n_bug_introducing_szz / Math.max(1, data.n_commits), 1)} · traced by szz`}
        />
        <Tile label="Authors" value={num(authors.length)} sub="humans only" />
        <Tile label="Model" value={fixed(data.model.test_pr_auc)} sub={`test pr-auc · roc ${fixed(data.model.test_roc_auc)}`} />
      </div>

      <Section
        index="01"
        title="Weekly strain"
        intro="One cell per author-week: a weighted blend of off-hours share, commit risk, burstiness and message negativity, from 0 to 1. Hover a cell to isolate it."
      >
        <div className="mb-6 grid gap-8 sm:grid-cols-2">
          <Slider label="Weeks shown" value={weeks} min={8} max={156} onChange={setWeeks} hint="Changes only how many weeks are drawn, never a value." />
          <Slider label="Most active authors" value={top} min={3} max={30} onChange={setTop} hint="Changes only how many authors get a row, never a value." />
        </div>
        <Card className="p-5">
          <Heatmap strain={data.strain} weeks={weeks} top={top} />
        </Card>
      </Section>

      {current && (
        <Section
          index="02"
          title="Strain profile"
          intro="One author against the repository. Values are commit-weighted averages from 0 to 1; differences are shown neutrally because a higher value is a pattern to understand, not a fault."
        >
          <Label>Author</Label>
          <select
            value={current.email}
            onChange={(e) => setEmail(e.target.value)}
            className="mt-2 w-full cursor-pointer rounded-lg border border-line bg-surface px-4 py-3 text-ink transition-colors duration-200 outline-none hover:border-line-strong focus:border-accent"
          >
            {authors.slice(0, 200).map((a) => (
              <option key={a.email} value={a.email}>{a.name} · {num(a.commits)} commits</option>
            ))}
          </select>
          <div className="mt-6 grid grid-cols-2 gap-4 md:grid-cols-5">
            <Tile label="Overall" value={fixed(mine.strain)} sub={`${signed(mine.strain - repoAvg.strain)} vs repo`} />
            {COMPONENTS.map((c) => (
              <Tile key={c} label={COMPONENT_LABELS[c]} value={fixed(mine[c])} sub={`${signed(mine[c] - repoAvg[c])} vs repo`} />
            ))}
          </div>
          <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
            <Card className="p-5"><Radar author={mine} repo={repoAvg} name={current.name} /></Card>
            <Card className="p-5"><WeeklyLine strain={data.strain} email={current.email} reference={ref} label={`${current.name}, weekly`} /></Card>
          </div>
        </Section>
      )}

      <Section
        index="03"
        title="Riskiest commits"
        intro="The five commits the model ranks highest. Each reason is a SHAP contribution: + raises risk, − lowers it, with its weight in log-odds."
      >
        <RiskCarousel commits={risky} repoUrl={data.repo} />
      </Section>
    </div>
  );
}
