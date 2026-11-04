import { PushRhythm, WeeklyLine } from "@/components/charts";
import { RiskCarousel } from "@/components/RiskCarousel";
import { Card, Chip, Section, Tile } from "@/components/ui";
import { num, pct, percentile, topRisky } from "@/lib/data";
import type { AccountResult } from "@/types";

export function AccountView({ data }: { data: AccountResult }) {
  const t = data.totals;
  return (
    <div>
      <header className="flex flex-wrap items-center gap-4">
        {data.avatar_url && <img src={data.avatar_url} alt="" className="h-14 w-14 rounded-full border border-line" />}
        <div>
          <a href={data.profile_url} target="_blank" rel="noreferrer" className="font-mono text-3xl font-bold text-ink transition-colors hover:text-accent">
            {data.name || data.login} <span className="text-accent">@{data.login} ↗</span>
          </a>
          <div className="mt-2 flex flex-wrap gap-2">
            <Chip>{t.repos_analyzed} repos analyzed</Chip>
            <Chip>{t.forks} forks</Chip>
            <Chip>{data.api_calls} api calls</Chip>
          </div>
        </div>
      </header>

      <div className="mt-6 grid grid-cols-2 gap-4 lg:grid-cols-5">
        <Tile label="My commits" value={num(t.my_commits)} sub="across analyzed repos" />
        <Tile label="Later fixed" value={pct(t.my_bug_rate)} sub={`${t.my_bug_introducing} traced by szz`} />
        <Tile label="Via pull request" value={pct(t.my_via_pr)} sub="rest pushed directly" />
        <Tile label="Off-hours commits" value={pct(t.my_off_hours)} sub="nights + weekends" />
        <Tile label="Pushes · 90 days" value={num(t.pushes_90d)} sub={`${num(t.push_commits_90d)} commits pushed`} />
      </div>

      {t.my_commits === 0 && (
        <p className="mt-6 rounded-lg border border-warm/40 bg-warm/10 px-4 py-3 text-sm text-warm">
          None of the analyzed commits matched this account's emails. Commits made with an email that isn't linked to the GitHub account can't be attributed.
        </p>
      )}

      <Section index="01" title="Repositories" intro={'Your share of each repo and how your commits compare with everyone\'s. "Later fixed" is the share a subsequent fix traced a bug back to (SZZ); fork drift is commits ahead (↑) and behind (↓) the upstream default branch.'}>
        <Card className="overflow-x-auto">
          <table className="w-full min-w-[820px] text-sm">
            <thead>
              <tr className="border-b border-line text-left font-mono text-[11px] tracking-[0.15em] text-muted uppercase">
                {["Repository", "Commits mine / all", "Later fixed mine vs repo", "Via PR", "Off-hours", "Risk pctl mine vs repo", "Fork drift"].map((h) => (
                  <th key={h} className="px-4 py-3 font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.repos.map((r) => (
                <tr key={r.full_name} className="border-b border-line/60 transition-colors duration-150 last:border-0 hover:bg-raised">
                  <td className="px-4 py-3">
                    <a href={r.url} target="_blank" rel="noreferrer" className="font-mono text-accent hover:text-ink">{r.full_name}</a>
                    <span className="ml-2 text-xs text-faint">{r.language ?? ""}{r.fork ? " · fork" : ""}{r.private ? " · private" : ""}</span>
                    {r.status !== "ok" && <div className="mt-1 font-mono text-xs text-up">{r.error}</div>}
                  </td>
                  <td className="px-4 py-3 font-mono tabular-nums">{num(r.my_commits)} <span className="text-faint">/ {num(r.repo_commits)}</span></td>
                  <td className="px-4 py-3 font-mono tabular-nums">{pct(r.my_bug_rate)} <span className="text-faint">vs {pct(r.repo_bug_rate)}</span></td>
                  <td className="px-4 py-3 font-mono tabular-nums">{pct(r.my_via_pr)}</td>
                  <td className="px-4 py-3 font-mono tabular-nums">{pct(r.my_off_hours)}</td>
                  <td className="px-4 py-3 font-mono tabular-nums">
                    {r.my_risk_pct == null ? "–" : percentile(r.my_risk_pct)} <span className="text-faint">vs {r.repo_risk_pct == null ? "–" : percentile(r.repo_risk_pct)}</span>
                  </td>
                  <td className="px-4 py-3 font-mono tabular-nums">{r.parent ? `↑${r.ahead_by ?? "?"} ↓${r.behind_by ?? "?"}` : "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      </Section>

      <Section index="02" title="Push rhythm" intro="Your pushes from GitHub's public event log over the last 90 days, by weekday and hour (UTC).">
        <Card className="p-5">{data.pushes.length ? <PushRhythm pushes={data.pushes} /> : <p className="text-muted">No public push events in the last 90 days.</p>}</Card>
      </Section>

      <Section index="03" title="Weekly strain, all repos" intro="Your strain per week, combined across every analyzed repository.">
        <Card className="p-5"><WeeklyLine strain={data.strain} label={`@${data.login}, weekly`} /></Card>
      </Section>

      <Section index="04" title="Your riskiest commits" intro="Your five highest-risk commits across all analyzed repos.">
        <RiskCarousel commits={topRisky(data.commits)} />
      </Section>
    </div>
  );
}
