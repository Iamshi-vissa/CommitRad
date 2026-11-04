import { useEffect, useState } from "react";
import { commitUrl, percentile, riskBand } from "@/lib/data";
import type { Commit } from "@/types";

/** The riskiest commits as a sliding carousel; reasons read like a diff. */
export function RiskCarousel({ commits, repoUrl }: { commits: Commit[]; repoUrl?: string }) {
  const [i, setI] = useState(0);
  useEffect(() => setI(0), [commits]);
  if (!commits.length) return <p className="text-muted">No commits to show.</p>;
  const go = (step: number) => setI((v) => (v + step + commits.length) % commits.length);

  return (
    <div>
      <div className="overflow-hidden rounded-xl border border-line bg-surface">
        <div className="flex transition-transform duration-500 ease-[cubic-bezier(.22,.9,.3,1)]" style={{ transform: `translateX(-${i * 100}%)` }}>
          {commits.map((c, k) => (
            <CommitCard key={c.hash} commit={c} rank={k + 1} url={commitUrl(c.repo_url ?? repoUrl ?? "", c.hash)} />
          ))}
        </div>
      </div>
      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 font-mono text-sm">
        <div className="flex gap-1.5">
          {commits.map((c, k) => (
            <button
              key={c.hash}
              onClick={() => setI(k)}
              className={`rounded-md border px-3 py-1.5 transition-colors duration-200 ${
                k === i ? "border-accent bg-accent/10 text-accent" : "border-line text-muted hover:border-line-strong hover:text-ink"
              }`}
            >
              {String(k + 1).padStart(2, "0")}
            </button>
          ))}
        </div>
        <div className="flex gap-1.5">
          <button onClick={() => go(-1)} className="rounded-md border border-line px-3 py-1.5 text-muted transition-colors duration-200 hover:border-line-strong hover:text-ink">← prev</button>
          <button onClick={() => go(1)} className="rounded-md border border-line px-3 py-1.5 text-muted transition-colors duration-200 hover:border-line-strong hover:text-ink">next →</button>
        </div>
      </div>
    </div>
  );
}

function CommitCard({ commit: c, rank, url }: { commit: Commit; rank: number; url?: string }) {
  const p = c.risk_percentile;
  return (
    <article className="w-full shrink-0 p-6 sm:p-8">
      <div className="flex flex-wrap items-center justify-between gap-3 font-mono text-xs text-muted">
        <span>RISK RANK #{rank}{c.repo ? ` · ${c.repo}` : ""}</span>
        <span className="text-warm">{percentile(p)}th percentile · score {c.risk.toFixed(2)}</span>
      </div>
      <h3 className="mt-3 text-xl font-semibold text-ink sm:text-2xl">{c.subject || "(no message)"}</h3>
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-xs text-faint">
        {url ? (
          <a href={url} target="_blank" rel="noreferrer" className="text-accent transition-colors hover:text-ink">{c.hash.slice(0, 7)} ↗</a>
        ) : (
          <span>{c.hash.slice(0, 7)}</span>
        )}
        <span>{c.author}</span>
        <span>{c.timestamp.slice(0, 10)}</span>
        <span>{c.via_pr ? "via pull request" : "direct push"}</span>
        {c.bug_introducing && <span className="rounded border border-up/40 px-1.5 text-up">later fixed (SZZ)</span>}
      </div>
      <div className="mt-5 h-1.5 overflow-hidden rounded-full bg-raised">
        <div className="h-full rounded-full bg-gradient-to-r from-accent-strong to-warm" style={{ width: `${Math.max(3, p * 100)}%` }} />
      </div>
      <p className="mt-2 font-mono text-xs text-muted">{riskBand(p)}</p>
      <div className="mt-6 rounded-lg border border-line bg-canvas p-4 font-mono text-sm">
        <div className="mb-2 text-faint">@@ why this commit ranks here @@</div>
        {c.top_reasons.map((r) => (
          <div key={r.feature} className={`flex justify-between gap-4 rounded px-2 py-1 ${r.effect === "raises" ? "bg-up/10 text-up" : "bg-down/10 text-down"}`}>
            <span>{r.effect === "raises" ? "+" : "−"} {r.text}</span>
            <span className="shrink-0 tabular-nums opacity-80">{r.shap >= 0 ? "+" : "−"}{Math.abs(r.shap).toFixed(2)}</span>
          </div>
        ))}
      </div>
    </article>
  );
}
