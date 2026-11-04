import { useState } from "react";
import { Card, Section } from "@/components/ui";

const BLOCKS: { title: string; text: string; cmd: string }[] = [
  { title: "Score your unpushed commits", text: "Ranks the commits between your upstream branch and HEAD, riskiest first, each with its three largest reasons.", cmd: "uv run commitrad check --repo ." },
  { title: "Run it on every push", text: "Installs a git pre-push hook. It warns by default and never blocks unless you ask it to.", cmd: "uv run commitrad hook install --repo ." },
  { title: "Block the riskiest 1%", text: "The hook refuses a push when any commit reaches the 99th percentile. Skip once with git push --no-verify.", cmd: "uv run commitrad hook install --repo . --block-above 0.99" },
  { title: "Use it in CI", text: "Exits with code 1 when a commit in the range reaches the chosen percentile.", cmd: "uv run commitrad check --repo . --base origin/main --fail-above 0.99" },
];

export function PrePushView() {
  return (
    <Section index="01" title="Pre-push check" intro={<>The same model, before your commits leave your machine. Run these from the CommitRad folder, pointing <code className="font-mono text-ink">--repo</code> at your project.</>}>
      <div className="grid gap-4 md:grid-cols-2">
        {BLOCKS.map((b) => (
          <Card key={b.title} className="p-5">
            <h3 className="font-semibold text-ink">{b.title}</h3>
            <p className="mt-1 text-sm text-muted">{b.text}</p>
            <CopyLine cmd={b.cmd} />
          </Card>
        ))}
      </div>
    </Section>
  );
}

function CopyLine({ cmd }: { cmd: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(cmd);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      /* clipboard unavailable */
    }
  };
  return (
    <button onClick={copy} className="group mt-4 flex w-full items-center justify-between gap-3 rounded-lg border border-line bg-canvas px-3 py-2.5 text-left font-mono text-[13px] transition-colors duration-200 hover:border-accent">
      <span className="truncate"><span className="text-accent">$</span> {cmd}</span>
      <span className="shrink-0 text-xs text-faint group-hover:text-accent">{copied ? "copied" : "copy"}</span>
    </button>
  );
}
