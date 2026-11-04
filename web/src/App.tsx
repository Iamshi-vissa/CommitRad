import { useEffect, useRef, useState, type FormEvent } from "react";
import { analyzeAccount, analyzeRepo } from "@/api";
import { AccountView } from "@/components/AccountView";
import GradientWaves from "@/components/GradientWaves";
import { PrePushView } from "@/components/PrePushView";
import { RepoView } from "@/components/RepoView";
import { ErrorBox, Spinner } from "@/components/ui";
import type { AccountResult, RepoResult } from "@/types";

type Mode = "repo" | "account" | "prepush";
const MODES: { id: Mode; label: string; short: string }[] = [
  { id: "repo", label: "Repository", short: "Repo" },
  { id: "account", label: "GitHub account", short: "Account" },
  { id: "prepush", label: "Pre-push check", short: "Pre-push" },
];

const field =
  "w-full rounded-lg border border-line-strong bg-canvas/80 px-4 py-3.5 font-mono text-[15px] text-ink backdrop-blur outline-none transition-colors duration-200 hover:border-faint focus:border-accent";

export default function App() {
  const params = new URLSearchParams(window.location.search);
  const [mode, setMode] = useState<Mode>((params.get("view") as Mode) || "repo");
  const [url, setUrl] = useState(params.get("repo") ?? "");
  const [login, setLogin] = useState("");
  const [token, setToken] = useState("");
  const [repos, setRepos] = useState(6);
  const [repo, setRepo] = useState<RepoResult | null>(null);
  const [account, setAccount] = useState<AccountResult | null>(null);
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const results = useRef<HTMLDivElement>(null);

  const scrollToResults = () => requestAnimationFrame(() => results.current?.scrollIntoView({ behavior: "smooth", block: "start" }));

  async function runRepo(target: string) {
    if (!target.trim()) return;
    setError(null);
    setRepo(null);
    setLoading("Cloning and scoring commits… large repos take a minute.");
    scrollToResults();
    window.history.replaceState(null, "", `?${new URLSearchParams({ repo: target.trim() })}`);
    try {
      setRepo(await analyzeRepo(target.trim()));
      scrollToResults();
    } catch (e) {
      setError(`Could not analyze this repository: ${(e as Error).message}`);
    } finally {
      setLoading(null);
    }
  }

  async function runAccount() {
    if (!login.trim()) return;
    setError(null);
    setAccount(null);
    setLoading(`Cloning and scoring @${login.trim()}'s ${repos} most recently pushed repos…`);
    scrollToResults();
    try {
      setAccount(await analyzeAccount(login.trim(), repos, token.trim() || undefined));
      scrollToResults();
    } catch (e) {
      setError(`GitHub: ${(e as Error).message}`);
    } finally {
      setLoading(null);
    }
  }

  useEffect(() => {
    if (params.get("repo")) void runRepo(params.get("repo")!);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (mode === "repo") void runRepo(url);
    if (mode === "account") void runAccount();
  };

  const switchMode = (m: Mode) => {
    setMode(m);
    setError(null);
  };

  return (
    <div className="min-h-screen">
      <nav className="sticky top-0 z-30 border-b border-line bg-canvas/85 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center gap-6 px-4 py-3 sm:px-8">
          <a href="/" className="flex items-center gap-2.5">
            <img src="/logo.svg" alt="CommitRad" className="h-7 w-7 shrink-0" />
            <span className="hidden font-mono text-lg font-bold tracking-tight sm:inline">CommitRad</span>
          </a>
          <div className="ml-auto flex gap-1 rounded-lg border border-line p-1">
            {MODES.map((m) => (
              <button
                key={m.id}
                onClick={() => switchMode(m.id)}
                className={`rounded-md px-2.5 py-1.5 font-mono text-xs whitespace-nowrap sm:text-sm transition-colors duration-200 sm:text-sm ${
                  mode === m.id ? "bg-raised text-ink" : "text-muted hover:text-ink"
                }`}
              >
                <span className="sm:hidden">{m.short}</span>
                <span className="hidden sm:inline">{m.label}</span>
              </button>
            ))}
          </div>
        </div>
      </nav>

      <header className="relative isolate overflow-hidden border-b border-line">
        <div className="absolute inset-0 -z-10 opacity-70">
          <GradientWaves
            horizonColor="#0b1a33"
            waveColor="#1f6feb"
            crestColor="#a5d6ff"
            backgroundColor="#0b0f14"
            speed={0.25}
            amplitude={2}
            grain={0.12}
            quality="medium"
            swellOnClick={false}
          />
        </div>
        <div className="absolute inset-0 -z-10 bg-gradient-to-b from-canvas/30 via-canvas/40 to-canvas" />
        <div className="mx-auto max-w-7xl px-4 pt-20 pb-16 sm:px-8 sm:pt-28">
          <p className="font-mono text-xs tracking-[0.3em] text-accent uppercase">Explainable commit risk</p>
          <h1 className="mt-4 text-5xl font-bold tracking-tight sm:text-7xl">
            CommitRad<span className="text-accent">_</span>
          </h1>
          <p className="mt-5 max-w-2xl text-lg leading-relaxed text-muted">
            Ranks every commit by the chance it will need a bug fix, and tells you why. Learned from 27,384 commits in
            16 open-source projects; runs entirely on your machine.
          </p>

          {mode !== "prepush" && (
            <form onSubmit={submit} className="mt-10 max-w-3xl">
              {mode === "repo" ? (
                <div className="relative">
                  <span className="pointer-events-none absolute top-1/2 left-4 z-10 -translate-y-1/2 font-mono text-[15px] text-accent">$ analyze</span>
                  <input
                    className={`${field} pl-[6.5rem]`}
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    placeholder="https://github.com/owner/repo"
                    aria-label="Repository URL"
                    autoFocus
                  />
                </div>
              ) : (
                <div className="grid gap-3 sm:grid-cols-[1fr_1fr_110px]">
                  <input className={field} value={login} onChange={(e) => setLogin(e.target.value)} placeholder="GitHub username" aria-label="GitHub username" autoFocus />
                  <input className={field} type="password" value={token} onChange={(e) => setToken(e.target.value)} placeholder="token (optional)" aria-label="GitHub token" autoComplete="off" />
                  <input className={field} type="number" min={1} max={15} value={repos} onChange={(e) => setRepos(Number(e.target.value))} aria-label="Repositories to analyze" title="Most recently pushed repos to analyze" />
                </div>
              )}
              <div className="mt-3 flex flex-wrap items-center gap-4">
                <button type="submit" disabled={!!loading} className="rounded-lg bg-accent-strong px-5 py-2.5 font-mono text-sm font-medium text-white transition-colors duration-200 hover:bg-accent disabled:opacity-50">
                  {loading ? "analyzing…" : "Analyze ↵"}
                </button>
                <span className="font-mono text-xs text-faint">
                  {mode === "repo"
                    ? "public GitHub, GitLab or Codeberg · press Enter"
                    : "token stays in this tab's memory and is sent only to the local API"}
                </span>
              </div>
            </form>
          )}
        </div>
      </header>

      <main ref={results} className="mx-auto max-w-7xl scroll-mt-16 px-4 pb-24 sm:px-8">
        {error && <ErrorBox>{error}</ErrorBox>}
        {loading && <Spinner text={loading} />}
        <div className="pt-12">
          {mode === "repo" && repo && !loading && <RepoView data={repo} />}
          {mode === "account" && account && !loading && <AccountView data={account} />}
          {mode === "prepush" && <PrePushView />}
        </div>
      </main>

      <footer className="border-t border-line">
        <div className="mx-auto flex max-w-7xl flex-wrap justify-between gap-3 px-4 py-6 font-mono text-xs text-faint sm:px-8">
          <span>Signals, not judgments: use risk to aim review attention, never to rank people.</span>
          <a href="https://github.com/Iamshi-vissa/CommitRad" target="_blank" rel="noreferrer" className="transition-colors hover:text-accent">
            github.com/Iamshi-vissa/CommitRad
          </a>
        </div>
      </footer>
    </div>
  );
}
