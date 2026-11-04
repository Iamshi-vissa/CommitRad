import type { AccountResult, RepoResult } from "@/types";

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      /* keep the status text */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const analyzeRepo = (repo: string) => call<RepoResult>(`/analyze?${new URLSearchParams({ repo })}`);

export const analyzeAccount = (user: string, repos: number, token?: string) =>
  call<AccountResult>(`/account?${new URLSearchParams({ user, repos: String(repos) })}`, {
    headers: token ? { "X-GitHub-Token": token } : {},
  });
