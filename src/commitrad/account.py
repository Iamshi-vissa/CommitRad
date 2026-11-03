"""Personal, cross-repo view of one GitHub account ("link your GitHub").

GitHub shows how much you commit. This shows how your changes behave: which of your commits
carry the most risk and why, how many later needed a bug fix compared with each repo's
norm, how much of your work went through pull requests, when you push, and how far your
forks have drifted from upstream. Everything is computed locally; only the GitHub API
metadata calls leave your machine.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from commitrad.analyze import DISCLAIMER, Settings, analyze_repo
from commitrad.github import GitHubClient, GitHubError

_LOGIN_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
Analyzer = Callable[[str, Settings, str | None], dict[str, Any]]


class InvalidLogin(ValueError):
    pass


def validate_login(login: str) -> str:
    login = login.strip().lstrip("@")
    if not _LOGIN_RE.match(login) or "--" in login or login.endswith("-"):
        raise InvalidLogin("Enter a GitHub username, e.g. octocat")
    return login


def noreply_emails(user: dict[str, Any]) -> set[str]:
    login = str(user.get("login", "")).lower()
    emails = {f"{login}@users.noreply.github.com"}
    if user.get("id"):
        emails.add(f"{user['id']}+{login}@users.noreply.github.com")
    if user.get("email"):
        emails.add(str(user["email"]).lower())
    return emails


def select_repos(repos: Iterable[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Most recently pushed, non-archived, non-empty repos (forks included)."""
    usable = [r for r in repos if not r.get("archived") and r.get("size", 0) > 0]
    usable.sort(key=lambda r: r.get("pushed_at") or "", reverse=True)
    return usable[:limit]


@dataclass
class RepoSummary:
    full_name: str
    url: str
    fork: bool
    private: bool
    language: str | None
    pushed_at: str | None
    parent: str | None = None
    ahead_by: int | None = None
    behind_by: int | None = None
    status: str = "ok"
    error: str | None = None
    my_commits: int = 0
    repo_commits: int = 0
    my_bug_rate: float | None = None
    repo_bug_rate: float | None = None
    my_via_pr: float | None = None
    my_off_hours: float | None = None
    my_risk_pct: float | None = None
    repo_risk_pct: float | None = None
    in_training_data: bool = False

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class AccountResult:
    login: str
    name: str | None
    avatar_url: str | None
    profile_url: str
    repos: list[RepoSummary] = field(default_factory=list)
    commits: list[dict[str, Any]] = field(default_factory=list)
    strain: list[dict[str, Any]] = field(default_factory=list)
    pushes: list[dict[str, Any]] = field(default_factory=list)
    emails_matched: int = 0
    api_calls: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "login": self.login,
            "name": self.name,
            "avatar_url": self.avatar_url,
            "profile_url": self.profile_url,
            "repos": [r.as_dict() for r in self.repos],
            "commits": self.commits,
            "strain": self.strain,
            "pushes": self.pushes,
            "totals": totals(self.repos, self.commits, self.pushes),
            "emails_matched": self.emails_matched,
            "api_calls": self.api_calls,
            "disclaimer": DISCLAIMER,
        }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_repo(
    summary: RepoSummary, result: dict[str, Any], emails: set[str]
) -> list[dict[str, Any]]:
    """Fill ``summary`` from one repo analysis; return the user's commits (tagged with repo)."""
    commits = result["commits"]
    mine = [c for c in commits if c.get("author_email", "").lower() in emails]
    summary.repo_commits, summary.my_commits = len(commits), len(mine)
    summary.in_training_data = bool(result["model"].get("in_training_data"))
    summary.repo_bug_rate = _mean([float(c.get("bug_introducing", False)) for c in commits])
    summary.repo_risk_pct = _mean([c["risk_percentile"] for c in commits])
    if mine:
        summary.my_bug_rate = _mean([float(c.get("bug_introducing", False)) for c in mine])
        summary.my_via_pr = _mean([float(c.get("via_pr", False)) for c in mine])
        summary.my_off_hours = _mean([float(c.get("off_hours", False)) for c in mine])
        summary.my_risk_pct = _mean([c["risk_percentile"] for c in mine])
    return [{**c, "repo": summary.full_name, "repo_url": summary.url} for c in mine]


def merge_strain(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Combine the user's weekly strain across repos, commit-weighted per week."""
    weeks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        weeks[r["week"]].append(r)
    out = []
    for week, items in sorted(weeks.items()):
        n = sum(i["n_commits"] for i in items)
        comps = {
            k: sum(i["components"][k] * i["n_commits"] for i in items) / n
            for k in items[0]["components"]
        }
        strain = sum(i["strain"] * i["n_commits"] for i in items) / n
        out.append({"week": week, "n_commits": n, "strain": strain, "components": comps})
    return out


def push_activity(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per push: when (UTC), where, and how many commits it carried."""
    rows = []
    for e in events:
        if e.get("type") != "PushEvent":
            continue
        payload = e.get("payload") or {}
        when = datetime.fromisoformat(str(e["created_at"]).replace("Z", "+00:00"))
        rows.append(
            {
                "at": when.isoformat(),
                "weekday": when.weekday(),
                "hour": when.hour,
                "repo": (e.get("repo") or {}).get("name", ""),
                "branch": str(payload.get("ref", "")).removeprefix("refs/heads/"),
                "commits": int(payload.get("size", len(payload.get("commits") or []))),
            }
        )
    return rows


def totals(
    repos: list[RepoSummary], commits: list[dict[str, Any]], pushes: list[dict[str, Any]]
) -> dict[str, Any]:
    ok = [r for r in repos if r.status == "ok"]
    weekend_or_night = [p for p in pushes if p["weekday"] >= 5 or p["hour"] >= 22 or p["hour"] < 6]
    return {
        "repos_analyzed": len(ok),
        "forks": sum(r.fork for r in ok),
        "my_commits": len(commits),
        "my_bug_introducing": sum(bool(c.get("bug_introducing")) for c in commits),
        "my_bug_rate": _mean([float(c.get("bug_introducing", False)) for c in commits]),
        "my_via_pr": _mean([float(c.get("via_pr", False)) for c in commits]),
        "my_off_hours": _mean([float(c.get("off_hours", False)) for c in commits]),
        "my_risk_pct": _mean([c["risk_percentile"] for c in commits]),
        "pushes_90d": len(pushes),
        "push_commits_90d": sum(p["commits"] for p in pushes),
        "off_hours_pushes": len(weekend_or_night) / len(pushes) if pushes else None,
        "busiest_push_hour": Counter(p["hour"] for p in pushes).most_common(1)[0][0]
        if pushes
        else None,
    }


def fork_drift(client: GitHubClient, repo: dict[str, Any], summary: RepoSummary) -> None:
    """For a fork: its upstream, and how many commits it is ahead of / behind it."""
    detail = client.repo(repo["full_name"])
    parent = (detail or {}).get("parent") if isinstance(detail, dict) else None
    if not parent:
        return
    summary.parent = parent["full_name"]
    try:
        cmp = client.compare(
            parent["full_name"], parent.get("default_branch", "main"),
            repo["owner"]["login"], repo.get("default_branch", "main"),
        )  # fmt: skip
    except GitHubError:
        return
    if isinstance(cmp, dict):
        summary.ahead_by, summary.behind_by = cmp.get("ahead_by"), cmp.get("behind_by")


def analyze_account(
    login: str,
    settings: Settings,
    token: str | None = None,
    max_repos: int = 6,
    client: GitHubClient | None = None,
    analyze: Analyzer = analyze_repo,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Analyze the user's most recently pushed repos and their own commits in them."""
    login = validate_login(login)
    client = client or GitHubClient(token=token)
    say = progress or (lambda _msg: None)
    if token:
        viewer = client.viewer()
        if isinstance(viewer, dict) and str(viewer.get("login", "")).lower() != login.lower():
            raise GitHubError(f"That token belongs to @{viewer.get('login')}, not @{login}")
    user = client.user(login)
    assert isinstance(user, dict)
    result = AccountResult(
        login=str(user.get("login", login)),
        name=user.get("name"),
        avatar_url=user.get("avatar_url"),
        profile_url=str(user.get("html_url", f"https://github.com/{login}")),
    )
    repos = select_repos(client.repos(login, include_private=bool(token)), max_repos)
    emails = noreply_emails(user)
    strain_rows: list[dict[str, Any]] = []
    for i, repo in enumerate(repos, 1):
        say(f"[{i}/{len(repos)}] {repo['full_name']}")
        summary = RepoSummary(
            full_name=repo["full_name"], url=repo["html_url"], fork=bool(repo.get("fork")),
            private=bool(repo.get("private")), language=repo.get("language"),
            pushed_at=repo.get("pushed_at"),
        )  # fmt: skip
        result.repos.append(summary)
        try:
            emails |= client.author_emails(repo["full_name"], login)
            if summary.fork:
                fork_drift(client, repo, summary)
            analysis = analyze(repo["html_url"], settings, token)
        except Exception as exc:  # one bad repo must not sink the whole account view
            summary.status, summary.error = "failed", f"{type(exc).__name__}: {exc}"
            continue
        result.commits += summarize_repo(summary, analysis, emails)
        strain_rows += [s for s in analysis["strain"] if s["author_email"].lower() in emails]
    result.emails_matched = len(emails)
    result.commits.sort(key=lambda c: c["risk"], reverse=True)
    result.strain = merge_strain(strain_rows)
    try:
        result.pushes = push_activity(client.events(login))
    except GitHubError:
        result.pushes = []
    result.api_calls = client.calls
    return result.as_dict()
