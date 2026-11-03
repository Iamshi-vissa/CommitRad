"""Minimal GitHub REST client (stdlib only) for the "link your GitHub" account view.

Unauthenticated calls get 60 requests/hour, which covers one account analysis; a personal
access token raises that to 5,000/hour and unlocks private repositories. The token is only
sent to api.github.com and github.com, is never written to disk, and is never logged.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

API = "https://api.github.com"
USER_AGENT = "commitrad"
Json = dict[str, Any] | list[Any] | str | int | float | bool | None
# (url, headers) -> (status, response headers, body)
Transport = Callable[[str, dict[str, str]], tuple[int, dict[str, str], bytes]]


class GitHubError(Exception):
    def __init__(self, message: str, status: int = 0) -> None:
        super().__init__(message)
        self.status = status


class NotFound(GitHubError):
    pass


class RateLimited(GitHubError):
    pass


class BadToken(GitHubError):
    pass


def urllib_transport(url: str, headers: dict[str, str]) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, dict(resp.headers.items()), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers.items()), exc.read()


def git_auth_config(token: str | None) -> list[str]:
    """``git -c`` arguments that authenticate an HTTPS clone without storing the token."""
    if not token:
        return []
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return ["-c", f"http.https://github.com/.extraHeader=Authorization: Basic {basic}"]


@dataclass
class GitHubClient:
    token: str | None = None
    transport: Transport = urllib_transport
    calls: int = 0
    _cache: dict[str, Json] = field(default_factory=dict, repr=False)

    def _headers(self) -> dict[str, str]:
        h = {
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def get(self, path: str, **params: str | int) -> Json:
        query = f"?{urllib.parse.urlencode(params)}" if params else ""
        url = f"{API}{path}{query}"
        if url in self._cache:
            return self._cache[url]
        status, headers, body = self.transport(url, self._headers())
        self.calls += 1
        lower = {k.lower(): v for k, v in headers.items()}
        if status == 401:
            raise BadToken("GitHub rejected the token (expired or revoked?)", status)
        if status in (403, 429) and lower.get("x-ratelimit-remaining") == "0":
            reset = int(lower.get("x-ratelimit-reset", "0") or 0)
            when = datetime.fromtimestamp(reset, UTC).strftime("%H:%M UTC") if reset else "later"
            hint = "" if self.token else " Add a personal access token for 5,000 requests/hour."
            raise RateLimited(f"GitHub API rate limit reached; it resets at {when}.{hint}", status)
        if status == 404:
            raise NotFound(f"Not found on GitHub: {path}", status)
        if status >= 400:
            raise GitHubError(f"GitHub API error {status} for {path}", status)
        data = json.loads(body or b"null")
        self._cache[url] = data
        return data

    def paginate(self, path: str, max_pages: int = 3, **params: str | int) -> Iterator[Json]:
        for page in range(1, max_pages + 1):
            items = self.get(path, per_page=100, page=page, **params)
            if not isinstance(items, list):
                return
            yield from items
            if len(items) < 100:
                return

    # --- endpoints ----------------------------------------------------------------------

    def viewer(self) -> dict[str, Any]:
        """The account the token belongs to (validates the token)."""
        return self.get("/user")

    def user(self, login: str) -> dict[str, Any]:
        return self.get(f"/users/{urllib.parse.quote(login)}")

    def repos(self, login: str, include_private: bool = False) -> list[dict[str, Any]]:
        """Repos the user owns (forks included), most recently pushed first."""
        if include_private:
            return list(self.paginate("/user/repos", affiliation="owner", sort="pushed"))
        return list(
            self.paginate(f"/users/{urllib.parse.quote(login)}/repos", type="owner", sort="pushed")
        )

    def repo(self, full_name: str) -> dict[str, Any]:
        return self.get(f"/repos/{full_name}")

    def compare(self, base_repo: str, base: str, head_owner: str, head: str) -> dict[str, Any]:
        spec = urllib.parse.quote(f"{base}...{head_owner}:{head}", safe=":.")
        return self.get(f"/repos/{base_repo}/compare/{spec}")

    def events(self, login: str) -> list[dict[str, Any]]:
        """The user's public events (GitHub keeps the last 90 days, up to 300)."""
        return list(self.paginate(f"/users/{urllib.parse.quote(login)}/events/public"))

    def author_emails(self, full_name: str, login: str) -> set[str]:
        """Emails ``login`` used as commit author in ``full_name`` (one API call)."""
        commits = self.get(f"/repos/{full_name}/commits", author=login, per_page=100)
        emails = set()
        for c in commits if isinstance(commits, list) else []:
            email = ((c.get("commit") or {}).get("author") or {}).get("email")
            if email:
                emails.add(email.lower())
        return emails
