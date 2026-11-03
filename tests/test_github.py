from __future__ import annotations

import base64
import json

import pytest

from commitrad.github import (
    BadToken,
    GitHubClient,
    NotFound,
    RateLimited,
    git_auth_config,
)


class FakeTransport:
    """Serves canned JSON by URL path+query; records every request."""

    def __init__(
        self,
        routes: dict[str, object],
        status: dict[str, int] | None = None,
        headers: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.routes, self.status, self.headers = routes, status or {}, headers or {}
        self.requests: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, headers: dict[str, str]) -> tuple[int, dict[str, str], bytes]:
        self.requests.append((url, headers))
        key = url.removeprefix("https://api.github.com")
        body = self.routes.get(key, {"message": "Not Found"})
        status = self.status.get(key, 200 if key in self.routes else 404)
        return status, self.headers.get(key, {}), json.dumps(body).encode()


def test_get_sends_headers_and_caches() -> None:
    t = FakeTransport({"/users/ann": {"login": "ann"}})
    c = GitHubClient(token="tok", transport=t)
    assert c.user("ann") == {"login": "ann"} and c.user("ann") == {"login": "ann"}
    assert len(t.requests) == 1 and c.calls == 1  # second call served from cache
    hdrs = t.requests[0][1]
    assert hdrs["Authorization"] == "Bearer tok" and hdrs["User-Agent"] == "commitrad"


def test_no_token_sends_no_authorization() -> None:
    t = FakeTransport({"/users/ann": {"login": "ann"}})
    GitHubClient(transport=t).user("ann")
    assert "Authorization" not in t.requests[0][1]


def test_errors_are_specific() -> None:
    t = FakeTransport(
        {"/users/ann": {}, "/user": {}},
        status={"/users/ann": 403, "/user": 401},
        headers={"/users/ann": {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "0"}},
    )
    c = GitHubClient(transport=t)
    with pytest.raises(RateLimited, match="personal access token"):
        c.user("ann")
    with pytest.raises(BadToken):
        c.viewer()
    with pytest.raises(NotFound):
        c.user("nobody")


def test_paginate_stops_on_short_page() -> None:
    page1 = [{"id": i} for i in range(100)]
    t = FakeTransport(
        {
            "/users/ann/repos?per_page=100&page=1&type=owner&sort=pushed": page1,
            "/users/ann/repos?per_page=100&page=2&type=owner&sort=pushed": [{"id": 100}],
        }
    )
    assert len(GitHubClient(transport=t).repos("ann")) == 101 and len(t.requests) == 2


def test_author_emails_and_compare() -> None:
    commits = [{"commit": {"author": {"email": "Ann@X.org"}}}, {"commit": {"author": {}}}]
    t = FakeTransport(
        {
            "/repos/ann/r/commits?author=ann&per_page=100": commits,
            "/repos/up/r/compare/main...ann:main": {"ahead_by": 2, "behind_by": 5},
        }
    )
    c = GitHubClient(transport=t)
    assert c.author_emails("ann/r", "ann") == {"ann@x.org"}
    assert c.compare("up/r", "main", "ann", "main")["behind_by"] == 5


def test_git_auth_config_keeps_token_out_of_urls() -> None:
    assert git_auth_config(None) == []
    flag, value = git_auth_config("s3cret")
    assert flag == "-c" and value.startswith("http.https://github.com/.extraHeader=")
    encoded = value.rsplit(" ", 1)[1]
    assert base64.b64decode(encoded).decode() == "x-access-token:s3cret"
