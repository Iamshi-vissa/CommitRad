from __future__ import annotations

from typing import Any

import pytest
from test_github import FakeTransport

from commitrad.account import (
    InvalidLogin,
    analyze_account,
    merge_strain,
    push_activity,
    select_repos,
    validate_login,
)
from commitrad.analyze import Settings
from commitrad.github import GitHubClient, GitHubError

USER = {
    "login": "ann",
    "id": 42,
    "name": "Ann",
    "email": None,
    "avatar_url": "https://avatars/ann",
    "html_url": "https://github.com/ann",
}
REPOS = [
    {
        "full_name": "ann/tool",
        "html_url": "https://github.com/ann/tool",
        "fork": False,
        "private": False,
        "size": 10,
        "pushed_at": "2026-10-01T00:00:00Z",
        "language": "Python",
        "owner": {"login": "ann"},
        "default_branch": "main",
    },
    {
        "full_name": "ann/lib",
        "html_url": "https://github.com/ann/lib",
        "fork": True,
        "private": False,
        "size": 10,
        "pushed_at": "2026-09-01T00:00:00Z",
        "language": "Go",
        "owner": {"login": "ann"},
        "default_branch": "main",
    },
    {
        "full_name": "ann/old",
        "html_url": "https://github.com/ann/old",
        "fork": False,
        "archived": True,
        "size": 10,
        "pushed_at": "2026-11-01T00:00:00Z",
        "owner": {"login": "ann"},
    },
    {
        "full_name": "ann/broken",
        "html_url": "https://github.com/ann/broken",
        "fork": False,
        "private": False,
        "size": 10,
        "pushed_at": "2026-08-01T00:00:00Z",
        "owner": {"login": "ann"},
    },
]
EVENTS = [
    {
        "type": "PushEvent",
        "created_at": "2026-10-04T23:15:00Z",
        "repo": {"name": "ann/tool"},
        "payload": {"ref": "refs/heads/main", "size": 3},
    },
    {
        "type": "PushEvent",
        "created_at": "2026-10-06T10:00:00Z",
        "repo": {"name": "ann/lib"},
        "payload": {"ref": "refs/heads/fix", "size": 1},
    },
    {"type": "WatchEvent", "created_at": "2026-10-06T11:00:00Z", "repo": {"name": "x/y"}},
]


def routes() -> dict[str, object]:
    return {
        "/users/ann": USER,
        "/users/ann/repos?per_page=100&page=1&type=owner&sort=pushed": REPOS,
        "/repos/ann/tool/commits?author=ann&per_page=100": [
            {"commit": {"author": {"email": "ann@work.com"}}}
        ],
        "/repos/ann/lib/commits?author=ann&per_page=100": [],
        "/repos/ann/broken/commits?author=ann&per_page=100": [],
        "/repos/ann/lib": {"parent": {"full_name": "up/lib", "default_branch": "master"}},
        "/repos/up/lib/compare/master...ann:main": {"ahead_by": 4, "behind_by": 17},
        "/users/ann/events/public?per_page=100&page=1": EVENTS,
    }


def commit(email: str, risk: float, bug: bool, via_pr: bool, off: bool) -> dict[str, Any]:
    return {
        "hash": f"h{risk}",
        "author": email.split("@")[0],
        "author_email": email,
        "timestamp": "2026-10-01T00:00:00+00:00",
        "subject": "s",
        "risk": risk,
        "risk_percentile": risk,
        "via_pr": via_pr,
        "off_hours": off,
        "bug_introducing": bug,
        "top_reasons": [],
    }


def fake_analyze(url: str, settings: Settings, token: str | None) -> dict[str, Any]:
    if url.endswith("/broken"):
        raise RuntimeError("clone failed")
    mine = "ann@work.com" if url.endswith("/tool") else "42+ann@users.noreply.github.com"
    commits = [
        commit(mine, 0.9, True, False, True),
        commit(mine, 0.5, False, True, False),
        commit("bob@x.org", 0.1, False, True, False),
        commit("bob@x.org", 0.2, False, True, False),
    ]
    week = {
        "author": "Ann",
        "author_email": mine,
        "week": "2026-09-28",
        "n_commits": 2,
        "strain": 0.4,
        "components": {"off_hours": 0.5, "risk": 0.7, "burstiness": 0.0, "negativity": 0.0},
    }
    return {"commits": commits, "strain": [week], "model": {"in_training_data": False}}


def test_validate_login() -> None:
    assert validate_login(" @Ann-1 ") == "Ann-1"
    for bad in ["", "a--b", "-ann", "ann-", "a b", "x" * 40, "ann/../x"]:
        with pytest.raises(InvalidLogin):
            validate_login(bad)


def test_select_repos_skips_archived_and_empty_and_sorts_by_push() -> None:
    picked = select_repos([*REPOS, {"full_name": "ann/empty", "size": 0, "pushed_at": "2027"}], 3)
    assert [r["full_name"] for r in picked] == ["ann/tool", "ann/lib", "ann/broken"]


def test_analyze_account_end_to_end() -> None:
    client = GitHubClient(transport=FakeTransport(routes()))
    seen: list[str] = []
    out = analyze_account(
        "ann", Settings(), client=client, analyze=fake_analyze, max_repos=3, progress=seen.append
    )
    assert seen == ["[1/3] ann/tool", "[2/3] ann/lib", "[3/3] ann/broken"]
    repos = {r["full_name"]: r for r in out["repos"]}
    assert (
        repos["ann/broken"]["status"] == "failed" and "clone failed" in repos["ann/broken"]["error"]
    )
    tool, lib = repos["ann/tool"], repos["ann/lib"]
    assert (tool["my_commits"], tool["repo_commits"]) == (2, 4)  # matched via commit-search email
    assert lib["my_commits"] == 2  # matched via the id+login noreply address
    assert (lib["parent"], lib["ahead_by"], lib["behind_by"]) == ("up/lib", 4, 17)
    assert tool["my_bug_rate"] == 0.5 and tool["repo_bug_rate"] == 0.25
    assert tool["my_via_pr"] == 0.5 and tool["my_off_hours"] == 0.5

    t = out["totals"]
    assert (t["repos_analyzed"], t["forks"], t["my_commits"], t["my_bug_introducing"]) == (
        2,
        1,
        4,
        2,
    )
    assert t["pushes_90d"] == 2 and t["push_commits_90d"] == 4 and t["off_hours_pushes"] == 0.5
    assert [c["risk"] for c in out["commits"]] == [0.9, 0.9, 0.5, 0.5]  # riskiest first
    assert {c["repo"] for c in out["commits"]} == {"ann/tool", "ann/lib"}
    assert out["strain"] == [
        {
            "week": "2026-09-28",
            "n_commits": 4,
            "strain": pytest.approx(0.4),
            "components": pytest.approx(
                {"off_hours": 0.5, "risk": 0.7, "burstiness": 0.0, "negativity": 0.0}
            ),
        }
    ]
    assert out["api_calls"] > 0 and "not judgments" in out["disclaimer"]


def test_token_must_belong_to_the_login() -> None:
    client = GitHubClient(
        token="t", transport=FakeTransport({**routes(), "/user": {"login": "mallory"}})
    )
    with pytest.raises(GitHubError, match="mallory"):
        analyze_account("ann", Settings(), token="t", client=client, analyze=fake_analyze)


def test_token_lists_private_repos() -> None:
    r = {
        **routes(),
        "/user": {"login": "ann"},
        "/user/repos?per_page=100&page=1&affiliation=owner&sort=pushed": [
            {**REPOS[0], "private": True}
        ],
    }
    out = analyze_account(
        "ann",
        Settings(),
        token="t",
        client=GitHubClient(token="t", transport=FakeTransport(r)),
        analyze=fake_analyze,
    )
    assert [x["private"] for x in out["repos"]] == [True]


def test_push_activity_and_merge_strain_units() -> None:
    pushes = push_activity(EVENTS)
    assert [(p["weekday"], p["hour"], p["branch"], p["commits"]) for p in pushes] == [
        (6, 23, "main", 3),
        (1, 10, "fix", 1),
    ]
    merged = merge_strain(
        [
            {"week": "w", "n_commits": 1, "strain": 1.0, "components": {"a": 1.0}},
            {"week": "w", "n_commits": 3, "strain": 0.0, "components": {"a": 0.0}},
        ]
    )
    assert merged == [{"week": "w", "n_commits": 4, "strain": 0.25, "components": {"a": 0.25}}]
