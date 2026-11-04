from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from gitrepo import GitRepo

import commitrad.api as api_module
from commitrad.analyze import DISCLAIMER, InvalidRepoURL, Settings, validate_repo_url
from commitrad.api import create_app


@pytest.fixture
def repo(tmp_path: Path) -> GitRepo:
    r = GitRepo(tmp_path / "proj")
    r.write("calc.py", "def add(a, b):\n    return a - b\n")
    r.commit("add calculator", "2024-01-01T23:30:00+00:00")
    r.write("calc.py", "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n")
    r.commit("add mul", "2024-01-02T10:00:00+00:00", "Bob", "bob@x.org")
    r.write("calc.py", "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n")
    r.commit("Fix add returning difference", "2024-01-03T10:00:00+00:00", "Bob", "bob@x.org")
    return r


@pytest.fixture
def settings(tmp_path: Path, trained_model_dir: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "api",
        model_dir=trained_model_dir,
        db_path=tmp_path / "api.duckdb",
        strain_config=Path(__file__).parents[1] / "config" / "strain.toml",
        allow_local=True,
        sentiment="lexicon",
    )


def get(client: TestClient, **params: str | int) -> httpx.Response:
    return client.get("/analyze", params=params)


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/a/b",
        "https://evil.example.com/a/b",
        "https://github.com/a",
        "https://github.com/a/b/c",
        "https://github.com/../b",
        "https://user:pw@github.com/a/b",
        "https://github.com:8443/a/b",
        "https://github.com/a/b?x=1",
        "file:///etc/passwd",
        "git@github.com:a/b.git",
    ],
)
def test_validation_rejects(url: str) -> None:
    with pytest.raises(InvalidRepoURL):
        validate_repo_url(url, Settings())


def test_validation_normalizes() -> None:
    s = Settings()
    assert (
        validate_repo_url(" https://GitHub.com/jd/tenacity.git/ ", s)
        == "https://github.com/jd/tenacity"
    )


def test_local_paths_rejected_unless_enabled(repo: GitRepo) -> None:
    with pytest.raises(InvalidRepoURL):
        validate_repo_url(str(repo.path), Settings())


def test_api_rejects_bad_input(settings: Settings) -> None:
    client = TestClient(create_app(replace(settings, allow_local=False)))
    assert get(client, repo="https://evil.example.com/a/b").status_code == 422
    assert client.get("/analyze").status_code == 422  # missing parameter
    assert get(client, repo="x" * 400).status_code == 422  # too long
    assert get(client, repo="https://github.com/a/b", limit=0).status_code == 422


def test_health(settings: Settings) -> None:
    assert TestClient(create_app(settings)).get("/health").json()["status"] == "ok"


def test_analyze_end_to_end_and_cache(repo: GitRepo, settings: Settings) -> None:
    client = TestClient(create_app(settings))
    res = get(client, repo=str(repo.path))
    assert res.status_code == 200
    body = res.json()
    assert body["n_commits"] == 3 and body["cached"] is False
    assert body["n_bug_introducing_szz"] == 1
    assert body["disclaimer"] == DISCLAIMER
    assert body["model"]["in_training_data"] is False

    risks = [c["risk"] for c in body["commits"]]
    assert risks == sorted(risks, reverse=True)
    for c in body["commits"]:
        assert 0 <= c["risk"] <= 1 and 0 <= c["risk_percentile"] <= 1
        assert len(c["top_reasons"]) == 3
        assert all(r["text"].endswith(("(raises risk)", "(lowers risk)")) for r in c["top_reasons"])

    authors = {s["author"] for s in body["strain"]}
    assert authors == {"Alice", "Bob"}
    assert all(0 <= s["strain"] <= 1 and set(s["components"]) == {
        "off_hours", "risk", "burstiness", "negativity"} for s in body["strain"])  # fmt: skip

    again = get(client, repo=str(repo.path), limit=1).json()
    assert again["cached"] is True and len(again["commits"]) == 1

    repo.write("new.py", "x = 1\n")
    repo.commit("add new module", "2024-01-04T10:00:00+00:00")
    fresh = get(client, repo=str(repo.path)).json()
    assert fresh["cached"] is False and fresh["n_commits"] == 4


def test_too_many_commits_is_413(repo: GitRepo, settings: Settings) -> None:
    client = TestClient(create_app(replace(settings, max_commits=2)))
    res = get(client, repo=str(repo.path))
    assert res.status_code == 413 and "limit" in res.json()["detail"]


def test_timeout_is_504(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    def slow(url: str, s: Settings) -> dict[str, Any]:
        time.sleep(0.5)
        return {}

    monkeypatch.setattr(api_module, "analyze_repo", slow)
    client = TestClient(create_app(replace(settings, timeout_seconds=0.05)))
    res = get(client, repo="https://github.com/a/b")
    assert res.status_code == 504 and "retry" in res.json()["detail"]


def test_snapshot_served_without_cloning(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from commitrad.analyze import save_snapshot

    demo = tmp_path / "demo"
    save_snapshot({"repo": "https://github.com/o/r", "slug": "o__r", "commits": [1, 2, 3]}, demo)

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("must not clone a pre-computed repo")

    monkeypatch.setattr(api_module, "analyze_repo", boom)
    client = TestClient(create_app(settings, demo_dir=demo, web_dir=tmp_path / "none"))
    body = client.get("/analyze", params={"repo": "https://github.com/o/r/", "limit": 2}).json()
    assert body["slug"] == "o__r"
    assert body["commits"] == [1, 2]
    assert "snapshot_at" in body


def test_account_endpoint(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    from commitrad.github import RateLimited

    seen: dict[str, Any] = {}

    def fake(login: str, _settings: Settings, token: str | None, max_repos: int) -> dict[str, Any]:
        seen.update(login=login, token=token, max_repos=max_repos)
        if login == "limited":
            raise RateLimited("rate limit reached")
        return {"login": login, "repos": []}

    monkeypatch.setattr(api_module, "analyze_account", fake)
    client = TestClient(create_app(settings))
    headers = {"X-GitHub-Token": "t"}
    ok = client.get("/account", params={"user": "octo", "repos": 3}, headers=headers)
    assert ok.json() == {"login": "octo", "repos": []}
    assert seen == {"login": "octo", "token": "t", "max_repos": 3}
    assert client.get("/account", params={"user": "limited"}).status_code == 429
    assert client.get("/account", params={"user": "octo", "repos": 99}).status_code == 422


def test_built_ui_served_at_root(settings: Settings, tmp_path: Path) -> None:
    web = tmp_path / "dist"
    web.mkdir()
    (web / "index.html").write_text("<!doctype html><title>CommitRad</title>")
    client = TestClient(create_app(settings, web_dir=web))
    assert "CommitRad" in client.get("/").text
    assert client.get("/health").json()["status"] == "ok"
