"""FastAPI service: ``GET /analyze?repo=<https URL>`` -> per-commit risk and weekly strain.

Also ``GET /account?user=<login>`` for a GitHub account, and, when the React UI has been built
into ``web/dist``, the UI itself at ``/``.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles

from commitrad import __version__
from commitrad.account import InvalidLogin, analyze_account
from commitrad.analyze import (
    DEMO_DIR,
    DISCLAIMER,
    InvalidRepoURL,
    RepoTooLarge,
    Settings,
    analyze_repo,
    list_snapshots,
    load_snapshot,
    validate_repo_url,
)
from commitrad.github import BadToken, GitHubError, NotFound, RateLimited

WEB_DIR = Path("web/dist")

RepoParam = Annotated[
    str,
    Query(
        min_length=8,
        max_length=300,
        description="HTTPS URL of a public repo, e.g. https://github.com/jd/tenacity",
    ),
]
LimitParam = Annotated[
    int | None, Query(ge=1, le=10_000, description="Return only the N riskiest commits.")
]
UserParam = Annotated[str, Query(min_length=1, max_length=39, description="GitHub login")]
ReposParam = Annotated[int, Query(ge=1, le=15, description="Most recently pushed repos to analyze")]
TokenHeader = Annotated[
    str | None,
    Header(alias="X-GitHub-Token", description="Optional token: private repos, higher rate limit"),
]


def find_snapshot(repo: str, settings: Settings, demo_dir: Path) -> dict[str, Any] | None:
    """A pre-computed result for this repo, if one exists (opens instantly)."""
    snapshots = list_snapshots(demo_dir) if demo_dir.is_dir() else {}
    if repo not in snapshots:
        try:
            repo = validate_repo_url(repo, settings)
        except InvalidRepoURL:
            return None
    path = snapshots.get(repo)
    return load_snapshot(path) if path else None


def github_status(exc: GitHubError) -> int:
    if isinstance(exc, NotFound):
        return 404
    if isinstance(exc, RateLimited):
        return 429
    if isinstance(exc, BadToken):
        return 401
    return 502


def create_app(
    settings: Settings | None = None, demo_dir: Path = DEMO_DIR, web_dir: Path = WEB_DIR
) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="CommitRad", version=__version__, description=DISCLAIMER)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/analyze")
    async def analyze(repo: RepoParam, limit: LimitParam = None) -> dict[str, Any]:
        snapshot = find_snapshot(repo, settings, demo_dir)
        try:
            result = snapshot or await asyncio.wait_for(
                run_in_threadpool(analyze_repo, repo, settings), timeout=settings.timeout_seconds
            )
        except InvalidRepoURL as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RepoTooLarge as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except TimeoutError as exc:
            detail = (
                f"Analysis exceeded {settings.timeout_seconds:.0f}s. It continues in the "
                "background; retry shortly to get the cached result."
            )
            raise HTTPException(status_code=504, detail=detail) from exc
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail="Cloning the repository timed out") from exc
        except subprocess.CalledProcessError as exc:
            detail = "Could not clone the repository (does it exist and is it public?)"
            raise HTTPException(status_code=502, detail=detail) from exc
        if limit is not None:
            result = {**result, "commits": result["commits"][:limit]}
        return result

    @app.get("/account")
    async def account(
        user: UserParam, repos: ReposParam = 6, token: TokenHeader = None
    ) -> dict[str, Any]:
        try:
            return await run_in_threadpool(
                analyze_account, user, settings, token=token or None, max_repos=repos
            )
        except InvalidLogin as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GitHubError as exc:
            raise HTTPException(status_code=github_status(exc), detail=str(exc)) from exc

    if (web_dir / "index.html").is_file():  # built React UI, served last so the API wins
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app


app = create_app()
