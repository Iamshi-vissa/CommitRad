"""FastAPI service: ``GET /analyze?repo=<https URL>`` -> per-commit risk and weekly strain."""

from __future__ import annotations

import asyncio
import subprocess
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from commitrad import __version__
from commitrad.analyze import DISCLAIMER, InvalidRepoURL, RepoTooLarge, Settings, analyze_repo

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


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="CommitRad", version=__version__, description=DISCLAIMER)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/analyze")
    async def analyze(repo: RepoParam, limit: LimitParam = None) -> dict[str, Any]:
        try:
            result = await asyncio.wait_for(
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

    return app


app = create_app()
