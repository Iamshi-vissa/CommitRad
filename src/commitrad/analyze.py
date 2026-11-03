"""Analyze one repo end to end: clone -> mine -> SZZ -> features -> risk + SHAP -> strain."""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import pyarrow as pa

from commitrad.explain import explain_frame
from commitrad.features import FEATURE_COLUMNS, build_features
from commitrad.mining import SCHEMA, iter_records
from commitrad.model import RiskModel, load_model
from commitrad.pipeline import score_commits
from commitrad.repos import ensure_clone, repo_slug
from commitrad.sentiment import Backend, get_scorer
from commitrad.store import connect, replace_repos
from commitrad.strain import load_config, weekly_strain
from commitrad.szz import label_table

DISCLAIMER = (
    "Strain scores are signals derived from commit metadata, not judgments about people. "
    "They can reflect deadlines, time zones or personal schedules, and must not be used to "
    "evaluate, rank or discipline anyone."
)
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class InvalidRepoURL(ValueError):
    pass


class RepoTooLarge(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data/api")
    model_dir: Path = Path("models")
    strain_config: Path = Path("config/strain.toml")
    db_path: Path = Path("data/commitrad.duckdb")
    allowed_hosts: frozenset[str] = frozenset({"github.com", "gitlab.com", "codeberg.org"})
    allow_local: bool = False  # tests only: accept local directories
    timeout_seconds: float = 120.0
    git_timeout_seconds: float = 90.0
    max_commits: int = 20_000
    sentiment: Backend = "auto"

    @property
    def clone_dir(self) -> Path:
        return self.data_dir / "repos"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "results"


# --- validation & git --------------------------------------------------------------------


def validate_repo_url(url: str, settings: Settings) -> str:
    """Normalize ``https://host/owner/repo[.git]`` on an allowed host, or raise."""
    url = url.strip()
    if settings.allow_local and Path(url).is_dir():
        return str(Path(url).resolve())
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise InvalidRepoURL("Only https:// repository URLs are accepted")
    if parsed.username or parsed.password or parsed.port or parsed.query or parsed.fragment:
        raise InvalidRepoURL("URL must not contain credentials, ports, queries or fragments")
    host = (parsed.hostname or "").lower()
    if host not in settings.allowed_hosts:
        raise InvalidRepoURL(f"Host {host!r} is not allowed; use {sorted(settings.allowed_hosts)}")
    path = parsed.path.strip("/").removesuffix(".git")
    parts = path.split("/")
    if len(parts) != 2 or not all(_SEGMENT_RE.match(p) and p not in {".", ".."} for p in parts):
        raise InvalidRepoURL("URL path must be /<owner>/<repo>")
    return f"https://{host}/{parts[0]}/{parts[1]}"


def _git(repo: Path, *args: str, timeout: float) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True, timeout=timeout
    ).stdout.strip()


def clone_or_update(source: str, settings: Settings) -> Path:
    """Cached clone; refreshed with a fast-forward pull (kept as-is if the pull fails)."""
    if settings.allow_local and Path(source).is_dir():
        return Path(source)
    target = settings.clone_dir / repo_slug(source)
    if (target / ".git").is_dir():
        # A failed refresh serves the cached snapshot rather than failing the request.
        with contextlib.suppress(subprocess.CalledProcessError, subprocess.TimeoutExpired):
            _git(target, "pull", "--ff-only", "--quiet", timeout=settings.git_timeout_seconds)
        return target
    return ensure_clone(source, settings.clone_dir, timeout=settings.git_timeout_seconds)


# --- pipeline ----------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _load_model(model_dir: str) -> tuple[RiskModel, dict[str, Any]]:
    return load_model(Path(model_dir))


def commit_payload(
    scored: pd.DataFrame, reasons: list[list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    rows = []
    for (_, r), why in zip(scored.iterrows(), reasons, strict=True):
        msg = (r["message"] or "").strip()
        rows.append(
            {
                "hash": r["hash"],
                "author": r["author"],
                "timestamp": r["timestamp"].isoformat(),
                "subject": msg.splitlines()[0][:200] if msg else "",
                "risk": round(float(r["risk"]), 4),
                "risk_percentile": round(float(r["risk_pct"]), 4),
                "top_reasons": why,
            }
        )
    return sorted(rows, key=lambda c: c["risk"], reverse=True)


def strain_payload(strain: pd.DataFrame) -> list[dict[str, Any]]:
    cols = ["off_hours", "risk", "burstiness", "negativity"]
    return [
        {
            "author": r["author"],
            "author_email": r["author_email"],
            "week": r["week"].isoformat(),
            "n_commits": int(r["n_commits"]),
            "strain": round(float(r["strain"]), 4),
            "components": {c: round(float(r[c]), 4) for c in cols},
        }
        for _, r in strain.sort_values(["week", "author"]).iterrows()
    ]


def run_pipeline(source: str, repo: Path, head: str, settings: Settings) -> dict[str, Any]:
    slug = repo_slug(source)
    raw = pa.Table.from_pylist(list(iter_records(repo)), schema=SCHEMA)
    labeled = label_table(raw, repo).to_pandas()
    features = build_features(labeled, get_scorer(settings.sentiment), slug)
    if features.empty:
        raise ValueError("Repository has no non-merge commits to analyze")
    model, meta = _load_model(str(settings.model_dir))
    scored = score_commits(features, model, meta["risk_quantiles"])
    reasons = explain_frame(model.shap_values(scored), scored[FEATURE_COLUMNS])
    strain = weekly_strain(scored, load_config(settings.strain_config))
    with connect(settings.db_path) as con:
        replace_repos(con, "commit_risk", scored)
        replace_repos(con, "strain", strain)
    return {
        "repo": source,
        "slug": slug,
        "head": head,
        "n_commits": len(scored),
        "n_bug_introducing_szz": int(labeled["is_bug_introducing"].sum()),
        "model": {
            "label_mode": meta["label_mode"],
            "test_pr_auc": meta["metrics"]["test"]["lightgbm"]["pr_auc"],
            "test_roc_auc": meta["metrics"]["test"]["lightgbm"]["roc_auc"],
            # Scores for repos the model was trained on are in-sample (optimistic).
            "in_training_data": slug in meta.get("training_repos", []),
        },
        "commits": commit_payload(scored, reasons),
        "strain": strain_payload(strain),
        "disclaimer": DISCLAIMER,
    }


_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _repo_lock(slug: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(slug, threading.Lock())


def analyze_repo(url: str, settings: Settings) -> dict[str, Any]:
    """Validated, cached analysis. Results are cached per (repo, HEAD commit)."""
    source = validate_repo_url(url, settings)
    slug = repo_slug(source)
    with _repo_lock(slug):
        repo = clone_or_update(source, settings)
        head = _git(repo, "rev-parse", "HEAD", timeout=settings.git_timeout_seconds)
        cached = settings.cache_dir / f"{slug}-{head}.json"
        if cached.exists():
            return {**json.loads(cached.read_text()), "cached": True}
        count = int(_git(repo, "rev-list", "--count", "HEAD", timeout=settings.git_timeout_seconds))
        if count > settings.max_commits:
            raise RepoTooLarge(f"{count:,} commits exceeds the limit of {settings.max_commits:,}")
        result = run_pipeline(source, repo, head, settings)
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_suffix(".tmp")
        tmp.write_text(json.dumps(result))
        tmp.replace(cached)
        return {**result, "cached": False}
