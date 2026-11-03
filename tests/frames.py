"""Builders for in-memory labeled-commit DataFrames (the input to feature building)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

BASE = pd.Timestamp("2024-01-01T00:00:00Z")


def commit(
    hash: str,
    t_hours: float,
    author: str = "alice",
    files: dict[str, tuple[int, int]] | None = None,
    message: str = "change something specific here",
    found_after_hours: float | None = None,
    tz_offset_minutes: int = 0,
    is_merge: bool = False,
) -> dict[str, Any]:
    files = files or {"a.py": (1, 0)}
    ts = BASE + pd.Timedelta(hours=t_hours)
    found = ts + pd.Timedelta(hours=found_after_hours) if found_after_hours is not None else pd.NaT
    return {
        "hash": hash,
        "is_merge": is_merge,
        "author": author.title(),
        "author_email": f"{author}@x.org",
        "timestamp": ts,
        "tz_offset_minutes": tz_offset_minutes,
        "message": message,
        "lines_added": sum(a for a, _ in files.values()),
        "lines_deleted": sum(d for _, d in files.values()),
        "files_changed": len(files),
        "modified_files": list(files),
        "file_lines_added": [a for a, _ in files.values()],
        "file_lines_deleted": [d for _, d in files.values()],
        "is_bug_introducing": found_after_hours is not None,
        "bug_found_at": found,
    }


def frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["bug_found_at"] = pd.to_datetime(df["bug_found_at"], utc=True)
    return df


def random_history(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Random commits with timestamp ties, shared files, and ~20% later-found bugs."""
    rng = np.random.default_rng(seed)
    authors = ["alice", "bob", "carol", "dave", "erin"]
    pool = [f"pkg/m{i}.py" for i in range(10)]
    rows = []
    for i in range(n):
        picked = rng.choice(pool, size=int(rng.integers(1, 4)), replace=False)
        files = {str(p): (int(rng.integers(0, 50)), int(rng.integers(0, 20))) for p in picked}
        found = float(rng.uniform(1, 2000)) if rng.random() < 0.2 else None
        rows.append(
            commit(
                f"h{i:04d}",
                t_hours=float(rng.integers(0, 4000)),  # integer hours -> plenty of ties
                author=str(rng.choice(authors)),
                files=files,
                message=str(rng.choice(["fix stuff", "Refactor parser for clarity", "ugh hack"])),
                found_after_hours=found,
                tz_offset_minutes=int(rng.choice([-300, 0, 60, 330])),
            )
        )
    return frame(rows)


def planted_features(n: int = 900, seed: int = 3) -> pd.DataFrame:
    """Feature table with a planted signal: large diffs are usually bug-introducing."""
    from commitrad.features import build_features
    from commitrad.sentiment import LexiconSentiment

    df = build_features(random_history(n=n, seed=seed), LexiconSentiment(), "synthetic")
    rng = np.random.default_rng(0)
    risky = (df["lines_added"] > 70) | (rng.random(len(df)) < 0.03)
    df["is_bug_introducing"] = risky
    df["bug_found_at"] = df["timestamp"].where(risky) + pd.Timedelta(days=3)
    return df
