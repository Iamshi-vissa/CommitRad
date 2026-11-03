"""Weekly per-author strain: a weighted blend of off-hours work, risk, burstiness, negativity.

Strain is a *signal* that a week looked unusually pressured. It is not a measure of a
person's performance, health or worth, and should never be used to rank people.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from commitrad.features import author_key, local_time

COMPONENTS = ("off_hours", "risk", "burstiness", "negativity")
DEFAULT_CONFIG = Path("config/strain.toml")
STRAIN_COLUMNS = [
    "repo",
    "author",
    "author_email",
    "week",
    "n_commits",
    "off_hours",
    "risk",
    "burstiness",
    "negativity",
    "strain",
]


@dataclass(frozen=True)
class StrainConfig:
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "off_hours": 0.3,
            "risk": 0.3,
            "burstiness": 0.2,
            "negativity": 0.2,
        }
    )
    min_commits_for_burstiness: int = 3

    def __post_init__(self) -> None:
        unknown = set(self.weights) - set(COMPONENTS)
        if unknown:
            raise ValueError(f"Unknown strain components: {sorted(unknown)}")
        if any(w < 0 for w in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("Strain weights must be non-negative and not all zero")
        if self.min_commits_for_burstiness < 3:
            raise ValueError("Burstiness needs at least 3 commits (2 gaps)")


def load_config(path: Path = DEFAULT_CONFIG) -> StrainConfig:
    data = tomllib.loads(path.read_text())
    options = data.get("options", {})
    return StrainConfig(
        weights={k: float(v) for k, v in data["weights"].items()},
        min_commits_for_burstiness=int(options.get("min_commits_for_burstiness", 3)),
    )


def burstiness(timestamps: np.ndarray, min_commits: int = 3) -> float:
    """Finite-size-corrected burstiness of inter-commit gaps, clipped to [0, 1].

    Uses Kim & Jo (2016), "Measuring burstiness for finite event sequences", because the
    classic Goh-Barabasi B = (r - 1) / (r + 1), with r = sigma / mu, cannot exceed ~0.4 for
    the handful of commits typical in one week. With n gaps:
    A_n = (sqrt(n+1) r - sqrt(n-1)) / ((sqrt(n+1) - 2) r + sqrt(n-1)).
    Perfectly regular timing gives -1 (clipped to 0), Poisson-like timing ~0, and all
    commits in one burst plus one long pause gives 1. Fewer than ``min_commits`` gives 0.
    """
    t = np.sort(np.asarray(timestamps, dtype=float))
    if len(t) < min_commits:
        return 0.0
    gaps = np.diff(t)
    mu = gaps.mean()
    if mu == 0:  # all commits at the same instant: one burst
        return 1.0
    r, n = gaps.std() / mu, len(gaps)
    a, b = np.sqrt(n + 1), np.sqrt(n - 1)
    return float(np.clip((a * r - b) / ((a - 2) * r + b), 0.0, 1.0))


def week_start(df: pd.DataFrame) -> pd.Series:
    """Monday (author local time) of each commit's week, as a date."""
    local = local_time(df)
    return (local.dt.normalize() - pd.to_timedelta(local.dt.day_of_week, unit="D")).dt.date


def commit_components(commits: pd.DataFrame) -> pd.DataFrame:
    """Per-commit inputs. Needs feature columns plus ``risk_pct`` in [0, 1]."""
    return pd.DataFrame(
        {
            "repo": commits["repo"],
            "author_id": author_key(commits),
            "author": commits["author"],
            "week": week_start(commits),
            "t": (commits["timestamp"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds(),
            "off_hours": ((commits["is_night"] == 1) | (commits["is_weekend"] == 1)).astype(float),
            "risk": commits["risk_pct"].astype(float),
            "negativity": (-commits["msg_sentiment"]).clip(lower=0).astype(float),
        },
        index=commits.index,
    )


def blend(components: pd.DataFrame, config: StrainConfig) -> pd.Series:
    total = sum(config.weights.values())
    return sum(components[c] * w for c, w in config.weights.items()) / total


def weekly_strain(commits: pd.DataFrame, config: StrainConfig | None = None) -> pd.DataFrame:
    """One row per (repo, author, week) with components and the blended ``strain``."""
    config = config or StrainConfig()
    per = commit_components(commits)
    rows = []
    for (repo, author_id, week), g in per.groupby(["repo", "author_id", "week"], sort=True):
        rows.append(
            {
                "repo": repo,
                "author": g["author"].mode().iloc[0],
                "author_email": author_id,
                "week": week,
                "n_commits": len(g),
                "off_hours": g["off_hours"].mean(),
                "risk": g["risk"].mean(),
                "burstiness": burstiness(g["t"].to_numpy(), config.min_commits_for_burstiness),
                "negativity": g["negativity"].mean(),
            }
        )
    out = pd.DataFrame(rows, columns=STRAIN_COLUMNS[:-1])
    out["strain"] = blend(out, config) if len(out) else pd.Series(dtype=float)
    out["week"] = pd.to_datetime(out["week"]).dt.date
    return out[STRAIN_COLUMNS]
