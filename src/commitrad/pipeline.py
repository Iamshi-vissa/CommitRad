"""Glue: score commits with the saved model, compute weekly strain, persist to DuckDB."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from commitrad.model import RiskModel, load_model, risk_percentile
from commitrad.store import connect, replace_repos
from commitrad.strain import StrainConfig, weekly_strain


def score_commits(features: pd.DataFrame, model: RiskModel, quantiles: list[float]) -> pd.DataFrame:
    """Features plus ``risk`` (model score) and ``risk_pct`` (reference percentile)."""
    risk = model.predict(features)
    return features.assign(risk=risk, risk_pct=risk_percentile(risk, quantiles))


@dataclass(frozen=True)
class StrainRun:
    n_commits: int
    n_weeks: int
    scored: pd.DataFrame
    strain: pd.DataFrame


def run_strain(
    features: pd.DataFrame, model_dir: Path, config: StrainConfig, db_path: Path
) -> StrainRun:
    model, meta = load_model(model_dir)
    scored = score_commits(features, model, meta["risk_quantiles"])
    strain = weekly_strain(scored, config)
    with connect(db_path) as con:
        replace_repos(con, "commit_risk", scored)
        replace_repos(con, "strain", strain)
    return StrainRun(len(scored), len(strain), scored, strain)
