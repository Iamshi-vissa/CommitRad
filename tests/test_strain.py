from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from commitrad.model import reference_quantiles, risk_percentile
from commitrad.store import connect, read_table, replace_repos
from commitrad.strain import STRAIN_COLUMNS, StrainConfig, burstiness, load_config, weekly_strain

HOUR = 3600.0


def commits(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Minimal scored-commit frame; ``t`` is an ISO UTC timestamp string."""
    base = {
        "repo": "r",
        "author": "Alice",
        "author_email": "alice@x.org",
        "tz_offset_minutes": 0,
        "is_night": 0,
        "is_weekend": 0,
        "msg_sentiment": 0.0,
        "risk_pct": 0.5,
    }
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["timestamp"] = pd.to_datetime(df.pop("t"), utc=True)
    return df


def test_burstiness_regular_vs_bursty_vs_few() -> None:
    regular = np.arange(10) * 24 * HOUR
    bursty = np.array([0, 60, 120, 180, 5 * 24 * HOUR, 5 * 24 * HOUR + 60, 5 * 24 * HOUR + 120])
    assert burstiness(regular) == 0.0
    assert burstiness(bursty) > 0.9
    assert burstiness(np.array([0, 1, 2, 3, 1000.0])) == pytest.approx(1.0, abs=0.01)  # max burst
    rng = np.random.default_rng(0)
    poisson = np.cumsum(rng.exponential(HOUR, size=(400, 6)), axis=1)
    assert np.mean([burstiness(p) for p in poisson]) < 0.25  # random timing is not bursty
    assert burstiness(np.array([0.0, 10.0])) == 0.0  # too few commits
    assert burstiness(np.array([5.0, 5.0, 5.0])) == 1.0  # all at once
    assert burstiness(bursty[::-1]) == burstiness(bursty)  # order-independent


def test_weekly_grouping_uses_author_local_monday() -> None:
    # Sunday 23:30 local at UTC-5 is Monday 04:30 UTC: belongs to the *previous* local week.
    df = commits(
        [
            {"t": "2024-01-08T04:30:00Z", "tz_offset_minutes": -300},
            {"t": "2024-01-08T12:00:00Z", "tz_offset_minutes": -300},
        ]
    )
    out = weekly_strain(df)
    assert out["week"].tolist() == [dt.date(2024, 1, 1), dt.date(2024, 1, 8)]


def test_components_and_weighted_blend() -> None:
    df = commits(
        [
            {"t": "2024-01-01T10:00:00Z", "is_night": 1, "risk_pct": 0.9, "msg_sentiment": -0.5},
            {"t": "2024-01-02T10:00:00Z", "is_weekend": 1, "risk_pct": 0.7, "msg_sentiment": 0.4},
            {"t": "2024-01-03T10:00:00Z", "risk_pct": 0.2},
            {"t": "2024-01-03T10:00:00Z", "author_email": "bob@x.org", "author": "Bob"},
        ]
    )
    cfg = StrainConfig(weights={"off_hours": 2, "risk": 1, "burstiness": 0, "negativity": 1})
    out = weekly_strain(df, cfg).set_index("author")
    alice = out.loc["Alice"]
    assert alice["n_commits"] == 3
    assert alice["off_hours"] == pytest.approx(2 / 3)
    assert alice["risk"] == pytest.approx(0.6)
    assert alice["negativity"] == pytest.approx(0.5 / 3)  # positive sentiment counts as 0
    assert alice["strain"] == pytest.approx((2 * 2 / 3 + 0.6 + 0.5 / 3) / 4)
    assert out.loc["Bob", "n_commits"] == 1
    assert list(weekly_strain(df).columns) == STRAIN_COLUMNS


def test_strain_is_bounded() -> None:
    worst = commits([{"t": f"2024-01-06T0{i}:00:00Z", "is_night": 1, "risk_pct": 1.0,
                      "msg_sentiment": -1.0} for i in range(4)])  # fmt: skip
    worst.loc[3, "timestamp"] += pd.Timedelta(days=1)  # makes the gaps bursty
    s = weekly_strain(worst)["strain"]
    assert ((s >= 0) & (s <= 1)).all() and s.iloc[0] > 0.8


@pytest.mark.parametrize(
    "weights",
    [{"off_hours": -1.0}, {"off_hours": 0.0, "risk": 0.0}, {"mood": 1.0}],
)
def test_config_validation(weights: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        StrainConfig(weights=weights)


def test_load_config_from_toml(tmp_path: Path) -> None:
    p = tmp_path / "s.toml"
    p.write_text("[weights]\noff_hours = 1\nrisk = 3\n[options]\nmin_commits_for_burstiness = 4\n")
    cfg = load_config(p)
    assert cfg.weights == {"off_hours": 1.0, "risk": 3.0}
    assert cfg.min_commits_for_burstiness == 4


def test_repo_config_file_is_valid() -> None:
    cfg = load_config(Path(__file__).parents[1] / "config" / "strain.toml")
    assert set(cfg.weights) == {"off_hours", "risk", "burstiness", "negativity"}


def test_risk_percentile_scale() -> None:
    q = reference_quantiles(np.linspace(0.1, 0.9, 1000))
    pct = risk_percentile(np.array([0.0, 0.1, 0.5, 0.9, 1.0]), q)
    np.testing.assert_allclose(pct, [0.0, 0.0, 0.5, 1.0, 1.0], atol=0.01)


def test_duckdb_replace_is_per_repo(tmp_path: Path) -> None:
    db = tmp_path / "t.duckdb"
    week = commits([{"t": "2024-01-01T10:00:00Z"}, {"t": "2024-01-01T11:00:00Z", "repo": "s"}])
    with connect(db) as con:
        replace_repos(con, "strain", weekly_strain(week))
        replace_repos(con, "strain", weekly_strain(week[week["repo"] == "r"]))  # rerun r only
        stored = read_table(con, "strain")
        assert sorted(stored["repo"]) == ["r", "s"]  # no duplicates, s untouched
        assert len(read_table(con, "strain", repo="s")) == 1
    with connect(db, read_only=True) as con:
        cols = con.execute("DESCRIBE strain").df()["column_name"].tolist()
        assert cols[:4] == ["repo", "author", "author_email", "week"] and "strain" in cols


def test_read_table_rejects_unknown_table(tmp_path: Path) -> None:
    with connect(tmp_path / "t.duckdb") as con, pytest.raises(ValueError):
        read_table(con, "users; DROP TABLE strain")


def test_bots_are_excluded_from_strain() -> None:
    df = commits(
        [
            {"t": "2024-01-01T10:00:00Z"},
            {"t": "2024-01-01T11:00:00Z", "author": "dependabot[bot]",
             "author_email": "49699333+dependabot[bot]@users.noreply.github.com"},
        ]
    )  # fmt: skip
    assert weekly_strain(df)["author"].tolist() == ["Alice"]
