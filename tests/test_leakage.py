"""Leakage tests: a commit's features may only depend on strictly earlier information."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from frames import commit, frame, random_history

from commitrad.features import FEATURE_COLUMNS, build_features
from commitrad.sentiment import LexiconSentiment

SCORER = LexiconSentiment()


def features(df: pd.DataFrame) -> pd.DataFrame:
    return build_features(df, SCORER, "r").set_index("hash")[FEATURE_COLUMNS]


def assert_rows_equal(a: pd.Series, b: pd.Series) -> None:
    pd.testing.assert_series_equal(a, b, check_names=False, check_dtype=False)


@pytest.fixture(scope="module")
def history() -> pd.DataFrame:
    return random_history(n=300, seed=7)


@pytest.fixture(scope="module")
def full(history: pd.DataFrame) -> pd.DataFrame:
    return features(history)


def as_known_at(history: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """The data a perfect observer would have at time ``t``: no later commits, no later fixes."""
    past = history[history["timestamp"] <= t].copy()
    unknown = past["bug_found_at"] >= t
    past.loc[unknown, "bug_found_at"] = pd.NaT
    past["is_bug_introducing"] = past["bug_found_at"].notna()
    return past


def test_features_match_point_in_time_recomputation(
    history: pd.DataFrame, full: pd.DataFrame
) -> None:
    for i in range(0, len(history), 5):
        row = history.iloc[i]
        truncated = features(as_known_at(history, row["timestamp"]))
        assert_rows_equal(truncated.loc[row["hash"]], full.loc[row["hash"]])


def test_scrambling_the_future_changes_nothing(history: pd.DataFrame, full: pd.DataFrame) -> None:
    """Rewrite everything about other commits at or after t (incl. same-time ties)."""
    rng = np.random.default_rng(1)
    for i in range(0, len(history), 15):
        row = history.iloc[i]
        t = row["timestamp"]
        noisy = history.copy()
        future = (noisy["timestamp"] >= t) & (noisy["hash"] != row["hash"])
        n = int(future.sum())
        noisy.loc[future, "author_email"] = rng.choice(["alice@x.org", "zed@x.org"], size=n)
        noisy.loc[future, "lines_added"] = rng.integers(0, 10_000, size=n)
        noisy.loc[future, "is_bug_introducing"] = rng.random(n) < 0.9
        noisy.loc[future, "bug_found_at"] = t - pd.Timedelta(hours=1)  # "found" before t!
        noisy.loc[future, "modified_files"] = pd.Series(
            [["pkg/m0.py"]] * n, index=noisy.index[future]
        )
        noisy.loc[future, "file_lines_added"] = pd.Series([[9999]] * n, index=noisy.index[future])
        noisy.loc[future, "file_lines_deleted"] = pd.Series([[0]] * n, index=noisy.index[future])
        assert_rows_equal(features(noisy).loc[row["hash"]], full.loc[row["hash"]])


def test_target_column_is_never_read(history: pd.DataFrame, full: pd.DataFrame) -> None:
    flipped = history.assign(is_bug_introducing=~history["is_bug_introducing"])
    pd.testing.assert_frame_equal(features(flipped), full)


def test_past_bug_counts_only_after_its_fix() -> None:
    df = frame(
        [
            commit("bug", 0, found_after_hours=240),  # fixed 10 days later
            commit("before_fix", 120),  # day 5: bug not known yet
            commit("after_fix", 480),  # day 20: bug known
        ]
    )
    rate = features(df)["author_past_bug_rate"]
    assert rate["bug"] == 0.0
    assert rate["before_fix"] == 0.0
    assert rate["after_fix"] == pytest.approx(1 / 2)


def test_same_timestamp_commits_do_not_see_each_other() -> None:
    df = frame(
        [
            commit("a", 10, files={"x.py": (5, 5)}),
            commit("b", 10, files={"x.py": (7, 0)}),
            commit("c", 11, files={"x.py": (1, 0)}),
        ]
    )
    f = features(df)
    assert f.loc["a", "author_prior_commits"] == f.loc["b", "author_prior_commits"] == 0
    assert f.loc["a", "file_churn_max"] == f.loc["b", "file_churn_max"] == 0
    assert np.isnan(f.loc["b", "hours_since_prev_commit"])
    assert f.loc["c", "file_churn_max"] == 17 and f.loc["c", "author_prior_commits"] == 2
