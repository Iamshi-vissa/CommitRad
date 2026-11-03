from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from frames import random_history

from commitrad.features import FEATURE_COLUMNS, build_features
from commitrad.model import (
    RiskModel,
    class_weight,
    labels_as_of,
    load_model,
    run_training,
    save_model,
    split_labels,
    time_split,
)
from commitrad.report import render_report
from commitrad.sentiment import LexiconSentiment


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    """Feature table with a planted signal: large diffs are usually bug-introducing."""
    df = build_features(random_history(n=900, seed=3), LexiconSentiment(), "synthetic")
    rng = np.random.default_rng(0)
    risky = (df["lines_added"] > 70) | (rng.random(len(df)) < 0.03)
    df["is_bug_introducing"] = risky
    df["bug_found_at"] = df["timestamp"].where(risky) + pd.Timedelta(days=3)
    return df


def test_time_split_is_ordered_and_keeps_ties_together(features: pd.DataFrame) -> None:
    s = time_split(features)
    assert s.train["timestamp"].max() < s.val["timestamp"].min() == s.val_start
    assert s.val["timestamp"].max() < s.test["timestamp"].min() == s.test_start
    assert len(s.train) + len(s.val) + len(s.test) == len(features)
    assert len(s.train) / len(features) == pytest.approx(0.70, abs=0.03)
    assert len(s.val) / len(features) == pytest.approx(0.15, abs=0.03)


def test_time_split_ignores_input_order(features: pd.DataFrame) -> None:
    reversed_input = features.iloc[::-1]
    a, b = time_split(features), time_split(reversed_input)
    assert set(a.test["hash"]) == set(b.test["hash"])


def test_time_split_rejects_tiny_data(features: pd.DataFrame) -> None:
    same_time = features.head(5).assign(timestamp=features["timestamp"].iloc[0])
    with pytest.raises(ValueError):
        time_split(same_time)


def test_strict_labels_ignore_fixes_from_later_splits() -> None:
    ts = pd.Timestamp("2024-01-10", tz="UTC")
    df = pd.DataFrame(
        {
            "is_bug_introducing": [True, True, False],
            "bug_found_at": [ts - pd.Timedelta(days=1), ts + pd.Timedelta(days=1), pd.NaT],
        }
    )
    assert labels_as_of(df, ts).tolist() == [True, False, False]
    assert labels_as_of(df, None).tolist() == [True, True, False]


def test_split_labels_modes(features: pd.DataFrame) -> None:
    s = time_split(features)
    strict, final = split_labels(s, "strict"), split_labels(s, "final")
    assert (strict["train"] <= final["train"]).all()  # strict only removes positives
    assert strict["test"].equals(final["test"])
    late = s.train["bug_found_at"] >= s.val_start
    assert not strict["train"][late].any()


def test_class_weight() -> None:
    assert class_weight(pd.Series([True, False, False, False])) == 3.0
    assert class_weight(pd.Series([False, False])) == 1.0


def test_model_learns_signal_and_shap_is_additive(features: pd.DataFrame, tmp_path: Path) -> None:
    result = run_training(features, "strict", n_seeds=2)
    test = result.metrics["test"]["lightgbm"]
    assert test["pr_auc"] > 2 * test["positive_rate"] and test["roc_auc"] > 0.8
    assert test["pr_auc_lo"] <= test["pr_auc"] <= test["pr_auc_hi"]
    assert result.importance.loc[0, "feature"] in {"lines_added", "file_churn_max"}
    assert max(result.model.best_iterations) > 1  # early stopping must not collapse to 1 tree

    sample = result.split.test.head(20)
    shap = result.model.shap_values(sample)
    np.testing.assert_allclose(shap.sum(axis=1), result.model.raw_score(sample), rtol=1e-6)

    save_model(result, tmp_path)
    loaded, meta = load_model(tmp_path)
    np.testing.assert_allclose(loaded.predict(sample), result.model.predict(sample), rtol=1e-9)
    assert meta["features"] == FEATURE_COLUMNS and meta["label_mode"] == "strict"
    assert len(list(tmp_path.glob("commitrad_seed*.lgb"))) == 2

    report = render_report(result, ["synthetic"])
    for heading in ("## Headline", "## Time-based split", "## Metrics", "SHAP", "## Caveats"):
        assert heading in report


def test_load_rejects_missing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        RiskModel.load(tmp_path)
