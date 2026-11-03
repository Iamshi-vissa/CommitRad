from __future__ import annotations

import math

import pandas as pd

from commitrad.explain import describe_value, top_reasons
from commitrad.features import FEATURE_COLUMNS


def test_describe_values_in_plain_english() -> None:
    assert describe_value("lines_added", 1) == "adds 1 line"
    assert describe_value("lines_added", 231) == "adds 231 lines"
    assert describe_value("hour", 3) == "committed at 03:00 local time"
    assert describe_value("weekday", 5) == "committed on a Saturday"
    assert describe_value("is_night", 1) == "committed late at night"
    assert (
        describe_value("hours_since_prev_commit", 0.25)
        == "15 minutes after the author's previous commit"
    )
    assert (
        describe_value("hours_since_prev_commit", 72) == "3 days after the author's previous commit"
    )
    assert (
        describe_value("hours_since_prev_commit", math.nan)
        == "the author's first commit in this repo"
    )
    assert (
        describe_value("author_past_bug_rate", 0.25)
        == "25% of the author's past commits were later fixed"
    )
    assert describe_value("msg_vagueness", 0.9) == "commit message is vague"


def test_every_feature_has_a_phrase() -> None:
    for f in FEATURE_COLUMNS:
        assert "=" not in describe_value(f, 1.0), f


def test_top_reasons_rank_by_absolute_shap() -> None:
    shap = pd.Series(0.0, index=[*FEATURE_COLUMNS, "base_value"])
    shap[["lines_added", "author_prior_commits", "hour", "is_night"]] = [0.8, -0.9, 0.1, 0.3]
    values = pd.Series(1.0, index=FEATURE_COLUMNS)
    reasons = top_reasons(shap, values, k=3)
    assert [r["feature"] for r in reasons] == ["author_prior_commits", "lines_added", "is_night"]
    assert reasons[0]["effect"] == "lowers" and reasons[0]["text"].endswith("(lowers risk)")
    assert reasons[1]["text"] == "Adds 1 line (raises risk)"
