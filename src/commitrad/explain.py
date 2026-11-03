"""Turn per-commit SHAP values into short, plain-English reasons."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from commitrad.features import FEATURE_COLUMNS

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _plural(n: float, word: str) -> str:
    count = round(n)
    return f"{count:,} {word}" if count == 1 else f"{count:,} {word}s"


def _hours(h: float) -> str:
    if h < 1:
        return f"{round(h * 60)} minutes"
    if h < 48:
        return f"{h:.0f} hours"
    return f"{h / 24:.0f} days"


def describe_value(feature: str, value: float | None) -> str:
    """What the feature value *is*, in words (no judgment about risk)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        if feature == "hours_since_prev_commit":
            return "the author's first commit in this repo"
        return f"{feature} unknown"
    v = float(value)
    phrases = {
        "hour": lambda: f"committed at {int(v):02d}:00 local time",
        "weekday": lambda: f"committed on a {WEEKDAYS[int(v) % 7]}",
        "is_night": lambda: "committed late at night" if v else "committed during daytime",
        "is_weekend": lambda: "committed on a weekend" if v else "committed on a weekday",
        "hours_since_prev_commit": lambda: f"{_hours(v)} after the author's previous commit",
        "lines_added": lambda: f"adds {_plural(v, 'line')}",
        "lines_deleted": lambda: f"deletes {_plural(v, 'line')}",
        "files_touched": lambda: f"touches {_plural(v, 'file')}",
        "file_churn_max": lambda: f"edits a file with {_plural(v, 'line')} of prior churn",
        "file_churn_mean": lambda: f"touched files average {_plural(v, 'line')} of prior churn",
        "author_prior_commits": lambda: (
            "author's first commit in this repo"
            if v == 0
            else f"author had {_plural(v, 'earlier commit')} here"
        ),
        "author_commit_rate_30d": lambda: f"author made {v * 30:.0f} commits in the prior 30 days",
        "author_past_bug_rate": lambda: f"{v:.0%} of the author's past commits were later fixed",
        "msg_length": lambda: f"commit message is {_plural(v, 'character')} long",
        "msg_word_count": lambda: f"commit message has {_plural(v, 'word')}",
        "msg_vagueness": lambda: (
            "commit message is vague"
            if v >= 0.6
            else "commit message is fairly specific"
            if v <= 0.3
            else "commit message is somewhat generic"
        ),
        "msg_sentiment": lambda: (
            "commit message sounds frustrated"
            if v < 0
            else "commit message sounds positive"
            if v > 0
            else "commit message is neutral in tone"
        ),
    }
    return phrases.get(feature, lambda: f"{feature} = {v:g}")()


def reason(feature: str, value: float | None, shap: float) -> dict[str, Any]:
    effect = "raises" if shap > 0 else "lowers"
    return {
        "feature": feature,
        "value": None if pd.isna(value) else float(value),
        "shap": float(shap),
        "effect": effect,
        "text": f"{describe_value(feature, value).capitalize()} ({effect} risk)",
    }


def top_reasons(shap_row: pd.Series, values: pd.Series, k: int = 3) -> list[dict[str, Any]]:
    """The ``k`` features with the largest absolute SHAP contribution for one commit."""
    contrib = shap_row[FEATURE_COLUMNS].astype(float)
    order = contrib.abs().sort_values(ascending=False, kind="stable").index[:k]
    return [reason(f, values[f], contrib[f]) for f in order]


def explain_frame(shap: pd.DataFrame, x: pd.DataFrame, k: int = 3) -> list[list[dict[str, Any]]]:
    """Top-``k`` reasons for every row (aligned with ``x``)."""
    return [top_reasons(shap.loc[i], x.loc[i], k) for i in x.index]
