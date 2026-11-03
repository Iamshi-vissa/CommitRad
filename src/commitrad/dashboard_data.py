"""Pure data shaping for the dashboard (kept out of the Streamlit script so it is testable)."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

COMPONENTS = ["off_hours", "risk", "burstiness", "negativity"]
COMPONENT_LABELS = {
    "off_hours": "Off-hours",
    "risk": "Commit risk",
    "burstiness": "Burstiness",
    "negativity": "Negativity",
}
REPO_AVERAGE = "Repo average"


def strain_frame(result: dict[str, Any]) -> pd.DataFrame:
    """Flatten the API's strain list into one row per author-week."""
    rows = [
        {k: v for k, v in s.items() if k != "components"} | s["components"]
        for s in result["strain"]
    ]
    df = pd.DataFrame(
        rows, columns=["author", "author_email", "week", "n_commits", "strain", *COMPONENTS]
    )
    df["week"] = pd.to_datetime(df["week"])
    return df


def heatmap_frame(strain: pd.DataFrame, weeks: int = 52, top_authors: int = 12) -> pd.DataFrame:
    """Last ``weeks`` weeks of activity for the ``top_authors`` most active authors.

    Weeks without commits are absent (no data), not zero strain.
    """
    if strain.empty:
        return strain.assign(week_label=pd.Series(dtype=str))
    end = strain["week"].max()
    window = strain[strain["week"] > end - pd.Timedelta(weeks=weeks)]
    active = window.groupby("author_email")["n_commits"].sum().nlargest(top_authors).index
    out = window[window["author_email"].isin(active)].copy()
    out["week_label"] = out["week"].dt.strftime("%Y-%m-%d")
    return out.sort_values(["week", "author"]).reset_index(drop=True)


def week_axis(heat: pd.DataFrame, weeks: int = 52) -> tuple[list[str], list[str]]:
    """All week labels in the window (so gaps show) and a sparse subset to print."""
    if heat.empty:
        return [], []
    end = heat["week"].max()
    end -= pd.Timedelta(days=end.day_of_week)  # snap to Monday
    all_weeks = pd.date_range(end=end, periods=weeks, freq="W-MON")
    labels = [w.strftime("%Y-%m-%d") for w in all_weeks]
    step = max(1, len(labels) // 8)
    return labels, labels[::-step][::-1]


def author_profile(strain: pd.DataFrame, author_email: str | None = None) -> dict[str, float]:
    """Commit-weighted mean of each component, for one author or the whole repo."""
    df = strain if author_email is None else strain[strain["author_email"] == author_email]
    if df.empty:
        return dict.fromkeys(COMPONENTS, 0.0)
    w = df["n_commits"].astype(float)
    return {c: float((df[c] * w).sum() / w.sum()) for c in COMPONENTS}


def _point(i: int, n: int, r: float) -> tuple[float, float]:
    angle = math.pi / 2 - 2 * math.pi * i / n  # first axis at 12 o'clock, clockwise
    return r * math.cos(angle), r * math.sin(angle)


def radar_frame(profiles: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Closed polygon coordinates per series (first vertex repeated at the end)."""
    rows = []
    for series, profile in profiles.items():
        for order, i in enumerate([*range(len(COMPONENTS)), 0]):
            comp = COMPONENTS[i]
            x, y = _point(i, len(COMPONENTS), profile[comp])
            rows.append(
                {
                    "series": series,
                    "order": order,
                    "component": COMPONENT_LABELS[comp],
                    "value": round(profile[comp], 3),
                    "x": x,
                    "y": y,
                }
            )
    return pd.DataFrame(rows)


def radar_grid(rings: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)) -> pd.DataFrame:
    rows = []
    for ring in rings:
        for order, i in enumerate([*range(len(COMPONENTS)), 0]):
            x, y = _point(i, len(COMPONENTS), ring)
            rows.append({"ring": ring, "order": order, "x": x, "y": y})
    return pd.DataFrame(rows)


def radar_labels(radius: float = 1.18) -> pd.DataFrame:
    rows = []
    for i, comp in enumerate(COMPONENTS):
        x, y = _point(i, len(COMPONENTS), radius)
        rows.append({"label": COMPONENT_LABELS[comp], "x": x, "y": y})
    return pd.DataFrame(rows)


def commit_url(repo_url: str, sha: str) -> str | None:
    if repo_url.startswith("https://gitlab.com/"):
        return f"{repo_url}/-/commit/{sha}"
    if repo_url.startswith(("https://github.com/", "https://codeberg.org/")):
        return f"{repo_url}/commit/{sha}"
    return None


def top_risky(result: dict[str, Any], n: int = 5) -> list[dict[str, Any]]:
    return sorted(result["commits"], key=lambda c: c["risk"], reverse=True)[:n]


def risk_band(percentile: float) -> str:
    """Plain-English position against the model's reference distribution."""
    top = max(1, math.ceil((1 - percentile) * 100))
    return f"in the top {top}% of reference commits by risk"
