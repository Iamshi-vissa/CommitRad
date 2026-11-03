"""Pure data shaping for the dashboard (kept out of the Streamlit script so it is testable)."""

from __future__ import annotations

import html
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
    """Unit-circle position of axis ``i`` at radius ``r`` (first axis at 12 o'clock, clockwise)."""
    angle = math.pi / 2 - 2 * math.pi * i / n
    return r * math.cos(angle), r * math.sin(angle)


def radar_extent(profiles: dict[str, dict[str, float]]) -> float:
    """Outer ring value: the largest component rounded up to a quarter, so shapes fill the chart."""
    peak = max((v for p in profiles.values() for v in p.values()), default=0.0)
    return min(1.0, max(0.25, math.ceil(peak * 4 - 1e-9) / 4))


def radar_frame(profiles: dict[str, dict[str, float]], extent: float = 1.0) -> pd.DataFrame:
    """Closed polygon coordinates per series, scaled so ``extent`` sits on the outer ring."""
    rows = []
    for series, profile in profiles.items():
        for order, i in enumerate([*range(len(COMPONENTS)), 0]):
            comp = COMPONENTS[i]
            x, y = _point(i, len(COMPONENTS), min(profile[comp], extent) / extent)
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


def radar_svg(
    profiles: dict[str, dict[str, float]],
    colors: list[str],
    ink: str,
    muted: str,
    grid: str,
) -> str:
    """A self-contained SVG radar: zoomed rings with values, filled shapes, labelled points.

    The first profile gets its value printed at each point; every point has a hover tooltip.
    """
    width, height, cx, cy, radius = 600, 520, 300, 272, 165
    extent = radar_extent(profiles)
    n = len(COMPONENTS)

    def xy(i: int, r: float) -> tuple[float, float]:
        x, y = _point(i, n, r)
        return cx + x * radius, cy - y * radius

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="Strain profile radar" style="max-width:{width}px;font-family:inherit">'
    ]
    for k in range(1, 5):  # rings + their values on the top spoke
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(i, k / 4) for i in range(n)))
        parts.append(f'<polygon points="{pts}" fill="none" stroke="{grid}" stroke-width="1"/>')
        _, ry = xy(0, k / 4)
        parts.append(  # left of the top spoke; the author's values sit to its right
            f'<text x="{cx - 6}" y="{ry + 13}" text-anchor="end" font-size="11" '
            f'fill="{muted}">{extent * k / 4:.2f}</text>'
        )
    for i, comp in enumerate(COMPONENTS):  # spokes + axis names
        x, y = xy(i, 1)
        parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="{grid}"/>')
        lx, ly = xy(i, 1.13)
        anchor = "middle" if abs(lx - cx) < 1 else ("start" if lx > cx else "end")
        parts.append(
            f'<text x="{lx:.1f}" y="{ly + 5:.1f}" text-anchor="{anchor}" font-size="14" '
            f'font-weight="600" fill="{ink}">{COMPONENT_LABELS[comp]}</text>'
        )
    for (series, profile), color in zip(profiles.items(), colors, strict=False):
        pts = [xy(i, min(profile[c], extent) / extent) for i, c in enumerate(COMPONENTS)]
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        parts.append(
            f'<polygon points="{poly}" fill="{color}" fill-opacity="0.16" stroke="{color}" '
            f'stroke-width="2.5" stroke-linejoin="round"/>'
        )
        name = html.escape(series)
        for (x, y), comp in zip(pts, COMPONENTS, strict=True):
            tip = f"{name} · {COMPONENT_LABELS[comp]}: {profile[comp]:.2f}"
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{color}" stroke="white" '
                f'stroke-width="1.5"><title>{tip}</title></circle>'
            )
    lx = 12
    for (series, _), color in zip(profiles.items(), colors, strict=False):  # legend
        label = html.escape(series)
        parts.append(f'<circle cx="{lx + 6}" cy="22" r="6" fill="{color}"/>')
        parts.append(
            f'<text x="{lx + 18}" y="27" font-size="14" font-weight="600" fill="{ink}">'
            f"{label}</text>"
        )
        lx += 30 + 8 * len(series)
    first = next(iter(profiles.values()))
    for i, comp in enumerate(COMPONENTS):  # value labels for the selected author
        r = min(first[comp], extent) / extent
        x, y = xy(i, r)
        dx, _ = _point(i, n, 1)
        if abs(dx) > 0.5 and r < 0.3:  # near the centre: outward, clear of the ring values
            tx, ty, anchor = x + 12 * dx, y + 18, "end" if dx < 0 else "start"
        elif abs(dx) > 0.5:  # side spokes: above the point, clear of the other shape's dot
            tx, ty, anchor = x, y - 13, "middle"
        else:  # top / bottom spokes: to the right of the point
            tx, ty, anchor = x + 10, y + (4 if i == 0 else 16), "start"
        parts.append(
            f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="{anchor}" font-size="13" '
            f'font-weight="700" fill="{colors[0]}">{first[comp]:.2f}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def profile_deltas(author: dict[str, float], repo: dict[str, float]) -> list[dict[str, Any]]:
    """Each component's value for the author, the repo average and the difference."""
    return [
        {
            "component": c,
            "label": COMPONENT_LABELS[c],
            "value": author[c],
            "repo": repo[c],
            "delta": author[c] - repo[c],
        }
        for c in COMPONENTS
    ]


GAP_WEEKS = 4  # a longer pause breaks the author's strain line


def author_weeks(strain: pd.DataFrame, author_email: str) -> pd.DataFrame:
    """The author's weekly rows, oldest first, with a week label and a run id.

    ``run`` increases after any pause longer than ``GAP_WEEKS``, so a line drawn per run
    never bridges months without commits.
    """
    df = strain[strain["author_email"] == author_email].sort_values("week").copy()
    df["week_label"] = df["week"].dt.strftime("%Y-%m-%d")
    df["run"] = (df["week"].diff() > pd.Timedelta(weeks=GAP_WEEKS)).cumsum()
    return df.reset_index(drop=True)


def step_carousel(index: int, n: int, step: int) -> int:
    """Next 1-based carousel position, wrapping around at both ends."""
    return (index - 1 + step) % n + 1 if n else 1


def repo_parts(repo_url: str) -> tuple[str, str]:
    """``https://github.com/owner/repo`` -> ``("owner", "repo")``; local paths -> ``("", name)``."""
    parts = [p for p in repo_url.rstrip("/").split("/") if p]
    if repo_url.startswith("https://") and len(parts) >= 4:
        return parts[-2], parts[-1]
    return "", parts[-1] if parts else repo_url


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
