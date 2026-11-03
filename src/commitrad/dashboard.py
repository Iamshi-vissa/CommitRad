"""CommitRad dashboard. Run with ``commitrad dashboard`` or ``streamlit run`` on this file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from commitrad.analyze import (
    DISCLAIMER,
    InvalidRepoURL,
    RepoTooLarge,
    Settings,
    analyze_repo,
    list_snapshots,
    load_snapshot,
)
from commitrad.dashboard_data import (
    COMPONENT_LABELS,
    COMPONENTS,
    REPO_AVERAGE,
    author_profile,
    commit_url,
    heatmap_frame,
    radar_frame,
    radar_grid,
    radar_labels,
    risk_band,
    strain_frame,
    top_risky,
    week_axis,
)
from commitrad.explain import reason

DEMO_DIR = Path(os.environ.get("COMMITRAD_DEMO_DIR", "demo_cache"))
# Reference palette: sequential blue ramp (steps 100..700); categorical slots 1-2.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SERIES = ["#2a78d6", "#eb6834"]
MUTED = "#8a8984"

st.set_page_config(page_title="CommitRad", page_icon="📡", layout="wide")


@st.cache_data(show_spinner=False)
def _snapshot(path: str) -> dict[str, Any]:
    return load_snapshot(Path(path))


@st.cache_data(show_spinner=False, ttl=3600)
def _live(url: str) -> dict[str, Any]:
    return analyze_repo(url, Settings())


def load(url: str, snapshots: dict[str, Path]) -> dict[str, Any] | None:
    if url in snapshots:
        return _snapshot(str(snapshots[url]))
    try:
        with st.spinner("Cloning and analyzing… large repos can take a minute."):
            return _live(url)
    except InvalidRepoURL as exc:
        st.error(f"Invalid repository URL: {exc}")
    except RepoTooLarge as exc:
        st.error(f"Repository too large: {exc}")
    except Exception as exc:  # surface git/network problems without a traceback
        st.error(f"Could not analyze this repository ({type(exc).__name__}). Is it public?")
    return None


# --- charts --------------------------------------------------------------------------------


def heatmap(heat: pd.DataFrame, weeks: int) -> alt.Chart:
    all_weeks, shown = week_axis(heat, weeks)
    authors = heat.groupby("author")["n_commits"].sum().sort_values(ascending=False).index.tolist()
    return (
        alt.Chart(heat)
        .mark_rect(cornerRadius=2)
        .encode(
            x=alt.X(
                "week_label:O",
                title=None,
                scale=alt.Scale(domain=all_weeks, paddingInner=0.15),
                axis=alt.Axis(values=shown, labelAngle=-40, ticks=False, domain=False),
            ),
            y=alt.Y(
                "author:N",
                title=None,
                sort=authors,
                scale=alt.Scale(paddingInner=0.15),
                axis=alt.Axis(ticks=False, domain=False),
            ),
            color=alt.Color(
                "strain:Q",
                title="Strain",
                scale=alt.Scale(domain=[0, 1], range=SEQUENTIAL),
                legend=alt.Legend(orient="top", gradientLength=220),
            ),
            tooltip=[
                alt.Tooltip("author:N", title="Author"),
                alt.Tooltip("week_label:N", title="Week of"),
                alt.Tooltip("strain:Q", title="Strain", format=".2f"),
                alt.Tooltip("n_commits:Q", title="Commits"),
                *[
                    alt.Tooltip(f"{c}:Q", title=COMPONENT_LABELS[c], format=".2f")
                    for c in COMPONENTS
                ],
            ],
        )
        .properties(height=alt.Step(22))
    )


def radar(author_name: str, profiles: dict[str, dict[str, float]]) -> alt.LayerChart:
    data = radar_frame(profiles)
    axis_scale = alt.Scale(domain=[-1.35, 1.35])
    hidden = alt.Axis(labels=False, ticks=False, grid=False, domain=False, title=None)
    x = alt.X("x:Q", scale=axis_scale, axis=hidden)
    y = alt.Y("y:Q", scale=axis_scale, axis=hidden)
    grid = (
        alt.Chart(radar_grid())
        .mark_line(color=MUTED, strokeWidth=0.6, opacity=0.5)
        .encode(x=x, y=y, detail="ring:N", order="order:Q")
    )
    labels = (
        alt.Chart(radar_labels())
        .mark_text(fontSize=12, color=MUTED)
        .encode(x=x, y=y, text="label:N")
    )
    color = alt.Color(
        "series:N",
        title=None,
        scale=alt.Scale(domain=[author_name, REPO_AVERAGE], range=SERIES),
        legend=alt.Legend(orient="top"),
    )
    tooltip = [
        alt.Tooltip("series:N", title="Who"),
        alt.Tooltip("component:N", title="Component"),
        alt.Tooltip("value:Q", title="Value", format=".2f"),
    ]
    lines = alt.Chart(data).mark_line(strokeWidth=2).encode(x=x, y=y, color=color, order="order:Q")
    points = (
        alt.Chart(data[data["order"] < len(COMPONENTS)])
        .mark_point(filled=True, size=70, opacity=1)
        .encode(x=x, y=y, color=color, tooltip=tooltip)
    )
    return (grid + labels + lines + points).properties(width=440, height=440)


# --- page ----------------------------------------------------------------------------------


def header(snapshots: dict[str, Path]) -> str:
    st.title("📡 CommitRad")
    st.caption("Risky-commit and developer-strain signals mined from git history.")
    st.warning(f"**Signals, not judgments.** {DISCLAIMER}", icon="⚠️")
    options = [*snapshots, "Another repository…"]
    left, right = st.columns([2, 3])
    choice = left.selectbox("Pre-computed demo repos (instant)", options, index=0)
    default = "" if choice == options[-1] else choice
    url = right.text_input(
        "Repository URL", value=default, placeholder="https://github.com/owner/repo"
    )
    return url.strip()


def summary(result: dict[str, Any], strain: pd.DataFrame) -> None:
    cols = st.columns(4)
    cols[0].metric("Commits analyzed", f"{result['n_commits']:,}")
    cols[1].metric("SZZ bug-introducing", f"{result['n_bug_introducing_szz']:,}")
    cols[2].metric("Authors", f"{strain['author_email'].nunique():,}")
    cols[3].metric("Author-weeks", f"{len(strain):,}")
    model = result["model"]
    notes = [
        f"Model test PR-AUC {model['test_pr_auc']:.2f}, ROC-AUC {model['test_roc_auc']:.2f} "
        "(labels from SZZ, which is noisy)."
    ]
    if model.get("in_training_data"):
        notes.append("This repo was in the training data, so its risk scores are in-sample.")
    if "snapshot_at" in result:
        notes.append(f"Pre-computed snapshot from {result['snapshot_at'][:10]}.")
    st.caption(" ".join(notes))


def strain_section(strain: pd.DataFrame) -> None:
    st.subheader("Weekly strain")
    st.caption(
        "Each cell is one author-week: a weighted blend of off-hours share, commit risk, "
        "burstiness and message negativity (0-1). Blank means no commits that week."
    )
    c1, c2 = st.columns(2)
    weeks = c1.slider("Weeks shown", 8, 156, 52, step=4)
    top = c2.slider("Most active authors", 3, 30, 12)
    heat = heatmap_frame(strain, weeks, top)
    if heat.empty:
        st.info("No human commits in this window.")
        return
    st.altair_chart(heatmap(heat, weeks), use_container_width=True)
    with st.expander("Table view"):
        st.dataframe(
            heat[["author", "week_label", "n_commits", "strain", *COMPONENTS]],
            hide_index=True,
            use_container_width=True,
        )


def radar_section(strain: pd.DataFrame) -> None:
    st.subheader("Strain profile by author")
    totals = (
        strain.groupby(["author_email", "author"])["n_commits"].sum().sort_values(ascending=False)
    )
    labels = {email: f"{name} ({n} commits)" for (email, name), n in totals.items()}
    email = st.selectbox("Author", list(labels), format_func=labels.__getitem__)
    name = strain.loc[strain["author_email"] == email, "author"].mode().iloc[0]
    profiles = {name: author_profile(strain, email), REPO_AVERAGE: author_profile(strain)}
    left, right = st.columns([1, 1])
    left.altair_chart(radar(name, profiles), use_container_width=False)
    table = pd.DataFrame(profiles).rename(index=COMPONENT_LABELS).round(2)
    right.caption("Commit-weighted averages over all of the author's weeks (0-1).")
    right.dataframe(table, use_container_width=True)


def risky_section(result: dict[str, Any]) -> None:
    st.subheader("Top 5 riskiest commits")
    st.caption("Ranked by model risk. Reasons are the three largest SHAP contributions.")
    for c in top_risky(result, 5):
        with st.container(border=True):
            url = commit_url(result["repo"], c["hash"])
            sha = f"[`{c['hash'][:8]}`]({url})" if url else f"`{c['hash'][:8]}`"
            title = c["subject"] or "(no message)"
            st.markdown(f"**{title}**  \n{sha} · {c['author']} · {c['timestamp'][:10]}")
            st.markdown(f"Risk {c['risk']:.2f}: {risk_band(c['risk_percentile'])}.")
            # Re-render from stored values so snapshots pick up wording improvements.
            texts = [reason(r["feature"], r["value"], r["shap"])["text"] for r in c["top_reasons"]]
            st.markdown("\n".join(f"- {t}" for t in texts))


def main() -> None:
    snapshots = list_snapshots(DEMO_DIR)
    url = header(snapshots)
    if not url:
        st.info("Pick a demo repo or paste a public GitHub, GitLab or Codeberg URL.")
        return
    result = load(url, snapshots)
    if result is None:
        return
    strain = strain_frame(result)
    summary(result, strain)
    if strain.empty:
        st.info("No human commits to compute strain for.")
    else:
        strain_section(strain)
        radar_section(strain)
    risky_section(result)


main()
