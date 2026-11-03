"""CommitRad dashboard. Run with ``commitrad dashboard`` or ``streamlit run`` on this file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from commitrad import ui
from commitrad.analyze import (
    DISCLAIMER,
    InvalidRepoURL,
    RepoTooLarge,
    Settings,
    analyze_repo,
    list_snapshots,
    load_snapshot,
    validate_repo_url,
)
from commitrad.dashboard_data import (
    COMPONENT_LABELS,
    COMPONENTS,
    REPO_AVERAGE,
    author_profile,
    author_weeks,
    commit_url,
    heatmap_frame,
    profile_deltas,
    radar_svg,
    repo_parts,
    risk_band,
    strain_frame,
    top_risky,
    week_axis,
)
from commitrad.explain import describe_value

DEMO_DIR = Path(os.environ.get("COMMITRAD_DEMO_DIR", "demo_cache"))
URL_KEY = "repo_url"
EXAMPLES = ("https://github.com/pallets/click", "https://github.com/psf/requests",
            "https://github.com/pallets/flask")  # fmt: skip

st.set_page_config(page_title="CommitRad", page_icon=str(ui.LOGO), layout="wide")


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
        url = validate_repo_url(url, Settings())  # .../repo/, .../repo.git -> canonical
    except InvalidRepoURL as exc:
        st.error(f"Invalid repository URL: {exc}")
        return None
    if url in snapshots:
        return _snapshot(str(snapshots[url]))
    try:
        with st.spinner("Cloning and scoring commits… large repos take a minute."):
            return _live(url)
    except RepoTooLarge as exc:
        st.error(f"Repository too large: {exc}")
    except Exception as exc:  # surface git/network problems without a traceback
        st.error(f"Could not analyze this repository ({type(exc).__name__}). Is it public?")
    return None


# --- charts --------------------------------------------------------------------------------


def _style(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    """Shared chart chrome: mono axis labels, hairline grid, no frame."""
    return (
        chart.configure(
            background="transparent",
        )
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelFont=ui.MONO, titleFont=ui.MONO, labelColor=ui.MUTED, titleColor=ui.MUTED,
            labelFontSize=11, titleFontSize=11, titleFontWeight=400, gridColor="#1a2029",
            domainColor=ui.BORDER, tickColor=ui.BORDER,
        )
        .configure_legend(
            labelFont=ui.MONO, titleFont=ui.MONO, labelColor=ui.MUTED, titleColor=ui.MUTED,
            titleFontWeight=400, labelFontSize=11, titleFontSize=11,
        )
    )  # fmt: skip


def heatmap(heat: pd.DataFrame, weeks: int) -> alt.TopLevelMixin:
    all_weeks, shown = week_axis(heat, weeks)
    authors = heat.groupby("author")["n_commits"].sum().sort_values(ascending=False).index.tolist()
    hover = alt.selection_point(
        on="pointerover", clear="pointerout", fields=["author", "week_label"]
    )
    chart = (
        alt.Chart(heat)
        .mark_rect(cornerRadius=3)
        .encode(
            x=alt.X(
                "week_label:O",
                title=None,
                scale=alt.Scale(domain=all_weeks, paddingInner=0.2),
                axis=alt.Axis(
                    values=shown,
                    labelAngle=0,
                    ticks=False,
                    domain=False,
                    labelExpr="slice(datum.value, 0, 7)",
                ),
            ),
            y=alt.Y(
                "author:N",
                title=None,
                sort=authors,
                scale=alt.Scale(paddingInner=0.2),
                axis=alt.Axis(ticks=False, domain=False, labelPadding=8),
            ),
            color=alt.Color(
                "strain:Q",
                title="strain",
                scale=alt.Scale(domain=[0, 1], range=list(ui.RAMP)),
                legend=alt.Legend(
                    orient="top",
                    direction="horizontal",
                    gradientLength=160,
                    titleOrient="left",
                    offset=10,
                ),
            ),
            opacity=alt.condition(hover, alt.value(1.0), alt.value(0.3)),
            tooltip=[
                alt.Tooltip("author:N", title="author"),
                alt.Tooltip("week_label:N", title="week of"),
                alt.Tooltip("strain:Q", title="strain", format=".2f"),
                alt.Tooltip("n_commits:Q", title="commits"),
                *[
                    alt.Tooltip(f"{c}:Q", title=COMPONENT_LABELS[c].lower(), format=".2f")
                    for c in COMPONENTS
                ],
            ],
        )
        .add_params(hover)
        .properties(height=alt.Step(22))
    )
    return _style(chart)


def strain_over_time(weeks: pd.DataFrame, repo_median: float) -> alt.TopLevelMixin:
    x = alt.X(
        "week:T",
        title=None,
        scale=alt.Scale(padding=14),  # keep edge points whole
        axis=alt.Axis(format="%Y", labelAngle=0, grid=False, tickCount="year"),
    )
    y = alt.Y("strain:Q", title="strain", scale=alt.Scale(domain=[0, 1]),
              axis=alt.Axis(tickCount=5))  # fmt: skip
    hover = alt.selection_point(on="pointerover", clear="pointerout", nearest=True, fields=["week"])
    tooltip = [
        alt.Tooltip("week_label:N", title="week of"),
        alt.Tooltip("strain:Q", title="strain", format=".2f"),
        alt.Tooltip("n_commits:Q", title="commits"),
    ]
    line = (
        alt.Chart(weeks)
        .mark_line(color=ui.AUTHOR, strokeWidth=1.75)
        .encode(x=x, y=y, detail="run:N")
    )
    points = (
        alt.Chart(weeks)
        .mark_circle(color=ui.AUTHOR, opacity=1)
        .encode(
            x=x,
            y=y,
            tooltip=tooltip,
            size=alt.condition(hover, alt.value(110), alt.value(28)),
        )
        .add_params(hover)
    )
    rule = (
        alt.Chart(pd.DataFrame({"median": [repo_median]}))
        .mark_rule(color=ui.REPO, strokeDash=[4, 4], strokeWidth=1.25)
        .encode(y="median:Q", tooltip=[alt.Tooltip("median:Q", title="repo median", format=".2f")])
    )
    return _style((rule + line + points).properties(height=430))


# --- page sections -------------------------------------------------------------------------


def _use_example(url: str) -> None:
    st.session_state[URL_KEY] = url


def header(snapshots: dict[str, Path]) -> str:
    st.html(ui.CSS)
    st.html(ui.nav_html("lightgbm · 5-seed ensemble · test pr-auc 0.22"))
    st.html(ui.hero_html(DISCLAIMER))
    url = st.text_input(
        "Repository URL",
        key=URL_KEY,
        placeholder="https://github.com/owner/repo",
        label_visibility="collapsed",
    )
    examples = [u for u in EXAMPLES if u in snapshots]
    if examples:
        with st.container(horizontal=True, horizontal_alignment="left", key="examples"):
            st.html(
                width="content",
                body='<span style="font-family:var(--mono);font-size:12px;color:var(--faint)">'
                    "try →</span>")  # fmt: skip
            for example in examples:
                owner, name = repo_parts(example)
                st.button(f"{owner}/{name}", key=f"example_{name}", on_click=_use_example,
                          args=(example,))  # fmt: skip
    return (url or "").strip()


def overview(result: dict[str, Any], strain: pd.DataFrame) -> None:
    owner, name = repo_parts(result["repo"])
    url = result["repo"] if result["repo"].startswith("https://") else None
    model = result["model"]
    chips: list[tuple[str, str | None]] = [(f"{result['n_commits']:,} commits", None)]
    if model.get("in_training_data"):
        chips.append(("in training data · in-sample scores", ui.REPO))
    else:
        chips.append(("out-of-sample", ui.DOWN))
    if "snapshot_at" in result:
        chips.append((f"snapshot {result['snapshot_at'][:10]}", None))
    st.html(ui.repo_html(owner, name, url, chips))
    bugs = result["n_bug_introducing_szz"]
    share = bugs / result["n_commits"] if result["n_commits"] else 0.0
    roc = f"test pr-auc · roc {model['test_roc_auc']:.2f}"
    st.html(
        ui.tiles_html(
            [
                ui.Tile("commits analyzed", f"{result['n_commits']:,}", "non-merge"),
                ui.Tile("bug-introducing", f"{bugs:,}", f"{share:.1%} · traced by szz"),
                ui.Tile("authors", f"{strain['author_email'].nunique():,}", "humans only"),
                ui.Tile("model", f"{model['test_pr_auc']:.2f}", roc),
            ]
        )
    )


def strain_section(strain: pd.DataFrame) -> None:
    st.html(
        ui.section_html(
            1,
            "Weekly strain",
            "One cell per author-week: a weighted blend of off-hours share, commit risk, "
            "burstiness and message negativity, from 0 to 1. Empty cells are weeks without "
            "commits. Hover a cell to isolate it.",
        )
    )
    c1, c2 = st.columns(2)
    weeks = c1.slider("Weeks shown", 8, 156, 52, step=4)
    top = c2.slider("Most active authors", 3, 30, 12)
    heat = heatmap_frame(strain, weeks, top)
    if heat.empty:
        st.info("No human commits in this window.")
        return
    with st.container(border=True, key="chart-heatmap"):  # padded card; chart fits inside
        st.altair_chart(heatmap(heat, weeks), use_container_width=True)
    with st.expander("Table view"):
        st.dataframe(
            heat[["author", "week_label", "n_commits", "strain", *COMPONENTS]],
            hide_index=True,
            use_container_width=True,
        )


def profile_section(strain: pd.DataFrame) -> None:
    st.html(
        ui.section_html(
            2,
            "Strain profile",
            "One author against the repository. Values are commit-weighted averages from 0 to "
            "1; differences are shown neutrally because a higher value is a pattern to "
            "understand, not a fault.",
        )
    )
    totals = (
        strain.groupby(["author_email", "author"])["n_commits"].sum().sort_values(ascending=False)
    )
    labels = {email: f"{name} · {n} commits" for (email, name), n in totals.items()}
    email = st.selectbox("Author", list(labels), format_func=labels.__getitem__)
    name = strain.loc[strain["author_email"] == email, "author"].mode().iloc[0]
    mine, repo = author_profile(strain, email), author_profile(strain)
    weeks = author_weeks(strain, email)

    overall = float((weeks["strain"] * weeks["n_commits"]).sum() / weeks["n_commits"].sum())
    repo_overall = float((strain["strain"] * strain["n_commits"]).sum() / strain["n_commits"].sum())
    tiles = [ui.Tile("overall", f"{overall:.2f}", f"{overall - repo_overall:+.2f} vs repo")]
    tiles += [
        ui.Tile(d["label"], f"{d['value']:.2f}", f"{d['delta']:+.2f} vs repo")
        for d in profile_deltas(mine, repo)
    ]
    st.html(ui.tiles_html(tiles))

    left, right = st.columns([1, 1], gap="medium")
    with left, st.container(border=True, key="chart-radar"):
        st.html(ui.legend_html([(name, ui.AUTHOR), (REPO_AVERAGE.lower(), ui.REPO)]))
        st.markdown(  # st.html's sanitizer strips SVG; markdown keeps it
            radar_svg(
                {name: mine, REPO_AVERAGE: repo}, [ui.AUTHOR, ui.REPO], ui.TEXT, ui.MUTED,
                "#232a33", surface=ui.SURFACE, mono=ui.MONO,
            ),
            unsafe_allow_html=True,
        )  # fmt: skip
        st.html(
            '<div class="cr-legend" style="margin-top:2px;color:var(--faint)">'
            "outer ring = largest value shown</div>"
        )
    with right:
        st.html(ui.legend_html([(f"{name}, weekly", ui.AUTHOR), ("repo median week", ui.REPO)]))
        with st.container(border=True, key="chart-weekly"):
            st.altair_chart(
                strain_over_time(weeks, float(strain["strain"].median())), use_container_width=True
            )


def risky_section(result: dict[str, Any]) -> None:
    st.html(
        ui.section_html(
            3,
            "Riskiest commits",
            "The five commits the model ranks highest. Each line is one SHAP contribution: "
            "+ raises risk, \u2212 lowers it, with its weight in log-odds.",
        )
    )
    commits = top_risky(result, 5)
    if not commits:
        st.info("No commits to rank.")
        return
    cards = [
        ui.CommitCard(
            rank=i,
            subject=c["subject"],
            author=c["author"],
            date=c["timestamp"][:10],
            sha=c["hash"],
            url=commit_url(result["repo"], c["hash"]),
            risk=c["risk"],
            percentile=c["risk_percentile"],
            band=risk_band(c["risk_percentile"]).replace(" of reference commits by risk", ""),
            reasons=[
                ui.Reason(describe_value(r["feature"], r["value"]), r["shap"])
                for r in c["top_reasons"]
            ],
        )
        for i, c in enumerate(commits, 1)
    ]
    st.html(ui.carousel_html(cards, uid=f"risky-{result.get('slug', 'repo')}"))


def main() -> None:
    snapshots = list_snapshots(DEMO_DIR)
    url = header(snapshots)
    if not url:
        st.html(ui.empty_html())
        st.html(ui.footer_html())
        return
    result = load(url, snapshots)
    if result is None:
        return
    strain = strain_frame(result)
    overview(result, strain)
    if strain.empty:
        st.info("No human commits to compute strain for.")
    else:
        strain_section(strain)
        profile_section(strain)
    risky_section(result)
    st.html(ui.footer_html())


main()
