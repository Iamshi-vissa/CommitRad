"""CommitRad dashboard. Run with ``commitrad dashboard`` or ``streamlit run`` on this file."""

from __future__ import annotations

import os
from dataclasses import dataclass
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
    author_weeks,
    commit_url,
    heatmap_frame,
    profile_deltas,
    radar_svg,
    repo_parts,
    risk_band,
    step_carousel,
    strain_frame,
    top_risky,
    week_axis,
)
from commitrad.explain import reason

DEMO_DIR = Path(os.environ.get("COMMITRAD_DEMO_DIR", "demo_cache"))
URL_KEY, DEMO_KEY = "repo_url", "demo_repo"

st.set_page_config(page_title="CommitRad", page_icon=":material/radar:", layout="wide")


@dataclass(frozen=True)
class Palette:
    """GitHub Primer colours for the current theme (charts can't read Streamlit's theme)."""

    ink: str
    muted: str
    grid: str
    series: tuple[str, str]  # selected author, repo average
    ramp: tuple[str, ...]  # sequential, low -> high strain


LIGHT = Palette(
    ink="#1f2328",
    muted="#59636e",
    grid="#d1d9e0",
    series=("#0969da", "#bc4c00"),
    ramp=("#ddf4ff", "#b6e3ff", "#80ccff", "#54aeff", "#218bff", "#0969da", "#0550ae", "#033d8b"),
)
DARK = Palette(
    ink="#f0f6fc",
    muted="#9198a1",
    grid="#3d444d",
    series=("#4493f8", "#db6d28"),
    ramp=("#0c2d6b", "#0d419d", "#1158c7", "#1f6feb", "#388bfd", "#58a6ff", "#79c0ff", "#a5d6ff"),
)


def palette() -> Palette:
    return DARK if st.context.theme.type == "dark" else LIGHT


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
        st.error(f"Invalid repository URL: {exc}", icon=":material/error:")
    except RepoTooLarge as exc:
        st.error(f"Repository too large: {exc}", icon=":material/error:")
    except Exception as exc:  # surface git/network problems without a traceback
        st.error(
            f"Could not analyze this repository ({type(exc).__name__}). Is it public?",
            icon=":material/error:",
        )
    return None


# --- charts --------------------------------------------------------------------------------


def heatmap(heat: pd.DataFrame, weeks: int, pal: Palette) -> alt.Chart:
    all_weeks, shown = week_axis(heat, weeks)
    authors = heat.groupby("author")["n_commits"].sum().sort_values(ascending=False).index.tolist()
    return (
        alt.Chart(heat)
        .mark_rect(cornerRadius=3)
        .encode(
            x=alt.X(
                "week_label:O",
                title=None,
                scale=alt.Scale(domain=all_weeks, paddingInner=0.18),
                axis=alt.Axis(values=shown, labelAngle=-40, ticks=False, domain=False),
            ),
            y=alt.Y(
                "author:N",
                title=None,
                sort=authors,
                scale=alt.Scale(paddingInner=0.18),
                axis=alt.Axis(ticks=False, domain=False),
            ),
            color=alt.Color(
                "strain:Q",
                title="Strain",
                scale=alt.Scale(domain=[0, 1], range=list(pal.ramp)),
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


def strain_over_time(weeks: pd.DataFrame, repo_median: float, pal: Palette) -> alt.LayerChart:
    x = alt.X("week:T", title=None, axis=alt.Axis(format="%b %Y", labelAngle=0, grid=False))
    y = alt.Y("strain:Q", title="Strain", scale=alt.Scale(domain=[0, 1]))
    color = pal.series[0]
    tooltip = [
        alt.Tooltip("week_label:N", title="Week of"),
        alt.Tooltip("strain:Q", title="Strain", format=".2f"),
        alt.Tooltip("n_commits:Q", title="Commits"),
    ]
    line = alt.Chart(weeks).mark_line(color=color, strokeWidth=2).encode(x=x, y=y, detail="run:N")
    points = (
        alt.Chart(weeks)
        .mark_point(filled=True, size=60, color=color)
        .encode(x=x, y=y, tooltip=tooltip)
    )
    median = pd.DataFrame({"median": [repo_median]})
    rule = (
        alt.Chart(median)
        .mark_rule(color=pal.series[1], strokeDash=[5, 4], strokeWidth=1.5)
        .encode(y="median:Q", tooltip=[alt.Tooltip("median:Q", title="Repo median", format=".2f")])
    )
    return (rule + line + points).properties(height=300)


# --- page sections -------------------------------------------------------------------------


def _pick_demo() -> None:
    if st.session_state.get(DEMO_KEY):
        st.session_state[URL_KEY] = ""


def _type_url() -> None:
    if st.session_state.get(URL_KEY):
        st.session_state[DEMO_KEY] = None


def header(snapshots: dict[str, Path]) -> str:
    st.title(":material/radar: CommitRad")
    st.caption("Risky-commit and developer-strain signals mined from git history.")
    st.warning(f"**Signals, not judgments.** {DISCLAIMER}", icon=":material/info:")
    left, right = st.columns([2, 3])
    demo = left.selectbox(
        "Demo repository (pre-computed, loads instantly)",
        list(snapshots),
        index=None,
        placeholder="Choose a demo repository",
        key=DEMO_KEY,
        on_change=_pick_demo,
    )
    url = right.text_input(
        "…or any public repository URL",
        key=URL_KEY,
        placeholder="https://github.com/owner/repo",
        on_change=_type_url,
    )
    return (url or demo or "").strip()


def empty_state() -> None:
    with st.container(border=True):
        st.markdown("#### :material/search: Choose a repository to analyze")
        st.markdown(
            "Pick a **demo repository** for an instant result, or paste a public **GitHub, "
            "GitLab or Codeberg** URL. A live analysis clones the repo and takes 20-60 seconds "
            "for a few thousand commits."
        )


def repo_header(result: dict[str, Any]) -> None:
    owner, name = repo_parts(result["repo"])
    url = result["repo"] if result["repo"].startswith("https://") else None
    title = f"{owner} / **{name}**" if owner else f"**{name}**"
    st.markdown(f"### :material/book_2: [{title}]({url})" if url else f"### {title}")
    badges = [f":blue-badge[:material/commit: {result['n_commits']:,} commits]"]
    if result["model"].get("in_training_data"):
        badges.append(":orange-badge[:material/school: In training data: scores are in-sample]")
    else:
        badges.append(":green-badge[:material/verified: Not in training data]")
    if "snapshot_at" in result:
        badges.append(f":gray-badge[:material/history: Snapshot {result['snapshot_at'][:10]}]")
    st.markdown(" ".join(badges))


def summary(result: dict[str, Any], strain: pd.DataFrame) -> None:
    cols = st.columns(4)
    cols[0].metric("Commits analyzed", f"{result['n_commits']:,}", border=True)
    cols[1].metric("SZZ bug-introducing", f"{result['n_bug_introducing_szz']:,}", border=True)
    cols[2].metric("Authors", f"{strain['author_email'].nunique():,}", border=True)
    cols[3].metric("Author-weeks", f"{len(strain):,}", border=True)
    model = result["model"]
    st.caption(
        f"Model test PR-AUC {model['test_pr_auc']:.2f}, ROC-AUC {model['test_roc_auc']:.2f}. "
        "Labels come from SZZ, which is noisy."
    )


def strain_section(strain: pd.DataFrame, pal: Palette) -> None:
    st.subheader("Weekly strain", divider="gray")
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
    st.altair_chart(heatmap(heat, weeks, pal), use_container_width=True)
    with st.expander("Table view"):
        st.dataframe(
            heat[["author", "week_label", "n_commits", "strain", *COMPONENTS]],
            hide_index=True,
            use_container_width=True,
        )


def profile_section(strain: pd.DataFrame, pal: Palette) -> None:
    st.subheader("Strain profile by author", divider="gray")
    totals = (
        strain.groupby(["author_email", "author"])["n_commits"].sum().sort_values(ascending=False)
    )
    labels = {email: f"{name} ({n} commits)" for (email, name), n in totals.items()}
    email = st.selectbox("Author", list(labels), format_func=labels.__getitem__)
    name = strain.loc[strain["author_email"] == email, "author"].mode().iloc[0]
    mine, repo = author_profile(strain, email), author_profile(strain)
    weeks = author_weeks(strain, email)

    cols = st.columns(len(COMPONENTS) + 1)
    overall = float((weeks["strain"] * weeks["n_commits"]).sum() / weeks["n_commits"].sum())
    repo_overall = float((strain["strain"] * strain["n_commits"]).sum() / strain["n_commits"].sum())
    cols[0].metric(
        "Overall strain",
        f"{overall:.2f}",
        f"{overall - repo_overall:+.2f} vs repo",
        delta_color="off",
        border=True,
    )
    for col, d in zip(cols[1:], profile_deltas(mine, repo), strict=True):
        col.metric(
            d["label"], f"{d['value']:.2f}", f"{d['delta']:+.2f} vs repo", delta_color="off",
            border=True,
        )  # fmt: skip
    st.caption(
        f"Commit-weighted averages over {len(weeks)} active weeks (0-1). Differences are "
        "neutral on purpose: a higher value is a pattern to understand, not a fault."
    )

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown(":small[Scale zooms to the data: the outer ring is the largest value shown.]")
        st.markdown(
            radar_svg({name: mine, REPO_AVERAGE: repo}, list(pal.series), pal.ink, pal.muted,
                      pal.grid),
            unsafe_allow_html=True,
        )  # fmt: skip
    with right:
        st.markdown(
            f"**Weekly strain for {name}**  \n:small[Orange dashed line: repo median week. "
            "Gaps mark pauses of more than 4 weeks.]"
        )
        st.altair_chart(
            strain_over_time(weeks, float(strain["strain"].median()), pal),
            use_container_width=True,
        )


def _carousel_key(result: dict[str, Any]) -> str:
    return f"risky_{result.get('slug', 'repo')}"


def _step(key: str, n: int, step: int) -> None:
    st.session_state[key] = step_carousel(st.session_state.get(key) or 1, n, step)


def commit_card(result: dict[str, Any], c: dict[str, Any], rank: int) -> None:
    url = commit_url(result["repo"], c["hash"])
    sha = f"[`{c['hash'][:7]}`]({url})" if url else f"`{c['hash'][:7]}`"
    with st.container(border=True):
        st.markdown(f":red-badge[:material/warning: Risk rank #{rank}]")
        st.markdown(f"#### {c['subject'] or '(no message)'}")
        st.markdown(
            f":material/person: **{c['author']}** &nbsp;·&nbsp; :material/calendar_today: "
            f"{c['timestamp'][:10]} &nbsp;·&nbsp; :material/commit: {sha}"
        )
        st.progress(
            min(1.0, max(0.0, c["risk_percentile"])),
            text=f"Risk {c['risk']:.2f}: {risk_band(c['risk_percentile'])}",
        )
        st.markdown("**Why the model flagged it**")
        lines = []
        for r in c["top_reasons"]:  # re-rendered from stored values so wording stays current
            text = reason(r["feature"], r["value"], r["shap"])["text"]
            icon = (
                ":red[:material/trending_up:]"
                if r["shap"] > 0
                else ":green[:material/trending_down:]"
            )
            lines.append(f"- {icon} {text}")
        st.markdown("\n".join(lines))


def risky_section(result: dict[str, Any]) -> None:
    st.subheader("Top 5 riskiest commits", divider="gray")
    st.caption("Ranked by model risk. Reasons are the three largest SHAP contributions.")
    commits = top_risky(result, 5)
    if not commits:
        st.info("No commits to rank.")
        return
    key, n = _carousel_key(result), len(commits)
    if st.session_state.get(key) not in range(1, n + 1):
        st.session_state[key] = 1
    prev_col, pager_col, next_col = st.columns([1, 4, 1], vertical_alignment="center")
    prev_col.button(
        "Previous", icon=":material/chevron_left:", key=f"{key}_prev", on_click=_step,
        args=(key, n, -1), use_container_width=True,
    )  # fmt: skip
    next_col.button(
        "Next", icon=":material/chevron_right:", key=f"{key}_next", on_click=_step,
        args=(key, n, 1), use_container_width=True,
    )  # fmt: skip
    with pager_col.container(horizontal=True, horizontal_alignment="center"):
        st.segmented_control(
            "Commit", list(range(1, n + 1)), key=key, label_visibility="collapsed",
            format_func=lambda i: f"{i}",
        )  # fmt: skip
    current = st.session_state.get(key) or 1
    commit_card(result, commits[current - 1], current)
    st.caption(f"Commit {current} of {n}")


def main() -> None:
    pal = palette()
    snapshots = list_snapshots(DEMO_DIR)
    url = header(snapshots)
    if not url:
        empty_state()
        return
    result = load(url, snapshots)
    if result is None:
        return
    strain = strain_frame(result)
    repo_header(result)
    summary(result, strain)
    if strain.empty:
        st.info("No human commits to compute strain for.")
    else:
        strain_section(strain, pal)
        profile_section(strain, pal)
    risky_section(result)


main()
