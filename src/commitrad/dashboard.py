"""CommitRad dashboard. Run with ``commitrad dashboard`` or ``streamlit run`` on this file."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from commitrad import ui
from commitrad.account import InvalidLogin, analyze_account, validate_login
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
from commitrad.github import GitHubError

DEMO_DIR = Path(os.environ.get("COMMITRAD_DEMO_DIR", "demo_cache"))
URL_KEY = "repo_url"
MODES = ("Repository", "GitHub account", "Pre-push check")

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
            labelFont=ui.MONO,
            titleFont=ui.MONO,
            labelColor=ui.MUTED,
            titleColor=ui.MUTED,
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight=400,
            gridColor="#1a2029",
            domainColor=ui.BORDER,
            tickColor=ui.BORDER,
        )
        .configure_legend(
            labelFont=ui.MONO,
            titleFont=ui.MONO,
            labelColor=ui.MUTED,
            titleColor=ui.MUTED,
            titleFontWeight=400,
            labelFontSize=11,
            titleFontSize=11,
        )
    )


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
    y = alt.Y(
        "strain:Q", title="strain", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(tickCount=5)
    )
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


@st.cache_data(show_spinner=False)
def _model_note(model_dir: str = "models") -> str:
    try:
        meta = json.loads((Path(model_dir) / "metadata.json").read_text())
        pr_auc = meta["metrics"]["test"]["lightgbm"]["pr_auc"]
        return f"lightgbm · {len(meta['features'])} features · test pr-auc {pr_auc:.2f}"
    except (OSError, KeyError, ValueError):
        return "lightgbm ensemble"


def scroll_to(anchor: str, token: str) -> None:
    """Smooth-scroll the page to ``anchor`` once per ``token`` (e.g. once per new result)."""
    if st.session_state.get("_scrolled") == token:
        return
    st.session_state["_scrolled"] = token
    components.html(
        "<script>setTimeout(function(){var el=window.parent.document.getElementById("
        f"{json.dumps(anchor)});if(el)el.scrollIntoView({{behavior:'smooth',block:'start'}});}},120);"
        "</script>",
        height=0,
    )


def nav() -> str:
    st.html(ui.CSS)
    st.html(ui.nav_html(_model_note()))
    mode = st.segmented_control(
        "View", MODES, key="mode", default=MODES[0], label_visibility="collapsed"
    )
    return mode or MODES[0]


def repo_input(collapsed: bool) -> str:
    if not collapsed:
        st.html(
            ui.page_hero_html(
                "Commit risk and team strain,",
                "read straight from git history.",
                "CommitRad traces past bugs back to the commits that introduced them, learns "
                "what risky changes look like, and ranks every commit in a repository with "
                "the reasons attached.",
            )
        )
        st.html(ui.note_html(DISCLAIMER))
    st.html(ui.prompt_html("commitrad analyze", "repo-url"))
    url = st.text_input(
        "Repository URL",
        key=URL_KEY,
        placeholder="https://github.com/owner/repo  ·  press Enter",
        label_visibility="collapsed",
    )
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
                {name: mine, REPO_AVERAGE: repo},
                [ui.AUTHOR, ui.REPO],
                ui.TEXT,
                ui.MUTED,
                "#232a33",
                surface=ui.SURFACE,
                mono=ui.MONO,
            ),
            unsafe_allow_html=True,
        )
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
    cards = [_card(c, i, result["repo"]) for i, c in enumerate(commits, 1)]
    st.html(ui.carousel_html(cards, uid=f"risky-{result.get('slug', 'repo')}"))


# --- account view --------------------------------------------------------------------------


@st.cache_data(show_spinner=False, ttl=1800, max_entries=8)
def _account(login: str, token: str, max_repos: int) -> dict[str, Any]:
    return analyze_account(login, Settings(), token=token or None, max_repos=max_repos)


def account_inputs() -> tuple[str, str, int, bool]:
    st.html(
        ui.page_hero_html(
            "Your commits,",
            "beyond the contribution graph.",
            "GitHub counts your commits. CommitRad shows how they behave across all your "
            "repositories and forks: which carry the most risk and why, how often a later fix "
            "traced a bug back to you, how much went through review, when you push, and how "
            "far your forks have drifted.",
        )
    )
    st.html(
        ui.note_html(
            "A token is optional: it unlocks private repositories and 5,000 API requests/hour. It "
            "stays in this browser session's memory, goes only to github.com, and is never written "
            "to disk. A read-only, fine-grained token (Contents + Metadata: read) is enough."
        )
    )
    with st.form("account", border=True, enter_to_submit=True):
        c1, c2, c3 = st.columns([2, 2, 1])
        login = c1.text_input("GitHub username", placeholder="octocat", key="gh_login")
        token = c2.text_input(
            "Token (optional)", type="password", placeholder="github_pat_…", key="gh_token"
        )
        max_repos = c3.number_input("Repos", min_value=1, max_value=15, value=6, key="gh_repos")
        submitted = st.form_submit_button("Analyze account", use_container_width=True)
    return login.strip(), token.strip(), int(max_repos), submitted


def account_view() -> None:
    login, token, max_repos, submitted = account_inputs()
    if submitted:
        st.session_state["gh_active"] = (login, token, max_repos)
    active = st.session_state.get("gh_active")
    if not active or not active[0]:
        st.html(ui.footer_html(_model_note()))
        return
    login, token, max_repos = active
    try:
        validate_login(login)
        with st.spinner(f"Cloning and scoring @{login}'s {max_repos} most recently pushed repos…"):
            data = _account(login, token, max_repos)
    except InvalidLogin as exc:
        st.error(str(exc))
        return
    except GitHubError as exc:
        st.error(f"GitHub: {exc}")
        return
    st.html(ui.anchor_html("cr-results"))
    scroll_to("cr-results", f"acct:{login}:{max_repos}:{bool(token)}")
    account_report(data)
    st.html(ui.footer_html(_model_note()))


def account_report(data: dict[str, Any]) -> None:
    t = data["totals"]
    chips: list[tuple[str, str | None]] = [
        (f"{t['repos_analyzed']} repos analyzed", None),
        (f"{t['forks']} forks", None),
        (f"{data['api_calls']} api calls", None),
    ]
    st.html(
        ui.profile_html(data["login"], data["name"], data["avatar_url"], data["profile_url"], chips)
    )
    st.html(
        ui.tiles_html(
            [
                ui.Tile("my commits", f"{t['my_commits']:,}", "across analyzed repos"),
                ui.Tile(
                    "later fixed",
                    ui._pct(t["my_bug_rate"]),
                    f"{t['my_bug_introducing']} traced by szz",
                ),
                ui.Tile("via pull request", ui._pct(t["my_via_pr"]), "rest pushed directly"),
                ui.Tile("off-hours commits", ui._pct(t["my_off_hours"]), "nights + weekends"),
                ui.Tile(
                    "pushes · 90 days",
                    f"{t['pushes_90d']:,}",
                    f"{t['push_commits_90d']:,} commits pushed",
                ),
            ]
        )
    )
    st.html(
        ui.section_html(
            1,
            "Repositories",
            "Your share of each repo and how your commits compare with everyone's. \"Later "
            'fixed" is the share of commits a subsequent fix traced a bug back to (SZZ); '
            "fork drift is commits ahead (↑) and behind (↓) the upstream default branch.",
        )
    )
    st.html(ui.repo_table_html(data["repos"]))
    if not data["commits"]:
        st.info(
            "None of the analyzed commits matched this account's emails. Commits made with an "
            "email that isn't linked to the GitHub account can't be attributed."
        )
        return
    pushes = pd.DataFrame(data["pushes"])
    st.html(
        ui.section_html(
            2,
            "Push rhythm",
            "When you push, from GitHub's last 90 days of public events (UTC). GitHub's own "
            "contribution graph shows days, not hours.",
        )
    )
    if pushes.empty:
        st.info("No public push events in the last 90 days.")
    else:
        with st.container(border=True, key="chart-pushes"):
            st.altair_chart(punch_card(pushes), use_container_width=True)
    if data["strain"]:
        st.html(
            ui.section_html(
                3,
                "Weekly strain, all repos",
                "Your weekly strain across every analyzed repo, commit-weighted. Gaps are "
                "pauses of more than four weeks.",
            )
        )
        weeks = pd.DataFrame(data["strain"])
        weeks["week"] = pd.to_datetime(weeks["week"])
        weeks["week_label"] = weeks["week"].dt.strftime("%Y-%m-%d")
        weeks["run"] = (weeks["week"].diff() > pd.Timedelta(weeks=4)).cumsum()
        with st.container(border=True, key="chart-account-strain"):
            st.altair_chart(
                strain_over_time(weeks, float(weeks["strain"].median())), use_container_width=True
            )
    st.html(
        ui.section_html(
            4,
            "Your riskiest commits",
            "Across all analyzed repos. + raises risk, \u2212 lowers it, weighted in log-odds.",
        )
    )
    cards = [
        _card(c, i, c["repo_url"], context=c["repo"]) for i, c in enumerate(data["commits"][:5], 1)
    ]
    st.html(ui.carousel_html(cards, uid=f"acct-{data['login']}"))


def punch_card(pushes: pd.DataFrame) -> alt.TopLevelMixin:
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    grid = pd.DataFrame([(d, h) for d in range(7) for h in range(24)], columns=["weekday", "hour"])
    counts = pushes.groupby(["weekday", "hour"]).agg(
        pushes=("commits", "size"), commits=("commits", "sum")
    )
    grid = grid.join(counts, on=["weekday", "hour"]).fillna(0)
    grid["day"] = grid["weekday"].map(lambda d: days[d])
    hover = alt.selection_point(on="pointerover", clear="pointerout", fields=["day", "hour"])
    chart = (
        alt.Chart(grid)
        .mark_rect(cornerRadius=3)
        .encode(
            x=alt.X(
                "hour:O", title="hour (utc)", axis=alt.Axis(labelAngle=0, ticks=False, domain=False)
            ),
            y=alt.Y("day:N", title=None, sort=days, axis=alt.Axis(ticks=False, domain=False)),
            color=alt.Color(
                "pushes:Q",
                title="pushes",
                scale=alt.Scale(range=["#161b22", *ui.RAMP[2:]]),
                legend=alt.Legend(
                    orient="top",
                    direction="horizontal",
                    gradientLength=140,
                    titleOrient="left",
                    format="d",
                    tickMinStep=1,
                ),
            ),
            opacity=alt.condition(hover, alt.value(1.0), alt.value(0.35)),
            tooltip=[
                alt.Tooltip("day:N", title="day"),
                alt.Tooltip("hour:O", title="hour (utc)"),
                alt.Tooltip("pushes:Q", title="pushes"),
                alt.Tooltip("commits:Q", title="commits"),
            ],
        )
        .add_params(hover)
        .properties(height=alt.Step(26))
    )
    return _style(chart)


def _card(c: dict[str, Any], rank: int, repo_url: str, context: str = "") -> ui.CommitCard:
    return ui.CommitCard(
        rank=rank,
        subject=c["subject"],
        author=c["author"],
        date=c["timestamp"][:10],
        sha=c["hash"],
        url=commit_url(repo_url, c["hash"]),
        risk=c["risk"],
        percentile=c["risk_percentile"],
        band=risk_band(c["risk_percentile"]).replace(" of reference commits by risk", ""),
        reasons=[
            ui.Reason(describe_value(r["feature"], r["value"]), r["shap"]) for r in c["top_reasons"]
        ],
        context=context,
    )


# --- pre-push view -------------------------------------------------------------------------


def prepush_view() -> None:
    st.html(
        ui.page_hero_html(
            "Catch the risky commit,",
            "before you push it.",
            "Dashboards tell you about problems after they ship. CommitRad's pre-push check "
            "scores the commits you are about to push, using your repository's full history, "
            "and explains each score, so the riskiest change gets a second look first.",
        )
    )
    st.html(ui.section_html(1, "Run a check", "From inside any git repository with an upstream:"))
    st.html(
        ui.code_card_html(
            "terminal",
            [
                "$ uv run commitrad check --repo .",
                "",
                "commitrad check · 2 unpushed commit(s) vs origin/main",
                "",
                "   RISK   PCTL  COMMIT   SUBJECT",
                "   0.89   98.9  9fa165b  Add retry cache layer",
                "                + adds 340 lines (+1.78)",
                "                + author's first commit in this repo (+0.41)",
                "                - landed through a pull request (-0.12)",
            ],
            note="Add --base origin/main to compare against another ref. Takes about a second.",
        )
    )
    st.html(ui.section_html(2, "Install the git hook", "Runs the check on every git push."))
    st.html(
        ui.code_card_html(
            "terminal",
            [
                "$ uv run commitrad hook install --repo <repo>                     # warn only",
                "$ uv run commitrad hook install --repo <repo> --block-above 0.99  # block top 1%",
                "$ uv run commitrad hook uninstall --repo <repo>",
            ],
            note="An existing pre-push hook is never overwritten unless you pass --force. "
            "Skip the check for one push with git push --no-verify.",
        )
    )
    st.html(
        ui.section_html(
            3, "Use it in CI", "Fail a pull-request job when a commit is in the top 1%:"
        )
    )
    st.html(
        ui.code_card_html(
            "github actions step",
            [
                "- run: >-",
                "    uv run commitrad check --fail-above 0.99",
                "    --base origin/${{ github.base_ref }}",
            ],
            note="Check out with fetch-depth: 0 so the full history is available.",
        )
    )
    st.html(ui.footer_html(_model_note()))


# --- page ----------------------------------------------------------------------------------


def repo_view() -> None:
    snapshots = list_snapshots(DEMO_DIR)
    current = (st.session_state.get(URL_KEY) or "").strip()
    url = repo_input(collapsed=bool(current))
    if not url:
        st.html(ui.empty_html())
        st.html(ui.footer_html(_model_note()))
        return
    result = load(url, snapshots)
    if result is None:
        return
    st.html(ui.anchor_html("cr-results"))
    scroll_to("cr-results", f"repo:{result.get('slug')}:{result.get('head', '')}")
    strain = strain_frame(result)
    overview(result, strain)
    if strain.empty:
        st.info("No human commits to compute strain for.")
    else:
        strain_section(strain)
        profile_section(strain)
    risky_section(result)
    st.html(ui.footer_html(_model_note()))


def main() -> None:
    mode = nav()
    if mode == MODES[1]:
        account_view()
    elif mode == MODES[2]:
        prepush_view()
    else:
        repo_view()


main()
