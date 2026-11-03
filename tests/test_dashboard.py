from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from gitrepo import GitRepo
from streamlit.testing.v1 import AppTest

from commitrad.analyze import Settings, analyze_repo, list_snapshots, load_snapshot, save_snapshot
from commitrad.dashboard_data import (
    COMPONENTS,
    REPO_AVERAGE,
    author_profile,
    author_weeks,
    commit_url,
    heatmap_frame,
    profile_deltas,
    radar_extent,
    radar_frame,
    radar_svg,
    repo_parts,
    risk_band,
    step_carousel,
    strain_frame,
    top_risky,
    week_axis,
)

DASHBOARD = Path(__file__).parents[1] / "src" / "commitrad" / "dashboard.py"


def result_with(strain: list[dict[str, Any]], commits: list[dict[str, Any]] | None = None) -> dict:
    return {"strain": strain, "commits": commits or []}


def week(author: str, day: str, n: int, strain: float, **comp: float) -> dict[str, Any]:
    components = dict.fromkeys(COMPONENTS, 0.0) | comp
    return {"author": author.title(), "author_email": f"{author}@x", "week": day,
            "n_commits": n, "strain": strain, "components": components}  # fmt: skip


def test_strain_frame_flattens_components() -> None:
    df = strain_frame(result_with([week("a", "2024-01-01", 2, 0.4, risk=0.7)]))
    assert df.loc[0, "risk"] == 0.7 and df["week"].dtype.kind == "M"
    assert strain_frame(result_with([])).empty


def test_heatmap_window_and_top_authors() -> None:
    rows = [week("old", "2023-01-02", 50, 0.9)]
    mondays = ["2024-01-01", "2024-02-05", "2024-03-04", "2024-04-01", "2024-05-06"]
    rows += [week("busy", d, 5, 0.2) for d in mondays]
    rows += [week("quiet", "2024-05-06", 1, 0.5)]
    heat = heatmap_frame(strain_frame(result_with(rows)), weeks=26, top_authors=1)
    assert set(heat["author"]) == {"Busy"}  # "old" is outside the window, "quiet" not top-1
    labels, shown = week_axis(heat, 26)
    assert len(labels) == 26 and labels[-1] == "2024-05-06" and shown[-1] == labels[-1]


def test_author_profile_is_commit_weighted() -> None:
    df = strain_frame(
        result_with([week("a", "2024-01-01", 3, 0, risk=1.0), week("a", "2024-01-08", 1, 0)])
    )
    assert author_profile(df, "a@x")["risk"] == pytest.approx(0.75)
    assert author_profile(df, "nobody@x") == dict.fromkeys(COMPONENTS, 0.0)


def test_radar_polygon_closes_and_scales() -> None:
    profile = {"off_hours": 1.0, "risk": 0.5, "burstiness": 0.0, "negativity": 0.25}
    df = radar_frame({"A": profile, REPO_AVERAGE: profile})
    a = df[df["series"] == "A"].sort_values("order")
    assert len(a) == len(COMPONENTS) + 1
    assert a.iloc[0][["x", "y"]].tolist() == a.iloc[-1][["x", "y"]].tolist()
    assert a.iloc[0]["x"] == pytest.approx(0) and a.iloc[0]["y"] == pytest.approx(1.0)  # 12 o'clock
    assert math.hypot(a.iloc[1]["x"], a.iloc[1]["y"]) == pytest.approx(0.5)


def test_radar_zooms_to_the_data() -> None:
    small = {"A": {"off_hours": 0.45, "risk": 0.51, "burstiness": 0.2, "negativity": 0.0}}
    assert radar_extent(small) == 0.75
    assert radar_extent({"A": dict.fromkeys(COMPONENTS, 0.0)}) == 0.25
    assert radar_extent({"A": dict.fromkeys(COMPONENTS, 0.9)}) == 1.0
    assert radar_extent({"A": dict.fromkeys(COMPONENTS, 0.5)}) == 0.5  # exact quarter kept
    df = radar_frame(small, extent=0.75)
    risk = df[(df["component"] == "Commit risk")].iloc[0]
    assert math.hypot(risk["x"], risk["y"]) == pytest.approx(0.51 / 0.75)


def test_radar_svg_labels_values_and_escapes_names() -> None:
    profiles = {
        "<Ann & Co>": {"off_hours": 0.45, "risk": 0.51, "burstiness": 0.49, "negativity": 0.01},
        REPO_AVERAGE: {"off_hours": 0.36, "risk": 0.61, "burstiness": 0.28, "negativity": 0.01},
    }
    svg = radar_svg(profiles, ["#0969da", "#bc4c00"], "#111", "#666", "#ddd")
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert "&lt;Ann &amp; Co&gt;" in svg and "<Ann" not in svg
    for label in ("Off-hours", "Commit risk", "Burstiness", "Negativity", "0.75", "0.45"):
        assert label in svg
    assert svg.count("<title>") == 8  # a tooltip on every point of both shapes
    assert svg.count("<polygon") == 4 + 2  # 4 rings + 2 shapes


def test_author_weeks_breaks_runs_at_long_pauses() -> None:
    rows = [week("a", d, 1, 0.3) for d in ("2024-01-01", "2024-01-08", "2024-06-03", "2024-06-10")]
    runs = author_weeks(strain_frame(result_with(rows)), "a@x")["run"].tolist()
    assert runs == [0, 0, 1, 1]


def test_profile_deltas_and_carousel_and_repo_parts() -> None:
    a = dict.fromkeys(COMPONENTS, 0.5)
    r = dict.fromkeys(COMPONENTS, 0.2)
    d = profile_deltas(a, r)
    assert [x["component"] for x in d] == COMPONENTS and d[0]["delta"] == pytest.approx(0.3)
    assert [step_carousel(i, 5, 1) for i in (1, 5)] == [2, 1]
    assert [step_carousel(i, 5, -1) for i in (1, 3)] == [5, 2]
    assert repo_parts("https://github.com/pallets/click") == ("pallets", "click")
    assert repo_parts("/tmp/x/demo") == ("", "demo")


def test_commit_urls_and_risk_band() -> None:
    assert commit_url("https://github.com/a/b", "abc") == "https://github.com/a/b/commit/abc"
    assert commit_url("https://gitlab.com/a/b", "abc") == "https://gitlab.com/a/b/-/commit/abc"
    assert commit_url("/local/path", "abc") is None
    assert risk_band(0.999) == "in the top 1% of reference commits by risk"
    assert risk_band(0.9) == "in the top 10% of reference commits by risk"


def test_top_risky_sorts() -> None:
    commits = [{"risk": r} for r in (0.1, 0.9, 0.5)]
    assert [c["risk"] for c in top_risky(result_with([], commits), 2)] == [0.9, 0.5]


@pytest.fixture
def demo_dir(tmp_path: Path, trained_model_dir: Path) -> Path:
    """A real snapshot from a synthetic repo, analyzed by the full pipeline."""
    repo = GitRepo(tmp_path / "demo")
    repo.write("a.py", "x = 1\n")
    repo.commit("start", "2024-01-01T23:00:00+00:00")
    repo.write("a.py", "x = 2\ny = 3\n")
    repo.commit("tweak values", "2024-01-02T10:00:00+00:00", "Bob", "bob@x.org")
    repo.write("a.py", "x = 1\ny = 3\n")
    repo.commit("Fix x regression", "2024-01-09T10:00:00+00:00")
    settings = Settings(
        data_dir=tmp_path / "api", model_dir=trained_model_dir, db_path=tmp_path / "d.duckdb",
        strain_config=Path(__file__).parents[1] / "config" / "strain.toml",
        allow_local=True, sentiment="lexicon",
    )  # fmt: skip
    out = tmp_path / "demo_cache"
    save_snapshot(analyze_repo(str(repo.path), replace(settings)), out)
    return out


def test_snapshot_roundtrip_and_listing(demo_dir: Path) -> None:
    snaps = list_snapshots(demo_dir)
    assert len(snaps) == 1
    ((url, path),) = snaps.items()
    data = load_snapshot(path)
    assert data["repo"] == url and "snapshot_at" in data and "cached" not in data


def html_of(at: AppTest) -> str:
    """Everything rendered through st.html and st.markdown, as one string."""
    return " ".join([h.proto.body for h in at.get("html")] + [m.value for m in at.markdown])


def test_dashboard_renders_demo_snapshot(demo_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMMITRAD_DEMO_DIR", str(demo_dir))
    at = AppTest.from_file(str(DASHBOARD), default_timeout=60).run()
    assert not at.exception
    assert not at.selectbox and at.text_input[0].value == ""  # one empty URL box, no defaults
    page = html_of(at)
    assert "cr-nav" in page and "commitrad analyze" in page and "Signals, not judgments" not in page
    assert "not judgments about people" in page  # the disclaimer note
    assert 'class="cr cr-sec"' not in page  # nothing analyzed until a repo is entered

    at.text_input[0].input(next(iter(list_snapshots(demo_dir)))).run()
    assert not at.exception
    page = html_of(at)
    for section in ("Weekly strain", "Strain profile", "Riskiest commits"):
        assert section in page
    assert "cr-radar" in page and "cr-kpi" in page and "cr-foot" in page
    assert page.count('type="radio"') == 3  # one carousel slide per commit (3 in this repo)
    assert "@@ why the model flagged it @@" in page


def test_entering_a_repo_collapses_the_intro(
    demo_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COMMITRAD_DEMO_DIR", str(demo_dir))
    at = AppTest.from_file(str(DASHBOARD), default_timeout=60).run()
    assert 'class="cr cr-hero"' in html_of(at) and not at.button  # no example chips
    at.text_input[0].input(next(iter(list_snapshots(demo_dir)))).run()
    page = html_of(at)
    assert 'class="cr cr-hero"' not in page  # results move to the top
    assert 'id="cr-results"' in page  # scroll target for the smooth scroll


def fake_account(
    login: str, settings: object, token: str | None = None, max_repos: int = 6
) -> dict:
    from commitrad.account import AccountResult, RepoSummary

    repo = RepoSummary(
        "ann/tool",
        "https://github.com/ann/tool",
        True,
        False,
        "Python",
        None,
        parent="up/tool",
        ahead_by=3,
        behind_by=9,
        my_commits=1,
        repo_commits=4,
        my_bug_rate=0.5,
        repo_bug_rate=0.25,
        my_via_pr=1.0,
        my_off_hours=0.0,
        my_risk_pct=0.9,
        repo_risk_pct=0.5,
    )
    commit = {
        "hash": "abc1234def",
        "author": "Ann",
        "author_email": "a@x",
        "subject": "Add cache",
        "timestamp": "2026-10-01T00:00:00+00:00",
        "risk": 0.8,
        "risk_percentile": 0.99,
        "via_pr": True,
        "off_hours": False,
        "bug_introducing": True,
        "repo": "ann/tool",
        "repo_url": "https://github.com/ann/tool",
        "top_reasons": [
            {"feature": "lines_added", "value": 300.0, "shap": 1.2, "effect": "raises", "text": ""}
        ],
    }
    out = AccountResult(
        login,
        "Ann",
        None,
        f"https://github.com/{login}",
        repos=[repo],
        commits=[commit],
        strain=[
            {
                "week": "2026-09-28",
                "n_commits": 1,
                "strain": 0.4,
                "components": {"off_hours": 0, "risk": 0.9, "burstiness": 0, "negativity": 0},
            }
        ],
        pushes=[
            {
                "at": "2026-10-04T23:00:00+00:00",
                "weekday": 6,
                "hour": 23,
                "repo": "ann/tool",
                "branch": "main",
                "commits": 2,
            }
        ],
        api_calls=7,
    )
    return out.as_dict()


def test_account_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import commitrad.account

    monkeypatch.setattr(commitrad.account, "analyze_account", fake_account)
    monkeypatch.setenv("COMMITRAD_DEMO_DIR", str(tmp_path / "empty"))
    at = AppTest.from_file(str(DASHBOARD), default_timeout=60).run()
    at.button_group[0].set_value("GitHub account").run()
    assert "beyond the contribution graph" in html_of(at)
    at.text_input(key="gh_login").input("ann")  # form: value is sent with the submit
    at.button[0].click().run()  # the form's submit button
    assert not at.exception
    page = html_of(at)
    assert "@ann" in page and "cr-table" in page and "↑3" in page and "↓9" in page
    for section in (
        "Repositories",
        "Push rhythm",
        "Weekly strain, all repos",
        "Your riskiest commits",
    ):
        assert section in page
    assert "risk rank #1 · ann/tool" in page


def test_account_mode_rejects_bad_login(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMMITRAD_DEMO_DIR", str(tmp_path / "empty"))
    at = AppTest.from_file(str(DASHBOARD), default_timeout=60).run()
    at.button_group[0].set_value("GitHub account").run()
    at.text_input(key="gh_login").input("not a user!")
    at.button[0].click().run()
    assert "GitHub username" in at.error[0].value


def test_prepush_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMMITRAD_DEMO_DIR", str(tmp_path / "empty"))
    at = AppTest.from_file(str(DASHBOARD), default_timeout=60).run()
    at.button_group[0].set_value("Pre-push check").run()
    page = html_of(at)
    assert "commitrad hook install" in page and "--fail-above 0.99" in page


def test_dashboard_without_snapshots_prompts_for_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COMMITRAD_DEMO_DIR", str(tmp_path / "empty"))
    at = AppTest.from_file(str(DASHBOARD), default_timeout=60).run()
    assert not at.exception
    assert 'class="cr cr-empty"' in html_of(at)


def test_dashboard_rejects_bad_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMMITRAD_DEMO_DIR", str(tmp_path / "empty"))
    at = AppTest.from_file(str(DASHBOARD), default_timeout=60).run()
    at.text_input[0].input("http://evil.example.com/x/y").run()
    assert not at.exception
    assert "Invalid repository URL" in at.error[0].value
