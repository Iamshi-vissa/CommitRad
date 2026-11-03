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
    commit_url,
    heatmap_frame,
    radar_frame,
    risk_band,
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


def test_dashboard_renders_demo_snapshot(demo_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMMITRAD_DEMO_DIR", str(demo_dir))
    at = AppTest.from_file(str(DASHBOARD), default_timeout=60).run()
    assert not at.exception
    assert "Signals, not judgments" in at.warning[0].value
    subheaders = [s.value for s in at.subheader]
    assert subheaders == ["Weekly strain", "Strain profile by author", "Top 5 riskiest commits"]
    assert at.metric[0].value == "3"
    text = " ".join(m.value for m in at.markdown)
    assert "(raises risk)" in text or "(lowers risk)" in text


def test_dashboard_without_snapshots_prompts_for_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COMMITRAD_DEMO_DIR", str(tmp_path / "empty"))
    at = AppTest.from_file(str(DASHBOARD), default_timeout=60).run()
    assert not at.exception
    assert "Signals, not judgments" in at.warning[0].value
    assert any("paste a public" in i.value for i in at.info)


def test_dashboard_rejects_bad_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMMITRAD_DEMO_DIR", str(tmp_path / "empty"))
    at = AppTest.from_file(str(DASHBOARD), default_timeout=60).run()
    at.text_input[0].input("http://evil.example.com/x/y").run()
    assert not at.exception
    assert "Invalid repository URL" in at.error[0].value
