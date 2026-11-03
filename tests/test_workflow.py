from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from frames import commit, frame
from gitrepo import GitRepo

from commitrad.features import build_features, change_entropy
from commitrad.mining import mine_table
from commitrad.sentiment import LexiconSentiment
from commitrad.workflow import (
    has_coauthor,
    is_config_path,
    is_revert,
    is_test_path,
    merge_pr_number,
    pr_membership,
    squash_pr_number,
    workflow_columns,
)


@pytest.fixture
def pr_repo(tmp_path: Path) -> tuple[GitRepo, dict[str, str]]:
    """main: init -> direct push -> PR #7 merge (2 commits) -> squash merge (#9)."""
    r, h = GitRepo(tmp_path / "pr_repo"), {}
    r.write("a.py", "x = 1\n")
    h["init"] = r.commit("init", "2024-01-01T10:00:00+00:00")
    r.write("a.py", "x = 2\n")
    h["direct"] = r.commit("tweak x directly", "2024-01-02T10:00:00+00:00")
    r.git("checkout", "-q", "-b", "feature")
    r.write("b.py", "y = 1\n")
    h["pr_a"] = r.commit("add b", "2024-01-03T10:00:00+00:00", "Bob", "bob@x.org")
    r.write("tests/test_b.py", "def test_b(): pass\n")
    h["pr_b"] = r.commit("test b", "2024-01-03T11:00:00+00:00", "Bob", "bob@x.org")
    r.git("checkout", "-q", "main")
    h["merge"] = r.merge(
        "feature", "Merge pull request #7 from bob/feature", "2024-01-04T10:00:00+00:00"
    )
    r.write("c.py", "z = 1\n")
    h["squash"] = r.commit("Add c module (#9)", "2024-01-05T10:00:00+00:00")
    return r, h


def test_pr_membership_from_git_topology(pr_repo: tuple[GitRepo, dict[str, str]]) -> None:
    repo, h = pr_repo
    members = pr_membership(repo.path)
    assert members[h["pr_a"]].number == 7 and members[h["pr_a"]].size == 2
    assert h["pr_b"] in members and h["direct"] not in members and h["squash"] not in members


def test_mined_table_carries_workflow_columns(pr_repo: tuple[GitRepo, dict[str, str]]) -> None:
    repo, h = pr_repo
    rows = {r["hash"]: r for r in mine_table(repo.path).to_pylist()}
    assert (rows[h["pr_a"]]["via_pr"], rows[h["pr_a"]]["pr_size"]) == (1, 2)
    assert (rows[h["squash"]]["via_pr"], rows[h["squash"]]["pr_size"]) == (1, 1)
    assert (rows[h["direct"]]["via_pr"], rows[h["direct"]]["pr_size"]) == (0, 0)


@pytest.mark.parametrize(
    ("msg", "pr", "squash"),
    [
        ("Merge pull request #12 from a/b", 12, None),
        ("Merge branch 'x' into 'main'\n\nSee merge request group/proj!34", 34, None),
        ("Fix parser crash (#56)", None, 56),
        ("Fix parser crash (#56) and more", None, None),
        ("Merge branch 'main' into feature", None, None),
    ],
)
def test_pr_number_parsing(msg: str, pr: int | None, squash: int | None) -> None:
    assert merge_pr_number(msg) == pr
    assert squash_pr_number(msg) == squash


def test_workflow_columns_fall_back_to_squash_marker() -> None:
    cols = workflow_columns(["a", "b"], ["Add thing (#3)", "plain"], {})
    assert cols == {"via_pr": [1, 0], "pr_size": [1, 0]}


def test_path_and_message_classifiers() -> None:
    for p in [
        "tests/test_x.py",
        "pkg/test_x.py",
        "src/x_test.go",
        "web/x.spec.ts",
        "a/__tests__/b.js",
    ]:
        assert is_test_path(p), p
    for p in ["src/contest.py", "testing_utils.py"]:
        assert not is_test_path(p), p
    for p in [".github/workflows/ci.yml", "pyproject.toml", "setup.py", "go.mod", "Dockerfile"]:
        assert is_config_path(p), p
    assert not is_config_path("src/app.py")
    assert is_revert('Revert "add cache"\n\nThis reverts commit abc1234.')
    assert not is_revert("Reverting to old style is discussed")
    assert has_coauthor("Fix\n\nCo-authored-by: Ann <a@x.org>")


def test_change_entropy() -> None:
    assert change_entropy([10]) == 0.0
    assert change_entropy([5, 5]) == pytest.approx(1.0)
    assert change_entropy([9, 1]) == pytest.approx(0.469, abs=1e-3)
    assert change_entropy([0, 0]) == 0.0


def test_history_and_diffusion_features() -> None:
    day = 24
    f = build_features(
        frame(
            [
                commit("c0", 0, author="ann", files={"pkg/a.py": (10, 0)}),
                commit(
                    "c1", 2 * day, author="bob", files={"pkg/a.py": (1, 1), "docs/x.md": (3, 0)}
                ),
                commit(
                    "c2",
                    5 * day,
                    author="cat",
                    files={"pkg/a.py": (1, 0), "tests/test_a.py": (1, 0)},
                ),
                commit("c3", 6 * day, author="ann", files={"new.py": (4, 0)}),
            ]
        ),
        LexiconSentiment(),
        "r",
    ).set_index("hash")
    assert f.loc["c0", ["file_prior_authors", "file_age_days", "author_tenure_days"]].tolist() == [
        0,
        0,
        0,
    ]
    assert f.loc["c1", "file_prior_authors"] == 1 and f.loc["c1", "dirs_touched"] == 2
    assert f.loc["c1", "file_age_days"] == pytest.approx((2 + 0) / 2)  # docs/x.md is new
    assert f.loc["c2", "file_prior_authors"] == 2  # ann + bob edited pkg/a.py before
    assert f.loc["c2", "file_age_days"] == pytest.approx((3 + 0) / 2)
    assert f.loc["c2", "test_file_share"] == 0.5
    assert f.loc["c3", "author_tenure_days"] == pytest.approx(6)
    assert f.loc["c3", "file_prior_authors"] == 0


def test_new_history_features_ignore_same_time_and_future_commits() -> None:
    rows = [
        commit("a", 10, author="ann", files={"x.py": (1, 0)}),
        commit("b", 10, author="bob", files={"x.py": (1, 0)}),
        commit("c", 11, author="cat", files={"x.py": (1, 0)}),
    ]
    f = build_features(frame(rows), LexiconSentiment(), "r").set_index("hash")
    assert f.loc["a", "file_prior_authors"] == f.loc["b", "file_prior_authors"] == 0
    assert f.loc["c", "file_prior_authors"] == 2
    future = frame([*rows, commit("d", 99, author="dan", files={"x.py": (100, 0)})])
    g = build_features(future, LexiconSentiment(), "r").set_index("hash")
    pd.testing.assert_frame_equal(g.loc[["a", "b", "c"]], f.loc[["a", "b", "c"]])
