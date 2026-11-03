from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from gitrepo import GitRepo

from commitrad.cli import main
from commitrad.mining import mine_repo
from commitrad.szz import (
    find_bug_introducers,
    is_bot,
    is_fix_commit,
    is_fix_message,
    is_ignorable_line,
    is_non_functional,
    is_source_file,
    label_repo,
    parse_blame_porcelain,
    parse_removed_ranges,
)

CALC_V1 = """# calculator helpers
def add(a, b):
    return a + b


def scale(x):
    return x*2
"""
SUB_BUGGY = """

def sub(a, b):
    return a + b
"""
MUL_BUGGY = """

def mul(a, b):
    return a * a
"""


@dataclass(frozen=True)
class History:
    repo: GitRepo
    hashes: dict[str, str]


@pytest.fixture
def szz_repo(tmp_path: Path) -> History:
    """A history with known bug-introducing commits (``sub`` and ``mul``) and decoy fixes."""
    r = GitRepo(tmp_path / "szz_repo")
    h: dict[str, str] = {}
    r.write("calc.py", CALC_V1)
    r.write("README.md", "Calc\nAdds numbers.\n")
    h["init"] = r.commit("initial calculator", "2024-01-01T10:00:00+00:00")

    r.write("calc.py", CALC_V1 + SUB_BUGGY)
    h["add_sub"] = r.commit("add subtract", "2024-01-02T10:00:00+00:00", "Bob", "bob@x.org")

    r.write("calc.py", CALC_V1 + SUB_BUGGY + MUL_BUGGY)
    h["add_mul"] = r.commit("add mul helper", "2024-01-03T10:00:00+00:00", "Carol", "c@x.org")

    def edit(old: str, new: str, path: str = "calc.py") -> None:
        r.write(path, (r.path / path).read_text().replace(old, new))

    edit("return a + b\n", "return a - b\n", path="calc.py")  # first match is add(); undo below
    edit("def add(a, b):\n    return a - b", "def add(a, b):\n    return a + b")
    h["fix_sub"] = r.commit("Fix subtraction bug", "2024-01-04T10:00:00+00:00", "Dave", "d@x.org")

    edit("x*2", "x * 2")
    h["fix_ws"] = r.commit("fix spacing in scale", "2024-01-05T10:00:00+00:00")

    edit("# calculator helpers", "# calculator helper functions")
    h["fix_comment"] = r.commit("fix wording of header comment", "2024-01-06T10:00:00+00:00")

    edit("Adds numbers.", "Adds and subtracts numbers.", path="README.md")
    h["fix_docs"] = r.commit("fix description", "2024-01-07T10:00:00+00:00")

    r.write("calc.py", (r.path / "calc.py").read_text().replace("def add", "def  add"))
    r.write("calc.py", (r.path / "calc.py").read_text().replace("def  add", "def add"))
    edit("return x * 2", "return x * 2  # double")
    h["typo_fix"] = r.commit("Fix typo", "2024-01-07T12:00:00+00:00")  # non-functional subject

    r.move("calc.py", "ops.py")
    h["rename"] = r.commit("rename module", "2024-01-08T10:00:00+00:00")

    r.git("checkout", "-q", "-b", "fix-things")
    r.write("other.py", "y = 1\n")
    h["branch"] = r.commit("add other", "2024-01-09T10:00:00+00:00")
    r.git("checkout", "-q", "main")
    h["merge"] = r.merge("fix-things", "Merge branch 'fix-things'", "2024-01-10T10:00:00+00:00")

    edit("return a * a", "return a * b", path="ops.py")
    h["fix_mul"] = r.commit("Patch mul, see issue #7", "2024-01-11T10:00:00+00:00")
    return History(r, h)


@pytest.mark.parametrize(
    "msg",
    ["Fix crash", "fixed it", "Fixes #12", "bugfix: x", "Bug in parser", "patched", "issue #42"],
)
def test_fix_messages(msg: str) -> None:
    assert is_fix_message(msg)


@pytest.mark.parametrize(
    "msg", ["add prefix option", "debug logging", "dispatch events", "new issue template"]
)
def test_non_fix_messages(msg: str) -> None:
    assert not is_fix_message(msg)


@pytest.mark.parametrize(
    ("msg", "expected"),
    [
        ("Fix typo in README", True),
        ("Fix pep8 errors", True),
        ("fix CI on windows", True),
        ("Fix docs for wait_random", True),
        ("Fix format string bug", False),
        ("Fix crash when stop is None\n\nAlso updated the docs.", False),  # subject only
    ],
)
def test_non_functional_subjects(msg: str, expected: bool) -> None:
    assert is_non_functional(msg) is expected


def test_is_bot() -> None:
    assert is_bot("dependabot[bot]", "49699333+dependabot[bot]@users.noreply.github.com")
    assert is_bot("pre-commit-ci[bot]", "x@y.z")
    assert not is_bot("Abbott", "abbott@x.org")


def test_is_fix_commit_filters() -> None:
    base = {"is_merge": False, "parents": ["p"], "author": "A", "author_email": "a@x"}
    assert is_fix_commit({**base, "message": "Fix crash"})
    assert not is_fix_commit({**base, "message": "Fix typo"})
    assert not is_fix_commit({**base, "message": "Fix crash", "is_merge": True})
    assert not is_fix_commit({**base, "message": "Fix crash", "parents": []})
    assert not is_fix_commit(
        {**base, "message": "Bump x to fix crash", "author": "dependabot[bot]"}
    )


def test_source_and_ignorable_lines() -> None:
    assert is_source_file("pkg/mod.py") and is_source_file("a/B.JAVA")
    assert not is_source_file("README.md") and not is_source_file("CHANGES.rst")
    for line in ["", "   ", "# note", "  // note", " * doc", "/* x */", "-- sql comment"]:
        assert is_ignorable_line(line), line
    assert not is_ignorable_line("x = 1  # trailing comment")


def test_parse_removed_ranges() -> None:
    diff = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -3 +3 @@ def f():
-    old
+    new
@@ -10,2 +9,0 @@
-x
-y
@@ -20,0 +18,3 @@
+added only
diff --git a/new.py b/new.py
--- /dev/null
+++ b/new.py
@@ -0,0 +1 @@
+z
"""
    assert parse_removed_ranges(diff) == {"a.py": [(3, 1), (10, 2)]}


def test_parse_blame_porcelain() -> None:
    a, b = "a" * 40, "b" * 40
    text = (
        f"{a} 3 3 1\nauthor A\nfilename x.py\n\tfirst\n"
        f"{b} 9 4 2\nauthor B\n\tsecond\n"
        f"{b} 10 5\n\tthird\n"
    )
    assert parse_blame_porcelain(text) == [(a, "first"), (b, "second"), (b, "third")]


def test_find_bug_introducers_on_each_fix(szz_repo: History) -> None:
    repo, h = szz_repo.repo.path, szz_repo.hashes
    assert find_bug_introducers(repo, h["fix_sub"]) == {h["add_sub"]}
    assert find_bug_introducers(repo, h["fix_ws"]) == set()  # whitespace-only
    assert find_bug_introducers(repo, h["fix_comment"]) == set()  # comment-only
    assert find_bug_introducers(repo, h["fix_docs"]) == set()  # non-source file
    assert find_bug_introducers(repo, h["fix_mul"]) == {h["add_mul"]}  # through the rename


def test_label_repo_end_to_end(szz_repo: History, tmp_path: Path) -> None:
    src, h = str(szz_repo.repo.path), szz_repo.hashes
    raw, clones, out = tmp_path / "raw", tmp_path / "clones", tmp_path / "labeled"
    assert mine_repo(src, raw, clones).status == "mined"

    result = label_repo(src, raw, clones, out)
    assert result.status == "labeled"
    rows = {r["hash"]: r for r in pq.read_table(result.path).to_pylist()}
    by_name = {name: rows[sha] for name, sha in h.items()}

    fixes = {n for n, r in by_name.items() if r["is_fix"]}
    assert fixes == {"fix_sub", "fix_ws", "fix_comment", "fix_docs", "fix_mul"}  # not the merge
    bugs = {n for n, r in by_name.items() if r["is_bug_introducing"]}
    assert bugs == {"add_sub", "add_mul"}

    assert by_name["add_sub"]["fixed_by"] == [h["fix_sub"]]
    assert by_name["add_sub"]["bug_found_at"] == datetime(2024, 1, 4, 10, tzinfo=UTC)
    assert by_name["init"]["fixed_by"] == [] and by_name["init"]["bug_found_at"] is None

    assert not by_name["typo_fix"]["is_fix"]
    assert (result.n_commits, result.n_fixes, result.n_bug_introducing) == (11, 5, 2)
    assert result.label_rate == pytest.approx(2 / 11)
    assert label_repo(src, raw, clones, out).status == "skipped"


def test_label_repo_without_mined_data_fails_cleanly(szz_repo: History, tmp_path: Path) -> None:
    result = label_repo(str(szz_repo.repo.path), tmp_path / "raw", tmp_path / "c", tmp_path / "o")
    assert result.status == "failed" and "parquet" in (result.error or "").lower()


def test_cli_label(szz_repo: History, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    repos_txt = tmp_path / "repos.txt"
    repos_txt.write_text(f"{szz_repo.repo.path}\n")
    dirs = ["--clones", str(tmp_path / "c")]
    assert main(["mine", str(repos_txt), "--out", str(tmp_path / "raw"), *dirs]) == 0
    label_args = ["--raw", str(tmp_path / "raw"), "--out", str(tmp_path / "lab"), *dirs]
    assert main(["label", str(repos_txt), *label_args]) == 0
    assert "overall label rate: 2/11 = 18.2%" in capsys.readouterr().out
