from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from gitrepo import GitRepo

from commitrad.cli import main
from commitrad.mining import SCHEMA, mine_repo, mine_repos


def _rows(path: Path) -> list[dict[str, object]]:
    return pq.read_table(path).to_pylist()


def test_mine_repo_extracts_expected_fields(small_repo: GitRepo, tmp_path: Path) -> None:
    result = mine_repo(str(small_repo.path), tmp_path / "out", tmp_path / "clones")
    assert result.status == "mined" and result.n_commits == 3
    table = pq.read_table(result.path)
    assert table.schema.names == SCHEMA.names
    table.cast(SCHEMA)  # types are compatible (Parquet only renames list children)

    first, fix, drop = _rows(result.path)
    assert first["message"] == "initial commit" and first["parents"] == []
    assert sorted(first["modified_files"]) == ["README.md", "app.py"]
    assert (first["lines_added"], first["lines_deleted"], first["files_changed"]) == (4, 0, 2)
    assert first["tz_offset_minutes"] == 120
    assert first["timestamp"] == datetime(2024, 1, 1, 7, 0, tzinfo=UTC)

    assert fix["author"] == "Bob" and fix["author_email"] == "bob@x.org"
    assert fix["parents"] == [first["hash"]] and not fix["is_merge"]
    assert fix["modified_files"] == ["app.py"]
    assert (fix["lines_added"], fix["lines_deleted"]) == (2, 1)
    assert fix["tz_offset_minutes"] == -300

    assert drop["modified_files"] == ["README.md"]  # deleted file keeps its old path
    assert (drop["lines_added"], drop["lines_deleted"]) == (0, 1)


def test_mine_repo_is_resumable(small_repo: GitRepo, tmp_path: Path) -> None:
    out = tmp_path / "out"
    first = mine_repo(str(small_repo.path), out, tmp_path / "clones")
    mtime = first.path.stat().st_mtime_ns
    small_repo.write("new.py", "x = 1\n")
    small_repo.commit("more", "2024-02-01T10:00:00+00:00")

    again = mine_repo(str(small_repo.path), out, tmp_path / "clones")
    assert again.status == "skipped" and again.n_commits == 3
    assert first.path.stat().st_mtime_ns == mtime

    forced = mine_repo(str(small_repo.path), out, tmp_path / "clones", force=True)
    assert forced.status == "mined" and forced.n_commits == 4


def test_mine_repo_leaves_no_partial_file_on_failure(tmp_path: Path) -> None:
    bad = tmp_path / "not_a_repo"
    bad.mkdir()
    result = mine_repo(str(bad), tmp_path / "out", tmp_path / "clones")
    assert result.status == "failed" and result.error
    out = tmp_path / "out"
    assert not out.exists() or not any(out.iterdir())


@pytest.mark.parametrize("workers", [1, 3])
def test_mine_repos_parallel_keeps_order_and_isolates_failures(
    tmp_path: Path, workers: int
) -> None:
    sources = []
    for name in ("r1", "r2"):
        repo = GitRepo(tmp_path / name)
        repo.write("f.txt", f"{name}\n")
        repo.commit(f"{name} init", "2024-01-01T00:00:00+00:00")
        sources.append(str(repo.path))
    (tmp_path / "broken").mkdir()
    sources.insert(1, str(tmp_path / "broken"))

    results = mine_repos(sources, tmp_path / "out", tmp_path / "clones", workers=workers)
    assert [r.slug for r in results] == ["r1", "broken", "r2"]
    assert [r.status for r in results] == ["mined", "failed", "mined"]
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["r1.parquet", "r2.parquet"]


def test_cli_mine(small_repo: GitRepo, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    repos_txt = tmp_path / "repos.txt"
    repos_txt.write_text(f"{small_repo.path}\n")
    args = ["mine", str(repos_txt), "--out", str(tmp_path / "o"), "--clones", str(tmp_path / "c")]
    assert main(args) == 0
    assert "mined" in capsys.readouterr().out
    assert main(args) == 0
    assert "skipped" in capsys.readouterr().out


def test_cli_rejects_malformed_entry(tmp_path: Path) -> None:
    repos_txt = tmp_path / "repos.txt"
    repos_txt.write_text("definitely not a repo\n")
    assert main(["mine", str(repos_txt), "--out", str(tmp_path / "o")]) == 2
