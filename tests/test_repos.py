from __future__ import annotations

from pathlib import Path

import pytest
from gitrepo import GitRepo

from commitrad.repos import ensure_clone, read_repo_list, repo_slug


@pytest.mark.parametrize(
    ("url", "slug"),
    [
        ("https://github.com/jd/tenacity", "jd__tenacity"),
        ("https://github.com/jd/tenacity.git", "jd__tenacity"),
        ("https://github.com/jd/tenacity/", "jd__tenacity"),
        ("git@github.com:pallets/markupsafe.git", "pallets__markupsafe"),
        ("https://gitlab.com/group/sub/proj", "group__sub__proj"),
    ],
)
def test_repo_slug_urls(url: str, slug: str) -> None:
    assert repo_slug(url) == slug


def test_repo_slug_local_dir(small_repo: GitRepo) -> None:
    assert repo_slug(str(small_repo.path)) == "small_repo"


def test_repo_slug_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        repo_slug("not a repo")


def test_read_repo_list_skips_comments_blanks_dupes(tmp_path: Path) -> None:
    f = tmp_path / "repos.txt"
    f.write_text("# header\nhttps://a/x/y\n\n  https://a/x/z  # trailing\nhttps://a/x/y\n")
    assert read_repo_list(f) == ["https://a/x/y", "https://a/x/z"]


def test_ensure_clone_local_is_used_in_place(small_repo: GitRepo, tmp_path: Path) -> None:
    assert ensure_clone(str(small_repo.path), tmp_path / "clones") == small_repo.path.resolve()
    assert not (tmp_path / "clones").exists()


def test_ensure_clone_clones_once(small_repo: GitRepo, tmp_path: Path) -> None:
    url = f"file://{small_repo.path}"
    clones = tmp_path / "clones"
    first = ensure_clone(url, clones)
    assert (first / "app.py").exists()
    marker = first / ".cached"
    marker.write_text("x")
    assert ensure_clone(url, clones) == first and marker.exists()
