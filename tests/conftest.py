from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from gitrepo import GitRepo


@pytest.fixture
def small_repo(tmp_path: Path) -> GitRepo:
    """Three commits by two authors in different timezones, incl. a delete."""
    repo = GitRepo(tmp_path / "small_repo")
    repo.write("app.py", "a = 1\nb = 2\nc = 3\n")
    repo.write("README.md", "hello\n")
    repo.commit("initial commit", "2024-01-01T09:00:00+02:00")
    repo.write("app.py", "a = 1\nb = 20\nc = 3\nd = 4\n")
    repo.commit("fix bug in b", "2024-01-02T23:30:00-05:00", author="Bob", email="bob@x.org")
    repo.remove("README.md")
    repo.commit("drop readme", "2024-01-06T12:00:00+00:00")
    return repo
