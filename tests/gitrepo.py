"""Tiny builder for deterministic synthetic git repos used in tests."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


class GitRepo:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.mkdir(parents=True, exist_ok=True)
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.com")
        self.git("config", "commit.gpgsign", "false")

    def git(self, *args: str, env: dict[str, str] | None = None) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.path,
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, **(env or {})},
        )
        return result.stdout.strip()

    def write(self, rel: str, content: str) -> None:
        target = self.path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def remove(self, rel: str) -> None:
        self.git("rm", "-q", rel)

    def commit(
        self,
        message: str,
        date: str,
        author: str = "Alice",
        email: str = "alice@example.com",
    ) -> str:
        """Stage everything and commit with a fixed ISO-8601 date; returns the hash."""
        self.git("add", "-A")
        env = {
            "GIT_AUTHOR_NAME": author,
            "GIT_AUTHOR_EMAIL": email,
            "GIT_AUTHOR_DATE": date,
            "GIT_COMMITTER_NAME": author,
            "GIT_COMMITTER_EMAIL": email,
            "GIT_COMMITTER_DATE": date,
        }
        self.git("commit", "-q", "--allow-empty", "-m", message, env=env)
        return self.git("rev-parse", "HEAD")
