from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from gitrepo import GitRepo

from commitrad.check import (
    HOOK_MARKER,
    CheckError,
    check_repo,
    format_report,
    install_hook,
    resolve_base,
    should_block,
    uninstall_hook,
)
from commitrad.cli import main


@pytest.fixture
def clone_with_work(tmp_path: Path) -> Path:
    """A clone of an upstream repo, with two local commits not yet pushed."""
    up = GitRepo(tmp_path / "upstream")
    up.write("app.py", "def f():\n    return 1\n")
    up.commit("init", "2024-01-01T10:00:00+00:00")
    up.write("app.py", "def f():\n    return 2\n")
    up.commit("tweak", "2024-01-02T10:00:00+00:00")
    work = tmp_path / "work"
    subprocess.run(["git", "clone", "-q", str(up.path), str(work)], check=True)
    w = GitRepo.__new__(GitRepo)
    w.path = work
    w.git("config", "user.name", "Test")
    w.git("config", "user.email", "test@example.com")
    w.write("app.py", "def f():\n    return 3\n" + "x = 1\n" * 200)
    w.commit("big local change", "2024-01-03T10:00:00+00:00")
    w.write("util.py", "y = 1\n")
    w.commit("small local change", "2024-01-04T10:00:00+00:00")
    return work


def test_resolve_base_uses_upstream(clone_with_work: Path) -> None:
    assert resolve_base(clone_with_work, None) == "origin/main"
    with pytest.raises(CheckError):
        resolve_base(clone_with_work, "no-such-ref")


def test_check_scores_only_unpushed_commits(clone_with_work: Path, trained_model_dir: Path) -> None:
    ref, commits = check_repo(clone_with_work, trained_model_dir)
    assert ref == "origin/main"
    assert sorted(c.subject for c in commits) == ["big local change", "small local change"]
    assert commits[0].risk >= commits[1].risk  # riskiest first
    assert all(0 <= c.percentile <= 1 and len(c.reasons) == 3 for c in commits)
    report = format_report(ref, commits, fail_above=0.0)
    assert "2 unpushed commit(s) vs origin/main" in report and "<- above threshold" in report
    assert should_block(commits, 0.0) and not should_block(commits, None)


def test_nothing_to_push(clone_with_work: Path, trained_model_dir: Path) -> None:
    ref, commits = check_repo(clone_with_work, trained_model_dir, base="HEAD")
    assert commits == [] and "nothing to push" in format_report(ref, commits, None)


def test_cli_check_exit_codes(
    clone_with_work: Path, trained_model_dir: Path, capsys: pytest.CaptureFixture
) -> None:
    args = ["check", "--repo", str(clone_with_work), "--models", str(trained_model_dir)]
    assert main(args) == 0
    assert main([*args, "--fail-above", "0"]) == 1
    assert "Push blocked" in capsys.readouterr().out


def test_hook_install_and_uninstall(clone_with_work: Path, trained_model_dir: Path) -> None:
    hook = install_hook(clone_with_work, trained_model_dir, block_above=0.99)
    text = hook.read_text()
    assert hook.name == "pre-push" and HOOK_MARKER in text and "--fail-above 0.99" in text
    assert hook.stat().st_mode & 0o111  # executable
    install_hook(clone_with_work, trained_model_dir)  # re-installing our own hook is fine
    assert uninstall_hook(clone_with_work) == hook and not hook.exists()
    assert uninstall_hook(clone_with_work) is None


def test_hook_never_overwrites_a_foreign_hook(
    clone_with_work: Path, trained_model_dir: Path
) -> None:
    hooks = clone_with_work / ".git" / "hooks"
    (hooks / "pre-push").write_text("#!/bin/sh\necho mine\n")
    with pytest.raises(CheckError, match="--force"):
        install_hook(clone_with_work, trained_model_dir)
    assert (
        install_hook(clone_with_work, trained_model_dir, force=True).read_text().count(HOOK_MARKER)
        == 1
    )
    assert uninstall_hook(clone_with_work) is not None


def test_installed_hook_runs_under_git(clone_with_work: Path, trained_model_dir: Path) -> None:
    """The generated script itself executes and passes through check's exit code."""
    install_hook(clone_with_work, trained_model_dir, block_above=0.0)
    run = subprocess.run(
        ["sh", ".git/hooks/pre-push"], cwd=clone_with_work, capture_output=True, text=True
    )
    assert run.returncode == 1 and "Push blocked" in run.stdout
