"""How each commit reached the default branch: via a pull request, or pushed directly.

Derived from git alone (no API calls), so it works offline and for any host:

* GitHub merge commits ("Merge pull request #N ...") and GitLab merge commits ("See merge
  request !N") mark every commit on the merged branch as part of that PR; ``pr_size`` is
  the number of commits it carried.
* Squash merges are recognised by GitHub's ``(#N)`` subject suffix; they count as a PR of
  size 1 (the original commits are gone).

These describe a change as it landed on the default branch, which is the moment the risk
model scores it.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_PR_MERGE_RE = re.compile(r"^Merge pull request #(\d+)", re.MULTILINE)
_MR_MERGE_RE = re.compile(r"See merge request (?:[\w./-]+)?!(\d+)")
_SQUASH_RE = re.compile(r"\(#(\d+)\)\s*$")
TEST_PATH_RE = re.compile(
    r"(^|/)(tests?|spec|__tests__)/|(^|/)test_[^/]*$|_test\.[a-z]+$|\.(test|spec)\.[a-z]+$",
    re.IGNORECASE,
)
CONFIG_PATH_RE = re.compile(
    r"(^|/)\.github/|(^|/)\.gitlab-ci|\.(ya?ml|toml|cfg|ini|json)$|(^|/)(setup\.py|Makefile|"
    r"Dockerfile|requirements[^/]*\.txt|package\.json|go\.mod)$",
    re.IGNORECASE,
)
_REVERT_RE = re.compile(
    r"^Revert\b|^This reverts commit [0-9a-f]{7,}", re.IGNORECASE | re.MULTILINE
)
_COAUTHOR_RE = re.compile(r"^Co-authored-by:", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class PRInfo:
    number: int
    size: int


def merge_pr_number(message: str) -> int | None:
    """PR / MR number named by a merge commit's message, if any."""
    m = _PR_MERGE_RE.search(message) or _MR_MERGE_RE.search(message)
    return int(m.group(1)) if m else None


def squash_pr_number(message: str) -> int | None:
    subject = message.strip().splitlines()[0] if message.strip() else ""
    m = _SQUASH_RE.search(subject)
    return int(m.group(1)) if m else None


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True, errors="replace"
    ).stdout


def pr_membership(repo: Path, rev: str = "HEAD") -> dict[str, PRInfo]:
    """Map commit hash -> the PR whose merge brought it onto ``rev``'s first-parent line."""
    out = _git(repo, "log", "--merges", "--first-parent", "--format=%H%x00%P%x00%B%x1e", rev)
    members: dict[str, PRInfo] = {}
    for entry in filter(None, (e.strip("\n") for e in out.split("\x1e"))):
        _, parents, body = entry.split("\x00", 2)
        number = merge_pr_number(body)
        ps = parents.split()
        if number is None or len(ps) < 2:
            continue
        branch = _git(repo, "rev-list", "--no-merges", f"{ps[0]}..{ps[1]}").split()
        for sha in branch:
            members.setdefault(sha, PRInfo(number, len(branch)))
    return members


def workflow_columns(
    hashes: list[str], messages: list[str], members: dict[str, PRInfo]
) -> dict[str, list[int]]:
    """``via_pr`` (0/1) and ``pr_size`` for each commit, aligned with ``hashes``."""
    via_pr, pr_size = [], []
    for sha, msg in zip(hashes, messages, strict=True):
        info = members.get(sha)
        if info is None and squash_pr_number(msg or "") is not None:
            info = PRInfo(squash_pr_number(msg or "") or 0, 1)
        via_pr.append(int(info is not None))
        pr_size.append(info.size if info else 0)
    return {"via_pr": via_pr, "pr_size": pr_size}


def is_test_path(path: str) -> bool:
    return bool(TEST_PATH_RE.search(path))


def is_config_path(path: str) -> bool:
    return bool(CONFIG_PATH_RE.search(path))


def is_revert(message: str) -> bool:
    return bool(_REVERT_RE.search(message or ""))


def has_coauthor(message: str) -> bool:
    return bool(_COAUTHOR_RE.search(message or ""))
