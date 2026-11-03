"""SZZ: label bug-introducing commits by blaming the lines that fix commits removed.

Pipeline per repo:
1. Flag fix commits by message keywords (non-merge, non-bot, and not a
   style/docs/typo-only "fix").
2. For each fix F, diff ``F^..F`` ignoring whitespace, keep removed lines in
   source files that aren't blank or comment-only.
3. ``git blame F^`` on those lines; the blamed commits are bug-introducing.

Each introducer also records ``bug_found_at`` (the earliest implicating fix's
timestamp) so downstream features can avoid using labels that were not yet
knowable at a given point in time.
"""

from __future__ import annotations

import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq

from commitrad.mining import output_path as raw_output_path
from commitrad.mining import write_table
from commitrad.parallel import map_repos
from commitrad.repos import ensure_clone, repo_slug

FIX_PATTERN = re.compile(r"\b(?:fix\w*|bug\w*|patch\w*|issue\s*#\s*\d+)", re.IGNORECASE)

# Subjects like "fix typo" / "fix pep8" are fixes, but not of bugs: blaming them adds noise.
NON_FUNCTIONAL_PATTERN = re.compile(
    r"\b(?:typos?|spelling|grammar|pep8|flake8|lint\w*|style|formatting|whitespace|"
    r"docs?|docstrings?|documentation|readme|changelog|ci|tox|mypy|isort|black|ruff)\b",
    re.IGNORECASE,
)

# fmt: off
SOURCE_EXTENSIONS = frozenset({
    ".py", ".pyx", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".java", ".kt",
    ".scala", ".go", ".rs", ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift",
    ".m", ".sh", ".bash", ".lua", ".sql", ".r", ".jl", ".ex", ".exs", ".erl", ".hs", ".ml",
    ".clj", ".dart",
})
# fmt: on
_COMMENT_PREFIXES = ("#", "//", "/*", "*", "*/", "--", ";;", "%")
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")
_BLAME_HEADER_RE = re.compile(r"^([0-9a-f]{40}) \d+ \d+(?: \d+)?$")

LABEL_FIELDS = [
    pa.field("is_fix", pa.bool_()),
    pa.field("is_bug_introducing", pa.bool_()),
    pa.field("fixed_by", pa.list_(pa.string())),
    pa.field("bug_found_at", pa.timestamp("s", tz="UTC")),
]

Range = tuple[int, int]  # (start line, line count) on the parent side
Status = Literal["labeled", "skipped", "failed"]


def is_fix_message(message: str) -> bool:
    return bool(FIX_PATTERN.search(message))


def is_non_functional(message: str) -> bool:
    """True when the subject line says the change is about style, docs, typos or tooling."""
    subject = message.strip().splitlines()[0] if message.strip() else ""
    return bool(NON_FUNCTIONAL_PATTERN.search(subject))


def is_bot(author: str, email: str) -> bool:
    return "[bot]" in author.lower() or "[bot]" in email.lower() or email.lower().startswith("bot@")


def is_fix_commit(row: dict[str, Any]) -> bool:
    """A non-merge, non-root, human commit whose message reports fixing a functional bug."""
    return (
        not row["is_merge"]
        and bool(row["parents"])
        and not is_bot(row["author"], row["author_email"])
        and is_fix_message(row["message"])
        and not is_non_functional(row["message"])
    )


def is_source_file(path: str) -> bool:
    return Path(path).suffix.lower() in SOURCE_EXTENSIONS


def is_ignorable_line(content: str) -> bool:
    """Blank lines and comment-only lines carry no executable logic."""
    stripped = content.strip()
    return not stripped or stripped.startswith(_COMMENT_PREFIXES)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.quotePath=false", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout


def parse_removed_ranges(diff: str) -> dict[str, list[Range]]:
    """Map each pre-image path to the line ranges a ``-U0`` diff removed from it."""
    ranges: dict[str, list[Range]] = defaultdict(list)
    path: str | None = None
    for line in diff.splitlines():
        if line.startswith("--- "):
            old = line[4:].rstrip("\t")
            path = old[2:] if old.startswith("a/") else None
        elif path and (m := _HUNK_RE.match(line)):
            start, count = int(m.group(1)), int(m.group(2) or 1)
            if count:
                ranges[path].append((start, count))
    return dict(ranges)


def parse_blame_porcelain(text: str) -> list[tuple[str, str]]:
    """Return ``(commit_hash, line_content)`` pairs from ``git blame --porcelain``."""
    pairs: list[tuple[str, str]] = []
    current: str | None = None
    for line in text.splitlines():
        if line.startswith("\t"):
            if current:
                pairs.append((current, line[1:]))
        elif m := _BLAME_HEADER_RE.match(line):
            current = m.group(1)
    return pairs


def removed_ranges(repo: Path, fix_hash: str) -> dict[str, list[Range]]:
    """Removed line ranges per source file in a fix (modified/renamed files only)."""
    diff = _git(
        repo, "diff", "-U0", "-w", "-M", "--no-color", "--diff-filter=MR", f"{fix_hash}^", fix_hash
    )
    return {p: r for p, r in parse_removed_ranges(diff).items() if is_source_file(p)}


def blame_lines(repo: Path, rev: str, path: str, ranges: list[Range]) -> set[str]:
    """Commits that last touched the given non-trivial lines of ``path`` at ``rev``."""
    line_args = [arg for start, n in ranges for arg in ("-L", f"{start},+{n}")]
    out = _git(repo, "blame", "--porcelain", "-w", *line_args, rev, "--", path)
    return {sha for sha, content in parse_blame_porcelain(out) if not is_ignorable_line(content)}


def find_bug_introducers(repo: Path, fix_hash: str) -> set[str]:
    parent = f"{fix_hash}^"
    found: set[str] = set()
    for path, ranges in removed_ranges(repo, fix_hash).items():
        found |= blame_lines(repo, parent, path, ranges)
    return found


def compute_labels(rows: list[dict[str, Any]], repo: Path) -> dict[str, list[Any]]:
    """Label columns aligned with ``rows`` (dicts with hash/parents/is_merge/message/timestamp)."""
    is_fix = [is_fix_commit(r) for r in rows]
    fixed_by: dict[str, list[str]] = defaultdict(list)
    found_at: dict[str, datetime] = {}
    for row, fix in zip(rows, is_fix, strict=True):
        if not fix:
            continue
        for sha in find_bug_introducers(repo, row["hash"]):
            fixed_by[sha].append(row["hash"])
            prev = found_at.get(sha)
            found_at[sha] = row["timestamp"] if prev is None else min(prev, row["timestamp"])
    return {
        "is_fix": is_fix,
        "is_bug_introducing": [r["hash"] in fixed_by for r in rows],
        "fixed_by": [fixed_by.get(r["hash"], []) for r in rows],
        "bug_found_at": [found_at.get(r["hash"]) for r in rows],
    }


def label_table(table: pa.Table, repo: Path) -> pa.Table:
    labels = compute_labels(table.to_pylist(), repo)
    for field in LABEL_FIELDS:
        table = table.append_column(field, pa.array(labels[field.name], type=field.type))
    return table


@dataclass(frozen=True)
class LabelResult:
    source: str
    slug: str
    status: Status
    n_commits: int = 0
    n_fixes: int = 0
    n_bug_introducing: int = 0
    path: Path | None = None
    error: str | None = None

    @property
    def label_rate(self) -> float:
        """Share of commits labeled bug-introducing."""
        return self.n_bug_introducing / self.n_commits if self.n_commits else 0.0


def summarize(source: str, status: Status, path: Path) -> LabelResult:
    t = pq.read_table(path, columns=["is_merge", "is_fix", "is_bug_introducing"]).to_pydict()
    non_merge = [not m for m in t["is_merge"]]
    return LabelResult(
        source,
        repo_slug(source),
        status,
        n_commits=sum(non_merge),
        n_fixes=sum(t["is_fix"]),
        n_bug_introducing=sum(
            b and nm for b, nm in zip(t["is_bug_introducing"], non_merge, strict=True)
        ),
        path=path,
    )


def label_repo(
    source: str, raw_dir: Path, clone_dir: Path, out_dir: Path, force: bool = False
) -> LabelResult:
    out = out_dir / f"{repo_slug(source)}.parquet"
    if out.exists() and not force:
        return summarize(source, "skipped", out)
    try:
        raw = pq.read_table(raw_output_path(raw_dir, source))
        write_table(label_table(raw, ensure_clone(source, clone_dir)), out)
    except Exception as exc:  # report per-repo failures, keep the batch going
        return LabelResult(
            source, repo_slug(source), "failed", error=f"{type(exc).__name__}: {exc}"
        )
    return summarize(source, "labeled", out)


def label_repos(
    sources: list[str],
    raw_dir: Path,
    clone_dir: Path,
    out_dir: Path,
    workers: int = 4,
    force: bool = False,
) -> list[LabelResult]:
    job = partial(label_repo, raw_dir=raw_dir, clone_dir=clone_dir, out_dir=out_dir, force=force)
    return map_repos(job, sources, workers)
