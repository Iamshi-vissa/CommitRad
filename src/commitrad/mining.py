"""Mine commit metadata from git repos into one Parquet file per repo."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC
from functools import partial
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydriller import Commit, Repository

from commitrad.parallel import map_repos
from commitrad.repos import ensure_clone, repo_slug
from commitrad.workflow import pr_membership, workflow_columns

SCHEMA = pa.schema(
    [
        ("hash", pa.string()),
        ("parents", pa.list_(pa.string())),
        ("is_merge", pa.bool_()),
        ("author", pa.string()),
        ("author_email", pa.string()),
        ("timestamp", pa.timestamp("s", tz="UTC")),
        ("tz_offset_minutes", pa.int32()),
        ("message", pa.string()),
        ("lines_added", pa.int64()),
        ("lines_deleted", pa.int64()),
        ("files_changed", pa.int32()),
        ("modified_files", pa.list_(pa.string())),
        ("file_lines_added", pa.list_(pa.int64())),
        ("file_lines_deleted", pa.list_(pa.int64())),
        ("via_pr", pa.int8()),
        ("pr_size", pa.int32()),
    ]
)

Status = Literal["mined", "skipped", "failed"]


@dataclass(frozen=True)
class MineResult:
    source: str
    slug: str
    status: Status
    n_commits: int = 0
    path: Path | None = None
    error: str | None = None


def output_path(out_dir: Path, source: str) -> Path:
    return out_dir / f"{repo_slug(source)}.parquet"


def commit_to_record(commit: Commit) -> dict[str, Any]:
    """Flatten a PyDriller commit into a row matching ``SCHEMA``."""
    files = commit.modified_files
    paths = [f.new_path or f.old_path or "" for f in files]
    added = [f.added_lines for f in files]
    deleted = [f.deleted_lines for f in files]
    when = commit.author_date
    offset = when.utcoffset()
    return {
        "hash": commit.hash,
        "parents": list(commit.parents),
        "is_merge": commit.merge,
        "author": commit.author.name,
        "author_email": commit.author.email,
        "timestamp": when.astimezone(UTC),
        "tz_offset_minutes": int(offset.total_seconds() // 60) if offset else 0,
        "message": commit.msg,
        "lines_added": sum(added),
        "lines_deleted": sum(deleted),
        "files_changed": len(files),
        "modified_files": paths,
        "file_lines_added": added,
        "file_lines_deleted": deleted,
    }


def iter_records(repo_path: Path) -> Iterator[dict[str, Any]]:
    for commit in Repository(str(repo_path), order="topo-order").traverse_commits():
        yield commit_to_record(commit)


def write_table(table: pa.Table, path: Path) -> int:
    """Write a table atomically (temp file + rename) and return its row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)
    return table.num_rows


def write_parquet(records: Iterable[dict[str, Any]], path: Path) -> int:
    return write_table(pa.Table.from_pylist(list(records), schema=SCHEMA), path)


def mine_table(repo_path: Path) -> pa.Table:
    """All commits of ``repo_path`` as a table matching ``SCHEMA``, incl. PR workflow columns."""
    records = list(iter_records(repo_path))
    members = pr_membership(repo_path) if records else {}
    cols = workflow_columns([r["hash"] for r in records], [r["message"] for r in records], members)
    for i, r in enumerate(records):
        r["via_pr"], r["pr_size"] = cols["via_pr"][i], cols["pr_size"][i]
    return pa.Table.from_pylist(records, schema=SCHEMA)


def mine_repo(source: str, out_dir: Path, clone_dir: Path, force: bool = False) -> MineResult:
    """Mine one repo. Skips if its Parquet already exists (unless ``force``)."""
    slug = repo_slug(source)
    out = output_path(out_dir, source)
    if out.exists() and not force:
        return MineResult(source, slug, "skipped", pq.read_metadata(out).num_rows, out)
    try:
        repo_path = ensure_clone(source, clone_dir)
        n = write_table(mine_table(repo_path), out)
    except Exception as exc:  # report per-repo failures, keep the batch going
        return MineResult(source, slug, "failed", error=f"{type(exc).__name__}: {exc}")
    return MineResult(source, slug, "mined", n, out)


def mine_repos(
    sources: list[str],
    out_dir: Path,
    clone_dir: Path,
    workers: int = 4,
    force: bool = False,
) -> list[MineResult]:
    """Mine repos in parallel; results come back in input order."""
    job = partial(mine_repo, out_dir=out_dir, clone_dir=clone_dir, force=force)
    return map_repos(job, sources, workers)
