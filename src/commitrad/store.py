"""DuckDB storage for per-commit risk and weekly strain."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

DEFAULT_DB = Path("data/commitrad.duckdb")

SCHEMA = {
    "strain": """
        CREATE TABLE IF NOT EXISTS strain (
            repo VARCHAR, author VARCHAR, author_email VARCHAR, week DATE,
            n_commits INTEGER, off_hours DOUBLE, risk DOUBLE, burstiness DOUBLE,
            negativity DOUBLE, strain DOUBLE,
            PRIMARY KEY (repo, author_email, week)
        )""",
    "commit_risk": """
        CREATE TABLE IF NOT EXISTS commit_risk (
            repo VARCHAR, hash VARCHAR, author VARCHAR, author_email VARCHAR,
            timestamp TIMESTAMPTZ, message VARCHAR, risk DOUBLE, risk_pct DOUBLE,
            PRIMARY KEY (repo, hash)
        )""",
}
COLUMNS = {
    "strain": [
        "repo", "author", "author_email", "week", "n_commits",
        "off_hours", "risk", "burstiness", "negativity", "strain",
    ],
    "commit_risk": [
        "repo", "hash", "author", "author_email", "timestamp", "message", "risk", "risk_pct",
    ],
}  # fmt: skip


def connect(path: Path = DEFAULT_DB, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    if not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    if not read_only:
        for ddl in SCHEMA.values():
            con.execute(ddl)
    return con


def replace_repos(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> int:
    """Replace all rows of the repos present in ``df`` (other repos are left alone)."""
    rows = df[COLUMNS[table]]
    repos = sorted(rows["repo"].unique().tolist())
    con.execute("BEGIN")
    try:
        if repos:
            con.execute(f"DELETE FROM {table} WHERE repo IN (SELECT unnest(?))", [repos])
        con.register("incoming", rows)
        con.execute(f"INSERT INTO {table} SELECT * FROM incoming")
        con.unregister("incoming")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(rows)


def read_table(con: duckdb.DuckDBPyConnection, table: str, repo: str | None = None) -> pd.DataFrame:
    if table not in SCHEMA:
        raise ValueError(f"Unknown table {table!r}")
    if repo is None:
        return con.execute(f"SELECT * FROM {table}").df()
    return con.execute(f"SELECT * FROM {table} WHERE repo = ?", [repo]).df()
