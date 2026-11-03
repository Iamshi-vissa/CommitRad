"""Command-line entry point: ``commitrad <command> ...``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from commitrad.mining import MineResult, mine_repos
from commitrad.repos import read_repo_list, repo_slug

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_CLONE_DIR = Path("data/repos")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="commitrad", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    mine = sub.add_parser("mine", help="Mine commit history into Parquet (one file per repo).")
    mine.add_argument("repo_list", type=Path, help="Text file with one repo URL or path per line.")
    mine.add_argument("--out", type=Path, default=DEFAULT_RAW_DIR, help="Parquet output dir.")
    mine.add_argument("--clones", type=Path, default=DEFAULT_CLONE_DIR, help="Clone cache dir.")
    mine.add_argument("--workers", type=int, default=4, help="Parallel worker processes.")
    mine.add_argument("--force", action="store_true", help="Re-mine repos already on disk.")
    return parser


def format_result(r: MineResult) -> str:
    detail = r.error if r.status == "failed" else f"{r.n_commits} commits"
    return f"{r.status:<8} {r.slug:<40} {detail}"


def cmd_mine(args: argparse.Namespace) -> int:
    sources = read_repo_list(args.repo_list)
    for s in sources:
        repo_slug(s)  # fail fast on malformed entries before spawning workers
    results = mine_repos(sources, args.out, args.clones, args.workers, args.force)
    for r in results:
        print(format_result(r))
    failed = sum(r.status == "failed" for r in results)
    print(f"\n{len(results) - failed}/{len(results)} repos OK -> {args.out}")
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "mine":
            return cmd_mine(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
