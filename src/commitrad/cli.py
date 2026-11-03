"""Command-line entry point: ``commitrad <command> ...``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from commitrad.features import FeatureResult, featurize_repos, load_features
from commitrad.mining import MineResult, mine_repos
from commitrad.repos import read_repo_list, repo_slug
from commitrad.szz import LabelResult, label_repos

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_CLONE_DIR = Path("data/repos")
DEFAULT_LABELED_DIR = Path("data/labeled")
DEFAULT_FEATURES_DIR = Path("data/features")
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_REPORT = Path("reports/eval.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="commitrad", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    mine = sub.add_parser("mine", help="Mine commit history into Parquet (one file per repo).")
    mine.add_argument("repo_list", type=Path, help="Text file with one repo URL or path per line.")
    mine.add_argument("--out", type=Path, default=DEFAULT_RAW_DIR, help="Parquet output dir.")
    mine.add_argument("--clones", type=Path, default=DEFAULT_CLONE_DIR, help="Clone cache dir.")
    mine.add_argument("--workers", type=int, default=4, help="Parallel worker processes.")
    mine.add_argument("--force", action="store_true", help="Re-mine repos already on disk.")

    label = sub.add_parser("label", help="Add SZZ bug-introducing labels to mined commits.")
    label.add_argument("repo_list", type=Path, help="Same repo list used for `mine`.")
    label.add_argument("--raw", type=Path, default=DEFAULT_RAW_DIR, help="Mined Parquet dir.")
    label.add_argument("--clones", type=Path, default=DEFAULT_CLONE_DIR, help="Clone cache dir.")
    label.add_argument("--out", type=Path, default=DEFAULT_LABELED_DIR, help="Labeled output dir.")
    label.add_argument("--workers", type=int, default=4, help="Parallel worker processes.")
    label.add_argument("--force", action="store_true", help="Re-label repos already on disk.")

    feats = sub.add_parser("features", help="Build per-commit features from labeled commits.")
    feats.add_argument("repo_list", type=Path, help="Same repo list used for `mine`.")
    feats.add_argument("--labeled", type=Path, default=DEFAULT_LABELED_DIR, help="Labeled dir.")
    feats.add_argument("--out", type=Path, default=DEFAULT_FEATURES_DIR, help="Feature output dir.")
    feats.add_argument("--workers", type=int, default=4, help="Parallel worker processes.")
    feats.add_argument(
        "--sentiment",
        choices=["auto", "lexicon", "hf"],
        default="auto",
        help="Sentiment backend (auto: HuggingFace if available, else lexicon).",
    )
    feats.add_argument("--force", action="store_true", help="Rebuild repos already on disk.")

    train = sub.add_parser("train", help="Train the risk model (time split) and write the report.")
    train.add_argument("--features", type=Path, default=DEFAULT_FEATURES_DIR)
    train.add_argument("--models", type=Path, default=DEFAULT_MODEL_DIR)
    train.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    train.add_argument(
        "--labels",
        choices=["strict", "final"],
        default="strict",
        help="strict: train/val labels only use fixes known before the next split starts.",
    )
    return parser


def format_result(r: MineResult) -> str:
    detail = r.error if r.status == "failed" else f"{r.n_commits} commits"
    return f"{r.status:<8} {r.slug:<40} {detail}"


def format_label(r: LabelResult) -> str:
    if r.status == "failed":
        return f"{r.status:<8} {r.slug:<32} {r.error}"
    return (
        f"{r.status:<8} {r.slug:<32} commits={r.n_commits:<5} fixes={r.n_fixes:<4} "
        f"bug_introducing={r.n_bug_introducing:<4} rate={r.label_rate:6.1%}"
    )


def load_sources(repo_list: Path) -> list[str]:
    sources = read_repo_list(repo_list)
    for s in sources:
        repo_slug(s)  # fail fast on malformed entries before spawning workers
    return sources


def cmd_label(args: argparse.Namespace) -> int:
    sources = load_sources(args.repo_list)
    results = label_repos(sources, args.raw, args.clones, args.out, args.workers, args.force)
    for r in results:
        print(format_label(r))
    ok = [r for r in results if r.status != "failed"]
    total, bugs = sum(r.n_commits for r in ok), sum(r.n_bug_introducing for r in ok)
    if total:
        print(f"\noverall label rate: {bugs}/{total} = {bugs / total:.1%} (non-merge commits)")
    return 1 if len(ok) < len(results) else 0


def format_features(r: FeatureResult) -> str:
    if r.status == "failed":
        return f"{r.status:<8} {r.slug:<32} {r.error}"
    backend = f" sentiment={r.sentiment}" if r.sentiment else ""
    return f"{r.status:<8} {r.slug:<32} rows={r.n_rows}{backend}"


def cmd_features(args: argparse.Namespace) -> int:
    sources = load_sources(args.repo_list)
    results = featurize_repos(
        sources, args.labeled, args.out, args.workers, args.sentiment, args.force
    )
    for r in results:
        print(format_features(r))
    return 1 if any(r.status == "failed" for r in results) else 0


def cmd_train(args: argparse.Namespace) -> int:
    from commitrad.model import run_training, save_model  # heavy import, only when training
    from commitrad.report import render_report

    features = load_features(args.features)
    result = run_training(features, args.labels)
    path = save_model(result, args.models)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(result, sorted(features["repo"].unique())))
    for split_name, by_model in result.metrics.items():
        for model, m in by_model.items():
            print(
                f"{split_name:<5} {model:<14} PR-AUC={m['pr_auc']:.3f} ROC-AUC={m['roc_auc']:.3f} "
                f"P@10%={m['precision_at_10pct']:.3f} (random PR-AUC={m['positive_rate']:.3f})"
            )
    t = result.metrics["test"]["lightgbm"]
    print(f"test PR-AUC 95% CI [{t['pr_auc_lo']:.3f}, {t['pr_auc_hi']:.3f}]")
    for mode, m in result.sensitivity.items():
        print(
            f"sensitivity ({mode} labels): test PR-AUC={m['pr_auc']:.3f} ROC-AUC={m['roc_auc']:.3f}"
        )
    print(f"\nmodel -> {path}\nreport -> {args.report}")
    return 0


def cmd_mine(args: argparse.Namespace) -> int:
    sources = load_sources(args.repo_list)
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
        if args.command == "label":
            return cmd_label(args)
        if args.command == "features":
            return cmd_features(args)
        if args.command == "train":
            return cmd_train(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
