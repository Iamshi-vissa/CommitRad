"""Per-commit features for risk modeling.

Leakage rule: anything derived from *other* commits (author history, file churn,
time since previous commit) uses only commits with a strictly earlier timestamp,
and past bugginess only counts once the bug's fix exists (``bug_found_at < t``).
The training target ``is_bug_introducing`` is never read here.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from commitrad.mining import write_table
from commitrad.parallel import map_repos
from commitrad.repos import repo_slug
from commitrad.sentiment import Backend, SentimentScorer, get_scorer
from commitrad.workflow import has_coauthor, is_config_path, is_revert, is_test_path

NIGHT_START_HOUR, NIGHT_END_HOUR = 22, 6
WINDOW_DAYS = 30
SECONDS_PER_DAY = 86_400

TIME_FEATURES = ["hour", "weekday", "is_night", "is_weekend", "hours_since_prev_commit"]
DIFF_FEATURES = ["lines_added", "lines_deleted", "files_touched", "dirs_touched", "change_entropy"]
# Kamei et al. (2013) history features, computed from strictly earlier commits only.
HISTORY_FEATURES = ["file_churn_max", "file_churn_mean", "file_prior_authors", "file_age_days"]
# How the change landed and what it touches (git-derived "GitHub workflow" signals).
WORKFLOW_FEATURES = [
    "via_pr",
    "pr_size",
    "test_file_share",
    "touches_config",
    "is_revert",
    "has_coauthor",
]
AUTHOR_FEATURES = [
    "author_prior_commits",
    "author_tenure_days",
    "author_commit_rate_30d",
    "author_past_bug_rate",
]
MESSAGE_FEATURES = ["msg_length", "msg_word_count", "msg_vagueness", "msg_sentiment"]
FEATURE_COLUMNS = (
    TIME_FEATURES
    + DIFF_FEATURES
    + HISTORY_FEATURES
    + WORKFLOW_FEATURES
    + AUTHOR_FEATURES
    + MESSAGE_FEATURES
)
META_COLUMNS = [
    "repo",
    "hash",
    "author",
    "author_email",
    "timestamp",
    "tz_offset_minutes",
    "message",
    "bug_found_at",
]
TARGET = "is_bug_introducing"

GENERIC_WORDS = frozenset(
    "fix fixes fixed fixing stuff thing things update updates updated change changes changed "
    "wip misc minor tweak tweaks cleanup small some more it this that bug bugs issue work "
    "temp tmp commit edit edits various asdf done oops again try stuff a the and".split()
)
_WORD_RE = re.compile(r"[a-z0-9']+")


# --- helpers -------------------------------------------------------------------------------


def epoch_seconds(series: pd.Series) -> np.ndarray:
    """UTC timestamps -> float seconds since epoch (NaN for missing)."""
    utc = pd.to_datetime(series, utc=True)
    return (utc - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy(dtype=float)


def author_key(df: pd.DataFrame) -> pd.Series:
    email = df["author_email"].fillna("").str.strip().str.lower()
    return email.where(email != "", df["author"].fillna("").str.strip().str.lower())


def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


# --- time ----------------------------------------------------------------------------------


def local_time(df: pd.DataFrame) -> pd.Series:
    """Wall-clock time in the author's own timezone (tz-naive)."""
    utc = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    return utc + pd.to_timedelta(df["tz_offset_minutes"], unit="min")


def time_features(df: pd.DataFrame) -> pd.DataFrame:
    local = local_time(df)
    hour = local.dt.hour
    return pd.DataFrame(
        {
            "hour": hour,
            "weekday": local.dt.day_of_week,
            "is_night": ((hour >= NIGHT_START_HOUR) | (hour < NIGHT_END_HOUR)).astype(int),
            "is_weekend": (local.dt.day_of_week >= 5).astype(int),
        },
        index=df.index,
    )


# --- author history ------------------------------------------------------------------------


def _author_block(t: np.ndarray, found: np.ndarray) -> dict[str, np.ndarray]:
    """History features for one author's commits, ``t`` sorted ascending."""
    prior = np.searchsorted(t, t, side="left")  # commits strictly earlier
    window_start = np.searchsorted(t, t - WINDOW_DAYS * SECONDS_PER_DAY, side="left")
    prev_t = t[np.maximum(prior - 1, 0)]
    hours_since = np.where(prior > 0, (t - prev_t) / 3600.0, np.nan)

    is_bug = ~np.isnan(found)
    tb, fb = t[is_bug], found[is_bug]
    known = (tb[None, :] < t[:, None]) & (fb[None, :] < t[:, None])
    past_bugs = known.sum(axis=1)
    bug_rate = np.divide(past_bugs, prior, out=np.zeros(len(t)), where=prior > 0)
    first = t[0] if len(t) else 0.0
    tenure = np.where(prior > 0, (t - first) / SECONDS_PER_DAY, 0.0)
    return {
        "hours_since_prev_commit": hours_since,
        "author_prior_commits": prior.astype(float),
        "author_tenure_days": tenure,
        "author_commit_rate_30d": (prior - window_start) / WINDOW_DAYS,
        "author_past_bug_rate": bug_rate,
    }


def author_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Requires ``df`` sorted by timestamp. Uses ``bug_found_at``, never the label."""
    t = epoch_seconds(df["timestamp"])
    found = epoch_seconds(df["bug_found_at"])
    cols = ["hours_since_prev_commit", *AUTHOR_FEATURES]
    out = pd.DataFrame(np.nan, index=df.index, columns=cols)
    key = author_key(df)
    for idx in key.groupby(key, sort=False).indices.values():
        block = _author_block(t[idx], found[idx])
        for col in cols:
            out.iloc[idx, out.columns.get_loc(col)] = block[col]
    return out


# --- diff / churn --------------------------------------------------------------------------


def file_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per touched file, from strictly earlier commits: churn, distinct authors, age.

    * ``file_churn_max`` / ``file_churn_mean``: prior added+deleted lines of the files.
    * ``file_prior_authors`` (NDEV): distinct earlier authors across the touched files.
    * ``file_age_days`` (AGE): mean days since each touched file last changed (new files
      count as 0).

    Requires time-sorted ``df``; commits sharing a timestamp are all scored before any of
    them updates the running state.
    """
    churn: dict[str, int] = defaultdict(int)
    authors: dict[str, set[str]] = defaultdict(set)
    last_change: dict[str, float] = {}
    n = len(df)
    out = {c: np.zeros(n) for c in HISTORY_FEATURES}
    paths, added, deleted = (
        df["modified_files"].tolist(),
        df["file_lines_added"].tolist(),
        df["file_lines_deleted"].tolist(),
    )
    who = author_key(df).tolist()
    t = epoch_seconds(df["timestamp"])
    start = 0
    while start < n:
        end = start
        while end < n and t[end] == t[start]:
            end += 1
        for i in range(start, end):
            files = list(paths[i])
            if not files:
                continue
            prior = [churn[p] for p in files]
            out["file_churn_max"][i], out["file_churn_mean"][i] = (
                max(prior),
                sum(prior) / len(prior),
            )
            out["file_prior_authors"][i] = len(set().union(*(authors[p] for p in files)))
            ages = [
                (t[i] - last_change[p]) / SECONDS_PER_DAY if p in last_change else 0.0
                for p in files
            ]
            out["file_age_days"][i] = sum(ages) / len(ages)
        for i in range(start, end):
            for p, a, d in zip(paths[i], added[i], deleted[i], strict=True):
                churn[p] += int(a) + int(d)
                authors[p].add(who[i])
                last_change[p] = t[i]
        start = end
    return pd.DataFrame(out, index=df.index)


def change_entropy(lines: Sequence[int]) -> float:
    """Normalised Shannon entropy of how a change's lines spread over its files (0..1)."""
    total = sum(lines)
    if len(lines) < 2 or total <= 0:
        return 0.0
    p = np.asarray([x / total for x in lines if x > 0])
    return float(-(p * np.log2(p)).sum() / np.log2(len(lines)))


def top_dir(path: str) -> str:
    parts = path.split("/")
    return parts[0] if len(parts) > 1 else "."


def diff_features(df: pd.DataFrame) -> pd.DataFrame:
    lines = [
        [int(a) + int(d) for a, d in zip(fa, fd, strict=True)]
        for fa, fd in zip(df["file_lines_added"], df["file_lines_deleted"], strict=True)
    ]
    base = pd.DataFrame(
        {
            "lines_added": df["lines_added"],
            "lines_deleted": df["lines_deleted"],
            "files_touched": df["files_changed"],
            "dirs_touched": [len({top_dir(p) for p in ps}) for ps in df["modified_files"]],
            "change_entropy": [change_entropy(ls) for ls in lines],
        },
        index=df.index,
    )
    return base.join(file_history_features(df))


def workflow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-local signals about how the change landed and what kind of files it touches."""
    files = df["modified_files"].tolist()
    messages = df["message"].fillna("").tolist()
    via_pr = df["via_pr"] if "via_pr" in df else pd.Series(0, index=df.index)
    pr_size = df["pr_size"] if "pr_size" in df else pd.Series(0, index=df.index)
    return pd.DataFrame(
        {
            "via_pr": via_pr.fillna(0).astype(int),
            "pr_size": pr_size.fillna(0).astype(int),
            "test_file_share": [
                sum(is_test_path(p) for p in fs) / len(fs) if len(fs) else 0.0 for fs in files
            ],
            "touches_config": [int(any(is_config_path(p) for p in fs)) for fs in files],
            "is_revert": [int(is_revert(m)) for m in messages],
            "has_coauthor": [int(has_coauthor(m)) for m in messages],
        },
        index=df.index,
    )


# --- message -------------------------------------------------------------------------------


def vagueness(message: str) -> float:
    """0 (specific) .. 1 (vague): mixes subject shortness with share of generic words."""
    subject = message.strip().splitlines()[0].lower() if message.strip() else ""
    words = _WORD_RE.findall(subject)
    if not words:
        return 1.0
    shortness = max(0.0, 1.0 - (len(words) - 1) / 7)
    generic = sum(w in GENERIC_WORDS for w in words) / len(words)
    return 0.5 * shortness + 0.5 * generic


def message_features(messages: Sequence[str], scorer: SentimentScorer) -> pd.DataFrame:
    msgs = [m or "" for m in messages]
    return pd.DataFrame(
        {
            "msg_length": [len(m.strip()) for m in msgs],
            "msg_word_count": [len(m.split()) for m in msgs],
            "msg_vagueness": [vagueness(m) for m in msgs],
            "msg_sentiment": scorer.score(msgs),
        }
    )


# --- assembly ------------------------------------------------------------------------------


def build_features(labeled: pd.DataFrame, scorer: SentimentScorer, repo: str) -> pd.DataFrame:
    """Feature table for non-merge commits, sorted by time, with meta and target columns."""
    df = sort_chronologically(labeled[~labeled["is_merge"]])
    msg = message_features(df["message"].tolist(), scorer).set_axis(df.index)
    feats = pd.concat(
        [time_features(df), author_history_features(df), diff_features(df),
         workflow_features(df), msg],
        axis=1,
    )  # fmt: skip
    meta = df.reindex(columns=[c for c in META_COLUMNS if c != "repo"]).assign(repo=repo)
    target = df[TARGET].astype(bool) if TARGET in df else pd.Series(False, index=df.index)
    return pd.concat([meta[META_COLUMNS], feats[FEATURE_COLUMNS], target.rename(TARGET)], axis=1)


@dataclass(frozen=True)
class FeatureResult:
    source: str
    slug: str
    status: Literal["built", "skipped", "failed"]
    n_rows: int = 0
    sentiment: str = ""
    path: Path | None = None
    error: str | None = None


def featurize_repo(
    source: str,
    labeled_dir: Path,
    out_dir: Path,
    sentiment: Backend = "auto",
    force: bool = False,
) -> FeatureResult:
    slug = repo_slug(source)
    out = out_dir / f"{slug}.parquet"
    if out.exists() and not force:
        return FeatureResult(source, slug, "skipped", pq.read_metadata(out).num_rows, path=out)
    try:
        labeled = pq.read_table(labeled_dir / f"{slug}.parquet").to_pandas()
        scorer = get_scorer(sentiment)
        table = pa.Table.from_pandas(build_features(labeled, scorer, slug), preserve_index=False)
        n = write_table(table, out)
    except Exception as exc:  # report per-repo failures, keep the batch going
        return FeatureResult(source, slug, "failed", error=f"{type(exc).__name__}: {exc}")
    return FeatureResult(source, slug, "built", n, scorer.name, out)


def featurize_repos(
    sources: list[str],
    labeled_dir: Path,
    out_dir: Path,
    workers: int = 4,
    sentiment: Backend = "auto",
    force: bool = False,
) -> list[FeatureResult]:
    job = partial(
        featurize_repo, labeled_dir=labeled_dir, out_dir=out_dir, sentiment=sentiment, force=force
    )
    return map_repos(job, sources, workers)


def load_features(features_dir: Path) -> pd.DataFrame:
    """All repos' feature tables concatenated, in global time order."""
    frames = [pd.read_parquet(p) for p in sorted(features_dir.glob("*.parquet"))]
    if not frames:
        raise FileNotFoundError(f"No feature files in {features_dir}")
    return sort_chronologically(pd.concat(frames, ignore_index=True))
