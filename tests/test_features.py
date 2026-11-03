from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from frames import commit, frame
from gitrepo import GitRepo

from commitrad.cli import main
from commitrad.features import (
    FEATURE_COLUMNS,
    META_COLUMNS,
    TARGET,
    build_features,
    load_features,
    vagueness,
)
from commitrad.sentiment import LexiconSentiment, get_scorer

SCORER = LexiconSentiment()


def feats(rows: list[dict[str, object]]) -> pd.DataFrame:
    return build_features(frame(rows), SCORER, "r").set_index("hash")


def test_time_features_use_author_local_time() -> None:
    # 2024-01-01 is a Monday. 04:30 UTC at UTC-5 is Sunday 23:30 local.
    f = feats(
        [
            commit("night_local", 4.5, tz_offset_minutes=-300),
            commit("day_utc", 12, author="bob"),
            commit("sat", 24 * 5 + 10, author="carol", tz_offset_minutes=60),
        ]
    )
    assert f.loc["night_local", ["hour", "weekday", "is_night", "is_weekend"]].tolist() == [
        23,
        6,
        1,
        1,
    ]
    assert f.loc["day_utc", ["hour", "is_night", "is_weekend"]].tolist() == [12, 0, 0]
    assert f.loc["sat", ["hour", "weekday", "is_weekend"]].tolist() == [11, 5, 1]


def test_hours_since_prev_and_30_day_rate() -> None:
    f = feats(
        [
            commit("c0", 0),
            commit("c1", 5),
            commit("c2", 24 * 30 + 5),  # exactly 30 days after c1 -> c1 still inside window
            commit("c3", 24 * 31 + 6),  # c1 now outside
            commit("other", 1, author="bob"),
        ]
    )
    assert np.isnan(f.loc["c0", "hours_since_prev_commit"])
    assert f.loc["c1", "hours_since_prev_commit"] == 5
    assert f.loc["c2", "author_commit_rate_30d"] == pytest.approx(1 / 30)
    assert f.loc["c3", "author_commit_rate_30d"] == pytest.approx(1 / 30)  # only c2
    assert f.loc["c3", "author_prior_commits"] == 3
    assert f.loc["other", "author_prior_commits"] == 0


def test_author_identity_is_case_insensitive_email() -> None:
    rows = [commit("a", 0), commit("b", 1)]
    rows[1]["author_email"] = "ALICE@X.ORG"
    assert feats(rows).loc["b", "author_prior_commits"] == 1


def test_file_churn_is_prior_history_of_touched_files() -> None:
    f = feats(
        [
            commit("c0", 0, files={"a.py": (10, 0), "b.py": (2, 2)}),
            commit("c1", 1, files={"a.py": (5, 5)}),
            commit("c2", 2, files={"a.py": (1, 0), "b.py": (1, 0), "new.py": (3, 0)}),
        ]
    )
    assert f.loc["c0", ["file_churn_max", "file_churn_mean"]].tolist() == [0, 0]
    assert f.loc["c1", "file_churn_max"] == 10
    assert f.loc["c2", "file_churn_max"] == 20
    assert f.loc["c2", "file_churn_mean"] == pytest.approx((20 + 4 + 0) / 3)
    assert f.loc["c2", ["lines_added", "lines_deleted", "files_touched"]].tolist() == [5, 0, 3]


def test_merges_are_dropped_and_output_is_time_sorted() -> None:
    out = build_features(
        frame([commit("late", 9), commit("m", 5, is_merge=True), commit("early", 1)]), SCORER, "r"
    )
    assert out["hash"].tolist() == ["early", "late"]
    assert list(out.columns) == META_COLUMNS + FEATURE_COLUMNS + [TARGET]
    assert (out["repo"] == "r").all()


@pytest.mark.parametrize(
    ("vague", "specific"),
    [
        ("fix stuff", "Fix race in retry wait when stop is None"),
        ("wip", "Add jitter option to exponential backoff"),
        ("update", "Update humanize.naturaltime to accept timedelta"),
    ],
)
def test_vagueness_ranks_generic_messages_higher(vague: str, specific: str) -> None:
    assert vagueness(vague) > 0.8 > 0.3 > vagueness(specific)


def test_vagueness_uses_subject_and_handles_empty() -> None:
    assert vagueness("") == 1.0
    assert vagueness("fix stuff\n\nLong detailed body explaining the root cause.") > 0.8


def test_message_length_and_sentiment() -> None:
    f = feats(
        [
            commit("neg", 0, message="ugh, stupid hack to get this broken thing working"),
            commit("pos", 1, message="Nice cleaner retry API, finally"),
            commit("neutral", 2, message="Fix crash in parser"),
        ]
    )
    assert f.loc["neg", "msg_sentiment"] < 0 < f.loc["pos", "msg_sentiment"]
    assert f.loc["neutral", "msg_sentiment"] == 0
    assert f.loc["neutral", ["msg_length", "msg_word_count"]].tolist() == [19, 4]


def test_auto_sentiment_falls_back_to_lexicon_without_transformers() -> None:
    try:
        import transformers  # noqa: F401
    except ImportError:
        assert get_scorer("auto").name == "lexicon"
        with pytest.raises(ImportError):
            get_scorer("hf")
    else:
        pytest.skip("transformers installed")


def test_cli_features_end_to_end(
    small_repo: GitRepo, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    repos_txt = tmp_path / "repos.txt"
    repos_txt.write_text(f"{small_repo.path}\n")
    raw, lab, out, clones = (str(tmp_path / d) for d in ("raw", "lab", "feat", "c"))
    assert main(["mine", str(repos_txt), "--out", raw, "--clones", clones]) == 0
    assert main(["label", str(repos_txt), "--raw", raw, "--out", lab, "--clones", clones]) == 0
    args = ["features", str(repos_txt), "--labeled", lab, "--out", out, "--sentiment", "lexicon"]
    assert main(args) == 0
    assert "built" in capsys.readouterr().out

    table = pq.read_table(Path(out) / "small_repo.parquet").to_pandas()
    assert len(table) == 3 and set(FEATURE_COLUMNS) <= set(table.columns)
    assert table["timestamp"].is_monotonic_increasing
    assert len(load_features(Path(out))) == 3
