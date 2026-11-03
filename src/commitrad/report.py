"""Render the evaluation report (reports/eval.md) from a training run."""

from __future__ import annotations

import pandas as pd

from commitrad.model import TrainResult

FEATURE_NOTES = {
    "hour": "Hour of day (author local time)",
    "weekday": "Day of week (0 = Monday)",
    "is_night": "Committed 22:00-06:00 local",
    "is_weekend": "Committed on Sat/Sun local",
    "hours_since_prev_commit": "Gap since the author's previous commit",
    "lines_added": "Lines added",
    "lines_deleted": "Lines deleted",
    "files_touched": "Files touched",
    "file_churn_max": "Highest prior churn among touched files",
    "file_churn_mean": "Mean prior churn of touched files",
    "author_prior_commits": "Author's prior commits in this repo",
    "author_tenure_days": "Days since the author's first commit here",
    "dirs_touched": "Top-level directories touched",
    "change_entropy": "How evenly the change spreads over its files",
    "file_prior_authors": "Distinct earlier authors of the touched files",
    "file_age_days": "Mean days since touched files last changed",
    "via_pr": "Landed through a pull request",
    "pr_size": "Commits in its pull request",
    "test_file_share": "Share of touched files that are tests",
    "touches_config": "Touches build, CI or config files",
    "is_revert": "Reverts an earlier commit",
    "has_coauthor": "Has a Co-authored-by trailer",
    "author_commit_rate_30d": "Author's commits/day over the last 30 days",
    "author_past_bug_rate": "Share of author's past commits already known to be buggy",
    "msg_length": "Message length (chars)",
    "msg_word_count": "Message word count",
    "msg_vagueness": "Message vagueness (0 specific .. 1 vague)",
    "msg_sentiment": "Message sentiment (-1 .. 1)",
}


def _span(df: pd.DataFrame) -> str:
    return f"{df['timestamp'].min():%Y-%m-%d} → {df['timestamp'].max():%Y-%m-%d}"


def _split_table(result: TrainResult) -> list[str]:
    s, labels = result.split, result.labels
    lines = ["| split | commits | date range | positive rate |", "|---|---:|---|---:|"]
    for name, part in (("train", s.train), ("validation", s.val), ("test", s.test)):
        y = labels["val" if name == "validation" else name]
        lines.append(f"| {name} | {len(part):,} | {_span(part)} | {y.mean():.1%} |")
    return lines


def _metric_table(result: TrainResult) -> list[str]:
    lines = [
        "| split | model | PR-AUC | ROC-AUC | precision@top10% | random PR-AUC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split_name, by_model in result.metrics.items():
        for model, m in by_model.items():
            lines.append(
                f"| {split_name} | {model} | {m['pr_auc']:.3f} | {m['roc_auc']:.3f} | "
                f"{m['precision_at_10pct']:.3f} | {m['positive_rate']:.3f} |"
            )
    return lines


def _importance_table(result: TrainResult) -> list[str]:
    lines = ["| rank | feature | mean abs SHAP | effect of higher values |", "|---:|---|---:|---|"]
    for i, row in result.importance.iterrows():
        d = row["direction"]
        effect = (
            "n/a"
            if pd.isna(d)
            else ("↑ riskier" if d > 0.1 else "↓ safer" if d < -0.1 else "mixed")
        )
        note = FEATURE_NOTES.get(row["feature"], row["feature"])
        lines.append(
            f"| {i + 1} | `{row['feature']}`: {note} | {row['mean_abs_shap']:.3f} | {effect} |"
        )
    return lines


def _ci(m: dict[str, float], key: str) -> str:
    lo, hi = m.get(f"{key}_lo"), m.get(f"{key}_hi")
    return f"{m[key]:.3f} [{lo:.3f}, {hi:.3f}]" if lo is not None else f"{m[key]:.3f}"


def _headline(result: TrainResult) -> list[str]:
    t, base = result.metrics["test"]["lightgbm"], result.metrics["test"]["size_baseline"]
    lift = t["pr_auc"] / t["positive_rate"]
    return [
        f"On the held-out **test** period ({_span(result.split.test)}; {t['n']:,} commits, "
        f"{t['positives']} bug-introducing):",
        "",
        f"- **PR-AUC {_ci(t, 'pr_auc')}**: {lift:.1f}x the random baseline "
        f"({t['positive_rate']:.3f}).",
        f"- **ROC-AUC {_ci(t, 'roc_auc')}**.",
        f"- Diff-size heuristic: PR-AUC {_ci(base, 'pr_auc')}, ROC-AUC {_ci(base, 'roc_auc')}.",
        "",
        "Brackets are 95% bootstrap intervals over test commits. With this few test "
        "positives they are wide; differences inside them are not meaningful.",
    ]


def _label_section(result: TrainResult) -> list[str]:
    if result.label_mode == "strict":
        note = (
            "Train labels only count bugs whose fix landed before the validation cutoff, and "
            "validation labels only fixes before the test cutoff, so no training signal comes "
            "from a fix in a later split. Test uses final SZZ labels as ground truth."
        )
    else:
        note = "All splits use final SZZ labels (optimistic: train labels may use later fixes)."
    lines = [f"**Label timing ({result.label_mode}).** {note}", ""]
    for mode, m in result.sensitivity.items():
        lines.append(
            f"Sensitivity: the same pipeline trained with `{mode}` labels scores test PR-AUC "
            f"{m['pr_auc']:.3f} / ROC-AUC {m['roc_auc']:.3f}."
        )
    return lines


def render_report(result: TrainResult, repos: list[str]) -> str:
    s, p = result.split, result.params
    parts = [
        "# CommitRad: model evaluation",
        "",
        f"LightGBM on `is_bug_introducing` (SZZ) over {len(repos)} repos: "
        + ", ".join(f"`{r}`" for r in repos)
        + ".",
        "",
        "## Headline",
        "",
        *_headline(result),
        "",
        "## Time-based split",
        "",
        "All repos are pooled and ordered by commit time. Oldest 70% train, next 15% "
        "validation (early stopping), last 15% test, using timestamp cutoffs so commits at "
        "the same instant never straddle splits. No random splitting anywhere.",
        "",
        *_split_table(result),
        "",
        f"Cutoffs: validation starts {s.val_start:%Y-%m-%d %H:%M} UTC, test starts "
        f"{s.test_start:%Y-%m-%d %H:%M} UTC.",
        "",
        *_label_section(result),
        "",
        "## Metrics",
        "",
        *_metric_table(result),
        "",
        "PR-AUC is the primary metric under class imbalance; a random ranker scores the "
        "positive rate.",
        "",
        "## Training setup",
        "",
        f"- Class imbalance: `scale_pos_weight` = {p['scale_pos_weight']:.2f} "
        "(negatives / positives in train).",
        f"- Ensemble of {p['n_seeds']} seeds averaged in log-odds; early stopping on validation "
        f"ROC-AUC (best iterations {result.model.best_iterations}).",
        f"- `num_leaves={p['num_leaves']}`, `learning_rate={p['learning_rate']}`, "
        f"`min_data_in_leaf={p['min_data_in_leaf']}`, `feature_fraction={p['feature_fraction']}`.",
        "- Scores are class-weighted, so use them to rank commits, not as calibrated "
        "probabilities.",
        "",
        "## What drives predictions (SHAP, test set)",
        "",
        'Exact TreeSHAP contributions in log-odds (LightGBM `pred_contrib`). "Effect" is '
        "the Spearman sign between a feature's value and its SHAP value.",
        "",
        *_importance_table(result),
        "",
        "## Caveats",
        "",
        "- **Labels are noisy.** SZZ misses fixes that only add lines, and it can blame "
        "refactors or test code. The metrics measure agreement with SZZ, not true bugs.",
        "- **Recent commits are under-labeled.** Bugs in the test period may not be fixed "
        'yet, so some test "negatives" are undiscovered positives.',
        "- **The test set was viewed during development.** It was scored three times while "
        "the early-stopping metric changed (PR-AUC, then log-loss, which stopped at 1 tree "
        "under class weighting, then ROC-AUC). No hyperparameters were tuned on it, but "
        "treat the headline as mildly optimistic.",
        "- **Small sample.** A few thousand commits from 5 Python libraries; expect wide "
        "variance and limited transfer to other codebases.",
        "",
    ]
    return "\n".join(parts)
