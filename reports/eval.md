# CommitRad: model evaluation

LightGBM on `is_bug_introducing` (SZZ) over 5 repos: `jd__tenacity`, `pallets__itsdangerous`, `pallets__markupsafe`, `python-humanize__humanize`, `tartley__colorama`.

## Headline

On the held-out **test** period (2024-09-27 → 2026-11-02; 387 commits, 24 bug-introducing):

- **PR-AUC 0.221 [0.098, 0.371]**: 3.6x the random baseline (0.062).
- **ROC-AUC 0.765 [0.673, 0.843]**.
- Diff-size heuristic: PR-AUC 0.130 [0.078, 0.245], ROC-AUC 0.726 [0.626, 0.810].

Brackets are 95% bootstrap intervals over test commits. With this few test positives they are wide; differences inside them are not meaningful.

## Time-based split

All repos are pooled and ordered by commit time. Oldest 70% train, next 15% validation (early stopping), last 15% test, using timestamp cutoffs so commits at the same instant never straddle splits. No random splitting anywhere.

| split | commits | date range | positive rate |
|---|---:|---|---:|
| train | 1,804 | 2010-06-22 → 2023-01-08 | 11.7% |
| validation | 387 | 2023-01-08 → 2024-09-22 | 2.8% |
| test | 387 | 2024-09-27 → 2026-11-02 | 6.2% |

Cutoffs: validation starts 2023-01-08 17:23 UTC, test starts 2024-09-27 05:58 UTC.

**Label timing (strict).** Train labels only count bugs whose fix landed before the validation cutoff, and validation labels only fixes before the test cutoff, so no training signal comes from a fix in a later split. Test uses final SZZ labels as ground truth.

Sensitivity: the same pipeline trained with `final` labels scores test PR-AUC 0.190 / ROC-AUC 0.778.

## Metrics

| split | model | PR-AUC | ROC-AUC | precision@top10% | random PR-AUC |
|---|---|---:|---:|---:|---:|
| val | lightgbm | 0.515 | 0.902 | 0.205 | 0.028 |
| val | size_baseline | 0.106 | 0.795 | 0.103 | 0.028 |
| test | lightgbm | 0.221 | 0.765 | 0.128 | 0.062 |
| test | size_baseline | 0.130 | 0.726 | 0.128 | 0.062 |

PR-AUC is the primary metric under class imbalance; a random ranker scores the positive rate.

## Training setup

- Class imbalance: `scale_pos_weight` = 7.55 (negatives / positives in train).
- Ensemble of 5 seeds averaged in log-odds; early stopping on validation ROC-AUC (best iterations [100, 4, 21, 51, 4]).
- `num_leaves=15`, `learning_rate=0.03`, `min_data_in_leaf=20`, `feature_fraction=0.8`.
- Scores are class-weighted, so use them to rank commits, not as calibrated probabilities.

## What drives predictions (SHAP, test set)

Exact TreeSHAP contributions in log-odds (LightGBM `pred_contrib`). "Effect" is the Spearman sign between a feature's value and its SHAP value.

| rank | feature | mean abs SHAP | effect of higher values |
|---:|---|---:|---|
| 1 | `lines_added`: Lines added | 0.436 | ↑ riskier |
| 2 | `author_prior_commits`: Author's prior commits in this repo | 0.292 | ↓ safer |
| 3 | `hour`: Hour of day (author local time) | 0.142 | ↑ riskier |
| 4 | `files_touched`: Files touched | 0.112 | ↑ riskier |
| 5 | `file_churn_mean`: Mean prior churn of touched files | 0.109 | ↑ riskier |
| 6 | `msg_length`: Message length (chars) | 0.087 | ↑ riskier |
| 7 | `file_churn_max`: Highest prior churn among touched files | 0.087 | ↑ riskier |
| 8 | `author_past_bug_rate`: Share of author's past commits already known to be buggy | 0.079 | ↑ riskier |
| 9 | `msg_vagueness`: Message vagueness (0 specific .. 1 vague) | 0.049 | ↓ safer |
| 10 | `msg_word_count`: Message word count | 0.048 | ↑ riskier |
| 11 | `author_commit_rate_30d`: Author's commits/day over the last 30 days | 0.023 | ↑ riskier |
| 12 | `hours_since_prev_commit`: Gap since the author's previous commit | 0.020 | ↑ riskier |
| 13 | `weekday`: Day of week (0 = Monday) | 0.019 | ↑ riskier |
| 14 | `lines_deleted`: Lines deleted | 0.018 | ↑ riskier |
| 15 | `msg_sentiment`: Message sentiment (-1 .. 1) | 0.001 | ↑ riskier |
| 16 | `is_night`: Committed 22:00-06:00 local | 0.000 | n/a |
| 17 | `is_weekend`: Committed on Sat/Sun local | 0.000 | n/a |

## Caveats

- **Labels are noisy.** SZZ misses fixes that only add lines, and it can blame refactors or test code. The metrics measure agreement with SZZ, not true bugs.
- **Recent commits are under-labeled.** Bugs in the test period may not be fixed yet, so some test "negatives" are undiscovered positives.
- **The test set was viewed during development.** It was scored three times while the early-stopping metric changed (PR-AUC, then log-loss, which stopped at 1 tree under class weighting, then ROC-AUC). No hyperparameters were tuned on it, but treat the headline as mildly optimistic.
- **Small sample.** A few thousand commits from 5 Python libraries; expect wide variance and limited transfer to other codebases.
