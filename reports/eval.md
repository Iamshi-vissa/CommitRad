# CommitRad: model evaluation

LightGBM on `is_bug_introducing` (SZZ) over 16 repos: `BurntSushi__ripgrep`, `Textualize__rich`, `axios__axios`, `encode__httpx`, `expressjs__express`, `gin-gonic__gin`, `hynek__structlog`, `jd__tenacity`, `pallets__itsdangerous`, `pallets__jinja`, `pallets__markupsafe`, `python-attrs__attrs`, `python-humanize__humanize`, `spf13__cobra`, `tartley__colorama`, `tj__commander.js`.

## Headline

On the held-out **test** period (2023-11-28 → 2026-11-03; 4,108 commits, 352 bug-introducing):

- **PR-AUC 0.334 [0.293, 0.385]**: 3.9x the random baseline (0.086).
- **ROC-AUC 0.849 [0.831, 0.867]**.
- Diff-size heuristic: PR-AUC 0.213 [0.184, 0.253], ROC-AUC 0.764 [0.738, 0.786].

Brackets are 95% bootstrap intervals over test commits. With this few test positives they are wide; differences inside them are not meaningful.

## Time-based split

All repos are pooled and ordered by commit time. Oldest 70% train, next 15% validation (early stopping), last 15% test, using timestamp cutoffs so commits at the same instant never straddle splits. No random splitting anywhere.

| split | commits | date range | positive rate |
|---|---:|---|---:|
| train | 19,168 | 2007-02-26 → 2021-12-29 | 18.2% |
| validation | 4,108 | 2021-12-29 → 2023-11-28 | 8.1% |
| test | 4,108 | 2023-11-28 → 2026-11-03 | 8.6% |

Cutoffs: validation starts 2021-12-29 07:24 UTC, test starts 2023-11-28 21:23 UTC.

**Label timing (strict).** Train labels only count bugs whose fix landed before the validation cutoff, and validation labels only fixes before the test cutoff, so no training signal comes from a fix in a later split. Test uses final SZZ labels as ground truth.

Sensitivity: the same pipeline trained with `final` labels scores test PR-AUC 0.336 / ROC-AUC 0.853.

## Metrics

| split | model | PR-AUC | ROC-AUC | precision@top10% | random PR-AUC |
|---|---|---:|---:|---:|---:|
| val | lightgbm | 0.303 | 0.831 | 0.321 | 0.081 |
| val | size_baseline | 0.199 | 0.727 | 0.238 | 0.081 |
| test | lightgbm | 0.334 | 0.849 | 0.345 | 0.086 |
| test | size_baseline | 0.213 | 0.764 | 0.241 | 0.086 |

PR-AUC is the primary metric under class imbalance; a random ranker scores the positive rate.

## Training setup

- Class imbalance: `scale_pos_weight` = 4.50 (negatives / positives in train).
- Ensemble of 5 seeds averaged in log-odds; early stopping on validation ROC-AUC (best iterations [113, 140, 123, 156, 114]).
- `num_leaves=15`, `learning_rate=0.03`, `min_data_in_leaf=20`, `feature_fraction=0.8`.
- Scores are class-weighted, so use them to rank commits, not as calibrated probabilities.

## What drives predictions (SHAP, test set)

Exact TreeSHAP contributions in log-odds (LightGBM `pred_contrib`). "Effect" is the Spearman sign between a feature's value and its SHAP value.

| rank | feature | mean abs SHAP | effect of higher values |
|---:|---|---:|---|
| 1 | `lines_added`: Lines added | 0.713 | ↑ riskier |
| 2 | `touches_config`: Touches build, CI or config files | 0.442 | ↓ safer |
| 3 | `author_tenure_days`: Days since the author's first commit here | 0.342 | ↓ safer |
| 4 | `test_file_share`: Share of touched files that are tests | 0.212 | ↑ riskier |
| 5 | `lines_deleted`: Lines deleted | 0.156 | ↑ riskier |
| 6 | `change_entropy`: How evenly the change spreads over its files | 0.139 | ↑ riskier |
| 7 | `msg_length`: Message length (chars) | 0.115 | ↑ riskier |
| 8 | `author_past_bug_rate`: Share of author's past commits already known to be buggy | 0.103 | ↑ riskier |
| 9 | `files_touched`: Files touched | 0.091 | ↑ riskier |
| 10 | `author_prior_commits`: Author's prior commits in this repo | 0.069 | ↓ safer |
| 11 | `msg_word_count`: Message word count | 0.065 | ↑ riskier |
| 12 | `file_prior_authors`: Distinct earlier authors of the touched files | 0.054 | ↓ safer |
| 13 | `file_age_days`: Mean days since touched files last changed | 0.044 | ↓ safer |
| 14 | `via_pr`: Landed through a pull request | 0.042 | ↓ safer |
| 15 | `has_coauthor`: Has a Co-authored-by trailer | 0.038 | ↓ safer |
| 16 | `file_churn_mean`: Mean prior churn of touched files | 0.036 | ↓ safer |
| 17 | `file_churn_max`: Highest prior churn among touched files | 0.036 | ↑ riskier |
| 18 | `dirs_touched`: Top-level directories touched | 0.027 | ↑ riskier |
| 19 | `hours_since_prev_commit`: Gap since the author's previous commit | 0.023 | ↑ riskier |
| 20 | `pr_size`: Commits in its pull request | 0.013 | ↓ safer |
| 21 | `hour`: Hour of day (author local time) | 0.011 | ↑ riskier |
| 22 | `msg_vagueness`: Message vagueness (0 specific .. 1 vague) | 0.008 | mixed |
| 23 | `author_commit_rate_30d`: Author's commits/day over the last 30 days | 0.007 | ↓ safer |
| 24 | `weekday`: Day of week (0 = Monday) | 0.006 | ↑ riskier |
| 25 | `is_weekend`: Committed on Sat/Sun local | 0.000 | ↑ riskier |
| 26 | `msg_sentiment`: Message sentiment (-1 .. 1) | 0.000 | ↑ riskier |
| 27 | `is_night`: Committed 22:00-06:00 local | 0.000 | n/a |
| 28 | `is_revert`: Reverts an earlier commit | 0.000 | n/a |

## Caveats

- **Labels are noisy.** SZZ misses fixes that only add lines, and it can blame refactors or test code. The metrics measure agreement with SZZ, not true bugs.
- **Recent commits are under-labeled.** Bugs in the test period may not be fixed yet, so some test "negatives" are undiscovered positives.
- **The test set was viewed during development.** It was scored three times while the early-stopping metric changed (PR-AUC, then log-loss, which stopped at 1 tree under class weighting, then ROC-AUC). No hyperparameters were tuned on it, but treat the headline as mildly optimistic.
- **Small sample.** A few thousand commits from 5 Python libraries; expect wide variance and limited transfer to other codebases.
