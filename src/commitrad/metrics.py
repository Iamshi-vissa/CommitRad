"""Ranking metrics for imbalanced binary labels (tie-aware, numpy only)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def _check(y_true: ArrayLike, y_score: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(y_score, dtype=float)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("y_true and y_score must be 1-D and the same length")
    return y, s


def average_precision(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """PR-AUC as step-wise average precision; tied scores share one threshold."""
    y, s = _check(y_true, y_score)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    last_of_tie = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tps = np.cumsum(y)[last_of_tie]
    precision = tps / (last_of_tie + 1)
    recall = tps / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Probability a random positive outranks a random negative (ties count half)."""
    y, s = _check(y_true, y_score)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def precision_at_fraction(y_true: ArrayLike, y_score: ArrayLike, fraction: float = 0.1) -> float:
    """Precision among the top ``fraction`` highest-scored items."""
    y, s = _check(y_true, y_score)
    k = max(1, round(len(y) * fraction))
    return float(y[np.argsort(-s, kind="stable")[:k]].mean())


def bootstrap_ci(
    y_true: ArrayLike,
    y_score: ArrayLike,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI of ``metric`` over resampled evaluation rows (not a data split)."""
    y, s = _check(y_true, y_score)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].any() and not y[idx].all():
            stats.append(metric(y[idx], s[idx]))
    lo, hi = np.quantile(stats, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)
