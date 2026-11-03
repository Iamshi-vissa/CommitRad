from __future__ import annotations

import numpy as np
import pytest

from commitrad.metrics import average_precision, bootstrap_ci, precision_at_fraction, roc_auc

Y = [0, 0, 1, 1]
S = [0.1, 0.4, 0.35, 0.8]  # scikit-learn docs example: AP = 0.8333, ROC-AUC = 0.75


def test_reference_values() -> None:
    assert average_precision(Y, S) == pytest.approx(0.8333, abs=1e-4)
    assert roc_auc(Y, S) == pytest.approx(0.75)


def test_perfect_and_inverted_rankings() -> None:
    assert average_precision([0, 1, 1], [0.1, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 1, 1], [0.1, 0.8, 0.9]) == 1.0
    assert roc_auc([1, 0, 0], [0.1, 0.8, 0.9]) == 0.0


def test_ties_are_handled() -> None:
    y = [0, 1, 0, 1, 0]
    assert roc_auc(y, [0.5] * 5) == 0.5
    assert average_precision(y, [0.5] * 5) == pytest.approx(0.4)  # = positive rate


def test_random_scores_approach_positive_rate() -> None:
    rng = np.random.default_rng(0)
    y = rng.random(20_000) < 0.1
    s = rng.random(20_000)
    assert average_precision(y, s) == pytest.approx(y.mean(), abs=0.01)
    assert roc_auc(y, s) == pytest.approx(0.5, abs=0.02)


def test_degenerate_inputs() -> None:
    assert np.isnan(average_precision([0, 0], [0.1, 0.2]))
    assert np.isnan(roc_auc([1, 1], [0.1, 0.2]))
    with pytest.raises(ValueError):
        roc_auc([0, 1], [0.1])


def test_precision_at_fraction() -> None:
    y = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    s = [0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]
    assert precision_at_fraction(y, s, 0.1) == 1.0
    assert precision_at_fraction(y, s, 0.2) == 0.5


def test_bootstrap_ci_brackets_estimate() -> None:
    rng = np.random.default_rng(1)
    y = rng.random(500) < 0.2
    s = y * 0.5 + rng.random(500)
    lo, hi = bootstrap_ci(y, s, roc_auc, n_boot=300)
    assert lo < roc_auc(y, s) < hi and hi - lo < 0.2
