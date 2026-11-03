"""Train and evaluate the commit-risk model with a strictly time-based split."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd

from commitrad.features import FEATURE_COLUMNS, TARGET
from commitrad.metrics import average_precision, bootstrap_ci, precision_at_fraction, roc_auc

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
MODEL_GLOB, METADATA_FILE = "commitrad_seed*.lgb", "metadata.json"
N_SEEDS = 5
LabelMode = Literal["strict", "final"]

PARAMS: dict[str, Any] = {
    "objective": "binary",
    # Rank-based and unaffected by class weighting (unweighted logloss rises from round 1
    # when scale_pos_weight inflates scores); steadier than PR-AUC with few val positives.
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
    "verbose": -1,
}
MAX_ROUNDS, EARLY_STOPPING = 2000, 100


# --- split & labels ------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    val_start: pd.Timestamp
    test_start: pd.Timestamp


def time_split(
    df: pd.DataFrame, train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC
) -> TimeSplit:
    """Oldest ``train_frac`` / next ``val_frac`` / rest, by timestamp cutoffs.

    Rows sharing a timestamp always land in the same split. Never shuffles.
    """
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    ts = df["timestamp"]
    val_start = ts.iloc[int(len(df) * train_frac)]
    test_start = ts.iloc[int(len(df) * (train_frac + val_frac))]
    split = TimeSplit(
        train=df[ts < val_start],
        val=df[(ts >= val_start) & (ts < test_start)],
        test=df[ts >= test_start],
        val_start=val_start,
        test_start=test_start,
    )
    if split.train.empty or split.val.empty or split.test.empty:
        raise ValueError("Not enough distinct timestamps for a 3-way time split")
    return split


def labels_as_of(df: pd.DataFrame, cutoff: pd.Timestamp | None) -> pd.Series:
    """Bug labels as known at ``cutoff`` (its fix landed before it). ``None`` = final labels."""
    if cutoff is None:
        return df[TARGET].astype(bool)
    return (df["bug_found_at"].notna() & (df["bug_found_at"] < cutoff)).astype(bool)


def split_labels(split: TimeSplit, mode: LabelMode) -> dict[str, pd.Series]:
    """Train/val labels never use fixes from later splits in ``strict`` mode; test is final."""
    strict = mode == "strict"
    return {
        "train": labels_as_of(split.train, split.val_start if strict else None),
        "val": labels_as_of(split.val, split.test_start if strict else None),
        "test": labels_as_of(split.test, None),
    }


# --- training & scoring --------------------------------------------------------------------


def class_weight(y: pd.Series) -> float:
    pos = int(y.sum())
    return (len(y) - pos) / pos if pos else 1.0


def train_booster(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict[str, Any] | None = None,
) -> lgb.Booster:
    """One LightGBM model: ``scale_pos_weight`` for imbalance, early stopping on val."""
    p = {**PARAMS, **(params or {}), "scale_pos_weight": class_weight(y_train)}
    train_set = lgb.Dataset(x_train, label=y_train.astype(int))
    val_set = lgb.Dataset(x_val, label=y_val.astype(int), reference=train_set)
    return lgb.train(
        p,
        train_set,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class RiskModel:
    """Seed ensemble averaged in log-odds, so SHAP values stay additive."""

    boosters: list[lgb.Booster]

    @classmethod
    def fit(
        cls,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        n_seeds: int = N_SEEDS,
    ) -> RiskModel:
        seeds = range(PARAMS["seed"], PARAMS["seed"] + n_seeds)
        return cls([train_booster(x_train, y_train, x_val, y_val, {"seed": s}) for s in seeds])

    def _x(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[FEATURE_COLUMNS].astype(float)

    def raw_score(self, df: pd.DataFrame) -> np.ndarray:
        x = self._x(df)
        scores = [
            b.predict(x, raw_score=True, num_iteration=b.best_iteration) for b in self.boosters
        ]
        return np.mean(scores, axis=0)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Risk score in (0, 1). Class weighting inflates it: rank by it, not a P(bug)."""
        return _sigmoid(self.raw_score(df))

    def shap_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exact TreeSHAP (log-odds) per feature; rows sum with ``base_value`` to the raw score."""
        x = self._x(df)
        contribs = [
            b.predict(x, pred_contrib=True, num_iteration=b.best_iteration) for b in self.boosters
        ]
        return pd.DataFrame(
            np.mean(contribs, axis=0), columns=[*FEATURE_COLUMNS, "base_value"], index=df.index
        )

    @property
    def best_iterations(self) -> list[int]:
        return [b.best_iteration or b.current_iteration() for b in self.boosters]

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        for old in model_dir.glob(MODEL_GLOB):
            old.unlink()
        for i, b in enumerate(self.boosters):
            b.save_model(str(model_dir / f"commitrad_seed{i}.lgb"), num_iteration=b.best_iteration)

    @classmethod
    def load(cls, model_dir: Path) -> RiskModel:
        files = sorted(model_dir.glob(MODEL_GLOB))
        if not files:
            raise FileNotFoundError(f"No model files in {model_dir}; run `commitrad train`")
        return cls([lgb.Booster(model_file=str(f)) for f in files])


def shap_importance(shap: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per feature and whether higher values push risk up or down."""
    rows = []
    for col in FEATURE_COLUMNS:
        vals = x[col].astype(float)
        both = pd.DataFrame({"v": vals, "s": shap[col]}).dropna()
        varies = both["v"].nunique() > 1 and both["s"].nunique() > 1
        corr = both["v"].corr(both["s"], method="spearman") if varies else np.nan
        rows.append({"feature": col, "mean_abs_shap": shap[col].abs().mean(), "direction": corr})
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def evaluate(y: pd.Series, score: np.ndarray, with_ci: bool = False) -> dict[str, float]:
    out = {
        "n": len(y),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()),
        "pr_auc": average_precision(y, score),
        "roc_auc": roc_auc(y, score),
        "precision_at_10pct": precision_at_fraction(y, score, 0.10),
    }
    if with_ci:
        out["pr_auc_lo"], out["pr_auc_hi"] = bootstrap_ci(y, score, average_precision)
        out["roc_auc_lo"], out["roc_auc_hi"] = bootstrap_ci(y, score, roc_auc)
    return out


def size_baseline(df: pd.DataFrame) -> np.ndarray:
    """Heuristic: bigger diffs are riskier."""
    return (df["lines_added"] + df["lines_deleted"]).to_numpy(dtype=float)


def reference_quantiles(scores: np.ndarray, n: int = 101) -> list[float]:
    """Fixed reference distribution of model scores, for a repo-independent risk scale."""
    return np.quantile(scores, np.linspace(0, 1, n)).tolist()


def risk_percentile(scores: np.ndarray, quantiles: list[float]) -> np.ndarray:
    """Map raw risk scores to [0, 1] percentiles of the reference distribution."""
    q = np.asarray(quantiles, dtype=float)
    return np.interp(np.asarray(scores, dtype=float), q, np.linspace(0, 1, len(q)))


# --- end-to-end ----------------------------------------------------------------------------


@dataclass
class TrainResult:
    model: RiskModel
    split: TimeSplit
    label_mode: LabelMode
    labels: dict[str, pd.Series]
    metrics: dict[str, dict[str, dict[str, float]]]  # split -> model -> metric -> value
    importance: pd.DataFrame
    params: dict[str, Any] = field(default_factory=dict)
    sensitivity: dict[str, dict[str, float]] = field(default_factory=dict)
    risk_quantiles: list[float] = field(default_factory=list)


def fit_split(split: TimeSplit, labels: dict[str, pd.Series], n_seeds: int = N_SEEDS) -> RiskModel:
    x = {k: getattr(split, k)[FEATURE_COLUMNS] for k in ("train", "val")}
    return RiskModel.fit(x["train"], labels["train"], x["val"], labels["val"], n_seeds)


def run_training(
    features: pd.DataFrame, label_mode: LabelMode = "strict", n_seeds: int = N_SEEDS
) -> TrainResult:
    split = time_split(features)
    labels = split_labels(split, label_mode)
    model = fit_split(split, labels, n_seeds)
    metrics = {
        name: {
            "lightgbm": evaluate(labels[name], model.predict(part), with_ci=name == "test"),
            "size_baseline": evaluate(labels[name], size_baseline(part), with_ci=name == "test"),
        }
        for name, part in (("val", split.val), ("test", split.test))
    }
    other: LabelMode = "final" if label_mode == "strict" else "strict"
    other_model = fit_split(split, split_labels(split, other), n_seeds)
    sensitivity = {other: evaluate(labels["test"], other_model.predict(split.test))}
    importance = shap_importance(model.shap_values(split.test), split.test[FEATURE_COLUMNS])
    params = {**PARAMS, "scale_pos_weight": class_weight(labels["train"]), "n_seeds": n_seeds}
    everything = pd.concat([split.train, split.val, split.test])
    quantiles = reference_quantiles(model.predict(everything))
    return TrainResult(
        model, split, label_mode, labels, metrics, importance, params, sensitivity, quantiles
    )


def metadata(result: TrainResult) -> dict[str, Any]:
    s = result.split
    return {
        "features": FEATURE_COLUMNS,
        "label_mode": result.label_mode,
        "best_iterations": result.model.best_iterations,
        "val_start": s.val_start.isoformat(),
        "test_start": s.test_start.isoformat(),
        "params": result.params,
        "metrics": result.metrics,
        "sensitivity": result.sensitivity,
        "importance": result.importance.to_dict(orient="records"),
        "risk_quantiles": result.risk_quantiles,
        "training_repos": sorted(result.split.train["repo"].unique().tolist()),
    }


def save_model(result: TrainResult, model_dir: Path) -> Path:
    result.model.save(model_dir)
    (model_dir / METADATA_FILE).write_text(json.dumps(metadata(result), indent=2, default=str))
    return model_dir


def load_model(model_dir: Path) -> tuple[RiskModel, dict[str, Any]]:
    meta = json.loads((model_dir / METADATA_FILE).read_text())
    if meta["features"] != FEATURE_COLUMNS:
        raise ValueError("Saved model was trained on a different feature set; retrain it")
    return RiskModel.load(model_dir), meta
