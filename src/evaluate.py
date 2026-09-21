"""Cross-validated evaluation, shared by every model in the TFI revenue case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import BaseCrossValidator, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def rmse(y_true, y_pred) -> float:
    """Root mean squared error, on whatever scale the two arrays are given in."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


class StratifiedTargetKFold:
    """Repeated K-fold that stratifies a *continuous* target by quantile bins.

    `y` is ranked and cut into `n_bins` equal-count bins, and `StratifiedKFold` then keeps every
    bin represented in every fold, so each fold's validation set holds a share of the
    high-revenue restaurants (and of the low ones) instead of the outliers landing wherever a
    random split puts them. Each repeat reshuffles with its own seed (`random_state + repeat`).

    Works as a `GridSearchCV` `cv` too (with `n_repeats=1`): `y` there is the training labels,
    capped or not, and is re-binned on its own rows. Bins are built from ranks, so ties never
    collapse a bin.
    """

    def __init__(self, n_splits: int = 5, n_repeats: int = 1, n_bins: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.n_bins = n_bins
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits * self.n_repeats

    def bins(self, y) -> np.ndarray:
        return pd.qcut(pd.Series(np.asarray(y, dtype=float)).rank(method="first"), self.n_bins, labels=False).to_numpy()

    def split(self, X, y, groups=None):
        bins = self.bins(y)
        for repeat in range(self.n_repeats):
            skf = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state + repeat)
            yield from skf.split(np.zeros(len(bins)), bins)

    def __repr__(self) -> str:
        return (f"StratifiedTargetKFold(n_splits={self.n_splits}, n_repeats={self.n_repeats}, "
                f"n_bins={self.n_bins}, random_state={self.random_state})")


def cross_val_constant_baseline(
    df: pd.DataFrame,
    cv: BaseCrossValidator,
    target: str = "revenue",
    log_target: str = "log_revenue",
    strategy: str = "mean",
    use_log: bool = True,
) -> np.ndarray:
    """Per-fold RMSE (raw-target scale) of predicting one constant per fold.
    """
    if strategy not in {"mean", "median"}:
        raise ValueError(f"strategy must be 'mean' or 'median', got {strategy!r}")
    agg = np.mean if strategy == "mean" else np.median

    scores = []
    for train_idx, val_idx in cv.split(df, df[target]):
        if use_log:
            constant = np.expm1(agg(df[log_target].iloc[train_idx]))
        else:
            constant = agg(df[target].iloc[train_idx])
        pred = np.full(len(val_idx), constant)
        scores.append(rmse(df[target].iloc[val_idx], pred))
    return np.array(scores)


def build_pipeline(estimator, scale: bool = True, use_log: bool = False, prep=None):
    """Wrap `estimator` as the `"model"` step, preceded by a `StandardScaler` if `scale` and then
    by `prep` (a transformer, e.g. PCA / column selection from `variants.py`) if given.
    """
    steps = [("scaler", StandardScaler())] if scale else []
    if prep is not None:
        steps.append(("prep", clone(prep)))
    steps.append(("model", clone(estimator)))
    pipeline = Pipeline(steps)
    if use_log:
        return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)
    return pipeline


def _prefixed_param_grid(param_grid: dict, use_log: bool) -> dict:
    """`build_pipeline`'s `"model__..."` keys, prefixed with `"regressor__"` when `use_log`."""
    if not use_log:
        return dict(param_grid)
    return {f"regressor__{key}": value for key, value in param_grid.items()}


def nested_cv(
    df: pd.DataFrame,
    estimator,
    param_grid: dict,
    feature_cols: list[str],
    outer_cv: BaseCrossValidator,
    inner_cv,
    iqr_k: float | None = None,
    id_col: str = "Id",
    target: str = "revenue",
    scale: bool = True,
    n_jobs: int = -1,
    use_log: bool = True,
    on_fold=None,
    prep=None,
) -> tuple[np.ndarray, list[dict], pd.DataFrame]:
    """Nested CV: grid-search hyperparameters inside every outer training fold.

    For each `outer_cv` split, `GridSearchCV(cv=inner_cv)` picks the best `param_grid` entry
    using only that fold's training rows, then the refit best estimator predicts the held-out
    validation rows. Validation rows are always scored against the real, uncapped `target`, on
    the raw scale.

    `use_log=True`: the estimator is fit against `log1p(target)` with `expm1` applied to its
    predictions (see `build_pipeline`). `use_log=False`: it is fit against `target` directly.

    `iqr_k` set: the *training* labels of each outer fold are capped at that fold's own
    `Q3 + iqr_k x IQR` fence (Q1/Q3 from its training rows only, so no validation value leaks
    into the fence) before fitting. Capping happens before the optional `log1p`.

    `prep`, if given, is an extra pipeline step fitted inside every training fold (its tunable
    parameters go in `param_grid` as `prep__...`).

    `on_fold(fold, n_folds, fold_rmse)`, if given, is called after every outer fold (progress).

    Returns the per-fold RMSE array, each fold's `best_params_` (hyperparameters that vary wildly
    fold to fold are a sign the "optimal" choice is mostly noise at this sample size), and a
    per-row predictions frame: one row per (row, fold) it was held out in, with columns `id_col`,
    `fold`, `repeat`, `actual`, `predicted`, `residual` (`actual - predicted`; positive =
    underprediction), `abs_error`, `pct_error` (signed, vs actual), `is_outlier` (above the
    full-data 1.5xIQR fence) and `train_fence` (the cap used in that fold; NaN if uncapped).
    """
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    ids = df[id_col].to_numpy()

    template = build_pipeline(estimator, scale, use_log, prep)
    grid = _prefixed_param_grid(param_grid, use_log)
    n_folds = outer_cv.get_n_splits(df)
    n_splits = getattr(outer_cv, "n_splits", n_folds)
    # Descriptive only (never used for fitting): flags the rows above the full-data 1.5xIQR fence.
    q1_all, q3_all = np.percentile(y, [25, 75])
    is_outlier = y > q3_all + 1.5 * (q3_all - q1_all)

    scores = []
    best_params = []
    rows = []
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(df, y)):
        y_train = y[train_idx]
        fence = np.nan
        if iqr_k is not None:
            q1, q3 = np.percentile(y_train, [25, 75])
            fence = q3 + iqr_k * (q3 - q1)
            y_train = np.clip(y_train, a_min=None, a_max=fence)

        search = GridSearchCV(
            clone(template),
            grid,
            cv=inner_cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=n_jobs,
        )
        search.fit(X[train_idx], y_train)
        pred = search.predict(X[val_idx])

        scores.append(rmse(y[val_idx], pred))
        best_params.append(search.best_params_)
        rows.extend(
            {
                id_col: ids[i],
                "fold": fold,
                "repeat": fold // n_splits,
                "actual": float(y[i]),
                "predicted": float(p),
                "residual": float(y[i] - p),
                "abs_error": float(abs(y[i] - p)),
                "pct_error": float((p - y[i]) / y[i] * 100),
                "is_outlier": bool(is_outlier[i]),
                "train_fence": float(fence),
            }
            for i, p in zip(val_idx, pred)
        )
        if on_fold is not None:
            on_fold(fold + 1, n_folds, scores[-1])

    return np.array(scores), best_params, pd.DataFrame(rows)


def summarize_rows(predictions: pd.DataFrame, id_col: str = "Id") -> pd.DataFrame:
    """One row per restaurant, averaged over the repeats it was held out in — the frame for a
    predicted-vs-actual plot (one dot per restaurant, `predicted_std` as an error bar) and for
    ranking the hardest rows (`mean_abs_error`, descending).
    """
    grouped = predictions.groupby(id_col)
    summary = grouped.agg(
        actual=("actual", "first"),
        predicted_mean=("predicted", "mean"),
        predicted_std=("predicted", "std"),
        mean_residual=("residual", "mean"),
        mean_abs_error=("abs_error", "mean"),
        mean_pct_error=("pct_error", "mean"),
        is_outlier=("is_outlier", "first"),
        n_scored=("fold", "count"),
    ).reset_index()
    summary["rank_actual"] = summary["actual"].rank(ascending=False, method="first").astype(int)
    summary["rank_predicted"] = summary["predicted_mean"].rank(ascending=False, method="first").astype(int)
    return summary.sort_values("mean_abs_error", ascending=False).reset_index(drop=True)
