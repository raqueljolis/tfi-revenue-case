"""Cross-validated evaluation, shared by every model in the TFI revenue case.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import BaseCrossValidator, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def rmse(y_true, y_pred) -> float:
    """Root mean squared error, on whatever scale the two arrays are given in."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


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
    for train_idx, val_idx in cv.split(df):
        if use_log:
            constant = np.expm1(agg(df[log_target].iloc[train_idx]))
        else:
            constant = agg(df[target].iloc[train_idx])
        pred = np.full(len(val_idx), constant)
        scores.append(rmse(df[target].iloc[val_idx], pred))
    return np.array(scores)


def build_pipeline(estimator, scale: bool = True, use_log: bool = False):
    """Wrap `estimator` as the `"model"` step, preceded by a `StandardScaler` if `scale`.
    """
    steps = [("scaler", StandardScaler())] if scale else []
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


def nested_cv_grid_search_rmse(
    df: pd.DataFrame,
    estimator,
    param_grid: dict,
    feature_cols: list[str],
    outer_cv: BaseCrossValidator,
    inner_cv,
    target: str = "revenue",
    scale: bool = True,
    n_jobs: int = -1,
    use_log: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    """Nested CV: grid-search hyperparameters inside every outer training fold.

    For each `outer_cv` split, `GridSearchCV(cv=inner_cv)` picks the best `param_grid` entry
    using only that fold's training rows, then the refit best estimator predicts the held-out
    validation rows. RMSE is always computed on the raw `target` scale.

    `use_log=True` (default): the estimator is wrapped in a `TransformedTargetRegressor`
    (see `build_pipeline`), so it's fit and scored against `log1p(target)` with `expm1` applied
    to predictions automatically. `use_log=False`: the estimator is fit and predicts directly
    against `target` — no `log1p`/`expm1` anywhere — so the two settings can be compared to see
    whether modeling the log-target actually helps for this particular estimator.

    Either way, hyperparameter selection never sees the outer validation fold, so the
    resulting RMSE isn't optimistic the way tuning once against the full dataset (then
    reporting CV RMSE from that same fit) would be.

    Returns the per-fold RMSE array and each fold's `best_params_` — the latter is worth
    inspecting on its own: hyperparameters that vary wildly fold-to-fold are a sign the
    "optimal" choice is mostly noise at this sample size.
    """
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)

    template = build_pipeline(estimator, scale, use_log)
    grid = _prefixed_param_grid(param_grid, use_log)

    scores = []
    best_params = []
    for train_idx, val_idx in outer_cv.split(df):
        search = GridSearchCV(
            clone(template),
            grid,
            cv=inner_cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=n_jobs,
        )
        search.fit(X[train_idx], y[train_idx])
        pred = search.predict(X[val_idx])
        scores.append(rmse(y[val_idx], pred))
        best_params.append(search.best_params_)
    return np.array(scores), best_params


def nested_cv_predictions(
    df: pd.DataFrame,
    estimator,
    param_grid: dict,
    feature_cols: list[str],
    outer_cv: BaseCrossValidator,
    inner_cv,
    id_col: str = "Id",
    target: str = "revenue",
    scale: bool = True,
    n_jobs: int = -1,
    use_log: bool = True,
) -> pd.DataFrame:
    """Same nested CV as `nested_cv_grid_search_rmse` (no target capping), but returns every
    validation row's own prediction instead of only the aggregate RMSE per fold — the uncapped
    counterpart to `nested_cv_capped_predictions`, so the two can be compared side by side on
    the same rows (e.g. to see how much capping actually changes a prediction).

    One row per (row, fold) pair it was held out in — a given `id_col` value appears multiple
    times, once per outer fold it was scored in. Columns: `id_col`, `fold`, `actual`,
    `predicted` (back-transformed to the raw scale as usual), `residual` (`actual - predicted`).
    """
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    ids = df[id_col].to_numpy()

    template = build_pipeline(estimator, scale, use_log)
    grid = _prefixed_param_grid(param_grid, use_log)

    rows = []
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(df)):
        search = GridSearchCV(
            clone(template),
            grid,
            cv=inner_cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=n_jobs,
        )
        search.fit(X[train_idx], y[train_idx])
        pred = search.predict(X[val_idx])

        for row_id, actual, predicted in zip(ids[val_idx], y[val_idx], pred):
            rows.append({
                id_col: row_id,
                "fold": fold,
                "actual": float(actual),
                "predicted": float(predicted),
                "residual": float(actual - predicted),
            })

    return pd.DataFrame(rows)


def nested_cv_grid_search_rmse_capped(
    df: pd.DataFrame,
    estimator,
    param_grid: dict,
    feature_cols: list[str],
    outer_cv: BaseCrossValidator,
    inner_cv,
    iqr_k: float,
    target: str = "revenue",
    scale: bool = True,
    n_jobs: int = -1,
    use_log: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    """Same as `nested_cv_grid_search_rmse`, but caps the *training* target at each outer
    fold's own `Q3 + iqr_k x IQR` fence before fitting.

    Q1/Q3 are computed from that fold's training rows only — never from its validation rows —
    so no fold leaks its own held-out values into the fence used to cap its training data.
    Capping happens before the optional `log1p` (same order as fitting on `log1p` of an
    already-capped target would), and only ever touches the *training* labels: validation rows
    are always scored against the real, uncapped `target`, since the goal is a model that isn't
    dominated by a handful of extreme training rows — not one that's graded on an easier target.

    Fence granularity matches `cross_val_constant_baseline`: refit once per outer training
    fold, not separately inside each inner `GridSearchCV` fold.
    """
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)

    template = build_pipeline(estimator, scale, use_log)
    grid = _prefixed_param_grid(param_grid, use_log)

    scores = []
    best_params = []
    for train_idx, val_idx in outer_cv.split(df):
        y_train = y[train_idx]
        q1, q3 = np.percentile(y_train, [25, 75])
        upper_fence = q3 + iqr_k * (q3 - q1)
        y_train_capped = np.clip(y_train, a_min=None, a_max=upper_fence)

        search = GridSearchCV(
            clone(template),
            grid,
            cv=inner_cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=n_jobs,
        )
        search.fit(X[train_idx], y_train_capped)
        pred = search.predict(X[val_idx])
        scores.append(rmse(y[val_idx], pred))
        best_params.append(search.best_params_)
    return np.array(scores), best_params


def nested_cv_capped_predictions(
    df: pd.DataFrame,
    estimator,
    param_grid: dict,
    feature_cols: list[str],
    outer_cv: BaseCrossValidator,
    inner_cv,
    iqr_k: float,
    id_col: str = "Id",
    target: str = "revenue",
    scale: bool = True,
    n_jobs: int = -1,
    use_log: bool = True,
) -> pd.DataFrame:
    """Same per-fold-capped procedure as `nested_cv_grid_search_rmse_capped`, but returns every
    validation row's own prediction instead of only the aggregate RMSE per fold — for digging
    into whether a specific group of rows (e.g. the high-revenue outliers a capped target is
    meant to tame) is systematically over- or under-predicted, not just how the average RMSE
    moved.

    One row per (row, fold) pair it was held out in — since each row lands in validation in
    multiple outer folds (10 times, for this project's 5-fold x 10-repeat `outer_cv`), a given
    `id_col` value appears in the result multiple times, once per fold it was scored in. Columns:
    `id_col`, `fold`, `actual` (real revenue), `predicted` (back-transformed to the raw scale as
    usual), `residual` (`actual - predicted`; positive means underprediction).

    Same capping discipline as `nested_cv_grid_search_rmse_capped`: Q1/Q3 and the training cap
    are computed from each fold's training rows only, and every prediction here is still scored
    against the real, uncapped `target` — this only re-observes that procedure more closely, it
    doesn't change it.
    """
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    ids = df[id_col].to_numpy()

    template = build_pipeline(estimator, scale, use_log)
    grid = _prefixed_param_grid(param_grid, use_log)

    rows = []
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(df)):
        y_train = y[train_idx]
        q1, q3 = np.percentile(y_train, [25, 75])
        upper_fence = q3 + iqr_k * (q3 - q1)
        y_train_capped = np.clip(y_train, a_min=None, a_max=upper_fence)

        search = GridSearchCV(
            clone(template),
            grid,
            cv=inner_cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=n_jobs,
        )
        search.fit(X[train_idx], y_train_capped)
        pred = search.predict(X[val_idx])

        for row_id, actual, predicted in zip(ids[val_idx], y[val_idx], pred):
            rows.append({
                id_col: row_id,
                "fold": fold,
                "actual": float(actual),
                "predicted": float(predicted),
                "residual": float(actual - predicted),
            })

    return pd.DataFrame(rows)


def fit_final_model(
    df: pd.DataFrame,
    estimator,
    param_grid: dict,
    feature_cols: list[str],
    inner_cv,
    target: str = "revenue",
    scale: bool = True,
    use_log: bool = True,
    n_jobs: int = -1,
):
    """Refit on the *full* dataset for checkpointing — separate from the nested-CV RMSE estimate.

    Nested CV above never produces one deployable model: every outer fold's `GridSearchCV`
    picks its own best hyperparameters. This reruns the same `GridSearchCV(cv=inner_cv)` search
    against all of `df` and returns its refit best estimator (a `Pipeline` or, if `use_log`, a
    `TransformedTargetRegressor` wrapping one) plus `best_params_`, ready to `joblib.dump` and
    later `joblib.load` back without repeating the whole notebook.
    """
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)

    template = build_pipeline(estimator, scale, use_log)
    grid = _prefixed_param_grid(param_grid, use_log)

    search = GridSearchCV(
        clone(template),
        grid,
        cv=inner_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=n_jobs,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Stable hash of a dataframe's content, so a checkpoint can detect a changed `train`."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()).hexdigest()


def training_fingerprint(
    name: str,
    estimator,
    param_grid: dict,
    feature_cols: list[str],
    scale: bool,
    use_log: bool,
    inner_cv,
    data_fingerprint: str,
) -> str:
    """Stable hash of everything that determines `fit_final_model`'s output for one model.

    Used to skip a checkpoint refit when nothing that could change its result — the grid, the
    estimator's own fixed hyperparameters, the feature list, scaling, target scale, the inner CV
    scheme, or the training data itself — has changed since the checkpoint on disk was written.
    """
    payload = {
        "model": name,
        "estimator_repr": repr(estimator),
        "param_grid": {
            key: [float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v for v in values]
            for key, values in param_grid.items()
        },
        "feature_cols": list(feature_cols),
        "scale": scale,
        "use_log": use_log,
        "inner_cv_repr": repr(inner_cv),
        "data_fingerprint": data_fingerprint,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
