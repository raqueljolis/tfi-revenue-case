"""Experiment variants on how the `P1`-`P37` columns reach the model, run through `run_eval.py --variant`.

Each variant is tagged into every output filename (e.g. `ridge_raw_pca_predictions.csv`), so it sits
next to the standard runs and `aggregate_results.py` / the notebooks read it like any other.

- `widealpha`: no feature change; Ridge/Lasso/ElasticNet get an alpha grid that reaches past the
  old top edge (Ridge/Lasso raw kept picking the maximum). It is the fair reference for `pca`/`top10`.
- `pca`: PCA on the 37 P columns only (the other features pass through), `n_components` tuned.
- `nosparse`: drop the 17 sparse P columns (>=50% zeros); the `all_zero` flag that summarises them stays.
- `top10`: keep only the 10 P columns with the highest |Spearman| with the target.
- `nooutliers`: drop the revenue outliers (above the full-data Tukey fence, `Q3 + 1.5 x IQR`) from
  the dataset altogether, so the CV runs on the remaining rows only. The baseline is recomputed on
  those rows, so read it as "% vs the baseline of the same rows" -- the raw RMSE is NOT comparable
  with the runs on all 137 restaurants.

`pca` and `top10` are fitted *inside* each training fold (they are pipeline steps), so nothing about
the validation rows leaks into the choice of components / columns. `pca` and `top10` also use the
wide alpha grid on linear models.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

import data_prep as dp
import features as feat

VARIANTS: dict[str, str] = {
    "widealpha": "wider alpha grid on linear models (reference for pca/top10)",
    "pca": "PCA on P1-P37 inside the pipeline, n_components tuned",
    "nosparse": "drop the 17 sparse P columns, keep the all_zero flag",
    "top10": "top-10 P columns by |Spearman| with the target, chosen inside each training fold",
    "nooutliers": "revenue outliers (> Q3 + 1.5 IQR) removed from the dataset, CV on the remaining rows",
}

# (model, use_log) per variant, in run order. Linear models on the raw target (where they beat the
# baseline); RandomForest on the log target (twin of an existing run).
VARIANT_PLANS: dict[str, list[tuple[str, bool]]] = {
    "widealpha": [("Ridge", False), ("ElasticNet", False)],
    "pca": [("Ridge", False), ("ElasticNet", False)],
    "nosparse": [("RandomForest", True)],
    "top10": [("RandomForest", True), ("Ridge", False)],
    "nooutliers": [("RandomForest", True), ("RandomForest", False)],
}

WIDE_ALPHA_VARIANTS = {"widealpha", "pca", "top10"}
WIDE_ALPHA = {
    "Ridge": np.logspace(-3, 5, 17),
    "Lasso": np.logspace(-3, 5, 17),
    "ElasticNet": np.logspace(-2, 4, 10),
}
PCA_COMPONENTS = [4, 8, 12]
TOP_K = 10


class SpearmanTopK(BaseEstimator, TransformerMixin):
    """Keep the `k` columns with the highest |Spearman correlation| with `y` (learned in `fit`)."""

    def __init__(self, k: int = TOP_K):
        self.k = k

    def fit(self, X, y):
        ranks = rankdata(np.asarray(X, dtype=float), axis=0)
        ranks_y = rankdata(np.asarray(y, dtype=float))
        xc = ranks - ranks.mean(axis=0)
        yc = ranks_y - ranks_y.mean()
        denom = np.sqrt((xc**2).sum(axis=0) * (yc**2).sum())
        self.scores_ = np.divide(xc.T @ yc, denom, out=np.zeros(xc.shape[1]), where=denom > 0)
        self.support_ = np.sort(np.argsort(-np.abs(self.scores_))[: self.k])
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]


def variant_rows(train, variant: str):
    """`train` without the revenue outliers for `nooutliers`, reindexed; unchanged otherwise."""
    if variant != "nooutliers":
        return train
    return train[~feat.flag_revenue_outliers(train)].reset_index(drop=True)


def variant_feature_cols(feature_cols: list[str], variant: str) -> list[str]:
    """`feature_cols`, minus the sparse P columns for `nosparse`."""
    if variant != "nosparse":
        return feature_cols
    sparse = {c for cols in feat.SPARSE_P_GROUPS.values() for c in cols}
    return [c for c in feature_cols if c not in sparse]


def variant_spec(spec: dict, model_name: str, variant: str) -> dict:
    """Copy of a model spec with the variant's alpha grid applied (linear models only)."""
    if variant not in WIDE_ALPHA_VARIANTS or model_name not in WIDE_ALPHA:
        return spec
    return spec | {"param_grid": spec["param_grid"] | {"model__alpha": WIDE_ALPHA[model_name]}}


def variant_prep(variant: str, feature_cols: list[str]):
    """(`prep` transformer or None, extra param grid) for the pipeline step after the scaler.

    The step acts on the P columns only (`remainder="passthrough"` keeps every other feature).
    """
    p_idx = [i for i, c in enumerate(feature_cols) if c in dp.P_COLUMNS]
    if variant == "pca":
        prep = ColumnTransformer([("pca", PCA(random_state=42), p_idx)], remainder="passthrough")
        return prep, {"prep__pca__n_components": PCA_COMPONENTS}
    if variant == "top10":
        prep = ColumnTransformer([("topk", SpearmanTopK(k=TOP_K), p_idx)], remainder="passthrough")
        return prep, {}
    return None, {}
