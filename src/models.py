"""Model definitions and hyperparameter grids for the TFI revenue case.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge

try:
    from lightgbm import LGBMRegressor
except (ImportError, OSError):  # not installed, or its native library (libomp on macOS) is missing
    LGBMRegressor = None


def build_model_specs(random_state: int = 42) -> dict[str, dict]:
    """Estimator + `GridSearchCV` param grid + scaling flag, keyed by model name.
    """
    return {
        "Ridge": {
            "estimator": Ridge(),
            "param_grid": {"model__alpha": np.logspace(-3, 3, 13)},
            "scale": True,
        },
        "Lasso": {
            "estimator": Lasso(max_iter=20_000, random_state=random_state),
            "param_grid": {"model__alpha": np.logspace(-3, 3, 13)},
            "scale": True,
        },
        "ElasticNet": {
            "estimator": ElasticNet(max_iter=20_000, random_state=random_state),
            "param_grid": {
                "model__alpha": np.logspace(-2, 2, 7),
                "model__l1_ratio": [0.2, 0.5, 0.8, 1.0],
            },
            "scale": True,
        },
        "RandomForest": {
            "estimator": RandomForestRegressor(random_state=random_state, n_jobs=1),
            "param_grid": {
                "model__n_estimators": [200, 500],
                "model__max_depth": [None, 8],
                "model__min_samples_leaf": [1, 3, 5],
                "model__max_features": ["sqrt",0.5, 1.0],
                "model__max_samples": [None, 0.8],
            },
            "scale": False,
        },
        "ExtraTrees": {
            # RandomForest's sibling: split thresholds are drawn at random instead of searched,
            # which lowers the variance further. Same grid minus `max_samples` (no bootstrap by
            # default, so it doesn't apply); its `max_features` default is all features, so the
            # grid keeps sqrt / 0.5 as well.
            "estimator": ExtraTreesRegressor(random_state=random_state, n_jobs=1),
            "param_grid": {
                "model__n_estimators": [200, 500],
                "model__max_depth": [None, 8],
                "model__min_samples_leaf": [1, 3, 5],
                "model__max_features": ["sqrt", 0.5, 1.0],
            },
            "scale": False,
        },
        "GradientBoosting": {
            "estimator": GradientBoostingRegressor(random_state=random_state),
            "param_grid": {
                "model__n_estimators": [200, 300, 400],
                "model__max_depth": [2, 3],
                "model__learning_rate": [0.03, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__min_samples_leaf": [1, 5],
                "model__loss" : ["squared_error", "huber"],
            },
            "scale": False,
        },
        "RandomForestMAE": {
            # Same idea as RandomForest, but criterion="absolute_error" (MAE) instead of the
            # default squared_error -- meaningfully slower per split (needs the median at each
            # split, not the mean), so this grid is trimmed (max_samples dropped entirely) to
            # keep this comparison run's cost sane; this isn't the primary tuning pass for it.
            "estimator": RandomForestRegressor(random_state=random_state, n_jobs=1, criterion="absolute_error"),
            "param_grid": {
                "model__n_estimators": [200, 500],
                "model__max_depth": [None, 8],
                "model__min_samples_leaf": [1, 3, 5],
                "model__max_features": ["sqrt", 0.5, 1.0],
            },
            "scale": False,
        },
        "GradientBoostingHuber": {
            # Same grid as GradientBoosting, but loss pinned to huber only: isolates whether
            # Huber loss itself helps, rather than letting GridSearchCV pick per fold (where it
            # competes against squared_error and the choice can vary fold to fold). A deliberate
            # alternative to capping the target for the high-revenue outliers — Huber down-
            # weights large residuals without ever discarding the true label the way capping does.
            "estimator": GradientBoostingRegressor(random_state=random_state),
            "param_grid": {
                "model__n_estimators": [200, 300, 400],
                "model__max_depth": [2, 3],
                "model__learning_rate": [0.03, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__min_samples_leaf": [1, 5],
                "model__loss": ["huber"],
            },
            "scale": False,
        },
    } | _lightgbm_spec(random_state)


def _lightgbm_spec(random_state: int) -> dict[str, dict]:
    """LightGBM entry, empty if the library can't be loaded (so every other model still runs).

    Small-data settings: 137 rows, so few leaves and a small `min_child_samples`; the defaults
    (31 leaves, 20 samples per leaf) would leave a tree with almost no valid splits. `n_jobs=1`
    because `GridSearchCV` already parallelises over candidates. Unscaled: trees don't need it.
    """
    if LGBMRegressor is None:
        return {}
    return {
        "LightGBM": {
            "estimator": LGBMRegressor(random_state=random_state, n_jobs=1, verbose=-1),
            "param_grid": {
                "model__n_estimators": [200, 400],
                "model__learning_rate": [0.03, 0.1],
                "model__num_leaves": [4, 8, 16],
                "model__min_child_samples": [5, 10, 20],
                "model__colsample_bytree": [0.5, 1.0],
            },
            "scale": False,
        },
    }
