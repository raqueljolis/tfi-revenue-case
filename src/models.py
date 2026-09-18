"""Model definitions and hyperparameter grids for the TFI revenue case.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge


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
    }
