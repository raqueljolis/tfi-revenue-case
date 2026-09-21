"""Shared setup for nested-CV evaluation, used by `run_eval.py`, `aggregate_results.py`, and
the notebooks alike, so all agree on the training data, CV splits, feature list,
and baseline reference without duplicating that setup in each place.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import evaluate as ev
import variants

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
ERROR_LOG = RESULTS_DIR / "errors.log"

# Named datasets selectable with `run_eval.py --dataset`. "" is the original table; the others
# are tagged into every output filename (e.g. `randomforest_log_provinces_predictions.csv`).
DATASETS = {
    "": "train_clean.csv",
    "provinces": "train_clean_v2.csv",  # train + provinces.csv (built in 02_2_cleaning_features.ipynb)
}

RANDOM_STATE = 42

TARGET = "revenue"
LOG_TARGET = "log_revenue"
NON_FEATURE_COLS = ("Id", TARGET, LOG_TARGET)


def load_train(path: Path | str = PROCESSED_DIR / "train_clean.csv") -> pd.DataFrame:
    """Load the processed training table."""
    return pd.read_csv(path)


def iqr_tag(k: float) -> str:
    """Filesystem-safe tag for a `k`x-IQR revenue-capping threshold, e.g. 1.5 -> 'iqr1.5'."""
    return f"iqr{k:g}"


def feature_cols(train: pd.DataFrame) -> list[str]:
    """Every column except Id and the two target representations."""
    return [c for c in train.columns if c not in NON_FEATURE_COLS]


def build_outer_cv(random_state: int = RANDOM_STATE) -> ev.StratifiedTargetKFold:
    """5-fold x 10-repeat outer CV stratified on revenue quintiles, shared by the baseline and
    every real model, so each fold's validation set spans the whole revenue range."""
    return ev.StratifiedTargetKFold(n_splits=5, n_repeats=10, n_bins=5, random_state=random_state)


def build_inner_cv(random_state: int = RANDOM_STATE) -> ev.StratifiedTargetKFold:
    """5-fold inner CV, also stratified on (training-label) quintiles, for `GridSearchCV`."""
    return ev.StratifiedTargetKFold(n_splits=5, n_repeats=1, n_bins=5, random_state=random_state)


def compute_baseline(train: pd.DataFrame, outer_cv) -> dict:
    """Best of {mean, median} x {log, raw} constant baseline, scored on `outer_cv`.
    """
    candidates = []
    for use_log in [True, False]:
        for strategy in ["mean", "median"]:
            scores = ev.cross_val_constant_baseline(train, outer_cv, strategy=strategy, use_log=use_log)
            candidates.append({
                "strategy": strategy,
                "use_log": use_log,
                "target": LOG_TARGET if use_log else TARGET,
                "scores": scores,
            })

    best = min(candidates, key=lambda c: c["scores"].mean())
    return {
        "strategy": best["strategy"],
        "use_log": best["use_log"],
        "rmse": float(best["scores"].mean()),
        "label": f"baseline ({best['strategy']}, {'log' if best['use_log'] else 'raw'})",
    }


class EvalContext:
    """Everything a single (model_name, use_log) nested-CV run needs, built once and reused.
    """

    def __init__(
        self,
        train: pd.DataFrame,
        feature_cols: list[str],
        outer_cv,
        inner_cv,
        baseline: dict,
        dataset_tag: str = "",
        iqr_k: float | None = None,
        variant: str = "",
    ):
        self.train = train
        self.feature_cols = feature_cols
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.baseline = baseline
        self.dataset_tag = dataset_tag
        self.iqr_k = iqr_k
        self.variant = variant


def build_context(iqr_k: float | None = None, dataset: str = "", variant: str = "") -> EvalContext:
    """Load the processed table of `dataset` (see `DATASETS`) and construct the CV splits +
    baseline reference in one call.

    `iqr_k` set: the context of a revenue-capped variant (tagged `iqr{k}`, after the dataset tag).
    `variant` set (see `variants.VARIANTS`): a feature-treatment experiment, tagged after the dataset tag.
    """
    train = variants.variant_rows(load_train(PROCESSED_DIR / DATASETS[dataset]), variant)
    tag = "_".join(t for t in (dataset, variant, iqr_tag(iqr_k) if iqr_k is not None else "") if t)
    outer_cv = build_outer_cv()
    return EvalContext(
        train=train,
        feature_cols=variants.variant_feature_cols(feature_cols(train), variant),
        outer_cv=outer_cv,
        inner_cv=build_inner_cv(),
        baseline=compute_baseline(train, outer_cv),
        dataset_tag=tag,
        iqr_k=iqr_k,
        variant=variant,
    )
