"""Reassemble the nested-CV summary table from `run_eval.py`'s checkpoint files.

Loading `reports/results/checkpoints/*.pkl` and builds summary DataFrame with one row per checkpoint, sorted by RMSE mean. Also prints the summary to stdout.

Usage:
    python aggregate_results.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_setup as setup


def load_checkpoints(checkpoint_dir: Path = setup.CHECKPOINT_DIR) -> list[dict]:
    """Every checkpoint pickle currently on disk, in no particular order."""
    return [pickle.load(open(path, "rb")) for path in sorted(checkpoint_dir.glob("*.pkl"))]


def build_summary(checkpoints: list[dict]) -> pd.DataFrame:
    """model, use_log, dataset_tag, iqr_k, rmse_mean, rmse_std, vs_baseline_pct, sorted by
    rmse_mean.
    """
    rows = [
        {
            "model": c["model"],
            "use_log": c["use_log"],
            "dataset_tag": c.get("dataset_tag", ""),
            "iqr_k": c.get("iqr_k"),
            "rmse_mean": c["rmse_mean"],
            "rmse_std": c["rmse_std"],
            "vs_baseline_pct": c["vs_baseline_pct"],
        }
        for c in checkpoints
    ]
    return pd.DataFrame(rows).sort_values("rmse_mean").reset_index(drop=True)


def build_fold_scores(checkpoints: list[dict]) -> pd.DataFrame:
    """Long-format per-outer-fold RMSE: model, use_log, dataset_tag, fold, rmse.

    One row per (checkpoint, outer fold) pair.
    """
    rows = [
        {
            "model": c["model"],
            "use_log": c["use_log"],
            "dataset_tag": c.get("dataset_tag", ""),
            "fold": fold,
            "rmse": float(value),
        }
        for c in checkpoints
        for fold, value in enumerate(c["fold_rmse"])
    ]
    return pd.DataFrame(rows)


def main() -> None:
    checkpoints = load_checkpoints()
    if not checkpoints:
        print(f"No checkpoints found in {setup.CHECKPOINT_DIR}")
        return
    summary = build_summary(checkpoints)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
