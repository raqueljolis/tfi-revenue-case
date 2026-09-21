"""Average the held-out predictions of two already-evaluated runs and re-score the average.

No model is refit: every run in `reports/results/` already stores, for each (restaurant, outer fold),
the prediction it made while that restaurant was held out (`{stem}_predictions.csv`). All runs share
the same outer CV (same seed, same folds), so a `(Id, fold)` pair identifies the same held-out
restaurant in both files and the blend is `w * A + (1 - w) * B` on the raw-revenue scale. Per-fold
RMSE is then recomputed exactly as in `evaluate.nested_cv`.

Outputs (in `reports/results/`, tagged `blend_{A}__{B}` in the filenames, like a run): `_predictions.csv`,
`_row_summary.csv`, `_folds.csv`. No checkpoint `.pkl` is written, so `aggregate_results` / notebook 04
do not pick blends up (they are read from the CSVs, e.g. in notebook 06).

The weight is fixed in advance (0.5 by default) rather than tuned: tuning it on these same folds would
add one more layer of selection optimism on top of the choice of pair.

Usage:
    python -u src/blend.py                                   # DEFAULT_PAIRS, skipping missing runs
    python -u src/blend.py --a randomforest_raw_iqr1.5 --b ridge_raw
    python -u src/blend.py --a randomforest_log --b ridge_raw --weight 0.7   # 70 % A, 30 % B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import eval_setup as setup
import evaluate as ev

# (tree model, linear model): lowercase filename stems as in `reports/results/{stem}_predictions.csv`.
DEFAULT_PAIRS = [
    ("randomforest_raw_iqr1.5", "ridge_raw"),
    ("randomforest_log", "ridge_raw"),
    ("randomforest_raw", "ridge_raw"),
    ("randomforest_log_iqr1.5", "ridge_raw"),
    ("extratrees_raw_iqr1.5", "ridge_raw"),
    ("extratrees_log", "ridge_raw"),
    ("extratrees_raw", "ridge_raw"),
]
KEY = ["Id", "fold"]


def blend_name(stem_a: str, stem_b: str, weight_a: float = 0.5) -> str:
    return f"blend_{stem_a}__{stem_b}" + ("" if weight_a == 0.5 else f"_w{weight_a:g}")


def predictions_file(stem: str) -> Path:
    return setup.RESULTS_DIR / f"{stem}_predictions.csv"


def fold_rmse(pred: pd.DataFrame, col: str = "predicted") -> pd.Series:
    """Per-outer-fold RMSE on raw revenue, indexed by fold."""
    return pred.groupby("fold").apply(lambda g: ev.rmse(g["actual"], g[col]), include_groups=False)


def blend_predictions(stem_a: str, stem_b: str, weight_a: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(blended per-row predictions, the two components side by side) for one pair."""
    a, b = pd.read_csv(predictions_file(stem_a)), pd.read_csv(predictions_file(stem_b))
    both = a.merge(b[KEY + ["actual", "predicted"]], on=KEY, suffixes=("_a", "_b"), validate="one_to_one")
    if not (len(both) == len(a) == len(b)):
        raise ValueError(f"{stem_a} and {stem_b} do not cover the same (Id, fold) pairs")
    if not np.allclose(both["actual_a"], both["actual_b"]):
        raise ValueError(f"{stem_a} and {stem_b} disagree on the actual revenue: different folds/datasets")

    predicted = weight_a * both["predicted_a"] + (1 - weight_a) * both["predicted_b"]
    out = both[KEY + ["repeat", "actual_a", "is_outlier"]].rename(columns={"actual_a": "actual"})
    out["predicted"] = predicted
    out["residual"] = out["actual"] - predicted
    out["abs_error"] = out["residual"].abs()
    out["pct_error"] = (predicted - out["actual"]) / out["actual"] * 100
    out["train_fence"] = np.nan  # a blend has no single training fence (a component may be capped)
    cols = ["Id", "fold", "repeat", "actual", "predicted", "residual", "abs_error", "pct_error",
            "is_outlier", "train_fence"]
    return out[cols], both


def run_pair(stem_a: str, stem_b: str, weight_a: float, baseline_rmse: float) -> dict:
    blended, both = blend_predictions(stem_a, stem_b, weight_a)
    name = blend_name(stem_a, stem_b, weight_a)

    f_a = fold_rmse(both.rename(columns={"actual_a": "actual", "predicted_a": "predicted"}))
    f_b = fold_rmse(both.rename(columns={"actual_a": "actual", "predicted_b": "predicted"}))
    f_blend = fold_rmse(blended)
    d_a, d_b = f_blend - f_a, f_blend - f_b

    setup.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    blended.to_csv(setup.RESULTS_DIR / f"{name}_predictions.csv", index=False)
    ev.summarize_rows(blended).to_csv(setup.RESULTS_DIR / f"{name}_row_summary.csv", index=False)
    pd.DataFrame({"fold": f_blend.index, "rmse": f_blend.to_numpy(), f"rmse_{stem_a}": f_a.to_numpy(),
                  f"rmse_{stem_b}": f_b.to_numpy()}).to_csv(setup.RESULTS_DIR / f"{name}_folds.csv", index=False)

    # Are the two components' errors different enough for an average to help?
    corr = np.corrcoef(both["actual_a"] - both["predicted_a"], both["actual_a"] - both["predicted_b"])[0, 1]
    return {
        "blend": f"{weight_a:g}*{stem_a} + {1 - weight_a:g}*{stem_b}",
        "rmse_A": f_a.mean(), "rmse_B": f_b.mean(), "rmse_blend": f_blend.mean(),
        "vs_baseline_pct": (f_blend.mean() - baseline_rmse) / baseline_rmse * 100,
        "diff_vs_A": d_a.mean(), "se_vs_A": d_a.std() / np.sqrt(len(d_a)), "better_than_A": (d_a < 0).mean(),
        "diff_vs_B": d_b.mean(), "se_vs_B": d_b.std() / np.sqrt(len(d_b)), "better_than_B": (d_b < 0).mean(),
        "residual_corr": corr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", help="first run's filename stem (e.g. randomforest_raw_iqr1.5)")
    parser.add_argument("--b", help="second run's filename stem (e.g. ridge_raw)")
    parser.add_argument("--weight", type=float, default=0.5, help="weight of --a (default 0.5)")
    args = parser.parse_args()
    if (args.a is None) != (args.b is None):
        parser.error("--a and --b must be given together")

    pairs = [(args.a, args.b)] if args.a else DEFAULT_PAIRS
    baseline = setup.compute_baseline(setup.load_train(), setup.build_outer_cv())
    print(f"baseline: {baseline['label']}, RMSE = {baseline['rmse']:,.0f}\n")

    rows = []
    for stem_a, stem_b in pairs:
        missing = [s for s in (stem_a, stem_b) if not predictions_file(s).exists()]
        if missing:
            print(f"[skip] {stem_a} + {stem_b}: no predictions for {missing}")
            continue
        rows.append(run_pair(stem_a, stem_b, args.weight if args.a else 0.5, baseline["rmse"]))

    if rows:
        pd.set_option("display.width", 250, "display.max_columns", 20)
        print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
