"""Run nested-CV grid search for one (model, use_log) combination at a time, checkpointed.

Each combination is independent and idempotent: its result is pickled to
`reports/results/checkpoints/{model_name}_{log|raw}.pkl` as soon as it finishes, and a
combination whose checkpoint already exists is loaded from disk instead of recomputed. A
combination that raises is logged to `reports/results/errors.log` and skipped, so one bad
model doesn't stop the rest.

`--iqr` switches to a revenue-capped variant: the target is capped at `Q3 + iqr x IQR` inside
every outer fold (fence from that fold's training rows only; validation rows are always scored
against real, uncapped revenue — see `evaluate.nested_cv_grid_search_rmse_capped`), and the
checkpoint filename is tagged with it — `{model_name}_{log|raw}_{iqr_tag}.pkl` — so a variant's
results never collide with, or get mistaken for, the original uncapped run.

Usage:
    python run_eval.py --model RandomForest --use_log true            # one combination
    python run_eval.py --model RandomForest --use_log true --iqr 1.5  # same, on the 1.5x-IQR-capped variant
    python run_eval.py                                                 # every combination not yet checkpointed
    python run_eval.py --iqr 3.0                                       # ...for the 3x-IQR-capped variant
"""

from __future__ import annotations

import argparse
import pickle
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate as ev
import eval_setup as setup
from models import build_model_specs


def checkpoint_path(model_name: str, use_log: bool, dataset_tag: str = "") -> Path:
    suffix = "log" if use_log else "raw"
    stem = f"{model_name}_{suffix}" if not dataset_tag else f"{model_name}_{suffix}_{dataset_tag}"
    return setup.CHECKPOINT_DIR / f"{stem}.pkl"


def log_error(model_name: str, use_log: bool, exc: Exception, dataset_tag: str = "") -> None:
    setup.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    suffix = "log" if use_log else "raw"
    label = f"{model_name} ({suffix}{', ' + dataset_tag if dataset_tag else ''})"
    with open(setup.ERROR_LOG, "a") as f:
        f.write(f"[{timestamp}] {label}: {exc!r}\n")
        f.write("".join(traceback.format_exception(exc)))
        f.write("\n")


def run_combination(model_name: str, use_log: bool, ctx: setup.EvalContext, force: bool = False) -> dict:
    """Run (or load) nested CV for one (model_name, use_log) combination.

    Returns the same checkpoint dict that gets pickled: model name, use_log flag, dataset tag,
    IQR-capping multiplier (`None` if this isn't a capped variant), per-outer-fold RMSE,
    per-outer-fold best hyperparameters, and summary stats (rmse_mean, rmse_std,
    vs_baseline_pct).
    """
    label = f"{model_name} ({'log' if use_log else 'raw'}{', ' + ctx.dataset_tag if ctx.dataset_tag else ''})"
    path = checkpoint_path(model_name, use_log, ctx.dataset_tag)
    if path.exists() and not force:
        print(f"[skip] {label}: checkpoint already exists at {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    spec = build_model_specs()[model_name]

    print(f"[run] {label}")
    if ctx.iqr_k is not None:
        fold_rmse, fold_best_params = ev.nested_cv_grid_search_rmse_capped(
            ctx.train,
            estimator=spec["estimator"],
            param_grid=spec["param_grid"],
            feature_cols=ctx.feature_cols,
            outer_cv=ctx.outer_cv,
            inner_cv=ctx.inner_cv,
            iqr_k=ctx.iqr_k,
            scale=spec["scale"],
            use_log=use_log,
        )
    else:
        fold_rmse, fold_best_params = ev.nested_cv_grid_search_rmse(
            ctx.train,
            estimator=spec["estimator"],
            param_grid=spec["param_grid"],
            feature_cols=ctx.feature_cols,
            outer_cv=ctx.outer_cv,
            inner_cv=ctx.inner_cv,
            scale=spec["scale"],
            use_log=use_log,
        )

    rmse_mean = float(fold_rmse.mean())
    rmse_std = float(fold_rmse.std())
    baseline_rmse = ctx.baseline["rmse"]
    vs_baseline_pct = (rmse_mean - baseline_rmse) / baseline_rmse * 100

    result = {
        "model": model_name,
        "use_log": use_log,
        "dataset_tag": ctx.dataset_tag,
        "iqr_k": ctx.iqr_k,
        "fold_rmse": fold_rmse,
        "fold_best_params": fold_best_params,
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
        "vs_baseline_pct": vs_baseline_pct,
    }

    setup.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    print(f"[done] {label}: rmse_mean={rmse_mean:,.0f} -> {path}")

    return result


def run_all(ctx: setup.EvalContext) -> None:
    """Every (model_name, use_log) combination not yet checkpointed, in isolation."""
    for model_name in build_model_specs():
        for use_log in [True, False]:
            label = f"{model_name} ({'log' if use_log else 'raw'}{', ' + ctx.dataset_tag if ctx.dataset_tag else ''})"
            path = checkpoint_path(model_name, use_log, ctx.dataset_tag)
            if path.exists():
                print(f"[skip] {label}: checkpoint already exists at {path}")
                continue
            try:
                run_combination(model_name, use_log, ctx)
            except Exception as exc:
                print(f"[error] {label}: {exc!r} (logged, continuing)")
                log_error(model_name, use_log, exc, ctx.dataset_tag)


def _str2bool(value: str) -> bool:
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(build_model_specs()), help="model name to run alone")
    parser.add_argument("--use_log", type=_str2bool, help="true/false; required together with --model")
    parser.add_argument("--force", action="store_true", help="recompute even if a checkpoint already exists")
    parser.add_argument(
        "--iqr",
        type=float,
        help="IQR-capping multiplier for a revenue-capped variant (e.g. 1.5 or 3.0); "
        "omit for the original uncapped train_clean.csv",
    )
    args = parser.parse_args()

    if (args.model is None) != (args.use_log is None):
        parser.error("--model and --use_log must be given together")

    ctx = setup.build_context_for_iqr(args.iqr) if args.iqr is not None else setup.build_context()

    if args.model is not None:
        run_combination(args.model, args.use_log, ctx, force=args.force)
    else:
        run_all(ctx)


if __name__ == "__main__":
    main()
