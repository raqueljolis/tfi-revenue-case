"""Nested-CV grid search for every model combination of the TFI revenue case, checkpointed.

This is the only place training happens (it replaces running `03_modeling.ipynb`). Each
(model, use_log, variant) combination is independent and idempotent: its result is pickled to
`reports/results/checkpoints/{model}_{log|raw}[_iqr{k}].pkl` as soon as it finishes, together
with a per-row predictions CSV `reports/results/{model}_{log|raw}[_iqr{k}]_predictions.csv`
(lowercase name, as read by `04_results_summary.ipynb`), a `_row_summary.csv` (one line per
restaurant, averaged over repeats, hardest first) and a `_folds.csv` (RMSE + best params per fold). A combination whose checkpoint already
exists is skipped, so an interrupted run resumes where it stopped. A combination that raises is
written to `reports/results/errors.log` and skipped, so one bad model doesn't stop the rest.

Progress goes to stdout and to `reports/results/run.log`: one line per combination and one per
outer fold, with elapsed time and an ETA.

Splits: outer 5x10 and inner 5-fold CV are stratified on revenue quintiles (`evaluate.StratifiedTargetKFold`),
so every fold contains low, mid and high-revenue (outlier) restaurants.

Capped variants (`--iqr`): the training target is capped at `Q3 + k x IQR` inside every outer
fold (fence from that fold's training rows only); validation rows are always scored against real
revenue. See `evaluate.nested_cv`.

Usage:
    python -u src/run_eval.py                       # the full plan (PLAN below), skipping done ones
    python -u src/run_eval.py --model RandomForest --use_log true            # one combination
    python -u src/run_eval.py --model RandomForest --use_log true --iqr 1.5  # ...capped variant
    python -u src/run_eval.py --dataset provinces   # RandomForest (log) on train + provinces
    python -u src/run_eval.py --dataset provinces --model RandomForest --use_log false  # any single one
    python -u src/run_eval.py --variant pca         # a feature-treatment experiment's plan (variants.py)
    python -u src/run_eval.py --variant all         # all of them: widealpha, pca, nosparse, top10
    python -u src/run_eval.py --variant top10 --model RandomForest --use_log true   # one combination
    python -u src/run_eval.py --force               # recompute even if checkpointed
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import evaluate as ev
import eval_setup as setup
import variants
from models import build_model_specs

# (model, use_log, iqr_k or None), in run order.
PLAN: list[tuple[str, bool, float | None]] = (
    [(m, use_log, None) for m in ["Ridge", "Lasso", "ElasticNet", "RandomForest", "GradientBoosting"]
     for use_log in (True, False)]
    + [("GradientBoostingHuber", True, None), ("RandomForestMAE", True, None)]
    + [("LightGBM", use_log, None) for use_log in (True, False)]
    + [(m, use_log, 1.5) for m in ["Ridge", "ElasticNet", "RandomForest"] for use_log in (True, False)]
    + [("ExtraTrees", use_log, None) for use_log in (True, False)]
    + [("ExtraTrees", False, 1.5)]
)

# Plans for the alternative datasets in `eval_setup.DATASETS`, run with `--dataset NAME`.
DATASET_PLANS: dict[str, list[tuple[str, bool, float | None]]] = {
    "provinces": [("RandomForest", True, None), ("LightGBM", True, None)],
}

log = logging.getLogger("run_eval")


def setup_logging() -> None:
    setup.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(setup.RESULTS_DIR / "run.log")):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.setLevel(logging.INFO)


def combination_stem(model_name: str, use_log: bool, dataset_tag: str = "") -> str:
    stem = f"{model_name}_{'log' if use_log else 'raw'}"
    return f"{stem}_{dataset_tag}" if dataset_tag else stem


def checkpoint_path(model_name: str, use_log: bool, dataset_tag: str = "") -> Path:
    return setup.CHECKPOINT_DIR / f"{combination_stem(model_name, use_log, dataset_tag)}.pkl"


def predictions_path(model_name: str, use_log: bool, dataset_tag: str = "") -> Path:
    return setup.RESULTS_DIR / f"{combination_stem(model_name, use_log, dataset_tag).lower()}_predictions.csv"


def log_error(label: str, exc: Exception) -> None:
    with open(setup.ERROR_LOG, "a") as f:
        f.write(f"[{datetime.now(timezone.utc).isoformat()}] {label}: {exc!r}\n")
        f.write("".join(traceback.format_exception(exc)))
        f.write("\n")


def run_combination(model_name: str, use_log: bool, ctx: setup.EvalContext, force: bool = False) -> dict:
    """Run (or load) nested CV for one combination and checkpoint it.

    Returns the pickled dict: model, use_log, dataset_tag, iqr_k, fold_rmse, fold_best_params,
    rmse_mean, rmse_std, vs_baseline_pct.
    """
    label = combination_stem(model_name, use_log, ctx.dataset_tag)
    path = checkpoint_path(model_name, use_log, ctx.dataset_tag)
    if path.exists() and not force:
        log.info("[skip] %s: checkpoint exists", label)
        with open(path, "rb") as f:
            return pickle.load(f)

    spec = variants.variant_spec(build_model_specs()[model_name], model_name, ctx.variant)
    prep, prep_grid = variants.variant_prep(ctx.variant, ctx.feature_cols)
    param_grid = spec["param_grid"] | prep_grid
    n_combos = 1
    for values in param_grid.values():
        n_combos *= len(values)
    log.info("[run]  %s: grid=%d candidates x %d inner folds, %d outer folds",
             label, n_combos, ctx.inner_cv.get_n_splits(), ctx.outer_cv.get_n_splits(ctx.train))

    start = time.time()

    def on_fold(done: int, total: int, fold_rmse: float) -> None:
        elapsed = time.time() - start
        eta = elapsed / done * (total - done)
        log.info("       %s fold %d/%d  rmse=%s  elapsed=%s  eta=%s",
                 label, done, total, f"{fold_rmse:,.0f}",
                 timedelta(seconds=int(elapsed)), timedelta(seconds=int(eta)))

    fold_rmse, fold_best_params, predictions = ev.nested_cv(
        ctx.train,
        estimator=spec["estimator"],
        param_grid=param_grid,
        feature_cols=ctx.feature_cols,
        outer_cv=ctx.outer_cv,
        inner_cv=ctx.inner_cv,
        iqr_k=ctx.iqr_k,
        scale=spec["scale"],
        use_log=use_log,
        on_fold=on_fold,
        prep=prep,
    )

    rmse_mean = float(fold_rmse.mean())
    baseline_rmse = ctx.baseline["rmse"]
    result = {
        "model": model_name,
        "use_log": use_log,
        "dataset_tag": ctx.dataset_tag,
        "iqr_k": ctx.iqr_k,
        "fold_rmse": fold_rmse,
        "fold_best_params": fold_best_params,
        "rmse_mean": rmse_mean,
        "rmse_std": float(fold_rmse.std()),
        "vs_baseline_pct": (rmse_mean - baseline_rmse) / baseline_rmse * 100,
    }

    # Outputs first, checkpoint last: the pickle is the "done" marker for resuming.
    pred_path = predictions_path(model_name, use_log, ctx.dataset_tag)
    predictions.to_csv(pred_path, index=False)
    ev.summarize_rows(predictions).to_csv(pred_path.with_name(pred_path.name.replace("_predictions", "_row_summary")), index=False)
    pd.DataFrame({
        "fold": range(len(fold_rmse)),
        "rmse": fold_rmse,
        "best_params": [json.dumps(p, default=float) for p in fold_best_params],
    }).to_csv(pred_path.with_name(pred_path.name.replace("_predictions", "_folds")), index=False)
    setup.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    log.info("[done] %s: rmse_mean=%s (%+.1f%% vs baseline) in %s",
             label, f"{rmse_mean:,.0f}", result["vs_baseline_pct"], timedelta(seconds=int(time.time() - start)))
    return result


def run_variant_plan(variant: str, force: bool = False, dataset: str = "") -> None:
    """Every combination of `variants.VARIANT_PLANS[variant]` (all variants if `variant == "all"`),
    each isolated so one failure doesn't stop the rest."""
    names = list(variants.VARIANT_PLANS) if variant == "all" else [variant]
    failed = []
    t0 = time.time()
    for name in names:
        ctx = setup.build_context(None, dataset, name)
        log.info("variant %s (%s); baseline RMSE (%s): %s", name, variants.VARIANTS[name],
                 ctx.baseline["label"], f"{ctx.baseline['rmse']:,.0f}")
        for model_name, use_log in variants.VARIANT_PLANS[name]:
            label = combination_stem(model_name, use_log, ctx.dataset_tag)
            log.info("=== %s ===", label)
            try:
                run_combination(model_name, use_log, ctx, force=force)
            except Exception as exc:
                log.info("[error] %s: %r (logged to errors.log, continuing)", label, exc)
                log_error(label, exc)
                failed.append(label)
    log.info("ALL DONE in %s. Failed: %s", timedelta(seconds=int(time.time() - t0)), failed or "none")


def run_plan(force: bool = False, dataset: str = "") -> None:
    """Every combination in `PLAN` (or `DATASET_PLANS[dataset]`), each isolated so one failure
    doesn't stop the rest."""
    plan = DATASET_PLANS[dataset] if dataset else PLAN
    contexts: dict[float | None, setup.EvalContext] = {}
    failed = []
    t0 = time.time()
    for i, (model_name, use_log, iqr_k) in enumerate(plan, start=1):
        if iqr_k not in contexts:
            contexts[iqr_k] = setup.build_context(iqr_k, dataset)
            log.info("baseline RMSE (%s): %s", contexts[iqr_k].baseline["label"],
                     f"{contexts[iqr_k].baseline['rmse']:,.0f}")
        ctx = contexts[iqr_k]
        label = combination_stem(model_name, use_log, ctx.dataset_tag)
        log.info("=== combination %d/%d: %s ===", i, len(plan), label)
        try:
            run_combination(model_name, use_log, ctx, force=force)
        except Exception as exc:
            log.info("[error] %s: %r (logged to errors.log, continuing)", label, exc)
            log_error(label, exc)
            failed.append(label)
    log.info("ALL DONE in %s. Failed: %s", timedelta(seconds=int(time.time() - t0)), failed or "none")


def _str2bool(value: str) -> bool:
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(build_model_specs()), help="run this model alone")
    parser.add_argument("--use_log", type=_str2bool, help="true/false; required together with --model")
    parser.add_argument("--iqr", type=float, help="IQR-cap multiplier (e.g. 1.5); only with --model")
    parser.add_argument("--dataset", default="", choices=[d for d in setup.DATASETS if d],
                        help="alternative processed table; alone it runs that dataset's plan (DATASET_PLANS)")
    parser.add_argument("--variant", default="", choices=["all", *variants.VARIANTS],
                        help="feature-treatment experiment (variants.py); alone it runs that variant's plan")
    parser.add_argument("--force", action="store_true", help="recompute even if a checkpoint exists")
    args = parser.parse_args()

    if (args.model is None) != (args.use_log is None):
        parser.error("--model and --use_log must be given together")
    if args.iqr is not None and args.model is None:
        parser.error("--iqr only applies to a single --model run; the full plan already includes capped runs")

    if args.variant == "all" and args.model is not None:
        parser.error("--variant all runs the whole variant plan; drop --model")

    setup_logging()
    if args.model is not None:
        ctx = setup.build_context(args.iqr, args.dataset, args.variant)
        run_combination(args.model, args.use_log, ctx, force=args.force)
    elif args.variant:
        run_variant_plan(args.variant, force=args.force, dataset=args.dataset)
    else:
        run_plan(force=args.force, dataset=args.dataset)


if __name__ == "__main__":
    main()
