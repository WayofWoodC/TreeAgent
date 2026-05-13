from __future__ import annotations

import math

import pandas as pd

from pe_pipeline.modeling.splits import walk_forward_quarter_folds
from pe_pipeline.modeling.train_xgb import train_xgboost_model


def tune_xgboost(
    dataset: pd.DataFrame,
    target_name: str,
    metric_name: str = "rmse",
    n_trials: int = 25,
    random_state: int = 42,
) -> tuple[dict, pd.DataFrame]:
    import optuna

    unique_quarters = sorted(dataset["quarter_end"].dropna().unique())
    min_train_quarters = max(8, min(16, max(len(unique_quarters) // 2, 8)))
    folds = walk_forward_quarter_folds(dataset, min_train_quarters=min_train_quarters, val_quarters=4)
    results: list[dict] = []
    if not folds:
        raise ValueError("Not enough quarters to run walk-forward cross-validation")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "subsample": trial.suggest_float("subsample", 0.55, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 50.0, log=True),
            "early_stopping_rounds": 50,
        }
        rmse_scores = []
        ic_scores = []
        fold_iterations = []
        for fold_idx, (train_df, val_df) in enumerate(folds, start=1):
            result = train_xgboost_model(train_df, val_df, val_df, target_name=target_name, params=params, random_state=random_state)
            rmse_scores.append(result.metrics["val_rmse"])
            ic_scores.append(result.metrics["val_spearman_ic"])
            best_iteration = getattr(result.model, "best_iteration", None)
            fold_iterations.append(float(best_iteration) if best_iteration is not None else math.nan)
            results.append(
                {
                    "trial": trial.number,
                    "fold": fold_idx,
                    "val_rmse": result.metrics["val_rmse"],
                    "val_spearman_ic": result.metrics["val_spearman_ic"],
                    "best_iteration": best_iteration,
                    **params,
                }
            )
        mean_rmse = float(sum(rmse_scores) / len(rmse_scores))
        mean_ic = float(sum(ic_scores) / len(ic_scores))
        trial.set_user_attr("cv_rmse_mean", mean_rmse)
        trial.set_user_attr("cv_ic_mean", mean_ic)
        trial.set_user_attr("cv_best_iteration_mean", float(pd.Series(fold_iterations).dropna().mean()) if pd.Series(fold_iterations).dropna().size else math.nan)
        return mean_rmse if metric_name == "rmse" else -mean_ic

    direction = "minimize"
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)
    best_params = {
        **study.best_params,
        "cv_metric": metric_name,
        "cv_rmse_mean": study.best_trial.user_attrs.get("cv_rmse_mean"),
        "cv_ic_mean": study.best_trial.user_attrs.get("cv_ic_mean"),
        "cv_best_iteration_mean": study.best_trial.user_attrs.get("cv_best_iteration_mean"),
        "n_folds": len(folds),
    }
    trial_results = pd.DataFrame(results)
    trial_summary = (
        trial_results.groupby("trial", as_index=False)
        .agg(
            cv_rmse_mean=("val_rmse", "mean"),
            cv_rmse_std=("val_rmse", "std"),
            cv_ic_mean=("val_spearman_ic", "mean"),
            cv_ic_std=("val_spearman_ic", "std"),
            cv_best_iteration_mean=("best_iteration", "mean"),
            n_estimators=("n_estimators", "first"),
            learning_rate=("learning_rate", "first"),
            max_depth=("max_depth", "first"),
            min_child_weight=("min_child_weight", "first"),
            gamma=("gamma", "first"),
            subsample=("subsample", "first"),
            colsample_bytree=("colsample_bytree", "first"),
            reg_alpha=("reg_alpha", "first"),
            reg_lambda=("reg_lambda", "first"),
        )
    )
    sort_column = "cv_rmse_mean" if metric_name == "rmse" else "cv_ic_mean"
    ascending = metric_name == "rmse"
    return best_params, trial_summary.sort_values(sort_column, ascending=ascending).reset_index(drop=True)
