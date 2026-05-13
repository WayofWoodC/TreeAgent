from __future__ import annotations

import logging

from pe_pipeline.config import Settings
from pe_pipeline.modeling.splits import development_dataset
from pe_pipeline.modeling.tune_optuna import tune_xgboost
from pe_pipeline.utils.io import load_parquet, save_csv, save_json


def run_tune(settings: Settings, logger: logging.Logger, target_name: str, metric: str, n_trials: int) -> None:
    dataset = load_parquet(settings.processed_dir / "modeling_dataset.parquet")
    dev_dataset = development_dataset(dataset)
    logger.info("Starting walk-forward hyperparameter tuning with %s trials on %s development rows", n_trials, len(dev_dataset))
    best_params, trial_results = tune_xgboost(
        dataset=dev_dataset,
        target_name=target_name,
        metric_name=metric,
        n_trials=n_trials,
        random_state=settings.random_state,
    )
    save_csv(trial_results, settings.reports_dir / "tuning_results.csv")
    save_json(best_params, settings.reports_dir / "best_params.json")
