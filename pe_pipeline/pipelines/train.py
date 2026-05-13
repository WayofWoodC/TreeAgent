from __future__ import annotations

import logging

from pe_pipeline.config import Settings
from pe_pipeline.modeling.artifacts import save_model_artifacts
from pe_pipeline.modeling.importance import feature_importance_frame
from pe_pipeline.modeling.splits import time_based_split
from pe_pipeline.modeling.train_xgb import train_xgboost_model
from pe_pipeline.utils.io import load_json, load_parquet, save_csv, save_json


def run_train(settings: Settings, logger: logging.Logger, target_name: str) -> None:
    dataset = load_parquet(settings.processed_dir / "modeling_dataset.parquet")
    splits = time_based_split(dataset)
    logger.info("Training rows=%s validation rows=%s test rows=%s", len(splits["train"]), len(splits["val"]), len(splits["test"]))

    best_params_path = settings.reports_dir / "best_params.json"
    train_params = None
    if best_params_path.exists():
        loaded = load_json(best_params_path)
        train_params = {key: value for key, value in loaded.items() if key in {"n_estimators", "learning_rate", "max_depth", "min_child_weight", "gamma", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "early_stopping_rounds"}}
        logger.info("Using tuned hyperparameters from %s", best_params_path)

    result = train_xgboost_model(
        splits["train"],
        splits["val"],
        splits["test"],
        target_name=target_name,
        params=train_params,
        random_state=settings.random_state,
    )
    importance = feature_importance_frame(result.model, result.feature_columns)

    save_model_artifacts(
        result.model,
        result.imputer,
        {
            "target_name": target_name,
            "metrics": result.metrics,
            "feature_columns": result.feature_columns,
            "train_params": train_params or {},
        },
        settings.models_dir / "xgb_model.json",
        settings.models_dir / "preprocessor.joblib",
        settings.models_dir / "model_metadata.json",
    )
    save_csv(importance, settings.reports_dir / "feature_importance.csv")
    training_metrics_payload = {
        **result.metrics,
        "best_params_used": train_params or {},
    }
    save_json(training_metrics_payload, settings.reports_dir / "training_metrics.json")
    save_csv(result.validation_predictions, settings.reports_dir / "validation_predictions.csv")
    save_csv(result.test_predictions, settings.reports_dir / "test_predictions.csv")
