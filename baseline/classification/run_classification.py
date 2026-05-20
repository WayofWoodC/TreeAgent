from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from xgboost import XGBClassifier

from pe_pipeline.backtesting.reports import (
    save_backtest_plot,
    save_equity_curve_plot,
    stats_frame,
    summarize_backtest,
)
from pe_pipeline.backtesting.returns import attach_forward_return
from pe_pipeline.config import get_settings
from pe_pipeline.logging_utils import setup_logging
from pe_pipeline.modeling.preprocess import feature_columns_from_dataset, fit_preprocessor
from pe_pipeline.modeling.splits import time_based_split
from pe_pipeline.pipelines.build_dataset import run_build_dataset
from pe_pipeline.utils.io import load_parquet, save_csv, save_json


CLASS_LABELS = {
    0: "down",
    1: "flat",
    2: "up",
}


def _thresholds_from_train(target: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(target, errors="coerce").dropna()
    if clean.empty:
        raise ValueError("Training target is empty; cannot build tercile thresholds.")
    lower = float(clean.quantile(1.0 / 3.0))
    upper = float(clean.quantile(2.0 / 3.0))
    return lower, upper


def _classify_target(target: pd.Series, lower: float, upper: float) -> pd.Series:
    values = pd.to_numeric(target, errors="coerce")
    labels = pd.Series(np.full(len(values), np.nan), index=values.index, dtype=float)
    labels.loc[values <= lower] = 0.0
    labels.loc[(values > lower) & (values <= upper)] = 1.0
    labels.loc[values > upper] = 2.0
    return labels.astype("Int64")


def _prediction_frame(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    predicted_class: np.ndarray,
    true_class: pd.Series,
    return_column: str,
) -> pd.DataFrame:
    columns = ["ticker", "quarter_end", "target_pe_scaled_change", "pe_lag_1q", return_column]
    existing = [column for column in columns if column in frame.columns]
    result = frame[existing].copy()
    result["true_class"] = true_class.astype("Int64")
    result["true_label"] = result["true_class"].map(CLASS_LABELS)
    result["predicted_class"] = predicted_class.astype(int)
    result["predicted_label"] = pd.Series(predicted_class).map(CLASS_LABELS).to_numpy()
    result["prob_down"] = probabilities[:, 0]
    result["prob_flat"] = probabilities[:, 1]
    result["prob_up"] = probabilities[:, 2]
    result["prediction"] = probabilities[:, 2] - probabilities[:, 0]
    result["signal"] = result["prediction"]
    return result


def _classification_metrics(y_true: pd.Series, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    truth = y_true.dropna().astype(int)
    pred = pd.Series(y_pred, index=y_true.index).loc[truth.index].astype(int)
    return {
        f"{prefix}_accuracy": float(accuracy_score(truth, pred)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        f"{prefix}_macro_f1": float(f1_score(truth, pred, average="macro")),
    }


def _run_classification_backtest(predictions: pd.DataFrame, return_column: str) -> pd.DataFrame:
    rows: list[dict[str, float | str | pd.Timestamp | bool]] = []
    for quarter_end, frame in predictions.groupby("quarter_end"):
        working = frame.copy()
        working["side"] = "flat"
        working.loc[working["predicted_class"] == 2, "side"] = "long"
        working.loc[working["predicted_class"] == 0, "side"] = "short"

        long_bucket = working.loc[working["side"] == "long", return_column].dropna()
        short_bucket = working.loc[working["side"] == "short", return_column].dropna()

        long_return = float(long_bucket.mean()) if not long_bucket.empty else 0.0
        short_asset_return = float(short_bucket.mean()) if not short_bucket.empty else 0.0
        short_return = -short_asset_return
        long_short_return = 0.5 * long_return + 0.5 * short_return

        rows.append(
            {
                "quarter_end": quarter_end,
                "long_count": int((working["side"] == "long").sum()),
                "short_count": int((working["side"] == "short").sum()),
                "flat_count": int((working["side"] == "flat").sum()),
                "long_return": long_return,
                "short_asset_return": short_asset_return,
                "short_return": short_return,
                "long_short_return": long_short_return,
                "is_initial_row": False,
            }
        )

    result = pd.DataFrame(rows).sort_values("quarter_end").reset_index(drop=True)
    if result.empty:
        return result

    result["long_ret_cum"] = (1.0 + result["long_return"].fillna(0.0)).cumprod() - 1.0
    result["short_ret_cum"] = (1.0 + result["short_return"].fillna(0.0)).cumprod() - 1.0
    result["ls_ret_cum"] = (1.0 + result["long_short_return"].fillna(0.0)).cumprod() - 1.0
    result["long_equity_curve"] = result["long_ret_cum"] + 1.0
    result["short_equity_curve"] = result["short_ret_cum"] + 1.0
    result["ls_equity_curve"] = result["ls_ret_cum"] + 1.0
    result["equity_curve"] = result["ls_equity_curve"]

    start_row = {
        "quarter_end": pd.to_datetime(result["quarter_end"].iloc[0]) - pd.offsets.QuarterEnd(1),
        "long_count": 0,
        "short_count": 0,
        "flat_count": 0,
        "long_return": 0.0,
        "short_asset_return": 0.0,
        "short_return": 0.0,
        "long_short_return": 0.0,
        "long_ret_cum": 0.0,
        "short_ret_cum": 0.0,
        "ls_ret_cum": 0.0,
        "long_equity_curve": 1.0,
        "short_equity_curve": 1.0,
        "ls_equity_curve": 1.0,
        "equity_curve": 1.0,
        "is_initial_row": True,
    }
    return pd.concat([pd.DataFrame([start_row]), result], ignore_index=True)


def run_classification_baseline(start_year: int | None = None, periods: int | None = None) -> Path:
    settings = get_settings()
    output_dir = settings.baseline_dir / "classification"
    logger = setup_logging(output_dir / "classification.log")

    logger.info("Rebuilding dataset for classification workflow")
    run_build_dataset(settings, logger, target_name="target_pe_scaled_change", start_year=start_year)
    dataset = load_parquet(settings.processed_dir / "modeling_dataset.parquet")
    splits = time_based_split(dataset)

    train_df = splits["train"].copy()
    val_df = splits["val"].copy()
    test_df = splits["test"].copy()

    lower, upper = _thresholds_from_train(train_df["target_pe_scaled_change"])
    train_df["target_class"] = _classify_target(train_df["target_pe_scaled_change"], lower, upper)
    val_df["target_class"] = _classify_target(val_df["target_pe_scaled_change"], lower, upper)
    test_df["target_class"] = _classify_target(test_df["target_pe_scaled_change"], lower, upper)

    feature_columns = feature_columns_from_dataset(dataset, target_name="target_pe_scaled_change")
    preprocessor = fit_preprocessor(train_df, feature_columns)
    x_train = preprocessor.fit_transform(train_df)
    x_val = preprocessor.transform(val_df, update_history=True)
    x_test = preprocessor.transform(test_df, update_history=True)

    y_train = train_df["target_class"].astype(int)
    y_val = val_df["target_class"].astype(int)
    y_test = test_df["target_class"].astype(int)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=5,
        gamma=0.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=settings.random_state,
        tree_method="hist",
        early_stopping_rounds=50,
        eval_metric="mlogloss",
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    val_prob = model.predict_proba(x_val)
    test_prob = model.predict_proba(x_test)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    return_column = "next_quarter_return"
    if periods is not None:
        prices = load_parquet(settings.raw_dir / "prices" / "daily_prices.parquet")
        val_df = attach_forward_return(val_df, prices, periods=periods, output_column="holding_period_return")
        test_df = attach_forward_return(test_df, prices, periods=periods, output_column="holding_period_return")
        return_column = "holding_period_return"

    val_predictions = _prediction_frame(val_df, val_prob, val_pred, val_df["target_class"], return_column)
    test_predictions = _prediction_frame(test_df, test_prob, test_pred, test_df["target_class"], return_column)

    metrics = {
        "target_name": "target_pe_scaled_change",
        "classification_scheme": "train_terciles",
        "start_year": start_year,
        "holding_period_days": periods,
        "return_column": return_column,
        "lower_threshold": lower,
        "upper_threshold": upper,
        "train_class_counts": {CLASS_LABELS[k]: int((y_train == k).sum()) for k in sorted(CLASS_LABELS)},
    }
    metrics.update(_classification_metrics(val_df["target_class"], val_pred, "val"))
    metrics.update(_classification_metrics(test_df["target_class"], test_pred, "test"))

    backtest_returns = _run_classification_backtest(test_predictions, return_column=return_column)
    summary = summarize_backtest(backtest_returns, periods_per_year=4.0)
    summary.update(
        {
            "target_name": "target_pe_scaled_change",
            "classification_scheme": "train_terciles",
            "start_year": start_year,
            "holding_period_days": periods,
            "return_column": return_column,
            "periods_per_year": 4.0,
            "lower_threshold": lower,
            "upper_threshold": upper,
        }
    )

    save_json(metrics, output_dir / "classification_metrics.json")
    save_json(
        {
            "lower_threshold": lower,
            "upper_threshold": upper,
            "class_labels": CLASS_LABELS,
        },
        output_dir / "label_thresholds.json",
    )
    save_csv(val_predictions, output_dir / "validation_predictions.csv")
    save_csv(test_predictions, output_dir / "test_predictions.csv")
    save_csv(backtest_returns, output_dir / "quarterly_returns.csv")
    save_json(summary, output_dir / "portfolio_summary.json")
    save_csv(stats_frame(summary), output_dir / "portfolio_stats.csv")
    save_backtest_plot(backtest_returns, output_dir / "quarterly_returns_plot.svg")
    save_equity_curve_plot(backtest_returns, output_dir / "equity_curves_plot.svg")

    model.save_model(output_dir / "xgb_classifier.json")
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    with (output_dir / "model_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_type": "xgb_classifier",
                "target_name": "target_pe_scaled_change",
                "classification_scheme": "train_terciles",
                "feature_columns": feature_columns,
                "lower_threshold": lower,
                "upper_threshold": upper,
                "metrics": metrics,
            },
            handle,
            indent=2,
            default=str,
        )

    logger.info("Classification baseline outputs written to %s", output_dir)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3-class PE scaled-change classifier baseline")
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--periods", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_classification_baseline(start_year=args.start_year, periods=args.periods)


if __name__ == "__main__":
    main()
