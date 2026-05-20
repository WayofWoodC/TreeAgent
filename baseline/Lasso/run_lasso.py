from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pe_pipeline.backtesting.engine import run_quarterly_backtest
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
from pe_pipeline.utils.metrics import r2_score, rmse, spearman_ic


ALPHA_GRID = [
    0.0001,
    0.0003,
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
]


def _signal_from_prediction(frame: pd.DataFrame) -> pd.Series:
    lag = pd.to_numeric(frame["pe_lag_1q"], errors="coerce")
    pred = pd.to_numeric(frame["prediction"], errors="coerce")
    signal = pd.Series(0.0, index=frame.index, dtype=float)
    valid = lag.abs() > 0
    signal.loc[valid] = (pred.loc[valid] - lag.loc[valid]) / lag.loc[valid].abs()
    return signal


def _metrics_payload(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    truth = pd.to_numeric(frame["target_pe_current"], errors="coerce")
    pred = pd.to_numeric(frame["prediction"], errors="coerce")
    mask = truth.notna() & pred.notna()
    y_true = truth.loc[mask].to_numpy(dtype=float)
    y_pred = pred.loc[mask].to_numpy(dtype=float)
    return {
        f"{prefix}_rmse": rmse(y_true, y_pred),
        f"{prefix}_spearman_ic": spearman_ic(y_true, y_pred),
        f"{prefix}_r2": r2_score(y_true, y_pred),
    }


def _make_linear_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
            ("model", Lasso(alpha=alpha, max_iter=20000, random_state=42)),
        ]
    )


def _fit_best_lasso(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[Pipeline, float, list[dict[str, float]]]:
    trial_rows: list[dict[str, float]] = []
    best_model: Pipeline | None = None
    best_alpha: float | None = None
    best_rmse = float("inf")

    for alpha in ALPHA_GRID:
        model = _make_linear_pipeline(alpha)
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        val_score = rmse(y_val.to_numpy(dtype=float), np.asarray(val_pred, dtype=float))
        trial_rows.append({"alpha": alpha, "val_rmse": val_score})
        if val_score < best_rmse:
            best_rmse = val_score
            best_alpha = alpha
            best_model = model

    if best_model is None or best_alpha is None:
        raise RuntimeError("Failed to select a Lasso model.")

    return best_model, best_alpha, trial_rows


def run_lasso_baseline(
    start_year: int | None = None,
    periods: int | None = None,
    top_n: int = 5,
) -> Path:
    settings = get_settings()
    output_dir = settings.baseline_dir / "Lasso"
    logger = setup_logging(output_dir / "lasso.log")

    logger.info("Rebuilding dataset for Lasso baseline")
    run_build_dataset(settings, logger, target_name="target_pe_current", start_year=start_year)
    dataset = load_parquet(settings.processed_dir / "modeling_dataset.parquet")
    splits = time_based_split(dataset)

    train_df = splits["train"].copy()
    val_df = splits["val"].copy()
    test_df = splits["test"].copy()

    feature_columns = feature_columns_from_dataset(dataset, target_name="target_pe_current")
    preprocessor = fit_preprocessor(train_df, feature_columns)
    x_train = preprocessor.fit_transform(train_df)
    x_val = preprocessor.transform(val_df, update_history=True)
    x_test = preprocessor.transform(test_df, update_history=True)

    y_train = pd.to_numeric(train_df["target_pe_current"], errors="coerce")
    y_val = pd.to_numeric(val_df["target_pe_current"], errors="coerce")
    y_test = pd.to_numeric(test_df["target_pe_current"], errors="coerce")

    model, best_alpha, alpha_trials = _fit_best_lasso(x_train, y_train, x_val, y_val)

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    return_column = "next_quarter_return"
    if periods is not None:
        prices = load_parquet(settings.raw_dir / "prices" / "daily_prices.parquet")
        test_df = attach_forward_return(
            predictions=test_df,
            prices=prices,
            periods=periods,
            output_column="holding_period_return",
        )
        return_column = "holding_period_return"

    val_predictions = val_df[["ticker", "quarter_end", "pe_lag_1q", "target_pe_current"]].copy()
    val_predictions["prediction"] = val_pred
    val_predictions["signal"] = _signal_from_prediction(val_predictions)

    test_predictions = test_df[["ticker", "quarter_end", "pe_lag_1q", "target_pe_current", return_column]].copy()
    test_predictions["prediction"] = test_pred
    test_predictions["signal"] = _signal_from_prediction(test_predictions)

    metrics = {
        "model_type": "lasso_regression",
        "target_name": "target_pe_current",
        "start_year": start_year,
        "holding_period_days": periods,
        "return_column": return_column,
        "best_alpha": best_alpha,
    }
    metrics.update(_metrics_payload(val_predictions, "val"))
    metrics.update(_metrics_payload(test_predictions, "test"))

    backtest_returns = run_quarterly_backtest(
        predictions=test_predictions,
        top_n=top_n,
        return_column=return_column,
        signal_mode="current_pe_relative_change",
    )
    summary = summarize_backtest(backtest_returns, periods_per_year=4.0)
    summary.update(
        {
            "model_type": "lasso_regression",
            "target_name": "target_pe_current",
            "signal_mode": "current_pe_relative_change",
            "start_year": start_year,
            "holding_period_days": periods,
            "return_column": return_column,
            "periods_per_year": 4.0,
            "top_n": top_n,
            "best_alpha": best_alpha,
        }
    )

    coefficients = pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient": model.named_steps["model"].coef_,
            "abs_coefficient": np.abs(model.named_steps["model"].coef_),
        }
    ).sort_values("abs_coefficient", ascending=False)

    save_json(metrics, output_dir / "lasso_metrics.json")
    save_json({"alpha_grid": ALPHA_GRID, "trials": alpha_trials}, output_dir / "alpha_search.json")
    save_csv(val_predictions, output_dir / "validation_predictions.csv")
    save_csv(test_predictions, output_dir / "test_predictions.csv")
    save_csv(backtest_returns, output_dir / "quarterly_returns.csv")
    save_json(summary, output_dir / "portfolio_summary.json")
    save_csv(stats_frame(summary), output_dir / "portfolio_stats.csv")
    save_csv(coefficients, output_dir / "feature_coefficients.csv")
    save_backtest_plot(backtest_returns, output_dir / "quarterly_returns_plot.svg")
    save_equity_curve_plot(backtest_returns, output_dir / "equity_curves_plot.svg")

    joblib.dump(model, output_dir / "lasso_pipeline.joblib")
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    save_json(
        {
            "model_type": "lasso_regression",
            "target_name": "target_pe_current",
            "signal_mode": "current_pe_relative_change",
            "feature_columns": feature_columns,
            "best_alpha": best_alpha,
            "metrics": metrics,
        },
        output_dir / "model_metadata.json",
    )

    logger.info("Lasso baseline outputs written to %s", output_dir)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lasso direct-PE baseline")
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--periods", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_lasso_baseline(start_year=args.start_year, periods=args.periods, top_n=args.top_n)


if __name__ == "__main__":
    main()
