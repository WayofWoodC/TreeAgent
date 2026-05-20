from __future__ import annotations

import logging

import pandas as pd

from pe_pipeline.backtesting.engine import run_quarterly_backtest
from pe_pipeline.backtesting.returns import attach_forward_return
from pe_pipeline.backtesting.reports import save_backtest_plot, save_equity_curve_plot, stats_frame, summarize_backtest
from pe_pipeline.config import Settings
from pe_pipeline.modeling.splits import time_based_split
from pe_pipeline.pipelines.build_dataset import run_build_dataset
from pe_pipeline.utils.io import load_parquet, save_csv, save_json
from pe_pipeline.utils.metrics import r2_score, rmse, spearman_ic


def _baseline_prediction(frame: pd.DataFrame, target_name: str) -> pd.Series:
    if target_name == "target_pe_current":
        return pd.to_numeric(frame["pe_lag_1q"], errors="coerce")
    return pd.Series(0.0, index=frame.index, dtype=float)


def _metrics_payload(frame: pd.DataFrame, target_name: str, split_name: str) -> dict[str, float]:
    truth = pd.to_numeric(frame[target_name], errors="coerce")
    pred = pd.to_numeric(frame["prediction"], errors="coerce")
    mask = truth.notna() & pred.notna()
    y_true = truth.loc[mask].to_numpy(dtype=float)
    y_pred = pred.loc[mask].to_numpy(dtype=float)
    return {
        f"{split_name}_rmse": rmse(y_true, y_pred),
        f"{split_name}_spearman_ic": spearman_ic(y_true, y_pred),
        f"{split_name}_r2": r2_score(y_true, y_pred),
    }


def _periods_per_year(periods: int | None) -> float:
    return 4.0


def run_baseline(
    settings: Settings,
    logger: logging.Logger,
    target_name: str,
    top_n: int,
    start_year: int | None = None,
    periods: int | None = None,
) -> None:
    run_build_dataset(settings, logger, target_name=target_name, start_year=start_year)
    dataset = load_parquet(settings.processed_dir / "modeling_dataset.parquet")
    splits = time_based_split(dataset)
    logger.info(
        "Running pe_lag_1q baseline with rows train=%s val=%s test=%s",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )

    validation_predictions = splits["val"][["ticker", "quarter_end", target_name, "pe_lag_1q"]].copy()
    validation_predictions["prediction"] = _baseline_prediction(validation_predictions, target_name)

    test_predictions = splits["test"][["ticker", "quarter_end", target_name, "pe_lag_1q", "next_quarter_return"]].copy()
    test_predictions["prediction"] = _baseline_prediction(test_predictions, target_name)
    return_column = "next_quarter_return"
    if periods is not None:
        prices = load_parquet(settings.raw_dir / "prices" / "daily_prices.parquet")
        test_predictions = attach_forward_return(
            predictions=test_predictions,
            prices=prices,
            periods=periods,
            output_column="holding_period_return",
        )
        return_column = "holding_period_return"
    test_predictions["signal"] = pd.to_numeric(test_predictions["prediction"], errors="coerce")

    metrics = {}
    metrics.update(_metrics_payload(validation_predictions, target_name, "val"))
    metrics.update(_metrics_payload(test_predictions, target_name, "test"))
    metrics["baseline_name"] = "pe_lag_1q"
    metrics["target_name"] = target_name
    metrics["start_year"] = start_year
    metrics["holding_period_days"] = periods
    metrics["return_column"] = return_column

    backtest_returns = run_quarterly_backtest(test_predictions, top_n=top_n, return_column=return_column)
    summary = summarize_backtest(backtest_returns, periods_per_year=_periods_per_year(periods))
    summary["baseline_name"] = "pe_lag_1q"
    summary["target_name"] = target_name
    summary["signal_mode"] = "prediction"
    summary["start_year"] = start_year
    summary["holding_period_days"] = periods
    summary["return_column"] = return_column
    summary["periods_per_year"] = _periods_per_year(periods)

    baseline_dir = settings.baseline_dir / "prelag"
    save_json(metrics, baseline_dir / "baseline_metrics.json")
    save_csv(validation_predictions, baseline_dir / "validation_predictions.csv")
    save_csv(test_predictions, baseline_dir / "test_predictions.csv")
    quarterly_returns_path = baseline_dir / "quarterly_returns.csv"
    save_csv(backtest_returns, quarterly_returns_path)
    save_json(summary, baseline_dir / "portfolio_summary.json")
    save_csv(stats_frame(summary), baseline_dir / "portfolio_stats.csv")
    plot_source = pd.read_csv(quarterly_returns_path, parse_dates=["quarter_end"])
    save_backtest_plot(plot_source, baseline_dir / "quarterly_returns_plot.svg")
    save_equity_curve_plot(plot_source, baseline_dir / "equity_curves_plot.svg")
