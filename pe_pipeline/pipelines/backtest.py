from __future__ import annotations

import logging

from pe_pipeline.backtesting.engine import run_quarterly_backtest
from pe_pipeline.backtesting.returns import attach_forward_return
from pe_pipeline.backtesting.reports import save_backtest_plot, save_equity_curve_plot, stats_frame, summarize_backtest
from pe_pipeline.config import Settings
from pe_pipeline.utils.io import load_csv, load_json, load_parquet, save_csv, save_json


def _periods_per_year(periods: int | None) -> float:
    return 4.0


def _signal_mode_from_target_name(target_name: str) -> str:
    if target_name == "target_pe_current":
        return "current_pe_relative_change"
    if target_name == "target_pe_scaled_change":
        return "scaled_change_prediction"
    return "prediction"


def run_backtest(settings: Settings, logger: logging.Logger, top_n: int, periods: int | None = None) -> None:
    predictions = load_csv(settings.reports_dir / "test_predictions.csv")
    metadata = load_json(settings.models_dir / "model_metadata.json")
    target_name = metadata.get("target_name", "target_pe_current")
    signal_mode = _signal_mode_from_target_name(target_name)
    return_column = "next_quarter_return"
    if periods is not None:
        prices = load_parquet(settings.raw_dir / "prices" / "daily_prices.parquet")
        predictions = attach_forward_return(
            predictions=predictions,
            prices=prices,
            periods=periods,
            output_column="holding_period_return",
        )
        return_column = "holding_period_return"
        logger.info("Running %s-trading-day backtest on %s prediction rows", periods, len(predictions))
    else:
        logger.info("Running quarterly backtest on %s prediction rows", len(predictions))

    backtest_returns = run_quarterly_backtest(
        predictions,
        top_n=top_n,
        return_column=return_column,
        signal_mode=signal_mode,
    )
    periods_per_year = _periods_per_year(periods)
    summary = summarize_backtest(backtest_returns, periods_per_year=periods_per_year)
    summary["target_name"] = target_name
    summary["signal_mode"] = signal_mode
    summary["holding_period_days"] = periods
    summary["return_column"] = return_column
    summary["periods_per_year"] = periods_per_year
    quarterly_returns_path = settings.backtests_dir / "quarterly_returns.csv"
    save_csv(backtest_returns, quarterly_returns_path)
    save_json(summary, settings.backtests_dir / "portfolio_summary.json")
    save_csv(stats_frame(summary), settings.backtests_dir / "portfolio_stats.csv")
    plot_source = load_csv(quarterly_returns_path)
    save_backtest_plot(plot_source, settings.backtests_dir / "quarterly_returns_plot.svg")
    save_equity_curve_plot(plot_source, settings.backtests_dir / "equity_curves_plot.svg")
