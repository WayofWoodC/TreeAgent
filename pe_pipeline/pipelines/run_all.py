from __future__ import annotations

import logging

from pe_pipeline.config import Settings
from pe_pipeline.pipelines.backtest import run_backtest
from pe_pipeline.pipelines.build_dataset import run_build_dataset
from pe_pipeline.pipelines.fetch_data import run_fetch_data
from pe_pipeline.pipelines.predict_latest import run_predict_latest
from pe_pipeline.pipelines.train import run_train


def run_all(
    settings: Settings,
    logger: logging.Logger,
    target_name: str,
    top_n: int,
    limit: int | None = None,
    start_year: int | None = None,
    periods: int | None = None,
) -> None:
    run_fetch_data(settings, logger, limit=limit)
    run_build_dataset(settings, logger, target_name=target_name, start_year=start_year)
    run_train(settings, logger, target_name=target_name)
    run_backtest(settings, logger, top_n=top_n, periods=periods)
    run_predict_latest(settings, logger)
