from __future__ import annotations

import logging
from dataclasses import asdict

from pe_pipeline.config import Settings
from pe_pipeline.features.daily_features import build_daily_features
from pe_pipeline.features.dataset_builder import build_modeling_dataset
from pe_pipeline.features.fundamentals_features import build_fundamental_features
from pe_pipeline.features.monthly_features import build_monthly_macro_features
from pe_pipeline.features.pe_features import build_pe_feature_table
from pe_pipeline.utils.io import load_parquet, save_json, save_parquet


def run_build_dataset(settings: Settings, logger: logging.Logger, target_name: str, start_year: int | None = None) -> None:
    logger.info("Loading raw datasets")
    prices = load_parquet(settings.raw_dir / "prices" / "daily_prices.parquet")
    pe_ratios = load_parquet(settings.raw_dir / "pe_ratios" / "quarterly_pe_ratios.parquet")
    fundamentals = load_parquet(settings.raw_dir / "fundamentals" / "quarterly_fundamentals.parquet")
    macro = load_parquet(settings.raw_dir / "macro" / "macro_series.parquet")

    logger.info("Building quarterly feature blocks")
    daily_features, future_returns = build_daily_features(prices, settings.daily_windows)
    macro_features = build_monthly_macro_features(macro)
    fundamental_features = build_fundamental_features(fundamentals)
    pe_features = build_pe_feature_table(pe_ratios)

    save_parquet(daily_features, settings.intermediate_dir / "quarterly_prices.parquet")
    save_parquet(future_returns, settings.intermediate_dir / "quarterly_returns.parquet")
    save_parquet(macro_features, settings.intermediate_dir / "quarterly_macro.parquet")
    save_parquet(fundamental_features, settings.intermediate_dir / "quarterly_fundamentals.parquet")
    save_parquet(pe_features, settings.intermediate_dir / "quarterly_pe.parquet")

    dataset, drop_reports, missing_summary = build_modeling_dataset(
        daily_features=daily_features,
        monthly_macro_features=macro_features,
        fundamental_features=fundamental_features,
        pe_features=pe_features,
        future_returns=future_returns,
        epsilon=settings.epsilon,
    )
    if start_year is not None:
        cutoff = f"{start_year}-01-01"
        before_rows = len(dataset)
        dataset = dataset.loc[dataset["quarter_end"] >= cutoff].copy()
        dataset = dataset.sort_values(["quarter_end", "ticker"]).reset_index(drop=True)
        logger.info(
            "Applied start-year filter >= %s: kept %s of %s rows",
            cutoff,
            len(dataset),
            before_rows,
        )
    save_parquet(dataset, settings.processed_dir / "modeling_dataset.parquet")
    save_json(
        {
            "target_name": target_name,
            "start_year": start_year,
            "drop_reports": [asdict(report) for report in drop_reports],
            "missing_summary": missing_summary,
            "final_rows": len(dataset),
        },
        settings.reports_dir / "data_quality_report.json",
    )
