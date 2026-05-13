from __future__ import annotations

import pandas as pd

from pe_pipeline.features.target_builder import add_target_columns
from pe_pipeline.utils.validation import DropReport, assert_no_duplicate_keys, dropna_with_report, require_columns, summarize_missing


def build_modeling_dataset(
    daily_features: pd.DataFrame,
    monthly_macro_features: pd.DataFrame,
    fundamental_features: pd.DataFrame,
    pe_features: pd.DataFrame,
    future_returns: pd.DataFrame,
    epsilon: float,
) -> tuple[pd.DataFrame, list[DropReport], dict[str, int]]:
    require_columns(pe_features, ["ticker", "quarter_end", "pe_ratio", "target_pe_current"], "pe_features")
    assert_no_duplicate_keys(pe_features, ["ticker", "quarter_end"], "pe_features")

    dataset = pe_features.merge(daily_features, on=["ticker", "quarter_end"], how="left")
    dataset = dataset.merge(fundamental_features, on=["ticker", "quarter_end"], how="left")
    dataset = dataset.merge(monthly_macro_features, on="quarter_end", how="left")
    dataset = dataset.merge(future_returns[["ticker", "quarter_end", "next_quarter_return"]], on=["ticker", "quarter_end"], how="left")
    dataset = add_target_columns(dataset, epsilon=epsilon)
    missing_summary = summarize_missing(
        dataset,
        include_columns=["pe_ratio", "target_pe_current", "target_pe_change", "target_pe_scaled_change"],
    )

    cleaned, drop_report = dropna_with_report(
        dataset,
        subset=["pe_ratio", "pe_lag_1q", "target_pe_current"],
        stage="final_dataset_target_alignment",
    )
    cleaned = cleaned.sort_values(["quarter_end", "ticker"]).reset_index(drop=True)
    return cleaned, [drop_report], missing_summary
