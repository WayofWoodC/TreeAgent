from __future__ import annotations

import pandas as pd

from pe_pipeline.features.quarter_aggregation import add_quarter_end


def build_pe_feature_table(pe_ratios: pd.DataFrame) -> pd.DataFrame:
    if pe_ratios.empty:
        return pd.DataFrame(columns=["ticker", "quarter_end", "pe_ratio"])

    frame = add_quarter_end(pe_ratios)
    frame = frame.sort_values(["ticker", "quarter_end", "date"]).drop_duplicates(["ticker", "quarter_end"], keep="last")
    frame["pe_ratio"] = pd.to_numeric(frame["pe_ratio"], errors="coerce")
    frame["pe_lag_1q"] = frame.groupby("ticker")["pe_ratio"].shift(1)
    frame["pe_lag_2q"] = frame.groupby("ticker")["pe_ratio"].shift(2)
    frame["pe_change_1q"] = frame["pe_lag_1q"] - frame["pe_lag_2q"]
    frame["target_pe_current"] = frame["pe_ratio"]
    frame["target_pe_change"] = frame["pe_ratio"] - frame["pe_lag_1q"]
    return frame[
        [
            "ticker",
            "quarter_end",
            "pe_ratio",
            "pe_lag_1q",
            "pe_lag_2q",
            "pe_change_1q",
            "target_pe_current",
            "target_pe_change",
        ]
    ]
