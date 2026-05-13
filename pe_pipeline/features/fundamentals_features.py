from __future__ import annotations

import pandas as pd

from pe_pipeline.features.quarter_aggregation import add_quarter_end


def build_fundamental_features(fundamentals: pd.DataFrame) -> pd.DataFrame:
    if fundamentals.empty:
        return pd.DataFrame(columns=["ticker", "quarter_end"])

    frame = add_quarter_end(fundamentals)
    frame = frame.sort_values(["ticker", "quarter_end", "date"]).drop_duplicates(["ticker", "quarter_end"], keep="last")
    numeric_columns = [column for column in frame.columns if column not in {"date", "ticker", "quarter_end"}]
    rename_map = {column: column.lower() for column in numeric_columns}
    return frame[["ticker", "quarter_end", *numeric_columns]].rename(columns=rename_map)
