from __future__ import annotations

import pandas as pd

from pe_pipeline.utils.dates import to_quarter_end


def add_quarter_end(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    result["quarter_end"] = to_quarter_end(result[date_col])
    return result
