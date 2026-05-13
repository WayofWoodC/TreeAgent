from __future__ import annotations

import pandas as pd


def to_quarter_end(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.to_period("Q").dt.end_time.dt.normalize()


def quarter_label(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.to_period("Q").astype(str)
