from __future__ import annotations

import pandas as pd


FRED_SERIES = {
    "AAA10Y": "AAA10Y",
    "BAA10Y": "BAA10Y",
    "DCOILWTICO": "DCOILWTICO",
    "T3MFF": "T3MFF",
    "T10Y3M": "T10Y3M",
    "EFFR": "DFF",
    "INDPRO": "INDPRO",
    "CPI": "CPIAUCSL",
    "UNRATE": "UNRATE",
    "M2SL": "M2SL",
    "UMCSENT": "UMCSENT",
    "CSUSHPISA": "CSUSHPISA",
}


def fetch_fred_series(series_name: str, fred_code: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_code}"
    frame = pd.read_csv(url)
    frame.columns = ["date", "value"]
    frame["series_name"] = series_name
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame
