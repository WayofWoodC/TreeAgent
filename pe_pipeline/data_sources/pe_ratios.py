from __future__ import annotations

import pandas as pd

from pe_pipeline.data_sources.base import HttpClient


def fetch_quarterly_pe_ratios(ticker: str, api_key: str, client: HttpClient, limit: int = 120) -> pd.DataFrame:
    url = "https://finnhub.io/api/v1/stock/metric"
    payload = client.get_json(url, params={"symbol": ticker, "metric": "all", "token": api_key})
    quarterly = payload.get("series", {}).get("quarterly", {}) if isinstance(payload, dict) else {}
    pe_series = quarterly.get("peTTM", [])
    frame = pd.DataFrame(pe_series)
    if frame.empty or "period" not in frame.columns or "v" not in frame.columns:
        return pd.DataFrame(columns=["date", "ticker", "pe_ratio"])
    frame = frame.head(limit).copy()
    frame["date"] = frame["period"]
    frame["pe_ratio"] = frame["v"]
    frame["ticker"] = ticker
    return frame[["date", "ticker", "pe_ratio"]].copy()
