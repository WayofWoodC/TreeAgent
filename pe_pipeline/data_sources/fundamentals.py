from __future__ import annotations

import pandas as pd

from pe_pipeline.data_sources.base import HttpClient


def fetch_quarterly_fundamentals(ticker: str, api_key: str, client: HttpClient, limit: int = 120) -> pd.DataFrame:
    url = "https://finnhub.io/api/v1/stock/metric"
    payload = client.get_json(url, params={"symbol": ticker, "metric": "all", "token": api_key})
    quarterly = payload.get("series", {}).get("quarterly", {}) if isinstance(payload, dict) else {}
    if not quarterly:
        return pd.DataFrame(columns=["date", "ticker"])
    frames = []
    for feature_name, series in quarterly.items():
        series_frame = pd.DataFrame(series)
        if series_frame.empty or "period" not in series_frame.columns or "v" not in series_frame.columns:
            continue
        series_frame = series_frame.rename(columns={"period": "date", "v": feature_name})[["date", feature_name]]
        frames.append(series_frame)
    if not frames:
        return pd.DataFrame(columns=["date", "ticker"])
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")
    merged["ticker"] = ticker
    return merged.sort_values("date", ascending=False).head(limit).sort_values("date").reset_index(drop=True)
