from __future__ import annotations

from datetime import datetime

import pandas as pd

from pe_pipeline.data_sources.base import HttpClient


def fetch_daily_prices(ticker: str, start_date: str, end_date: str, client: HttpClient) -> pd.DataFrame:
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    payload = client.get_json(
        url,
        params={"period1": start_ts, "period2": end_ts, "interval": "1d", "includeAdjustedClose": "true"},
        headers={"User-Agent": "Mozilla/5.0"},
    )
    result = payload.get("chart", {}).get("result", []) if isinstance(payload, dict) else []
    if not result:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adjClose", "volume", "ticker"])
    node = result[0]
    timestamps = node.get("timestamp", [])
    quote = node.get("indicators", {}).get("quote", [{}])[0]
    adjclose = node.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s"),
            "open": quote.get("open", []),
            "high": quote.get("high", []),
            "low": quote.get("low", []),
            "close": quote.get("close", []),
            "adjClose": adjclose,
            "volume": quote.get("volume", []),
        }
    )
    frame["ticker"] = ticker
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return frame.dropna(subset=["date"]).copy()
