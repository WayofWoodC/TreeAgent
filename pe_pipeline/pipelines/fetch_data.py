from __future__ import annotations

import logging
from io import StringIO

import pandas as pd

from pe_pipeline.config import Settings
from pe_pipeline.data_sources.base import HttpClient
from pe_pipeline.data_sources.fundamentals import fetch_quarterly_fundamentals
from pe_pipeline.data_sources.macro import FRED_SERIES
from pe_pipeline.data_sources.pe_ratios import fetch_quarterly_pe_ratios
from pe_pipeline.data_sources.prices import fetch_daily_prices
from pe_pipeline.data_sources.universe import fetch_sp500_universe
from pe_pipeline.exceptions import PipelineError
from pe_pipeline.utils.io import load_parquet, save_csv, save_parquet


def _fetch_fred_frame_via_api(series_name: str, fred_code: str, fred_api_key: str, client: HttpClient) -> pd.DataFrame:
    url = "https://api.stlouisfed.org/fred/series/observations"
    payload = client.get_json(
        url,
        params={
            "series_id": fred_code,
            "api_key": fred_api_key,
            "file_type": "json",
            "observation_start": "1900-01-01",
            "sort_order": "asc",
        },
        headers={"User-Agent": "Mozilla/5.0"},
    )
    observations = payload.get("observations", []) if isinstance(payload, dict) else []
    frame = pd.DataFrame(observations)
    if frame.empty:
        return pd.DataFrame(columns=["date", "value", "series_name"])
    frame = frame[["date", "value"]].copy()
    frame["series_name"] = series_name
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame


def _fetch_fred_frame_via_graph(series_name: str, fred_code: str, client: HttpClient) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_code}"
    payload = client.get_text(url, headers={"User-Agent": "Mozilla/5.0"})
    frame = pd.read_csv(StringIO(payload))
    frame.columns = ["date", "value"]
    frame["series_name"] = series_name
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame


def _fetch_fred_frame(series_name: str, fred_code: str, client: HttpClient, fred_api_key: str | None) -> pd.DataFrame:
    if fred_api_key:
        try:
            return _fetch_fred_frame_via_api(series_name, fred_code, fred_api_key, client)
        except Exception:
            pass
    return _fetch_fred_frame_via_graph(series_name, fred_code, client)


def run_fetch_data(settings: Settings, logger: logging.Logger, limit: int | None = None) -> None:
    logger.info("Fetching S&P 500 universe")
    universe = fetch_sp500_universe()
    if limit is not None:
        universe = universe.head(limit).copy()
    save_parquet(universe, settings.raw_dir / "universe.parquet")
    save_csv(universe, settings.raw_dir / "universe.csv")

    if not settings.finnhub_api_key:
        raise PipelineError("FINNHUB_API_KEY is required for quarterly PE and fundamentals fetches")

    client = HttpClient()
    tickers = universe["ticker"].tolist()
    price_frames: list[pd.DataFrame] = []
    pe_frames: list[pd.DataFrame] = []
    fundamental_frames: list[pd.DataFrame] = []

    for idx, ticker in enumerate(tickers, start=1):
        logger.info("Fetching equity data for %s (%s/%s)", ticker, idx, len(tickers))
        try:
            price_frames.append(fetch_daily_prices(ticker, settings.start_date, settings.end_date, client))
            pe_frames.append(fetch_quarterly_pe_ratios(ticker, settings.finnhub_api_key, client))
            fundamental_frames.append(fetch_quarterly_fundamentals(ticker, settings.finnhub_api_key, client))
        except Exception as exc:
            logger.warning("Fetch failed for %s: %s", ticker, exc)

    prices = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame()
    pe_ratios = pd.concat(pe_frames, ignore_index=True) if pe_frames else pd.DataFrame()
    fundamentals = pd.concat(fundamental_frames, ignore_index=True) if fundamental_frames else pd.DataFrame()
    macro_frames: list[pd.DataFrame] = []
    for series_name, code in FRED_SERIES.items():
        try:
            macro_frames.append(_fetch_fred_frame(series_name, code, client, settings.fred_api_key))
        except Exception as exc:
            logger.warning("Macro fetch failed for %s (%s): %s", series_name, code, exc)

    macro = pd.concat(macro_frames, ignore_index=True) if macro_frames else pd.DataFrame()
    if macro.empty:
        macro_parquet_path = settings.raw_dir / "macro" / "macro_series.parquet"
        if macro_parquet_path.exists():
            logger.warning("All macro fetches failed, reusing existing raw macro dataset")
            macro = load_parquet(macro_parquet_path)

    save_parquet(prices, settings.raw_dir / "prices" / "daily_prices.parquet")
    save_csv(prices, settings.raw_dir / "prices" / "daily_prices.csv")
    save_parquet(pe_ratios, settings.raw_dir / "pe_ratios" / "quarterly_pe_ratios.parquet")
    save_csv(pe_ratios, settings.raw_dir / "pe_ratios" / "quarterly_pe_ratios.csv")
    save_parquet(fundamentals, settings.raw_dir / "fundamentals" / "quarterly_fundamentals.parquet")
    save_csv(fundamentals, settings.raw_dir / "fundamentals" / "quarterly_fundamentals.csv")
    save_parquet(macro, settings.raw_dir / "macro" / "macro_series.parquet")
    save_csv(macro, settings.raw_dir / "macro" / "macro_series.csv")
