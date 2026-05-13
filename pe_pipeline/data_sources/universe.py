from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests


WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
REPO_FALLBACK = Path(__file__).resolve().parents[2] / "data" / "raw" / "universe.parquet"


def _legacy_fallback_candidates() -> list[Path]:
    root = Path(__file__).resolve()
    return [
        root.parents[3] / "LASSO" / "Data" / "stock_data" / "sp500_tickers.txt",
        root.parents[2] / "Data" / "stock_data" / "sp500_tickers.txt",
        root.parents[3] / "Data" / "stock_data" / "sp500_tickers.txt",
    ]


def _load_legacy_universe() -> pd.DataFrame | None:
    for candidate in _legacy_fallback_candidates():
        if candidate.exists():
            tickers = pd.read_csv(candidate, header=None, names=["ticker"])
            tickers["ticker"] = tickers["ticker"].astype(str).str.strip().str.replace(".", "-", regex=False)
            tickers = tickers.loc[tickers["ticker"].ne("")].drop_duplicates(["ticker"]).reset_index(drop=True)
            tickers["security"] = pd.NA
            tickers["sector"] = pd.NA
            tickers["sub_industry"] = pd.NA
            return tickers[["ticker", "security", "sector", "sub_industry"]]
    return None


def _load_repo_fallback() -> pd.DataFrame | None:
    if not REPO_FALLBACK.exists():
        return None
    frame = pd.read_parquet(REPO_FALLBACK)
    if "ticker" not in frame.columns:
        return None
    frame = frame.drop_duplicates(["ticker"]).reset_index(drop=True)
    return frame if len(frame) >= 100 else None


def fetch_sp500_universe() -> pd.DataFrame:
    try:
        response = requests.get(
            WIKI_SP500_URL,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        universe = tables[0].rename(
            columns={
                "Symbol": "ticker",
                "Security": "security",
                "GICS Sector": "sector",
                "GICS Sub-Industry": "sub_industry",
            }
        )
        universe["ticker"] = universe["ticker"].str.replace(".", "-", regex=False)
        return universe[["ticker", "security", "sector", "sub_industry"]].sort_values("ticker").reset_index(drop=True)
    except Exception:
        legacy = _load_legacy_universe()
        if legacy is not None:
            return legacy
        repo_fallback = _load_repo_fallback()
        if repo_fallback is not None:
            return repo_fallback
        raise
