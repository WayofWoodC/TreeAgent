from __future__ import annotations

import pandas as pd

from pe_pipeline.features.quarter_aggregation import add_quarter_end


def _month_in_quarter(series: pd.Series) -> pd.Series:
    return series.dt.month.mod(3).replace({1: 1, 2: 2, 0: 3})


def _first_valid_n(values: pd.Series, n: int) -> pd.Series:
    clean = values.dropna()
    return clean.iloc[:n]


def _safe_last(values: pd.Series) -> float:
    clean = values.dropna()
    return float(clean.iloc[-1]) if not clean.empty else float("nan")


def _safe_mean(values: pd.Series) -> float:
    return float(values.mean()) if not values.empty else float("nan")


def _safe_std(values: pd.Series) -> float:
    return float(values.std(ddof=0)) if len(values) >= 2 else float("nan")


def _safe_min(values: pd.Series) -> float:
    return float(values.min()) if not values.empty else float("nan")


def _safe_max(values: pd.Series) -> float:
    return float(values.max()) if not values.empty else float("nan")


def build_daily_features(prices: pd.DataFrame, windows: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if prices.empty:
        empty = pd.DataFrame(columns=["ticker", "quarter_end"])
        return empty, empty

    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ticker", "date"])
    frame["adjClose"] = pd.to_numeric(frame["adjClose"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    frame["daily_return"] = frame.groupby("ticker")["adjClose"].pct_change()
    frame = add_quarter_end(frame)
    frame["month_in_quarter"] = _month_in_quarter(frame["date"])

    quarter_base = (
        frame.groupby(["ticker", "quarter_end"])
        .agg(
            adj_close_min_q=("adjClose", "min"),
            adj_close_max_q=("adjClose", "max"),
            adj_close_last_q=("adjClose", "last"),
            volume_last_q=("volume", "last"),
            quarter_return=("adjClose", lambda x: x.iloc[-1] / x.iloc[0] - 1.0 if len(x.dropna()) > 1 else 0.0),
        )
        .reset_index()
    )

    month1 = frame.loc[frame["month_in_quarter"] == 1].copy()
    sampled_rows = []
    for (ticker, quarter_end), group in month1.groupby(["ticker", "quarter_end"]):
        group = group.sort_values("date")
        adj_sample = _first_valid_n(group["adjClose"], 20)
        ret_sample = _first_valid_n(group["daily_return"], 20)
        vol_sample = _first_valid_n(group["volume"], 20)
        sampled_rows.append(
            {
                "ticker": ticker,
                "quarter_end": quarter_end,
                "adj_close_m1_first20_mean": _safe_mean(adj_sample),
                "adj_close_m1_first20_std": _safe_std(adj_sample),
                "adj_close_m1_first20_min": _safe_min(adj_sample),
                "adj_close_m1_first20_max": _safe_max(adj_sample),
                "adj_close_m1_first20_last": _safe_last(adj_sample),
                "daily_return_m1_first20_mean": _safe_mean(ret_sample),
                "daily_return_m1_first20_std": _safe_std(ret_sample),
                "daily_return_m1_first20_min": _safe_min(ret_sample),
                "daily_return_m1_first20_max": _safe_max(ret_sample),
                "daily_return_m1_first20_last": _safe_last(ret_sample),
                "volume_m1_first20_mean": _safe_mean(vol_sample),
                "volume_m1_first20_std": _safe_std(vol_sample),
                "volume_m1_first20_last": _safe_last(vol_sample),
                "month1_first_valid_trading_day": group["date"].dropna().iloc[0] if not group["date"].dropna().empty else pd.NaT,
                "month1_twentieth_valid_trading_day": group["date"].dropna().iloc[min(len(group["date"].dropna()), 20) - 1] if not group["date"].dropna().empty else pd.NaT,
            }
        )
    first_month_features = pd.DataFrame(sampled_rows)
    quarterly = quarter_base.merge(first_month_features, on=["ticker", "quarter_end"], how="left")

    returns = quarterly[["ticker", "quarter_end", "quarter_return"]].copy()
    returns["next_quarter_return"] = returns.groupby("ticker")["quarter_return"].shift(-1)
    return quarterly, returns
