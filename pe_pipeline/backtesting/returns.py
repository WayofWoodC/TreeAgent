from __future__ import annotations

import numpy as np
import pandas as pd


def attach_forward_return(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    periods: int,
    output_column: str = "holding_period_return",
) -> pd.DataFrame:
    if periods <= 0:
        raise ValueError("periods must be a positive integer")

    price_frame = prices.copy()
    price_frame["date"] = pd.to_datetime(price_frame["date"])
    price_frame["adjClose"] = pd.to_numeric(price_frame["adjClose"], errors="coerce")
    price_frame = price_frame.sort_values(["ticker", "date"])

    lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ticker, frame in price_frame.groupby("ticker"):
        valid = frame.dropna(subset=["adjClose"])
        lookup[ticker] = (
            valid["date"].to_numpy(dtype="datetime64[ns]"),
            valid["adjClose"].to_numpy(dtype=float),
        )

    result = predictions.copy()
    result["quarter_end"] = pd.to_datetime(result["quarter_end"])
    forward_returns: list[float] = []

    for row in result.itertuples(index=False):
        dates, values = lookup.get(row.ticker, (None, None))
        if dates is None or len(dates) == 0:
            forward_returns.append(float("nan"))
            continue

        entry_idx = int(np.searchsorted(dates, np.datetime64(row.quarter_end), side="right"))
        exit_idx = entry_idx + periods
        if entry_idx >= len(values) or exit_idx >= len(values):
            forward_returns.append(float("nan"))
            continue

        entry_price = float(values[entry_idx])
        exit_price = float(values[exit_idx])
        if entry_price == 0.0 or np.isnan(entry_price) or np.isnan(exit_price):
            forward_returns.append(float("nan"))
            continue
        forward_returns.append(exit_price / entry_price - 1.0)

    result[output_column] = forward_returns
    return result
