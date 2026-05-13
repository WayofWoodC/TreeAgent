from __future__ import annotations

import pandas as pd

from pe_pipeline.backtesting.portfolio import rank_long_short


def _build_signal(frame: pd.DataFrame, signal_mode: str) -> pd.DataFrame:
    signal_frame = frame.copy()
    if "signal" in signal_frame.columns:
        return signal_frame

    if signal_mode == "scaled_change_prediction":
        signal_frame["signal"] = signal_frame["prediction"]
        return signal_frame

    if signal_mode == "current_pe_relative_change" and "pe_lag_1q" in signal_frame.columns:
        lag_abs = signal_frame["pe_lag_1q"].abs()
        signal_frame["signal"] = 0.0
        valid = lag_abs > 0
        signal_frame.loc[valid, "signal"] = (
            signal_frame.loc[valid, "prediction"] - signal_frame.loc[valid, "pe_lag_1q"]
        ) / lag_abs.loc[valid]
        return signal_frame

    signal_frame["signal"] = signal_frame["prediction"]
    return signal_frame


def _safe_bucket_mean(frame: pd.DataFrame, side: str, return_column: str) -> float:
    bucket = frame.loc[frame["side"] == side, return_column].dropna()
    if bucket.empty:
        return 0.0
    return float(bucket.mean())


def run_quarterly_backtest(
    predictions: pd.DataFrame,
    top_n: int,
    return_column: str = "next_quarter_return",
    signal_mode: str = "prediction",
) -> pd.DataFrame:
    rows = []
    for quarter_end, frame in predictions.groupby("quarter_end"):
        eligible = frame.dropna(subset=["prediction"]).copy()
        eligible = _build_signal(eligible, signal_mode=signal_mode)
        ranked = rank_long_short(eligible, top_n=top_n)
        long_return = _safe_bucket_mean(ranked, "long", return_column=return_column)
        short_asset_return = _safe_bucket_mean(ranked, "short", return_column=return_column)
        short_return = -short_asset_return
        long_short_return = 0.5 * long_return + 0.5 * short_return
        rows.append(
            {
                "quarter_end": quarter_end,
                "long_return": long_return,
                "short_asset_return": short_asset_return,
                "short_return": short_return,
                "long_short_return": long_short_return,
                "is_initial_row": False,
            }
        )
    result = pd.DataFrame(rows).sort_values("quarter_end").reset_index(drop=True)
    result["long_ret_cum"] = (1.0 + result["long_return"].fillna(0.0)).cumprod() - 1.0
    result["short_ret_cum"] = (1.0 + result["short_return"].fillna(0.0)).cumprod() - 1.0
    result["ls_ret_cum"] = (1.0 + result["long_short_return"].fillna(0.0)).cumprod() - 1.0
    result["long_equity_curve"] = result["long_ret_cum"] + 1.0
    result["short_equity_curve"] = result["short_ret_cum"] + 1.0
    result["ls_equity_curve"] = result["ls_ret_cum"] + 1.0
    result["equity_curve"] = result["ls_equity_curve"]

    if result.empty:
        return result

    start_row = {
        "quarter_end": pd.to_datetime(result["quarter_end"].iloc[0]) - pd.offsets.QuarterEnd(1),
        "long_return": 0.0,
        "short_asset_return": 0.0,
        "short_return": 0.0,
        "long_short_return": 0.0,
        "long_ret_cum": 0.0,
        "short_ret_cum": 0.0,
        "ls_ret_cum": 0.0,
        "long_equity_curve": 1.0,
        "short_equity_curve": 1.0,
        "ls_equity_curve": 1.0,
        "equity_curve": 1.0,
        "is_initial_row": True,
    }
    result = pd.concat([pd.DataFrame([start_row]), result], ignore_index=True)
    return result
