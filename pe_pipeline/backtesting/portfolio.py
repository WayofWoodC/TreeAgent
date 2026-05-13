from __future__ import annotations

import pandas as pd


def rank_long_short(frame: pd.DataFrame, top_n: int) -> pd.DataFrame:
    signal_column = "signal" if "signal" in frame.columns else "prediction"
    ranked = frame.sort_values(signal_column, ascending=False).copy()
    ranked["side"] = "flat"
    if len(ranked) == 0:
        return ranked
    if ranked[signal_column].dropna().nunique() <= 1:
        return ranked
    effective_n = min(top_n, len(ranked) // 2)
    if effective_n == 0:
        return ranked
    ranked.iloc[:effective_n, ranked.columns.get_loc("side")] = "long"
    ranked.iloc[-effective_n:, ranked.columns.get_loc("side")] = "short"
    return ranked
