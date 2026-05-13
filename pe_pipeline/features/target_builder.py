from __future__ import annotations

import pandas as pd


TARGET_COLUMNS = {
    "target_pe_current": "target_pe_current",
    "target_pe_scaled_change": "target_pe_scaled_change",
}


def add_target_columns(df: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    result = df.copy()
    result["target_pe_scaled_change"] = (result["pe_ratio"] - result["pe_lag_1q"]) / (result["pe_lag_1q"].abs() + epsilon)
    return result
