from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def sharpe_ratio(returns: pd.Series, periods_per_year: float = 4.0) -> float:
    std = returns.std(ddof=0)
    if std == 0 or pd.isna(std):
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(cumulative_returns: pd.Series) -> float:
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / running_max - 1.0
    return float(drawdown.min())
