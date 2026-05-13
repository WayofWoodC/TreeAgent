from __future__ import annotations

from pathlib import Path
import subprocess

import pandas as pd

from pe_pipeline.utils.metrics import max_drawdown, sharpe_ratio


def _safe_mean(series: pd.Series) -> float:
    value = series.mean()
    return 0.0 if pd.isna(value) else float(value)


def _annualized_return(returns: pd.Series, periods_per_year: float = 4.0) -> float:
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    total_return = float((1.0 + clean).prod())
    years = len(clean) / periods_per_year
    if years <= 0 or total_return <= 0:
        return 0.0
    return total_return ** (1.0 / years) - 1.0


def _annualized_volatility(returns: pd.Series, periods_per_year: float = 4.0) -> float:
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    return float(clean.std(ddof=0) * (periods_per_year ** 0.5))


def _sortino_ratio(returns: pd.Series, periods_per_year: float = 4.0) -> float:
    clean = returns.dropna()
    downside = clean[clean < 0]
    if clean.empty or downside.empty:
        return 0.0
    downside_std = downside.std(ddof=0)
    if downside_std == 0 or pd.isna(downside_std):
        return 0.0
    return float(clean.mean() / downside_std * (periods_per_year ** 0.5))


def _value_at_risk(returns: pd.Series, alpha: float = 0.05) -> float:
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    return float(clean.quantile(alpha))


def _conditional_value_at_risk(returns: pd.Series, alpha: float = 0.05) -> float:
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    var = clean.quantile(alpha)
    tail = clean[clean <= var]
    if tail.empty:
        return 0.0
    return float(tail.mean())


def _win_rate(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    return float((clean > 0).mean())


def _return_metric_bundle(
    prefix: str,
    returns: pd.Series,
    equity_curve: pd.Series,
    periods_per_year: float,
) -> dict[str, float]:
    clean_returns = returns.fillna(0.0)
    return {
        f"{prefix}_mean_return": _safe_mean(returns),
        f"{prefix}_annualized_return": _annualized_return(clean_returns, periods_per_year=periods_per_year),
        f"{prefix}_annualized_volatility": _annualized_volatility(clean_returns, periods_per_year=periods_per_year),
        f"{prefix}_sharpe_ratio": sharpe_ratio(clean_returns, periods_per_year=periods_per_year),
        f"{prefix}_sortino_ratio": _sortino_ratio(clean_returns, periods_per_year=periods_per_year),
        f"{prefix}_var_95": _value_at_risk(clean_returns, alpha=0.05),
        f"{prefix}_cvar_95": _conditional_value_at_risk(clean_returns, alpha=0.05),
        f"{prefix}_win_rate": _win_rate(clean_returns),
        f"{prefix}_best_quarter": float(clean_returns.max()) if not clean_returns.empty else 0.0,
        f"{prefix}_worst_quarter": float(clean_returns.min()) if not clean_returns.empty else 0.0,
        f"{prefix}_max_drawdown": max_drawdown(equity_curve),
    }


def summarize_backtest(backtest_returns: pd.DataFrame, periods_per_year: float = 4.0) -> dict[str, float]:
    frame = backtest_returns.copy()
    if "is_initial_row" in frame.columns:
        frame = frame.loc[~frame["is_initial_row"].fillna(False)].copy()
    long_returns = frame["long_return"].fillna(0.0)
    short_returns = frame["short_return"].fillna(0.0)
    strategy_returns = frame["long_short_return"].fillna(0.0)
    summary = {
        "mean_long_return": _safe_mean(frame["long_return"]),
        "mean_short_return": _safe_mean(frame["short_return"]),
        "mean_long_short_return": _safe_mean(frame["long_short_return"]),
        "annualized_return": _annualized_return(strategy_returns, periods_per_year=periods_per_year),
        "annualized_volatility": _annualized_volatility(strategy_returns, periods_per_year=periods_per_year),
        "sharpe_ratio": sharpe_ratio(strategy_returns, periods_per_year=periods_per_year),
        "sortino_ratio": _sortino_ratio(strategy_returns, periods_per_year=periods_per_year),
        "var_95": _value_at_risk(strategy_returns, alpha=0.05),
        "cvar_95": _conditional_value_at_risk(strategy_returns, alpha=0.05),
        "win_rate": _win_rate(strategy_returns),
        "best_quarter": float(strategy_returns.max()) if not strategy_returns.empty else 0.0,
        "worst_quarter": float(strategy_returns.min()) if not strategy_returns.empty else 0.0,
        "num_quarters": int(len(frame)),
        "max_drawdown": max_drawdown(frame["equity_curve"]),
    }
    summary.update(_return_metric_bundle("long_only", long_returns, frame["long_equity_curve"], periods_per_year=periods_per_year))
    summary.update(_return_metric_bundle("short_only", short_returns, frame["short_equity_curve"], periods_per_year=periods_per_year))
    return summary


def stats_frame(summary: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": list(summary.keys()),
            "value": list(summary.values()),
        }
    )


def _save_series_plot(
    frame: pd.DataFrame,
    series_specs: list[tuple[str, str, str]],
    output_path: Path,
    title: str,
    subtitle: str,
    y_axis_label: str,
    start_value: float,
) -> None:
    width = 1280
    height = 680
    margin_left = 70
    margin_right = 40
    margin_top = 40
    margin_bottom = 140
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    frame = frame.copy()
    frame["quarter_end"] = pd.to_datetime(frame["quarter_end"])
    series_columns = [column for column, _, _ in series_specs]
    for column in series_columns:
        frame[column] = frame[column].fillna(start_value)

    if frame.empty:
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"></svg>'
        output_path.write_text(svg, encoding="utf-8")
        _save_png_from_svg(output_path)
        return

    def scale_value(value: float, min_v: float, max_v: float, y0: float, h: float) -> float:
        if min_v == max_v:
            min_v -= 1.0
            max_v += 1.0
        return y0 + h - ((value - min_v) / (max_v - min_v) * h)

    x_positions = [margin_left + (plot_width * i / max(len(frame) - 1, 1)) for i in range(len(frame))]
    y_min = min(float(frame[column].min()) for column in series_columns)
    y_max = max(float(frame[column].max()) for column in series_columns)
    y_min = min(y_min, start_value)
    y_max = max(y_max, start_value)
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    baseline_line = scale_value(start_value, y_min, y_max, margin_top, plot_height)
    scaled_series = {
        column: [scale_value(float(v), y_min, y_max, margin_top, plot_height) for v in frame[column]]
        for column in series_columns
    }

    tick_count = 6
    tick_values = [y_min + (y_max - y_min) * i / (tick_count - 1) for i in range(tick_count)]
    use_percent_axis = start_value == 0.0
    y_ticks = []
    for tick in tick_values:
        y = scale_value(tick, y_min, y_max, margin_top, plot_height)
        label = f"{tick * 100:.1f}%" if use_percent_axis else f"{tick:.2f}"
        y_ticks.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#ececec" />'
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" font-size="11" text-anchor="end" font-family="Arial, sans-serif" fill="#444">{label}</text>'
        )

    def polyline(points_y: list[float], color: str) -> str:
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(x_positions, points_y))
        return f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}" />'

    def markers(points_y: list[float], color: str) -> str:
        return "".join(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}" />'
            for x, y in zip(x_positions, points_y)
        )

    date_labels = []
    for x, dt in zip(x_positions, frame["quarter_end"]):
        date_labels.append(
            f'<text x="{x:.2f}" y="{height - 56}" font-size="10" text-anchor="middle" transform="rotate(-45 {x:.2f},{height - 56})" fill="#333">{dt.strftime("%Y-%m-%d")}</text>'
        )

    legend_x = margin_left + 10
    legend_y = margin_top + 20
    legend_entries = []
    for idx, (_, label, color) in enumerate(series_specs):
        cy = legend_y + 24 * idx
        legend_entries.append(
            f'<circle cx="{legend_x}" cy="{cy}" r="4" fill="{color}" />'
            f'<text x="{legend_x + 12}" y="{cy + 4}" font-size="13" font-family="Arial, sans-serif" fill="#111">{label}</text>'
        )
    legend = "".join(legend_entries)

    polyline_markup = []
    marker_markup = []
    for column, _, color in series_specs:
        polyline_markup.append(polyline(scaled_series[column], color))
        marker_markup.append(markers(scaled_series[column], color))

    svg = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect width="100%" height="100%" fill="white" />
  <text x="{margin_left}" y="24" font-size="18" font-family="Arial, sans-serif" fill="#111">{title}</text>
  <text x="{margin_left}" y="46" font-size="13" font-family="Arial, sans-serif" fill="#555">{subtitle}</text>

  <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#999" />
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#999" />
  {''.join(y_ticks)}
  <line x1="{margin_left}" y1="{baseline_line:.2f}" x2="{width - margin_right}" y2="{baseline_line:.2f}" stroke="#d0d0d0" stroke-dasharray="4,4" />
  {''.join(polyline_markup)}
  {''.join(marker_markup)}
  {legend}
  {''.join(date_labels)}
  <text x="{width / 2:.2f}" y="{height - 12}" font-size="13" text-anchor="middle" font-family="Arial, sans-serif" fill="#111">Quarter End</text>
  <text x="22" y="{height / 2:.2f}" font-size="13" text-anchor="middle" transform="rotate(-90 22,{height / 2:.2f})" font-family="Arial, sans-serif" fill="#111">{y_axis_label}</text>
</svg>
""".strip()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")
    _save_png_from_svg(output_path)


def save_backtest_plot(backtest_returns: pd.DataFrame, output_path: Path) -> None:
    _save_series_plot(
        frame=backtest_returns,
        series_specs=[
            ("long_return", "long_return", "#1f77b4"),
            ("short_return", "short_return", "#ff7f0e"),
            ("long_short_return", "long_short_return", "#2ca02c"),
        ],
        output_path=output_path,
        title="Quarterly Returns",
        subtitle="Quarterly long, short, and long-short strategy returns",
        y_axis_label="Quarterly Return",
        start_value=0.0,
    )


def save_equity_curve_plot(backtest_returns: pd.DataFrame, output_path: Path) -> None:
    _save_series_plot(
        frame=backtest_returns,
        series_specs=[
            ("long_equity_curve", "long_equity_curve", "#1f77b4"),
            ("short_equity_curve", "short_equity_curve", "#ff7f0e"),
            ("ls_equity_curve", "ls_equity_curve", "#2ca02c"),
        ],
        output_path=output_path,
        title="Equity Curves",
        subtitle="Quarterly long, short, and long-short NAV from a 1.0 starting point",
        y_axis_label="NAV",
        start_value=1.0,
    )


def _save_png_from_svg(svg_path: Path) -> None:
    png_path = svg_path.with_suffix(".png")
    try:
        subprocess.run(
            ["sips", "-s", "format", "png", str(svg_path), "--out", str(png_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return
