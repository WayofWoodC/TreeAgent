from __future__ import annotations

import numpy as np
import pandas as pd

from pe_pipeline.features.quarter_aggregation import add_quarter_end


ONE_MONTH_LAG_SERIES = {"CPI", "INDPRO", "M2SL", "UNRATE"}
TWO_MONTH_LAG_SERIES = {"CSUSHPISA"}


def _build_visible_month_slots(
    monthly: pd.DataFrame,
    series_name: str,
    offsets: list[int],
    labels: list[str],
) -> pd.DataFrame:
    quarter_index = pd.DataFrame({"quarter_end": sorted(monthly["quarter_end"].dropna().unique())})
    series_frame = monthly.loc[monthly["series_name"] == series_name, ["date", "value"]].copy()
    if series_frame.empty:
        return quarter_index

    series_frame = series_frame.rename(columns={"date": "source_date", "value": "source_value"})
    result = quarter_index.copy()
    quarter_end_ts = pd.to_datetime(result["quarter_end"])
    for offset, label in zip(offsets, labels):
        visible_dates = (quarter_end_ts + pd.offsets.MonthEnd(offset)).dt.normalize()
        lookup = pd.DataFrame({"source_date": visible_dates, "quarter_end": result["quarter_end"]})
        merged = lookup.merge(series_frame, on="source_date", how="left")
        result[f"{series_name.lower()}_{label}"] = merged["source_value"].to_numpy()
    return result


def build_monthly_macro_features(macro: pd.DataFrame) -> pd.DataFrame:
    if macro.empty:
        return pd.DataFrame(columns=["quarter_end"])

    frame = macro.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.sort_values(["series_name", "date"])
    frame["month_end"] = frame["date"].dt.to_period("M").dt.end_time.dt.normalize()

    monthly = (
        frame.groupby(["series_name", "month_end"], as_index=False)
        .agg(value=("value", "last"))
        .rename(columns={"month_end": "date"})
    )
    monthly = add_quarter_end(monthly)

    natural_monthly = monthly.loc[
        ~monthly["series_name"].isin(ONE_MONTH_LAG_SERIES | TWO_MONTH_LAG_SERIES)
    ].copy()
    natural_monthly["month_in_quarter"] = natural_monthly["date"].dt.month.mod(3).replace({1: 1, 2: 2, 0: 3})

    if natural_monthly.empty:
        pivot = pd.DataFrame({"quarter_end": sorted(monthly["quarter_end"].dropna().unique())})
    else:
        pivot = (
            natural_monthly.pivot_table(
                index="quarter_end",
                columns=["series_name", "month_in_quarter"],
                values="value",
                aggfunc="last",
            )
            .sort_index(axis=1)
        )
        pivot.columns = [f"{series.lower()}_m{slot}" for series, slot in pivot.columns]
        pivot = pivot.reset_index()

    for series_name in sorted(ONE_MONTH_LAG_SERIES):
        lagged = _build_visible_month_slots(
            monthly=monthly,
            series_name=series_name,
            offsets=[-3, -2, -1],
            labels=["m0", "m1", "m2"],
        )
        pivot = pivot.merge(lagged, on="quarter_end", how="left")

    for series_name in sorted(TWO_MONTH_LAG_SERIES):
        lagged = _build_visible_month_slots(
            monthly=monthly,
            series_name=series_name,
            offsets=[-4, -3, -2],
            labels=["m-1", "m0", "m1"],
        )
        pivot = pivot.merge(lagged, on="quarter_end", how="left")

    if {"cpi_m0", "cpi_m1", "cpi_m2"}.issubset(pivot.columns):
        pivot["inflation_1y_proxy"] = pivot["cpi_m2"].pct_change(4)
    if {"indpro_m0", "indpro_m1", "indpro_m2"}.issubset(pivot.columns):
        pivot["ind_growth_qoq"] = pivot["indpro_m2"].pct_change()
    if {"unrate_m0", "unrate_m1", "unrate_m2"}.issubset(pivot.columns):
        pivot["delta_unemployment"] = pivot["unrate_m2"].diff()
    if {"m2sl_m0", "m2sl_m1", "m2sl_m2"}.issubset(pivot.columns):
        pivot["delta_log_m2"] = np.log(pivot["m2sl_m2"].where(pivot["m2sl_m2"] > 0)).diff()
    if {"csushpisa_m-1", "csushpisa_m0", "csushpisa_m1"}.issubset(pivot.columns):
        pivot["delta_log_house_price"] = np.log(pivot["csushpisa_m1"].where(pivot["csushpisa_m1"] > 0)).diff()

    return pivot
