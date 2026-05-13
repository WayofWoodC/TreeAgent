from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


NON_FEATURE_COLUMNS = {
    "ticker",
    "quarter_end",
    "target_pe_current",
    "target_pe_change",
    "target_pe_scaled_change",
    "next_quarter_return",
}

LEAKY_FEATURE_COLUMNS = {
    "pe_ratio",
    "pe_lag_2q",
    "pe_change_1q",
    "pb",
    "pettm",
    "psttm",
    "pfcfttm",
    "evebitdattm",
    "evrevenuettm",
    "ptbv",
    "grossmargin",
    "operatingmargin",
    "pretaxmargin",
    "netmargin",
    "fcfmargin",
    "roattm",
    "roettm",
    "roicttm",
    "rotcttm",
    "cashratio",
    "currentratio",
    "quickratio",
    "longtermdebttotalasset",
    "longtermdebttotalcapital",
    "longtermdebttotalequity",
    "netdebttototalcapital",
    "netdebttototalequity",
    "totaldebttoequity",
    "totaldebttototalasset",
    "totaldebttototalcapital",
    "totalratio",
    "assetturnoverttm",
    "inventoryturnoverttm",
    "receivablesturnoverttm",
    "sgatosale",
    "ev",
    "bookvalue",
    "tangiblebookvalue",
    "ebitpershare",
    "eps",
    "fcfpersharettm",
    "salespershare",
    "payoutratiottm",
}


@dataclass(slots=True)
class PreparedData:
    X: pd.DataFrame
    y: pd.Series
    feature_columns: list[str]


@dataclass
class LastValuePreprocessor:
    feature_columns: list[str]
    history_: pd.DataFrame | None = None

    def _fill_with_history(self, frame: pd.DataFrame, keep_history: bool) -> pd.DataFrame:
        required = ["ticker", "quarter_end", *self.feature_columns]
        working = frame[required].copy()
        working["quarter_end"] = pd.to_datetime(working["quarter_end"])
        working["_row_order"] = range(len(working))
        working["_is_current"] = True

        if self.history_ is not None and not self.history_.empty:
            history = self.history_.copy()
            history["_row_order"] = range(-len(history), 0)
            history["_is_current"] = False
            combined = pd.concat([history, working], ignore_index=True, sort=False)
        else:
            combined = working

        combined = combined.sort_values(["ticker", "quarter_end", "_row_order"], kind="mergesort")
        combined[self.feature_columns] = combined.groupby("ticker")[self.feature_columns].ffill()

        current = combined.loc[combined["_is_current"]].copy()
        current = current.sort_values("_row_order", kind="mergesort")
        transformed = current[self.feature_columns].reset_index(drop=True)

        if keep_history:
            history_columns = ["ticker", "quarter_end", *self.feature_columns]
            self.history_ = combined[history_columns].copy().reset_index(drop=True)

        return transformed

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        self.history_ = None
        return self._fill_with_history(frame, keep_history=True)

    def transform(self, frame: pd.DataFrame, update_history: bool = False) -> pd.DataFrame:
        return self._fill_with_history(frame, keep_history=update_history)


def feature_columns_from_dataset(dataset: pd.DataFrame, target_name: str) -> list[str]:
    columns = []
    for column in dataset.columns:
        if column in NON_FEATURE_COLUMNS or column == target_name or column in LEAKY_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(dataset[column]):
            columns.append(column)
    return columns


def prepare_xy(dataset: pd.DataFrame, target_name: str) -> PreparedData:
    features = feature_columns_from_dataset(dataset, target_name)
    return PreparedData(X=dataset[features].copy(), y=dataset[target_name].copy(), feature_columns=features)


def fit_preprocessor(train_df: pd.DataFrame, feature_columns: list[str]) -> LastValuePreprocessor:
    preprocessor = LastValuePreprocessor(feature_columns=feature_columns)
    return preprocessor
