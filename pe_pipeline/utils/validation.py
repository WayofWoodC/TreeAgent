from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pe_pipeline.exceptions import DataValidationError


@dataclass(slots=True)
class DropReport:
    stage: str
    rows_before: int
    rows_after: int

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after


def require_columns(df: pd.DataFrame, required: list[str], stage: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise DataValidationError(f"{stage} missing required columns: {missing}")


def assert_no_duplicate_keys(df: pd.DataFrame, keys: list[str], stage: str) -> None:
    duplicates = int(df.duplicated(keys).sum())
    if duplicates:
        raise DataValidationError(f"{stage} has {duplicates} duplicate rows on keys {keys}")


def summarize_missing(df: pd.DataFrame, include_columns: list[str] | None = None) -> dict[str, int]:
    include_columns = include_columns or []
    missing_counts = {column: int(count) for column, count in df.isna().sum().items() if int(count) > 0}
    for column in include_columns:
        if column in df.columns and column not in missing_counts:
            missing_counts[column] = int(df[column].isna().sum())
    return missing_counts


def dropna_with_report(df: pd.DataFrame, subset: list[str], stage: str) -> tuple[pd.DataFrame, DropReport]:
    before = len(df)
    cleaned = df.dropna(subset=subset).copy()
    return cleaned, DropReport(stage=stage, rows_before=before, rows_after=len(cleaned))
