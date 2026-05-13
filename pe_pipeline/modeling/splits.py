from __future__ import annotations

import pandas as pd


def time_based_split(dataset: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2) -> dict[str, pd.DataFrame]:
    quarters = sorted(dataset["quarter_end"].dropna().unique())
    n_quarters = len(quarters)
    train_end = max(1, int(n_quarters * train_frac))
    val_end = max(train_end + 1, int(n_quarters * (train_frac + val_frac)))

    train_quarters = quarters[:train_end]
    val_quarters = quarters[train_end:val_end]
    test_quarters = quarters[val_end:]

    return {
        "train": dataset[dataset["quarter_end"].isin(train_quarters)].copy(),
        "val": dataset[dataset["quarter_end"].isin(val_quarters)].copy(),
        "test": dataset[dataset["quarter_end"].isin(test_quarters)].copy(),
    }


def walk_forward_quarter_folds(dataset: pd.DataFrame, min_train_quarters: int = 12, val_quarters: int = 4) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    quarters = sorted(dataset["quarter_end"].dropna().unique())
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for split_idx in range(min_train_quarters, len(quarters) - val_quarters + 1, val_quarters):
        train_q = quarters[:split_idx]
        val_q = quarters[split_idx:split_idx + val_quarters]
        if not val_q:
            continue
        folds.append(
            (
                dataset[dataset["quarter_end"].isin(train_q)].copy(),
                dataset[dataset["quarter_end"].isin(val_q)].copy(),
            )
        )
    return folds


def development_dataset(dataset: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2) -> pd.DataFrame:
    splits = time_based_split(dataset, train_frac=train_frac, val_frac=val_frac)
    return pd.concat([splits["train"], splits["val"]], ignore_index=True).sort_values(["quarter_end", "ticker"]).reset_index(drop=True)
