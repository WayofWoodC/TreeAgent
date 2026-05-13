import pandas as pd

from pe_pipeline.modeling.splits import time_based_split


def test_time_split_is_ordered():
    frame = pd.DataFrame(
        {
            "quarter_end": pd.to_datetime(["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31", "2024-03-31"]),
            "value": [1, 2, 3, 4, 5],
        }
    )
    splits = time_based_split(frame, train_frac=0.4, val_frac=0.2)
    assert splits["train"]["quarter_end"].max() < splits["test"]["quarter_end"].min()
