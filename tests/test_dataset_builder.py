import pandas as pd

from pe_pipeline.features.dataset_builder import build_modeling_dataset


def test_dataset_builder_keeps_valid_target_rows():
    daily = pd.DataFrame({"ticker": ["A"], "quarter_end": [pd.Timestamp("2024-03-31")], "x": [1.0]})
    monthly = pd.DataFrame({"quarter_end": [pd.Timestamp("2024-03-31")], "m": [2.0]})
    fundamentals = pd.DataFrame({"ticker": ["A"], "quarter_end": [pd.Timestamp("2024-03-31")], "f": [3.0]})
    pe = pd.DataFrame(
        {
            "ticker": ["A"],
            "quarter_end": [pd.Timestamp("2024-03-31")],
            "pe_ratio": [10.0],
            "pe_lag_1q": [9.0],
            "pe_lag_2q": [8.0],
            "pe_change_1q": [1.0],
            "target_pe_current": [10.0],
            "target_pe_change": [1.0],
        }
    )
    future_returns = pd.DataFrame({"ticker": ["A"], "quarter_end": [pd.Timestamp("2024-03-31")], "next_quarter_return": [0.05]})
    dataset, reports, missing = build_modeling_dataset(daily, monthly, fundamentals, pe, future_returns, epsilon=1.0)
    assert len(dataset) == 1
    assert reports[0].rows_removed == 0
    assert missing == {}
