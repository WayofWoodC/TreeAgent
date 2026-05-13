import pandas as pd

from pe_pipeline.backtesting.engine import run_quarterly_backtest


def test_backtest_long_short():
    frame = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D"],
            "quarter_end": pd.to_datetime(["2024-03-31"] * 4),
            "prediction": [4.0, 3.0, 2.0, 1.0],
            "next_quarter_return": [0.1, 0.05, -0.02, -0.08],
        }
    )
    result = run_quarterly_backtest(frame, top_n=1)
    assert round(result.loc[0, "long_short_return"], 6) == round(0.1 - (-0.08), 6)
