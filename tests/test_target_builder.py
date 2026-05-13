import pandas as pd

from pe_pipeline.features.target_builder import add_target_columns


def test_scaled_change_target():
    frame = pd.DataFrame({"pe_ratio": [15.0], "pe_lag_1q": [10.0]})
    result = add_target_columns(frame, epsilon=1.0)
    assert round(result.loc[0, "target_pe_scaled_change"], 6) == round(5.0 / 11.0, 6)
