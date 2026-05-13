import pandas as pd

from pe_pipeline.features.quarter_aggregation import add_quarter_end


def test_quarter_end_assignment():
    frame = pd.DataFrame({"date": ["2024-01-15", "2024-03-31", "2024-04-01"]})
    result = add_quarter_end(frame)
    assert result["quarter_end"].dt.strftime("%Y-%m-%d").tolist() == ["2024-03-31", "2024-03-31", "2024-06-30"]
