from __future__ import annotations

import pandas as pd


def feature_importance_frame(model, feature_columns: list[str]) -> pd.DataFrame:
    scores = model.feature_importances_
    frame = pd.DataFrame({"feature": feature_columns, "importance": scores})
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)
