from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pe_pipeline.modeling.preprocess import fit_preprocessor, prepare_xy
from pe_pipeline.utils.metrics import r2_score, rmse, spearman_ic


@dataclass(slots=True)
class TrainResult:
    model: object
    imputer: object
    metrics: dict[str, float]
    feature_columns: list[str]
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame


def train_xgboost_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_name: str,
    params: dict | None = None,
    random_state: int = 42,
) -> TrainResult:
    from xgboost import XGBRegressor

    params = params or {}
    prepared_train = prepare_xy(train_df, target_name)
    prepared_val = prepare_xy(val_df, target_name)
    prepared_test = prepare_xy(test_df, target_name)

    preprocessor = fit_preprocessor(train_df, prepared_train.feature_columns)
    x_train = preprocessor.fit_transform(train_df)
    x_val = preprocessor.transform(val_df, update_history=True)
    x_test = preprocessor.transform(test_df, update_history=True)

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=params.get("n_estimators", 500),
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", 4),
        min_child_weight=params.get("min_child_weight", 5),
        gamma=params.get("gamma", 0.0),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        reg_alpha=params.get("reg_alpha", 0.0),
        reg_lambda=params.get("reg_lambda", 1.0),
        random_state=random_state,
        tree_method="hist",
        early_stopping_rounds=params.get("early_stopping_rounds", 50),
        eval_metric="rmse",
    )
    model.fit(x_train, prepared_train.y, eval_set=[(x_val, prepared_val.y)], verbose=False)

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    metrics = {
        "val_rmse": rmse(prepared_val.y.to_numpy(dtype=float), np.asarray(val_pred, dtype=float)),
        "test_rmse": rmse(prepared_test.y.to_numpy(dtype=float), np.asarray(test_pred, dtype=float)),
        "val_spearman_ic": spearman_ic(prepared_val.y.to_numpy(dtype=float), np.asarray(val_pred, dtype=float)),
        "test_spearman_ic": spearman_ic(prepared_test.y.to_numpy(dtype=float), np.asarray(test_pred, dtype=float)),
        "val_r2": r2_score(prepared_val.y.to_numpy(dtype=float), np.asarray(val_pred, dtype=float)),
        "test_r2": r2_score(prepared_test.y.to_numpy(dtype=float), np.asarray(test_pred, dtype=float)),
    }

    prediction_columns = ["ticker", "quarter_end", target_name]
    if "pe_lag_1q" in val_df.columns:
        prediction_columns.insert(2, "pe_lag_1q")
    val_predictions = val_df[prediction_columns].copy()
    val_predictions["prediction"] = val_pred
    test_prediction_columns = ["ticker", "quarter_end", target_name, "next_quarter_return"]
    if "pe_lag_1q" in test_df.columns:
        test_prediction_columns.insert(2, "pe_lag_1q")
    test_predictions = test_df[test_prediction_columns].copy()
    test_predictions["prediction"] = test_pred

    return TrainResult(
        model=model,
        imputer=preprocessor,
        metrics=metrics,
        feature_columns=prepared_train.feature_columns,
        validation_predictions=val_predictions,
        test_predictions=test_predictions,
    )
