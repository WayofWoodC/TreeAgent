from __future__ import annotations

import logging

from pe_pipeline.config import Settings
from pe_pipeline.features.daily_features import build_daily_features
from pe_pipeline.features.fundamentals_features import build_fundamental_features
from pe_pipeline.features.monthly_features import build_monthly_macro_features
from pe_pipeline.features.pe_features import build_pe_feature_table
from pe_pipeline.modeling.preprocess import feature_columns_from_dataset
from pe_pipeline.utils.io import load_json, load_parquet, save_csv, save_parquet


def _apply_signal(latest, target_name: str):
    result = latest.copy()
    if target_name == "target_pe_current" and "pe_lag_1q" in result.columns:
        lag_abs = result["pe_lag_1q"].abs()
        result["signal"] = 0.0
        valid = lag_abs > 0
        result.loc[valid, "signal"] = (
            result.loc[valid, "prediction"] - result.loc[valid, "pe_lag_1q"]
        ) / lag_abs.loc[valid]
        return result
    result["signal"] = result["prediction"]
    return result


def run_predict_latest(settings: Settings, logger: logging.Logger) -> None:
    import joblib
    from xgboost import XGBRegressor

    prices = load_parquet(settings.raw_dir / "prices" / "daily_prices.parquet")
    pe_ratios = load_parquet(settings.raw_dir / "pe_ratios" / "quarterly_pe_ratios.parquet")
    fundamentals = load_parquet(settings.raw_dir / "fundamentals" / "quarterly_fundamentals.parquet")
    macro = load_parquet(settings.raw_dir / "macro" / "macro_series.parquet")

    daily_features, _ = build_daily_features(prices, settings.daily_windows)
    macro_features = build_monthly_macro_features(macro)
    fundamental_features = build_fundamental_features(fundamentals)
    pe_features = build_pe_feature_table(pe_ratios)

    latest = pe_features.merge(daily_features, on=["ticker", "quarter_end"], how="left")
    latest = latest.merge(fundamental_features, on=["ticker", "quarter_end"], how="left")
    latest = latest.merge(macro_features, on="quarter_end", how="left")
    latest = latest.sort_values(["ticker", "quarter_end"]).groupby("ticker").tail(1).reset_index(drop=True)

    model_metadata = load_json(settings.models_dir / "model_metadata.json")
    target_name = model_metadata.get("target_name", "target_pe_current")
    feature_columns = feature_columns_from_dataset(latest, target_name=target_name)
    imputer = joblib.load(settings.models_dir / "preprocessor.joblib")
    model = XGBRegressor()
    model.load_model(settings.models_dir / "xgb_model.json")
    latest["prediction"] = model.predict(imputer.transform(latest))
    latest = _apply_signal(latest, target_name=target_name)
    latest = latest.sort_values("signal", ascending=False).reset_index(drop=True)

    save_parquet(latest, settings.intermediate_dir / "latest_features.parquet")
    save_csv(latest[["ticker", "quarter_end", "prediction", "signal"]], settings.predictions_dir / "latest_predictions.csv")
