from __future__ import annotations

from pe_pipeline.utils.io import save_json


def save_model_artifacts(model, imputer, metadata: dict, model_path, imputer_path, metadata_path) -> None:
    import joblib

    model.save_model(model_path)
    joblib.dump(imputer, imputer_path)
    save_json(metadata, metadata_path)


def load_model_artifacts(model_cls, model_path, imputer_path):
    import joblib

    model = model_cls()
    model.load_model(model_path)
    imputer = joblib.load(imputer_path)
    return model, imputer
