
"""Generate DraftKings point predictions from a saved production model."""

from pathlib import Path

import joblib
import numpy as np

from src.modeling.model_registry import (
    DEFAULT_MODEL_TYPE,
    get_model_adapter,
)


DEFAULT_MODEL = Path(
    f"models/{DEFAULT_MODEL_TYPE}_production.joblib"
)


def predict_features(features, model_path=DEFAULT_MODEL):
    """Predict DK points using the model's saved feature schema."""

    artifact = joblib.load(model_path)

    model = artifact["model"]
    expected_features = artifact["features"]

    # Older production artifacts were LightGBM-only.
    model_type = artifact.get("model_type", "lightgbm")
    adapter = get_model_adapter(model_type)

    # Support both version 1 and version 2 artifacts.
    if "preprocessing" in artifact:
        preprocessing = artifact["preprocessing"]
    else:
        preprocessing = {
            "categories": artifact["categories"]
        }

    # Check for missing predictors before attempting prediction.
    missing = [
        column
        for column in expected_features
        if column not in features.columns
    ]

    if missing:
        raise ValueError(
            f"Missing {len(missing)} required prediction features: "
            f"{missing}"
        )

    # The adapter applies the original training preprocessing
    # and preserves the exact feature order.
    X = adapter.prepare_prediction_features(
        features,
        expected_features,
        preprocessing,
    )

    predictions = model.predict(X)

    result = features.copy()

    result["PROJECTED_DK_POINTS"] = np.maximum(
        np.asarray(predictions).reshape(-1),
        0,
    )

    return result
