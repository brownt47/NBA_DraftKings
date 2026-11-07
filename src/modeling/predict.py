
"""Generate DraftKings point predictions from a saved model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.modeling.features import prepare_model_features


DEFAULT_MODEL = Path("models/lightgbm_production.joblib")


def predict_features(features, model_path=DEFAULT_MODEL):
    """Predict DK points from pregame feature rows."""

    artifact = joblib.load(model_path)

    model = artifact["model"]
    expected_features = artifact["features"]

    X = prepare_model_features(features)

    # Match the exact training feature order.
    X = X[expected_features].copy()

    # Match categorical levels used during training.
    for column, categories in artifact["categories"].items():
        X[column] = X[column].cat.set_categories(categories)

    # Ensure numeric columns are numeric.
    for column in X.columns:
        if column not in artifact["categories"]:
            X[column] = pd.to_numeric(
                X[column], errors="coerce"
            )

    predictions = model.predict(X)

    result = features.copy()
    result["PROJECTED_DK_POINTS"] = np.maximum(
        predictions, 0
    )

    return result
