
"""LightGBM adapter for DraftKings production modeling."""

import lightgbm as lgb
import pandas as pd

from src.modeling.features import (
    CATEGORICAL_FEATURES,
    select_model_features,
)
from src.modeling.lightgbm_model import DEFAULT_PARAMS


def create_model():
    """Create LightGBM using validated hyperparameters."""
    return lgb.LGBMRegressor(**DEFAULT_PARAMS)


def prepare_training_features(df):
    """Prepare features and save categorical levels."""
    X = select_model_features(df)

    categories = {}

    for column in X.columns:
        if column in CATEGORICAL_FEATURES:
            X[column] = X[column].astype("category")
            categories[column] = list(X[column].cat.categories)
        else:
            X[column] = pd.to_numeric(
                X[column], errors="coerce"
            )

    return X, {"categories": categories}


def prepare_prediction_features(df, feature_names, preprocessing):
    """Prepare features exactly as expected by the trained model."""
    X = df[feature_names].copy()

    for column in X.columns:
        if column in preprocessing["categories"]:
            categories = preprocessing["categories"][column]
            X[column] = pd.Categorical(
                X[column], categories=categories
            )
        else:
            X[column] = pd.to_numeric(
                X[column], errors="coerce"
            )

    return X
