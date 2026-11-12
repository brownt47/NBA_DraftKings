
"""CatBoost adapter for DraftKings production modeling."""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.modeling.features import (
    CATEGORICAL_FEATURES,
    select_model_features,
)


DEFAULT_PARAMS = {
    "iterations": 500,
    "depth": 7,
    "learning_rate": 0.03,
    "loss_function": "RMSE",
    "random_seed": 42,
    "thread_count": -1,
    "verbose": False,
    "allow_writing_files": False,
}


def create_model():
    """Create CatBoost using validated hyperparameters."""
    return CatBoostRegressor(**DEFAULT_PARAMS)


def prepare_training_features(df):
    """Prepare features and record the categorical columns."""
    X = select_model_features(df)

    categories = [
        col for col in X.columns
        if col in CATEGORICAL_FEATURES
    ]

    for column in X.columns:
        if column in categories:
            X[column] = X[column].fillna("UNKNOWN").astype(str)
        else:
            X[column] = pd.to_numeric(
                X[column], errors="coerce"
            )
            X[column] = X[column].replace(
                [np.inf, -np.inf], np.nan
            )

    return X, {"categories": categories}


def prepare_prediction_features(df, feature_names, preprocessing):
    """Apply the same feature preparation used for training."""
    X = df[feature_names].copy()
    categories = preprocessing["categories"]

    for column in X.columns:
        if column in categories:
            X[column] = X[column].fillna("UNKNOWN").astype(str)
        else:
            X[column] = pd.to_numeric(
                X[column], errors="coerce"
            )
            X[column] = X[column].replace(
                [np.inf, -np.inf], np.nan
            )

    return X
