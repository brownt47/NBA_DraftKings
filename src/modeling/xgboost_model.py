"""XGBoost regression model for NBA DraftKings fantasy points."""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.modeling.features import prepare_model_features


DEFAULT_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "tree_method": "hist",
    "enable_categorical": True,
    "random_state": 42,
    "n_jobs": -1,
}

CATEGORICAL_COLUMNS = ["OPPONENT", "HOME_AWAY"]


def prepare_xgboost_features(train_df, validation_df):
    """Prepare pregame predictors with training-aligned categories."""

    X_train = prepare_model_features(train_df)
    X_validation = prepare_model_features(validation_df)

    for column in X_train.columns:
        if column in CATEGORICAL_COLUMNS:
            X_validation[column] = (
                X_validation[column].cat.set_categories(
                    X_train[column].cat.categories
                )
            )
        else:
            X_train[column] = pd.to_numeric(
                X_train[column], errors="coerce"
            )
            X_validation[column] = pd.to_numeric(
                X_validation[column], errors="coerce"
            )

    return X_train, X_validation


def train_xgboost(train_df, validation_df, model_params=None):
    """Train XGBoost and return nonnegative validation predictions."""

    params = DEFAULT_PARAMS.copy()

    if model_params:
        params.update(model_params)

    X_train, X_validation = prepare_xgboost_features(
        train_df, validation_df
    )

    y_train = pd.to_numeric(
        train_df["DK_POINTS"], errors="coerce"
    )

    if y_train.isna().any():
        raise ValueError("Training target contains missing values")

    model = XGBRegressor(**params)

    model.fit(X_train, y_train)

    predictions = np.maximum(
        model.predict(X_validation), 0
    )

    return model, predictions
