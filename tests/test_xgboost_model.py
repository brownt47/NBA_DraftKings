import numpy as np
import pandas as pd
import pytest

from src.modeling.features import PREGAME_FEATURES
from src.modeling.xgboost_model import (
    prepare_xgboost_features,
    train_xgboost,
)


def make_data(n=12):
    data = {
        column: np.arange(n, dtype=float)
        for column in PREGAME_FEATURES
    }

    data["OPPONENT"] = ["ATL", "BOS"] * (n // 2)
    data["HOME_AWAY"] = ["HOME", "AWAY"] * (n // 2)
    data["DK_POINTS"] = np.arange(n, dtype=float) * 2

    return pd.DataFrame(data)


def test_feature_columns():
    train = make_data()
    validation = make_data()

    X_train, X_validation = prepare_xgboost_features(
        train, validation
    )

    assert list(X_train.columns) == PREGAME_FEATURES
    assert list(X_validation.columns) == PREGAME_FEATURES
    assert len(X_train.columns) == 125


def test_categorical_alignment():
    train = make_data()
    validation = make_data()

    validation["OPPONENT"] = "NYK"

    _, X_validation = prepare_xgboost_features(
        train, validation
    )

    assert X_validation["OPPONENT"].isna().all()


def test_training_predictions():
    train = make_data()
    validation = make_data()

    model, predictions = train_xgboost(
        train,
        validation,
        model_params={
            "n_estimators": 5,
            "n_jobs": 1,
        },
    )

    assert len(predictions) == len(validation)
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()


def test_missing_target_rejected():
    train = make_data()
    validation = make_data()

    train.loc[0, "DK_POINTS"] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        train_xgboost(
            train,
            validation,
            model_params={"n_estimators": 5},
        )
