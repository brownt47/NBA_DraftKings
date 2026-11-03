"""Tests for pregame model feature selection."""

import pandas as pd
import pytest

from src.modeling.features import (
    PREGAME_FEATURES,
    select_model_features,
)


def test_select_model_features_excludes_outcomes():
    """Same-game statistics must not enter the model."""
    df = pd.DataFrame({
        **{col: [1.0, 2.0] for col in PREGAME_FEATURES},
        "DK_POINTS": [35.0, 42.0],
        "MIN": [32.0, 36.0],
        "PTS": [24.0, 30.0],
        "REB": [8.0, 10.0],
        "AST": [5.0, 7.0],
        "FGA": [18.0, 22.0],
    })

    result = select_model_features(df)

    assert result.columns.tolist() == PREGAME_FEATURES
    assert result.shape == (2, len(PREGAME_FEATURES))

    for column in ["DK_POINTS", "MIN", "PTS", "REB", "AST", "FGA"]:
        assert column not in result.columns


def test_select_model_features_rejects_missing_columns():
    """Missing required predictors should raise an error."""
    df = pd.DataFrame({"DK_POINTS": [30.0]})

    with pytest.raises(ValueError, match="Missing model features"):
        select_model_features(df)

def test_prepare_model_features():
    """Categorical conversion preserves missing values and excludes outcomes."""
    from src.modeling.features import prepare_model_features

    df = pd.DataFrame({
        **{col: [1.0, None] for col in PREGAME_FEATURES},
        "OPPONENT": ["BOS", "NYK"],
        "HOME_AWAY": ["HOME", "AWAY"],
        "DK_POINTS": [35.0, 42.0],
        "MIN": [32.0, 36.0],
    })

    result = prepare_model_features(df)

    assert len(result.columns) == len(PREGAME_FEATURES)
    assert str(result["OPPONENT"].dtype) == "category"
    assert str(result["HOME_AWAY"].dtype) == "category"

    assert pd.isna(result["DK_EMA_10"].iloc[1])
    assert "DK_POINTS" not in result.columns
    assert "MIN" not in result.columns