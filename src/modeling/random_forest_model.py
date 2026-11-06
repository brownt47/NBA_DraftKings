"""Frozen-season Random Forest benchmark for DraftKings points."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.modeling.features import select_model_features

TRAIN_PATH = Path("data/nba_features_2025_26.csv")
TEST_PATH = Path("data/nba_features_2026_27.csv")
BENCHMARK_PATH = Path(
    "reports/modeling/2026_27/frozen_model_predictions.csv"
)
OUTPUT_PATH = Path(
    "reports/modeling/2026_27/frozen_random_forest_predictions.csv"
)

CATEGORICAL = ["OPPONENT", "HOME_AWAY"]


def metrics(actual, predicted):
    error = predicted - actual
    return {
        "N": len(error),
        "MAE": np.abs(error).mean(),
        "RMSE": np.sqrt(np.mean(error ** 2)),
        "BIAS": error.mean(),
    }


def run():
    train = pd.read_csv(
        TRAIN_PATH,
        dtype={"GAME_ID": str, "PLAYER_ID": str},
    )
    test = pd.read_csv(
        TEST_PATH,
        dtype={"GAME_ID": str, "PLAYER_ID": str},
    )
    benchmark = pd.read_csv(
        BENCHMARK_PATH,
        dtype={"GAME_ID": str, "PLAYER_ID": str},
    )

    train = train.dropna(subset=["DK_POINTS"]).copy()

    # Evaluate exactly the existing frozen-model observations.
    test = test.merge(
        benchmark[["GAME_ID", "PLAYER_ID"]],
        on=["GAME_ID", "PLAYER_ID"],
        how="inner",
        validate="one_to_one",
    )

    X_train = select_model_features(train)
    X_test = select_model_features(test)

    numeric = [
        col for col in X_train.columns
        if col not in CATEGORICAL
    ]

    for col in numeric:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    for col in CATEGORICAL:
        X_train[col] = X_train[col].fillna("UNKNOWN").astype(str)
        X_test[col] = X_test[col].fillna("UNKNOWN").astype(str)

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL,
            ),
        ]
    )

    model = Pipeline([
        ("preprocessing", preprocessing),
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_leaf=5,
                max_features=0.85,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    print(f"Training on {len(train):,} player-games...")
    model.fit(X_train, train["DK_POINTS"])

    predictions = np.maximum(model.predict(X_test), 0)

    results = test[[
        "GAME_DATE", "GAME_ID", "PLAYER_ID", "PLAYER_NAME"
    ]].copy()

    results["RANDOM_FOREST"] = predictions

    comparison = benchmark.merge(
        results[["GAME_ID", "PLAYER_ID", "RANDOM_FOREST"]],
        on=["GAME_ID", "PLAYER_ID"],
        how="inner",
        validate="one_to_one",
    )

    if len(comparison) != len(benchmark):
        raise ValueError("Random Forest evaluation rows do not match benchmark")

    actual = comparison["ACTUAL"]

    summary = pd.DataFrame({
        "EMA_10": metrics(actual, comparison["EMA_10"]),
        "LIGHTGBM": metrics(actual, comparison["LIGHTGBM"]),
        "RANDOM_FOREST": metrics(actual, comparison["RANDOM_FOREST"]),
    }).T

    print("\n2026-27 FROZEN MODEL COMPARISON")
    print(summary.round(4).to_string())

    rf_error = np.abs(comparison["RANDOM_FOREST"] - actual)
    lgb_error = np.abs(comparison["LIGHTGBM"] - actual)

    print(
        "\nRandom Forest vs LightGBM win rate: "
        f"{(rf_error < lgb_error).mean():.2%}"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
