"""Frozen-season CatBoost benchmark for DraftKings points."""

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.modeling.features import select_model_features

TRAIN_PATH = Path("data/nba_features_2025_26.csv")
TEST_PATH = Path("data/nba_features_2026_27.csv")
BENCHMARK_PATH = Path(
    "reports/modeling/2026_27/frozen_model_predictions.csv"
)
OUTPUT_PATH = Path(
    "reports/modeling/2026_27/frozen_catboost_predictions.csv"
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

    test = test.merge(
        benchmark[["GAME_ID", "PLAYER_ID"]],
        on=["GAME_ID", "PLAYER_ID"],
        how="inner",
        validate="one_to_one",
    )

    X_train = select_model_features(train)
    X_test = select_model_features(test)

    for column in X_train.columns:
        if column in CATEGORICAL:
            X_train[column] = (
                X_train[column].fillna("UNKNOWN").astype(str)
            )
            X_test[column] = (
                X_test[column].fillna("UNKNOWN").astype(str)
            )
        else:
            X_train[column] = pd.to_numeric(
                X_train[column], errors="coerce"
            )
            X_test[column] = pd.to_numeric(
                X_test[column], errors="coerce"
            )

    model = CatBoostRegressor(
        iterations=500,
        depth=7,
        learning_rate=0.03,
        loss_function="RMSE",
        random_seed=42,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )

    print(f"Training on {len(train):,} player-games...")

    model.fit(
        X_train,
        train["DK_POINTS"],
        cat_features=CATEGORICAL,
    )

    predictions = np.maximum(model.predict(X_test), 0)

    results = test[[
        "GAME_DATE", "GAME_ID", "PLAYER_ID", "PLAYER_NAME"
    ]].copy()
    results["CATBOOST"] = predictions

    comparison = benchmark.merge(
        results[["GAME_ID", "PLAYER_ID", "CATBOOST"]],
        on=["GAME_ID", "PLAYER_ID"],
        how="inner",
        validate="one_to_one",
    )

    if len(comparison) != len(benchmark):
        raise ValueError("CatBoost evaluation rows do not match benchmark")

    actual = comparison["ACTUAL"]

    summary = pd.DataFrame({
        "EMA_10": metrics(actual, comparison["EMA_10"]),
        "LIGHTGBM": metrics(actual, comparison["LIGHTGBM"]),
        "CATBOOST": metrics(actual, comparison["CATBOOST"]),
    }).T

    print("\n2026-27 FROZEN MODEL COMPARISON")
    print(summary.round(4).to_string())

    cb_error = np.abs(comparison["CATBOOST"] - actual)
    lgb_error = np.abs(comparison["LIGHTGBM"] - actual)

    print(
        "\nCatBoost vs LightGBM win rate: "
        f"{(cb_error < lgb_error).mean():.2%}"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
