from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.modeling.model_registry import get_model_adapter


DATA_PATHS = [
    Path("data/nba_features_2025_26.csv"),
    Path("data/nba_features_2026_27.csv"),
]

CUTOFF = pd.Timestamp("2026-11-01")
TEST_END = pd.Timestamp("2026-11-19")

TARGETS = [
    "MIN", "PTS", "REB", "AST",
    "FG3M", "STL", "BLK", "TOV",
    "DK_BONUS",
]

WEIGHTS = {
    "PTS": 1.0,
    "REB": 1.25,
    "AST": 1.5,
    "FG3M": 0.5,
    "STL": 2.0,
    "BLK": 2.0,
    "TOV": -0.5,
    "DK_BONUS": 1.0,
}

OUTPUT = Path("reports/modeling/stat_models")
OUTPUT.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.concat(
        [pd.read_csv(path) for path in DATA_PATHS],
        ignore_index=True,
    )

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    train = df[df["GAME_DATE"] < CUTOFF].copy()
    test = df[
        (df["GAME_DATE"] >= CUTOFF)
        & (df["GAME_DATE"] < TEST_END)
    ].copy()

    adapter = get_model_adapter("catboost")

    X_train, preprocessing = adapter.prepare_training_features(train)

    # Apply the training feature schema to the holdout.
    X_test, _ = adapter.prepare_training_features(test)
    X_test = X_test.reindex(columns=X_train.columns)

    print(f"Training games: {len(train):,}")
    print(f"Test games: {len(test):,}")
    print(f"Predictors: {len(X_train.columns)}")

    predictions = test[
        ["GAME_DATE", "PLAYER_ID", "PLAYER_NAME", "TEAM", "DK_POINTS"]
    ].copy()

    metrics = []

    for target in TARGETS:
        print(f"\nTraining {target}...")

        y_train = pd.to_numeric(train[target], errors="coerce")
        y_test = pd.to_numeric(test[target], errors="coerce")

        valid_train = y_train.notna()
        valid_test = y_test.notna()

        model = adapter.create_model()

        model.fit(
            X_train.loc[valid_train],
            y_train.loc[valid_train],
            cat_features=preprocessing["categories"],
        )

        predicted = np.maximum(model.predict(X_test), 0)

        predictions[f"PRED_{target}"] = predicted

        mae = mean_absolute_error(
            y_test.loc[valid_test],
            predicted[valid_test.to_numpy()],
        )

        rmse = np.sqrt(mean_squared_error(
            y_test.loc[valid_test],
            predicted[valid_test.to_numpy()],
        ))

        metrics.append({
            "TARGET": target,
            "MAE": mae,
            "RMSE": rmse,
        })

        print(f"{target}: MAE={mae:.4f} RMSE={rmse:.4f}")

    predictions["STAT_DERIVED_DK"] = sum(
        predictions[f"PRED_{target}"] * weight
        for target, weight in WEIGHTS.items()
    )

    predictions["STAT_DERIVED_DK"] = predictions[
        "STAT_DERIVED_DK"
    ].clip(lower=0)

    actual = pd.to_numeric(predictions["DK_POINTS"], errors="coerce")
    predicted = predictions["STAT_DERIVED_DK"]

    valid = actual.notna()

    overall_mae = mean_absolute_error(actual[valid], predicted[valid])
    overall_rmse = np.sqrt(
        mean_squared_error(actual[valid], predicted[valid])
    )

    print("\nSTAT-DERIVED DK RESULTS")
    print(f"MAE:  {overall_mae:.4f}")
    print(f"RMSE: {overall_rmse:.4f}")

    predictions.to_csv(OUTPUT / "predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(
        OUTPUT / "target_metrics.csv", index=False
    )

    print(f"\nSaved results to {OUTPUT}")


if __name__ == "__main__":
    main()

