"""Compare baseline and ceiling-aware CatBoost using walk-forward CV."""

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.modeling.features import PREGAME_FEATURES
from src.modeling.splits import biweekly_walk_forward_splits

DATA_PATH = Path("data/nba_features_2025_26_ceiling.csv")
OUTPUT_DIR = Path("reports/modeling/minutes_ceiling")

NEW_FEATURES = [
    "MIN_TREND_RECENT_3",
    "MIN_P90_LAST_10",
    "MIN_HEADROOM_LAST_10",
    "MIN_CEILING_RATIO_LAST_10",
]

CATEGORICAL = ["OPPONENT", "HOME_AWAY"]

FEATURE_SETS = {
    "BASELINE": PREGAME_FEATURES,
    "CEILING": PREGAME_FEATURES + NEW_FEATURES,
}


def prepare_features(df, columns):
    X = df[columns].copy()

    for col in columns:
        if col in CATEGORICAL:
            X[col] = X[col].fillna("UNKNOWN").astype(str)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return X


def metrics(df, column):
    error = df[column] - df["ACTUAL"]

    return {
        "N": len(df),
        "MAE": error.abs().mean(),
        "RMSE": np.sqrt((error ** 2).mean()),
        "BIAS": error.mean(),
    }


def run():
    df = pd.read_csv(
        DATA_PATH,
        dtype={"GAME_ID": str, "PLAYER_ID": str},
    )

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    predictions = []

    for fold, (train_idx, val_idx) in enumerate(
        biweekly_walk_forward_splits(df), start=1
    ):
        train = df.loc[train_idx].dropna(
            subset=["DK_POINTS"]
        ).copy()

        val = df.loc[val_idx].dropna(
            subset=["DK_POINTS"]
        ).copy()

        result = val[[
            "GAME_DATE",
            "GAME_ID",
            "PLAYER_ID",
            "PLAYER_NAME",
            "MIN",
            "MIN_AVG_LAST_5",
            "MIN_TREND_RECENT_3",
            "MIN_P90_LAST_10",
            "MIN_HEADROOM_LAST_10",
            "MIN_CEILING_RATIO_LAST_10",
        ]].copy()

        result["FOLD"] = fold
        result["ACTUAL"] = val["DK_POINTS"]

        for name, columns in FEATURE_SETS.items():
            X_train = prepare_features(train, columns)
            X_val = prepare_features(val, columns)

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

            model.fit(
                X_train,
                train["DK_POINTS"],
                cat_features=CATEGORICAL,
            )

            result[name] = np.maximum(
                model.predict(X_val), 0
            )

        predictions.append(result)

        print(
            f"Fold {fold}: "
            f"{len(train):,} training, "
            f"{len(val):,} validation"
        )

    combined = pd.concat(
        predictions, ignore_index=True
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    combined.to_csv(
        OUTPUT_DIR / "predictions.csv",
        index=False,
    )

    overall = pd.DataFrame({
        name: metrics(combined, name)
        for name in FEATURE_SETS
    }).T

    overall.to_csv(
        OUTPUT_DIR / "overall_metrics.csv"
    )

    print("\nOVERALL RESULTS")
    print(overall.round(4).to_string())

    print("\nFOLD MAE")
    for fold, group in combined.groupby("FOLD"):
        base = metrics(group, "BASELINE")["MAE"]
        ceiling = metrics(group, "CEILING")["MAE"]

        print(
            f"Fold {fold}: "
            f"Baseline {base:.4f}, "
            f"Ceiling {ceiling:.4f}, "
            f"Improvement {base - ceiling:+.4f}"
        )

    print("\nRISING MINUTES ANALYSIS")

    for threshold in [0, 3, 5, 8]:
        group = combined.loc[
            combined["MIN_TREND_RECENT_3"] >= threshold
        ]

        if group.empty:
            continue

        base = metrics(group, "BASELINE")
        ceiling = metrics(group, "CEILING")

        print(
            f"Trend >= +{threshold}: "
            f"N={len(group):,}, "
            f"Baseline MAE={base['MAE']:.4f}, "
            f"Ceiling MAE={ceiling['MAE']:.4f}, "
            f"Improvement={base['MAE'] - ceiling['MAE']:+.4f}"
        )

    print("\nNEAR CEILING ANALYSIS")

    for threshold in [0.80, 0.90, 1.00]:
        group = combined.loc[
            combined["MIN_CEILING_RATIO_LAST_10"] >= threshold
        ]

        if group.empty:
            continue

        base = metrics(group, "BASELINE")
        ceiling = metrics(group, "CEILING")

        print(
            f"Ratio >= {threshold:.2f}: "
            f"N={len(group):,}, "
            f"Baseline MAE={base['MAE']:.4f}, "
            f"Ceiling MAE={ceiling['MAE']:.4f}, "
            f"Improvement={base['MAE'] - ceiling['MAE']:+.4f}"
        )

    print("\nSaved results to:", OUTPUT_DIR)


if __name__ == "__main__":
    run()
