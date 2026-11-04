
"""Walk-forward LightGBM model for NBA playing minutes."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.modeling.lightgbm_model import (
    DEFAULT_PARAMS,
    _minutes_to_float,
    _prepare_fold_features,
)
from src.modeling.splits import biweekly_walk_forward_splits


DATA_PATH = Path("data/nba_features_2025_26.csv")
OUTPUT_DIR = Path("reports/modeling/minutes")

TARGET = "MINUTES_TARGET"
BASELINE = "MIN_AVG_LAST_5"

ROLE_BINS = [-np.inf, 10, 20, 30, np.inf]
ROLE_LABELS = ["Under 10", "10-19", "20-29", "30+"]


def calculate_metrics(df):
    """Calculate prediction errors for both minutes models."""

    results = {}

    for name, column in [
        ("BASELINE", "BASELINE"),
        ("LIGHTGBM", "LIGHTGBM"),
    ]:
        errors = df[column] - df["ACTUAL"]

        results[f"MAE_{name}"] = errors.abs().mean()
        results[f"RMSE_{name}"] = np.sqrt(
            (errors ** 2).mean()
        )
        results[f"BIAS_{name}"] = errors.mean()

    results["N"] = len(df)
    results["MAE_IMPROVEMENT"] = (
        results["MAE_BASELINE"] - results["MAE_LIGHTGBM"]
    )

    return pd.Series(results)


def run_minutes_model():
    df = pd.read_csv(
        DATA_PATH,
        dtype={"GAME_ID": str, "PLAYER_ID": str},
    )

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Actual minutes are the target, never a predictor.
    df[TARGET] = df["MIN"].map(_minutes_to_float)

    predictions_by_fold = []

    for fold, (train_idx, validation_idx) in enumerate(
        biweekly_walk_forward_splits(df), start=1
    ):
        train_df = df.iloc[train_idx].copy()
        validation_df = df.iloc[validation_idx].copy()

        train_df = train_df.dropna(subset=[TARGET]).copy()

        X_train, X_validation = _prepare_fold_features(
            train_df, validation_df
        )

        # Ensure current-game outcomes cannot enter the model.
        forbidden = {
            TARGET,
            "MIN",
            "MINUTES_NUMERIC",
            "DK_POINTS",
        }

        leaked = forbidden.intersection(X_train.columns)
        if leaked:
            raise ValueError(
                f"Current-game outcome predictors found: {sorted(leaked)}"
            )

        y_train = train_df[TARGET]

        model = lgb.LGBMRegressor(**DEFAULT_PARAMS)
        model.fit(X_train, y_train)

        predicted_minutes = model.predict(X_validation)

        valid = (
            validation_df[TARGET].notna()
            & validation_df[BASELINE].notna()
            & np.isfinite(predicted_minutes)
        )

        results = validation_df.loc[
            valid,
            [
                "GAME_ID",
                "PLAYER_ID",
                "GAME_DATE",
                "PLAYER_NAME",
                BASELINE,
            ],
        ].copy()

        results["FOLD"] = fold
        results["ACTUAL"] = validation_df.loc[valid, TARGET]
        results["BASELINE"] = validation_df.loc[valid, BASELINE]
        results["LIGHTGBM"] = predicted_minutes[valid]

        results["PREGAME_ROLE"] = pd.cut(
            results["BASELINE"],
            bins=ROLE_BINS,
            labels=ROLE_LABELS,
            right=False,
        )

        predictions_by_fold.append(results)

        print(
            f"Fold {fold}: "
            f"{len(train_df):,} training rows, "
            f"{len(results):,} evaluation rows"
        )

    predictions = pd.concat(
        predictions_by_fold, ignore_index=True
    )

    overall = calculate_metrics(predictions).to_frame().T

    fold_metrics = (
        predictions.groupby("FOLD")
        .apply(calculate_metrics, include_groups=False)
        .reset_index()
    )

    role_metrics = (
        predictions.groupby("PREGAME_ROLE", observed=True)
        .apply(calculate_metrics, include_groups=False)
        .reset_index()
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(
        OUTPUT_DIR / "minutes_predictions.csv", index=False
    )
    overall.to_csv(
        OUTPUT_DIR / "minutes_overall_metrics.csv", index=False
    )
    fold_metrics.to_csv(
        OUTPUT_DIR / "minutes_fold_metrics.csv", index=False
    )
    role_metrics.to_csv(
        OUTPUT_DIR / "minutes_role_metrics.csv", index=False
    )

    print("\nOVERALL RESULTS")
    print(overall.to_string(index=False))

    print("\nPREGAME ROLE RESULTS")
    print(role_metrics.to_string(index=False))

    return predictions, overall, fold_metrics, role_metrics


if __name__ == "__main__":
    run_minutes_model()
