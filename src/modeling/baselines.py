"""Baseline prediction models for NBA DraftKings points."""

import numpy as np
import pandas as pd

from src.modeling.splits import monthly_walk_forward_splits


def evaluate_rolling_baseline(
    df,
    prediction_column="DK_POINTS_AVG_LAST_5",
    target_column="DK_POINTS",
):
    """Evaluate a historical rolling average across validation months."""

    results = []

    for fold, (train_idx, val_idx) in enumerate(
        monthly_walk_forward_splits(df), start=1
    ):
        validation = df.loc[val_idx]

        actual = pd.to_numeric(
            validation[target_column], errors="coerce"
        )
        predicted = pd.to_numeric(
            validation[prediction_column], errors="coerce"
        )

        valid = actual.notna() & predicted.notna()

        actual = actual.loc[valid]
        predicted = predicted.loc[valid]

        if actual.empty:
            raise ValueError(f"No valid predictions for fold {fold}")

        errors = predicted - actual

        results.append({
            "FOLD": fold,
            "MONTH": pd.to_datetime(
                validation["GAME_DATE"].iloc[0]
            ).strftime("%Y-%m"),
            "N": len(actual),
            "MAE": np.abs(errors).mean(),
            "RMSE": np.sqrt(np.mean(errors ** 2)),
            "BIAS": errors.mean(),
        })

    return pd.DataFrame(results)