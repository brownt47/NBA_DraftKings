
"""LightGBM walk-forward modeling for NBA DraftKings points."""

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.modeling.features import prepare_model_features
from src.modeling.splits import (
    monthly_walk_forward_splits,
    biweekly_walk_forward_splits,
)


DEFAULT_DATA_PATH = Path("data/nba_features_2025_26.csv")
DEFAULT_OUTPUT_DIR = Path("reports/modeling")

TARGET_COLUMN = "DK_POINTS"
BASELINE_COLUMN = "DK_EMA_10"
DATE_COLUMN = "GAME_DATE"
MINUTES_COLUMN = "MIN"

CATEGORICAL_COLUMNS = ["OPPONENT", "HOME_AWAY"]

MINUTE_BINS = [0, 10, 20, 30, 35, np.inf]
MINUTE_LABELS = [
    "Under 10",
    "10-19",
    "20-29",
    "30-34",
    "35+",
]

PREGAME_MINUTES_COLUMN = "MIN_AVG_LAST_5"

PREGAME_ROLE_BINS = [-np.inf, 10, 20, 30, np.inf]
PREGAME_ROLE_LABELS = [
    "Under 10",
    "10-19",
    "20-29",
    "30+",
]

DEFAULT_PARAMS = {
    "objective": "regression",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 7,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


def _prepare_fold_features(train_df, validation_df):
    """Prepare predictors and align categorical levels within each fold."""

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


def _minutes_to_float(value):
    """Convert numeric or MM:SS minutes into decimal minutes."""

    if pd.isna(value):
        return np.nan

    if isinstance(value, str) and ":" in value:
        parts = value.split(":")

        try:
            return float(parts[0]) + float(parts[1]) / 60
        except (ValueError, IndexError):
            return np.nan

    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def _calculate_metrics(actual, predicted):
    """Calculate regression metrics using prediction minus actual."""

    errors = predicted - actual

    return {
        "N": len(actual),
        "MAE": np.abs(errors).mean(),
        "RMSE": np.sqrt(np.mean(errors ** 2)),
        "BIAS": errors.mean(),
    }


def _build_comparison_metrics(predictions, group_columns):
    """Compare EMA-10 and LightGBM on identical player-games."""

    results = []

    if group_columns:
        grouped = predictions.groupby(
            group_columns, observed=True, dropna=False
        )
    else:
        grouped = [(None, predictions)]

    for group, frame in grouped:

        if not isinstance(group, tuple):
            group = (group,) if group is not None else ()

        row = {}

        for column, value in zip(group_columns, group):
            row[column] = value

        actual = frame["ACTUAL"]

        ema_metrics = _calculate_metrics(
            actual, frame["EMA_10"]
        )

        lgb_metrics = _calculate_metrics(
            actual, frame["LIGHTGBM"]
        )

        for metric in ["N", "MAE", "RMSE", "BIAS"]:
            row[f"{metric}_EMA"] = ema_metrics[metric]
            row[f"{metric}_LIGHTGBM"] = lgb_metrics[metric]

        row["MAE_IMPROVEMENT"] = (
            ema_metrics["MAE"] - lgb_metrics["MAE"]
        )

        results.append(row)

    return pd.DataFrame(results)


def run_walk_forward(df, model_params=None, split="monthly"):
    """
    Train and evaluate LightGBM using chronological monthly folds.

    April 2026 remains reserved for the final untouched holdout.
    """

    params = DEFAULT_PARAMS.copy()

    if model_params:
        params.update(model_params)

    required_columns = [
        TARGET_COLUMN,
        BASELINE_COLUMN,
        DATE_COLUMN,
        MINUTES_COLUMN,
        "PLAYER_NAME",
        PREGAME_MINUTES_COLUMN,
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    predictions_by_fold = []
    importance_by_fold = []

    if split == "monthly":
        splits = monthly_walk_forward_splits(df)
    elif split == "biweekly":
        splits = biweekly_walk_forward_splits(df)
    else:
        raise ValueError(f"Unknown split type: {split}")

    for fold, (train_idx, validation_idx) in enumerate(
        splits, start=1
    ):

        train_df = df.loc[train_idx].copy()
        validation_df = df.loc[validation_idx].copy()

        y_train = pd.to_numeric(
            train_df[TARGET_COLUMN], errors="coerce"
        )

        train_mask = y_train.notna()

        train_df = train_df.loc[train_mask]
        y_train = y_train.loc[train_mask]

        if train_df.empty:
            raise ValueError(f"No usable training records in fold {fold}")

        X_train, X_validation = _prepare_fold_features(
            train_df, validation_df
        )

        model = lgb.LGBMRegressor(**params)

        model.fit(X_train, y_train)

        # Collect normalized gain-based feature importance.
        importance = model.booster_.feature_importance(
            importance_type="gain"
        )

        importance_df = pd.DataFrame({
            "FEATURE": X_train.columns,
            "IMPORTANCE": importance,
        })

        total_importance = importance_df["IMPORTANCE"].sum()

        if total_importance > 0:
            importance_df["IMPORTANCE_PCT"] = (
                importance_df["IMPORTANCE"] / total_importance * 100
            )
        else:
            importance_df["IMPORTANCE_PCT"] = 0

        importance_df["FOLD"] = fold

        importance_by_fold.append(importance_df)

        validation_predictions = model.predict(X_validation)

        actual = pd.to_numeric(
            validation_df[TARGET_COLUMN], errors="coerce"
        )

        ema = pd.to_numeric(
            validation_df[BASELINE_COLUMN], errors="coerce"
        )

        # Restrict both models to exactly the same records.
        valid = (
            actual.notna()
            & ema.notna()
            & np.isfinite(validation_predictions)
        )

        results = validation_df.loc[
            valid,
        ["GAME_DATE", "PLAYER_NAME", "MIN", PREGAME_MINUTES_COLUMN, "GAME_ID","PLAYER_ID", "OPPONENT", "HOME_AWAY"]
        ].copy()

        results["FOLD"] = fold
        results["ACTUAL"] = actual.loc[valid]
        results["EMA_10"] = ema.loc[valid]
        results["LIGHTGBM"] = validation_predictions[valid]

        results["ERROR_EMA"] = (
            results["EMA_10"] - results["ACTUAL"]
        )

        results["ERROR_LIGHTGBM"] = (
            results["LIGHTGBM"] - results["ACTUAL"]
        )

        results["MONTH"] = results["GAME_DATE"].dt.strftime("%Y-%m")

        predictions_by_fold.append(results)

        print(
            f"Fold {fold}: "
            f"trained on {len(train_df):,} records, "
            f"evaluated {len(results):,} records "
            f"({results['MONTH'].iloc[0] if not results.empty else 'N/A'})"
        )

    predictions = pd.concat(
        predictions_by_fold, ignore_index=True
    )

    # Convert actual playing time for postgame analysis only.
    predictions["MINUTES_NUMERIC"] = (
        predictions[MINUTES_COLUMN].apply(_minutes_to_float)
    )

    predictions["MIN_RANGE"] = pd.cut(
        predictions["MINUTES_NUMERIC"],
        bins=MINUTE_BINS,
        labels=MINUTE_LABELS,
        right=False,
    )

    predictions["PREGAME_MINUTES"] = pd.to_numeric(
        predictions[PREGAME_MINUTES_COLUMN],
        errors="coerce",
    )

    predictions["PREGAME_ROLE"] = pd.cut(
        predictions["PREGAME_MINUTES"],
        bins=PREGAME_ROLE_BINS,
        labels=PREGAME_ROLE_LABELS,
        right=False,
    )

    pregame_role_metrics = _build_comparison_metrics(
        predictions.dropna(subset=["PREGAME_ROLE"]),
        ["PREGAME_ROLE"],
    )

    pregame_role_metrics["LIGHTGBM_WIN_RATE"] = [
        (
            predictions.loc[
                predictions["PREGAME_ROLE"] == role,
                "ERROR_LIGHTGBM",
            ].abs()
            <
            predictions.loc[
                predictions["PREGAME_ROLE"] == role,
                "ERROR_EMA",
            ].abs()
        ).mean()
        for role in pregame_role_metrics["PREGAME_ROLE"]
    ]

        # Existing pregame role analysis
    pregame_role_metrics["LIGHTGBM_WIN_RATE"] = [
        (
            predictions.loc[
                predictions["PREGAME_ROLE"] == role,
                "ERROR_LIGHTGBM",
            ].abs()
            <
            predictions.loc[
                predictions["PREGAME_ROLE"] == role,
                "ERROR_EMA",
            ].abs()
        ).mean()
        for role in pregame_role_metrics["PREGAME_ROLE"]
    ]

    # PASTE THE NEW MINUTES-CHANGE CODE HERE

    
    # Minutes change relative to pregame five-game average
    predictions["MINUTES_CHANGE"] = (
        predictions["MINUTES_NUMERIC"]
        - predictions["PREGAME_MINUTES"]
    )

    minutes_change_bins = [-np.inf, -10, -5, 5, 10, np.inf]
    minutes_change_labels = [
        "Major decrease",
        "Moderate decrease",
        "Stable",
        "Moderate increase",
        "Major increase",
    ]

    predictions["MINUTES_CHANGE_GROUP"] = pd.cut(
        predictions["MINUTES_CHANGE"],
        bins=minutes_change_bins,
        labels=minutes_change_labels,
        right=False,
    )

    minutes_change_metrics = _build_comparison_metrics(
        predictions.dropna(subset=["MINUTES_CHANGE_GROUP"]),
        ["MINUTES_CHANGE_GROUP"],
    )

    minutes_change_metrics["LIGHTGBM_WIN_RATE"] = [
        (
            predictions.loc[
                predictions["MINUTES_CHANGE_GROUP"] == group,
                "ERROR_LIGHTGBM",
            ].abs()
            <
            predictions.loc[
                predictions["MINUTES_CHANGE_GROUP"] == group,
                "ERROR_EMA",
            ].abs()
        ).mean()
        for group in minutes_change_metrics["MINUTES_CHANGE_GROUP"]
    ]

    # Existing monthly evaluation
    monthly_metrics = _build_comparison_metrics(
        predictions, ["MONTH"]
    )

    fold_metrics = _build_comparison_metrics(
        predictions, ["FOLD"]
    )

    minute_metrics = _build_comparison_metrics(
        predictions.dropna(subset=["MIN_RANGE"]),
        ["MIN_RANGE"],
    )

    overall_metrics = _build_comparison_metrics(
        predictions, []
    )


    importance_details = pd.concat(
        importance_by_fold,
        ignore_index=True
    )

    importance_summary = (
        importance_details
        .groupby("FEATURE")
        .agg(
            MEAN_IMPORTANCE=("IMPORTANCE_PCT", "mean"),
            STD_IMPORTANCE=("IMPORTANCE_PCT", "std"),
        )
        .reset_index()
        .sort_values("MEAN_IMPORTANCE", ascending=False)
    )

    importance_summary["RANK"] = range(
        1, len(importance_summary) + 1
    )

    return {
        "predictions": predictions,
        "monthly_metrics": monthly_metrics,
        "fold_metrics": fold_metrics,
        "minute_metrics": minute_metrics,
        "overall_metrics": overall_metrics,
        "importance_details": importance_details,
        "importance_summary": importance_summary,
        "pregame_role_metrics": pregame_role_metrics,
        "minutes_change_metrics": minutes_change_metrics
    }


def main():
    parser = argparse.ArgumentParser(
        description="NBA DraftKings LightGBM walk-forward evaluation"
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--split",
        choices=["monthly", "biweekly"],
        default="monthly",
        help="Walk-forward validation frequency",
    )

    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR / args.split
    
    print(f"Loading: {args.data}")

    df = pd.read_csv(args.data)

    print(f"Loaded {len(df):,} player-game records")

    results = run_walk_forward(df, split=args.split)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_files = {
        "predictions": "lightgbm_predictions.csv",
        "monthly_metrics": "lightgbm_monthly_metrics.csv",
        "fold_metrics": "lightgbm_fold_metrics.csv",
        "minute_metrics": "lightgbm_minute_metrics.csv",
        "overall_metrics": "lightgbm_overall_metrics.csv",
        "importance_details": "lightgbm_importance_by_fold.csv",
        "importance_summary": "lightgbm_importance_summary.csv",
        "pregame_role_metrics": "lightgbm_pregame_role_metrics.csv",
        "minutes_change_metrics": "lightgbm_minutes_change_metrics.csv"
    }

    for key, filename in output_files.items():

        path = args.output_dir / filename

        results[key].to_csv(path, index=False)

        print(f"Saved: {path}")

    print("\nOVERALL RESULTS")
    print(results["overall_metrics"].to_string(index=False))

    print("\nMONTHLY RESULTS")
    print(results["monthly_metrics"].to_string(index=False))

    print("\nMINUTE RANGE RESULTS")
    print(results["minute_metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
