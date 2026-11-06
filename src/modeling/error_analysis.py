"""Analyze historical model errors by pregame role and minutes changes."""

from pathlib import Path

import numpy as np
import pandas as pd


PREDICTIONS_PATH = Path(
    "reports/modeling/xgboost_comparison/predictions.csv"
)
FEATURES_PATH = Path("data/nba_features_2025_26.csv")
OUTPUT_DIR = Path("reports/modeling/error_analysis")

KEYS = ["GAME_ID", "PLAYER_ID"]

MODELS = [
    "LIGHTGBM",
    "CATBOOST",
    "XGBOOST",
    "ENSEMBLE_50_50",
]


def calculate_metrics(df):
    results = []

    for model in MODELS:
        error = df[model] - df["ACTUAL"]

        results.append({
            "MODEL": model,
            "N": len(df),
            "MAE": error.abs().mean(),
            "RMSE": np.sqrt((error ** 2).mean()),
            "BIAS": error.mean(),
        })

    return pd.DataFrame(results)


def grouped_metrics(df, group_column):
    results = []

    for group, frame in df.groupby(
        group_column, observed=True, dropna=False
    ):
        metrics = calculate_metrics(frame)
        metrics.insert(0, group_column, str(group))
        results.append(metrics)

    return pd.concat(results, ignore_index=True)


def minutes_to_float(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, str) and ":" in value:
        try:
            minutes, seconds = value.split(":")
            return float(minutes) + float(seconds) / 60
        except (ValueError, TypeError):
            return np.nan

    return pd.to_numeric(value, errors="coerce")


def run():
    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        dtype={"GAME_ID": str, "PLAYER_ID": str},
    )

    features = pd.read_csv(
        FEATURES_PATH,
        dtype={"GAME_ID": str, "PLAYER_ID": str},
    )

    if predictions.duplicated(KEYS).any():
        raise ValueError("Duplicate prediction keys")

    if features.duplicated(KEYS).any():
        raise ValueError("Duplicate feature keys")

    columns = KEYS + [
        "MIN",
        "MIN_AVG_LAST_5",
        "DK_EMA_10",
        "PRIOR_GAMES",
    ]

    df = predictions.merge(
        features[columns],
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not df["_merge"].eq("both").all():
        raise ValueError("Some predictions lack feature records")

    df = df.drop(columns="_merge")

    # Pregame projected minutes
    df["PREGAME_ROLE"] = pd.cut(
        pd.to_numeric(df["MIN_AVG_LAST_5"], errors="coerce"),
        bins=[-np.inf, 10, 20, 30, np.inf],
        labels=["Under 10", "10-19", "20-29", "30+"],
        right=False,
    ).astype(object).fillna("Unknown")

    # Pregame projected fantasy production
    df["PREGAME_PRODUCTION"] = pd.cut(
        pd.to_numeric(df["DK_EMA_10"], errors="coerce"),
        bins=[-np.inf, 10, 20, 30, 40, np.inf],
        labels=["Under 10", "10-19", "20-29", "30-39", "40+"],
        right=False,
    ).astype(object).fillna("No EMA")

    # Postgame diagnostic: actual minutes minus pregame average
    df["ACTUAL_MINUTES"] = df["MIN"].apply(minutes_to_float)

    df["MINUTES_CHANGE"] = (
        df["ACTUAL_MINUTES"]
        - pd.to_numeric(df["MIN_AVG_LAST_5"], errors="coerce")
    )

    df["MINUTES_CHANGE_GROUP"] = pd.cut(
        df["MINUTES_CHANGE"],
        bins=[-np.inf, -10, -5, 5, 10, np.inf],
        labels=[
            "Major decrease",
            "Moderate decrease",
            "Stable",
            "Moderate increase",
            "Major increase",
        ],
        right=False,
    ).astype(object).fillna("Unknown")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    analyses = {
        "overall": calculate_metrics(df),
        "pregame_role": grouped_metrics(df, "PREGAME_ROLE"),
        "pregame_production": grouped_metrics(
            df, "PREGAME_PRODUCTION"
        ),
        "minutes_change": grouped_metrics(
            df, "MINUTES_CHANGE_GROUP"
        ),
    }

    for name, result in analyses.items():
        path = OUTPUT_DIR / f"{name}.csv"
        result.to_csv(path, index=False)
        print(f"Saved: {path}")

    print("\nPREGAME ROLE MAE")
    print(
        analyses["pregame_role"]
        .pivot(index="PREGAME_ROLE", columns="MODEL", values="MAE")
        .round(3).to_string()
    )

    print("\nPREGAME PRODUCTION MAE")
    print(
        analyses["pregame_production"]
        .pivot(
            index="PREGAME_PRODUCTION",
            columns="MODEL",
            values="MAE",
        )
        .round(3).to_string()
    )

    print("\nMINUTES CHANGE MAE")
    print(
        analyses["minutes_change"]
        .pivot(
            index="MINUTES_CHANGE_GROUP",
            columns="MODEL",
            values="MAE",
        )
        .round(3).to_string()
    )


if __name__ == "__main__":
    run()
