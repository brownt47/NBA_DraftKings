
from pathlib import Path

import numpy as np
import pandas as pd


PREDICTIONS_PATH = Path(
    "reports/modeling/biweekly/lightgbm_predictions.csv"
)
FEATURES_PATH = Path("data/nba_features_2025_26.csv")
OUTPUT_PATH = Path(
    "reports/modeling/biweekly/minutes_diagnostics.csv"
)

PREDICTORS = [
    "MIN_TREND_3",
    "MIN_STD_LAST_5",
    "PLAYER_MINUTES_SHARE_TREND",
]


def run_diagnostics():
    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        dtype={"GAME_ID": str, "PLAYER_ID": str},
    )

    features = pd.read_csv(
        FEATURES_PATH,
        dtype={"GAME_ID": str, "PLAYER_ID": str},
        usecols=["GAME_ID", "PLAYER_ID", *PREDICTORS],
    )

    data = predictions.merge(
        features,
        on=["GAME_ID", "PLAYER_ID"],
        how="left",
        validate="one_to_one",
    )

    data["MAJOR_MINUTES_INCREASE"] = (
        data["MINUTES_CHANGE"] >= 10
    )

    data["MAJOR_MINUTES_DECREASE"] = (
        data["MINUTES_CHANGE"] < -10
    )

    baseline_rate = data["MAJOR_MINUTES_INCREASE"].mean()
    print(f"Overall major-increase rate: {baseline_rate:.2%}")

    results = []

    for predictor in PREDICTORS:
        subset = data.dropna(subset=[predictor]).copy()

        # Rank values to create three equally sized groups.
        subset["GROUP"] = pd.qcut(
            subset[predictor].rank(method="first"),
            q=3,
            labels=["Low", "Middle", "High"],
        )

        summary = (
            subset.groupby("GROUP", observed=True)
            .agg(
                N=("MAJOR_MINUTES_INCREASE", "size"),
                MAJOR_INCREASES=("MAJOR_MINUTES_INCREASE", "sum"),
                INCREASE_RATE=("MAJOR_MINUTES_INCREASE", "mean"),
                MAJOR_DECREASES=("MAJOR_MINUTES_DECREASE", "sum"),
                DECREASE_RATE=("MAJOR_MINUTES_DECREASE", "mean"),
                MEDIAN_FEATURE_VALUE=(predictor, "median"),
            )
            .reset_index()
        )

        summary["PREDICTOR"] = predictor
        summary["BASELINE_RATE"] = baseline_rate
        summary["LIFT"] = (
            summary["INCREASE_RATE"] / baseline_rate
            if baseline_rate > 0
            else np.nan
        )

        results.append(summary)

    
    # Analyze minutes changes within pregame playing-time ranges.
    role_bins = [-np.inf, 10, 20, 30, np.inf]
    role_labels = ["Under 10", "10-19", "20-29", "30+"]

    data["PREGAME_ROLE"] = pd.cut(
        data["PREGAME_MINUTES"],
        bins=role_bins,
        labels=role_labels,
        right=False,
    )

    role_results = []

    for predictor in PREDICTORS:
        subset = data.dropna(
            subset=[predictor, "PREGAME_ROLE"]
        ).copy()

        # Use the same global tertiles as the original report.
        subset["GROUP"] = pd.qcut(
            subset[predictor].rank(method="first"),
            q=3,
            labels=["Low", "Middle", "High"],
        )

        summary = (
            subset.groupby(
                ["PREGAME_ROLE", "GROUP"],
                observed=True,
            )
            .agg(
                N=("MAJOR_MINUTES_INCREASE", "size"),
                INCREASE_RATE=("MAJOR_MINUTES_INCREASE", "mean"),
                DECREASE_RATE=("MAJOR_MINUTES_DECREASE", "mean"),
            )
            .reset_index()
        )

        summary["PREDICTOR"] = predictor
        role_results.append(summary)

    role_report = pd.concat(role_results, ignore_index=True)

    role_output_path = OUTPUT_PATH.with_name(
        "minutes_diagnostics_by_role.csv"
    )
    role_report.to_csv(role_output_path, index=False)
    print(f"\nSaved: {role_output_path}")


    report = pd.concat(results, ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)

    print(report.to_string(index=False))
    print(f"\nSaved: {OUTPUT_PATH}")

    return report


if __name__ == "__main__":
    run_diagnostics()
