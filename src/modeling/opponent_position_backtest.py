"""Chronological opponent-position CatBoost residual adjustment backtest."""

from pathlib import Path
import numpy as np
import pandas as pd

PRED_PATH = Path("reports/modeling/boosting_comparison/predictions.csv")
FEATURE_PATH = Path("data/nba_features_2025_26.csv")
DIRECTORY_PATH = Path("data/espn_player_directory.csv")
OUT_DIR = Path("reports/modeling/opponent_position")

SHRINKAGE = [10, 20, 30, 50]
MIN_MINUTES = 22
MIN_PROJECTION = 10


def load_data():
    pred = pd.read_csv(
        PRED_PATH,
        dtype={"GAME_ID": str, "PLAYER_ID": str},
    )
    features = pd.read_csv(
        FEATURE_PATH,
        usecols=[
            "GAME_ID", "PLAYER_ID", "OPPONENT",
            "MIN_AVG_LAST_5",
        ],
        dtype={"GAME_ID": str, "PLAYER_ID": str},
    )
    directory = pd.read_csv(
        DIRECTORY_PATH,
        usecols=["PLAYER_ID", "POSITION"],
        dtype={"PLAYER_ID": str},
    )

    df = pred.merge(
        features,
        on=["GAME_ID", "PLAYER_ID"],
        how="inner",
        validate="one_to_one",
    )
    df = df.merge(
        directory,
        on="PLAYER_ID",
        how="left",
        validate="many_to_one",
    )

    position_map = {
        "G": "Guard",
        "PG": "Guard",
        "F": "Forward",
        "SF": "Forward",
        "C": "Center",
    }

    df["POSITION_GROUP"] = df["POSITION"].map(position_map)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["MONTH"] = df["GAME_DATE"].dt.to_period("M").astype(str)

    df = df.loc[
        (df["MIN_AVG_LAST_5"] >= MIN_MINUTES)
        & (df["CATBOOST"] >= MIN_PROJECTION)
        & df["POSITION_GROUP"].notna()
    ].copy()

    df["RESIDUAL"] = df["ACTUAL"] - df["CATBOOST"]

    return df.sort_values(
        ["GAME_DATE", "GAME_ID", "PLAYER_ID"]
    ).reset_index(drop=True)


def add_prior_adjustments(df):
    """
    Use strictly earlier game dates for residual estimates.

    All player-games on the same date receive the same
    pre-date history for their respective groups.
    """
    df = df.copy()

    keys = ["OPPONENT", "POSITION_GROUP"]

    # Aggregate by date first to prevent same-day leakage.
    daily = (
        df.groupby(keys + ["GAME_DATE"], as_index=False)
        .agg(
            DAILY_N=("RESIDUAL", "size"),
            DAILY_SUM=("RESIDUAL", "sum"),
        )
        .sort_values(keys + ["GAME_DATE"])
    )

    grouped = daily.groupby(keys, sort=False)

    daily["PRIOR_N"] = (
        grouped["DAILY_N"].cumsum() - daily["DAILY_N"]
    )
    daily["PRIOR_SUM"] = (
        grouped["DAILY_SUM"].cumsum() - daily["DAILY_SUM"]
    )

    df = df.merge(
        daily[keys + ["GAME_DATE", "PRIOR_N", "PRIOR_SUM"]],
        on=keys + ["GAME_DATE"],
        how="left",
        validate="many_to_one",
    )

    # Position-only historical residuals.
    pos_daily = (
        df.groupby(["POSITION_GROUP", "GAME_DATE"], as_index=False)
        .agg(
            POS_DAILY_N=("RESIDUAL", "size"),
            POS_DAILY_SUM=("RESIDUAL", "sum"),
        )
        .sort_values(["POSITION_GROUP", "GAME_DATE"])
    )

    pos_group = pos_daily.groupby("POSITION_GROUP", sort=False)

    pos_daily["POS_PRIOR_N"] = (
        pos_group["POS_DAILY_N"].cumsum()
        - pos_daily["POS_DAILY_N"]
    )
    pos_daily["POS_PRIOR_SUM"] = (
        pos_group["POS_DAILY_SUM"].cumsum()
        - pos_daily["POS_DAILY_SUM"]
    )

    df = df.merge(
        pos_daily[
            [
                "POSITION_GROUP", "GAME_DATE",
                "POS_PRIOR_N", "POS_PRIOR_SUM",
            ]
        ],
        on=["POSITION_GROUP", "GAME_DATE"],
        how="left",
        validate="many_to_one",
    )

    df["POS_ADJUSTMENT"] = (
        df["POS_PRIOR_SUM"]
        / (df["POS_PRIOR_N"] + 30)
    )

    df["POSITION_ONLY"] = (
        df["CATBOOST"] + df["POS_ADJUSTMENT"]
    ).clip(lower=0)

    for k in SHRINKAGE:
        # Shrink opponent-position residual estimates toward
        # the position-wide residual estimate.
        opponent_effect = (
            df["PRIOR_SUM"]
            - df["PRIOR_N"] * df["POS_ADJUSTMENT"]
        ) / (df["PRIOR_N"] + k)

        df[f"OPP_POS_K{k}"] = (
            df["CATBOOST"]
            + df["POS_ADJUSTMENT"]
            + opponent_effect
        ).clip(lower=0)

    return df


def metrics(group, column):
    error = group["ACTUAL"] - group[column]

    return {
        "N": len(group),
        "MAE": error.abs().mean(),
        "RMSE": np.sqrt((error ** 2).mean()),
        "BIAS": error.mean(),
        "BELOW_70_RATE": (
            group["ACTUAL"] < 0.70 * group[column]
        ).mean(),
    }


def main():
    df = load_data()
    df = add_prior_adjustments(df)

    models = [
        ("CATBOOST", "CATBOOST"),
        ("POSITION_ONLY", "POSITION_ONLY"),
    ] + [
        (f"OPP_POS_K{k}", f"OPP_POS_K{k}")
        for k in SHRINKAGE
    ]

    evaluation = df.loc[
        df["MONTH"].isin(["2026-02", "2026-03"])
    ].copy()

    results = []

    for month, subset in [
        ("ALL", evaluation),
        *list(evaluation.groupby("MONTH")),
    ]:
        for name, column in models:
            results.append({
                "MONTH": month,
                "MODEL": name,
                **metrics(subset, column),
            })

    summary = pd.DataFrame(results)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUT_DIR / "backtest_predictions.csv",
        index=False,
    )
    summary.to_csv(
        OUT_DIR / "backtest_metrics.csv",
        index=False,
    )

    print("\nCHRONOLOGICAL OPPONENT-POSITION BACKTEST")
    print(f"Total eligible player-games: {len(df):,}")
    print(f"February/March evaluation: {len(evaluation):,}")

    print("\nOVERALL RESULTS")
    print(
        summary.loc[summary["MONTH"] == "ALL"]
        .round(4).to_string(index=False)
    )

    print("\nMONTHLY RESULTS")
    print(
        summary.loc[summary["MONTH"] != "ALL"]
        .round(4).to_string(index=False)
    )

    print("\nPRIOR MATCHUP SAMPLE SIZES")
    print(
        evaluation["PRIOR_N"]
        .describe()
        .round(2)
        .to_string()
    )

    print(f"\nReports saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
