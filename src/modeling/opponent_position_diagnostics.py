"""Analyze out-of-sample CatBoost residuals by opponent and position."""

from pathlib import Path
import numpy as np
import pandas as pd

PRED_PATH = Path("reports/modeling/boosting_comparison/predictions.csv")
FEATURE_PATH = Path("data/nba_features_2025_26.csv")
DIRECTORY_PATH = Path("data/espn_player_directory.csv")
OUT_DIR = Path("reports/modeling/opponent_position")


def main():
    pred = pd.read_csv(
        PRED_PATH, dtype={"GAME_ID": str, "PLAYER_ID": str}
    )
    features = pd.read_csv(
        FEATURE_PATH, dtype={"GAME_ID": str, "PLAYER_ID": str}
    )
    directory = pd.read_csv(
        DIRECTORY_PATH, dtype={"PLAYER_ID": str}
    )

    df = pred.merge(
        features[
            ["GAME_ID", "PLAYER_ID", "OPPONENT", "MIN_AVG_LAST_5"]
        ],
        on=["GAME_ID", "PLAYER_ID"],
        validate="one_to_one",
    )

    df = df.merge(
        directory[["PLAYER_ID", "POSITION"]],
        on="PLAYER_ID",
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

    # Cash-eligible playing-time group.
    df = df.loc[
        (df["MIN_AVG_LAST_5"] >= 22)
        & (df["CATBOOST"] >= 10)
        & df["POSITION_GROUP"].notna()
    ].copy()

    df["RESIDUAL"] = df["ACTUAL"] - df["CATBOOST"]
    df["ABS_ERROR"] = df["RESIDUAL"].abs()

    df["OVER_5"] = df["RESIDUAL"] <= -5
    df["UNDER_5"] = df["RESIDUAL"] >= 5
    df["BELOW_70"] = df["ACTUAL"] < 0.70 * df["CATBOOST"]

    def summarize(group):
        return pd.Series({
            "N": len(group),
            "AVG_PROJECTION": group["CATBOOST"].mean(),
            "AVG_RESIDUAL": group["RESIDUAL"].mean(),
            "MAE": group["ABS_ERROR"].mean(),
            "OVER_5_RATE": group["OVER_5"].mean(),
            "UNDER_5_RATE": group["UNDER_5"].mean(),
            "BELOW_70_RATE": group["BELOW_70"].mean(),
        })

    overall = df.groupby("POSITION_GROUP").apply(
        summarize, include_groups=False
    ).reset_index()

    matchup = df.groupby(
        ["OPPONENT", "POSITION_GROUP"]
    ).apply(
        summarize, include_groups=False
    ).reset_index()

    monthly = df.groupby(
        ["MONTH", "OPPONENT", "POSITION_GROUP"]
    ).apply(
        summarize, include_groups=False
    ).reset_index()

    # Position-adjusted residual relative to league-wide results.
    matchup = matchup.merge(
        overall[["POSITION_GROUP", "AVG_RESIDUAL"]].rename(
            columns={"AVG_RESIDUAL": "POSITION_AVG_RESIDUAL"}
        ),
        on="POSITION_GROUP",
    )

    matchup["ADJUSTED_RESIDUAL"] = (
        matchup["AVG_RESIDUAL"]
        - matchup["POSITION_AVG_RESIDUAL"]
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    overall.to_csv(OUT_DIR / "position_summary.csv", index=False)
    matchup.to_csv(OUT_DIR / "opponent_position_summary.csv", index=False)
    monthly.to_csv(OUT_DIR / "opponent_position_monthly.csv", index=False)

    print("\nPOSITION SUMMARY")
    print(overall.round(3).to_string(index=False))

    print("\nMOST NEGATIVE MATCHUP RESIDUALS (N >= 30)")
    print(
        matchup.loc[matchup["N"] >= 30]
        .sort_values("ADJUSTED_RESIDUAL")
        .head(15)
        .round(3)
        .to_string(index=False)
    )

    print("\nMOST POSITIVE MATCHUP RESIDUALS (N >= 30)")
    print(
        matchup.loc[matchup["N"] >= 30]
        .sort_values("ADJUSTED_RESIDUAL", ascending=False)
        .head(15)
        .round(3)
        .to_string(index=False)
    )

    print("\nMIAMI AND SAN ANTONIO CENTER MATCHUPS")
    print(
        matchup.loc[
            matchup["OPPONENT"].isin(["MIA", "SA"])
            & matchup["POSITION_GROUP"].eq("Center")
        ].round(3).to_string(index=False)
    )

    print(f"\nReports saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
