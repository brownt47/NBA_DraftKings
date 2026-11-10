
"""Evaluate pregame opponent-position adjustments to frozen CatBoost."""

import numpy as np
import pandas as pd

FEATURES = "data/nba_features_2025_26.csv"
DIRECTORY = "data/espn_player_directory.csv"
PREDICTIONS = "reports/modeling/boosting_comparison/predictions.csv"

KEYS = ["GAME_ID", "PLAYER_ID"]
WINDOWS = [10, 20]
SHRINKAGE = 20


def main():
    features = pd.read_csv(
        FEATURES, dtype={"GAME_ID": str, "PLAYER_ID": str}
    )
    directory = pd.read_csv(
        DIRECTORY, dtype={"PLAYER_ID": str}
    )
    predictions = pd.read_csv(
        PREDICTIONS, dtype={"GAME_ID": str, "PLAYER_ID": str}
    )

    positions = directory[["PLAYER_ID", "POSITION"]].copy()
    positions["ESPN_POSITION"] = positions["POSITION"].replace({
        "PG": "G", "SG": "G", "SF": "F", "PF": "F"
    })
    positions = positions[["PLAYER_ID", "ESPN_POSITION"]]

    history = features.merge(
        positions, on="PLAYER_ID", how="left", validate="many_to_one"
    )
    history["GAME_DATE"] = pd.to_datetime(history["GAME_DATE"])
    history["DK_POINTS"] = pd.to_numeric(
        history["DK_POINTS"], errors="coerce"
    )
    history["DK_EMA_10"] = pd.to_numeric(
        history["DK_EMA_10"], errors="coerce"
    )
    history["RESIDUAL"] = (
        history["DK_POINTS"] - history["DK_EMA_10"]
    )

    history = history[
        history["ESPN_POSITION"].isin(["G", "F", "C"])
        & history["OPPONENT"].notna()
    ].copy()

    # Aggregate all player residuals from each opponent's game.
    daily = (
        history.dropna(subset=["RESIDUAL"])
        .groupby(["OPPONENT", "ESPN_POSITION", "GAME_DATE"])
        .agg(
            SUM_RESIDUAL=("RESIDUAL", "sum"),
            N=("RESIDUAL", "size"),
        )
        .reset_index()
        .sort_values(["OPPONENT", "ESPN_POSITION", "GAME_DATE"])
    )

    # Each daily value represents one completed game against an opponent.
    # Shift by one game so the current matchup cannot influence itself.
    for window in WINDOWS:
        daily[f"SUM_{window}"] = (
            daily.groupby(["OPPONENT", "ESPN_POSITION"])["SUM_RESIDUAL"]
            .transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).sum()
            )
        )
        daily[f"N_{window}"] = (
            daily.groupby(["OPPONENT", "ESPN_POSITION"])["N"]
            .transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).sum()
            )
        )
        daily[f"ADJ_{window}"] = (
            daily[f"SUM_{window}"]
            / (daily[f"N_{window}"] + SHRINKAGE)
        )

    # Use an as-of lookup so days without a valid residual still work.
    predictions["GAME_DATE"] = pd.to_datetime(predictions["GAME_DATE"])
    evaluation = predictions.merge(
        history[KEYS + ["OPPONENT", "ESPN_POSITION"]],
        on=KEYS, how="left", validate="one_to_one"
    )

    for window in WINDOWS:
        lookup = daily[
            ["OPPONENT", "ESPN_POSITION", "GAME_DATE", f"ADJ_{window}"]
        ].rename(columns={
            "GAME_DATE": "HISTORY_DATE",
            f"ADJ_{window}": f"ADJUSTMENT_{window}",
        })

        evaluation = pd.merge_asof(
            evaluation.sort_values("GAME_DATE"),
            lookup.sort_values("HISTORY_DATE"),
            left_on="GAME_DATE",
            right_on="HISTORY_DATE",
            by=["OPPONENT", "ESPN_POSITION"],
            direction="backward",
            allow_exact_matches=False,
        )

        evaluation[f"ADJUSTMENT_{window}"] = (
            evaluation[f"ADJUSTMENT_{window}"].fillna(0)
        )

        evaluation = evaluation.drop(
            columns=["HISTORY_DATE"]
        )

    def metrics(actual, predicted):
        error = predicted - actual
        return (
            np.mean(np.abs(error)),
            np.sqrt(np.mean(error ** 2)),
            np.mean(error),
        )

    print("EVALUATION ROWS:", len(evaluation))
    print("MISSING POSITIONS:", evaluation["ESPN_POSITION"].isna().sum())

    baseline = metrics(
        evaluation["ACTUAL"], evaluation["CATBOOST"]
    )
    print(
        f"\nBASELINE: MAE={baseline[0]:.4f} "
        f"RMSE={baseline[1]:.4f} BIAS={baseline[2]:+.4f}"
    )

    for window in WINDOWS:
        print(f"\nOPPONENT WINDOW: {window} games")
        print("WEIGHT    MAE     RMSE     BIAS    DELTA_MAE")

        for weight in [0.25, 0.5, 0.75, 1.0]:
            adjusted = (
                evaluation["CATBOOST"]
                + weight * evaluation[f"ADJUSTMENT_{window}"]
            ).clip(lower=0)

            mae, rmse, bias = metrics(
                evaluation["ACTUAL"], adjusted
            )

            print(
                f"{weight:>5.2f}   {mae:>7.4f} "
                f"{rmse:>8.4f} {bias:>+8.4f} "
                f"{mae - baseline[0]:>+10.4f}"
            )

    print("\nFOLD RESULTS (20-game window, weight 0.5)")
    adjusted = (
        evaluation["CATBOOST"]
        + 0.5 * evaluation["ADJUSTMENT_20"]
    ).clip(lower=0)

    for fold, group in evaluation.groupby("FOLD"):
        original = metrics(group["ACTUAL"], group["CATBOOST"])
        modified = metrics(
            group["ACTUAL"], adjusted.loc[group.index]
        )
        print(
            f"Fold {fold}: N={len(group):4d} "
            f"MAE {original[0]:.4f} -> {modified[0]:.4f} "
            f"Delta={modified[0] - original[0]:+.4f}"
        )


if __name__ == "__main__":
    main()
