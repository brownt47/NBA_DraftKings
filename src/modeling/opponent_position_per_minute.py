"""Test opponent-position DK-per-minute adjustments against CatBoost."""

import numpy as np
import pandas as pd

FEATURES = "data/nba_features_2025_26.csv"
DIRECTORY = "data/espn_player_directory.csv"
PREDICTIONS = "reports/modeling/boosting_comparison/predictions.csv"

KEYS = ["GAME_ID", "PLAYER_ID"]
WINDOWS = [10, 20]
SHRINKAGE = 20


def metrics(actual, predicted):
    error = predicted - actual
    return (
        np.mean(np.abs(error)),
        np.sqrt(np.mean(error ** 2)),
        np.mean(error),
    )


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
        positions, on="PLAYER_ID", how="left",
        validate="many_to_one"
    )

    history["GAME_DATE"] = pd.to_datetime(history["GAME_DATE"])
    history["MIN"] = pd.to_numeric(history["MIN"], errors="coerce")
    history["DK_POINTS"] = pd.to_numeric(
        history["DK_POINTS"], errors="coerce"
    )
    history["DK_EMA_10"] = pd.to_numeric(
        history["DK_EMA_10"], errors="coerce"
    )

    # Compare actual efficiency against a pregame efficiency baseline.
    # The expected minutes must be known before the game.
    minute_candidates = [
        "MIN_AVG_LAST_10",
        "MIN_AVG_LAST_5",
        "MIN_AVG_SEASON",
    ]
    minute_column = next(
        (c for c in minute_candidates if c in history.columns),
        None
    )
    if minute_column is None:
        raise ValueError("No pregame minutes baseline found")

    history["EXPECTED_MIN"] = pd.to_numeric(
        history[minute_column], errors="coerce"
    )

    valid = (
        history["MIN"].gt(0)
        & history["EXPECTED_MIN"].gt(0)
        & history["DK_EMA_10"].notna()
        & history["ESPN_POSITION"].isin(["G", "F", "C"])
        & history["OPPONENT"].notna()
    )

    history = history.loc[valid].copy()

    history["ACTUAL_PER_MIN"] = (
        history["DK_POINTS"] / history["MIN"]
    )
    history["EXPECTED_PER_MIN"] = (
        history["DK_EMA_10"] / history["EXPECTED_MIN"]
    )
    history["EFFICIENCY_RESIDUAL"] = (
        history["ACTUAL_PER_MIN"] - history["EXPECTED_PER_MIN"]
    )

    # Aggregate by opponent, position, and game date.
    daily = (
        history.groupby(
            ["OPPONENT", "ESPN_POSITION", "GAME_DATE"]
        )
        .agg(
            SUM_RESIDUAL=("EFFICIENCY_RESIDUAL", "sum"),
            N=("EFFICIENCY_RESIDUAL", "size"),
        )
        .reset_index()
        .sort_values(
            ["OPPONENT", "ESPN_POSITION", "GAME_DATE"]
        )
    )

    for window in WINDOWS:
        grouped = daily.groupby(
            ["OPPONENT", "ESPN_POSITION"]
        )

        daily[f"SUM_{window}"] = grouped["SUM_RESIDUAL"].transform(
            lambda s: s.shift(1).rolling(
                window, min_periods=1
            ).sum()
        )
        daily[f"N_{window}"] = grouped["N"].transform(
            lambda s: s.shift(1).rolling(
                window, min_periods=1
            ).sum()
        )

        daily[f"RATE_{window}"] = (
            daily[f"SUM_{window}"]
            / (daily[f"N_{window}"] + SHRINKAGE)
        )

    evaluation = predictions.merge(
        history[KEYS + [
            "OPPONENT", "ESPN_POSITION", "EXPECTED_MIN"
        ]],
        on=KEYS, how="left", validate="one_to_one"
    )

    evaluation["GAME_DATE"] = pd.to_datetime(
        evaluation["GAME_DATE"]
    )

    for window in WINDOWS:
        lookup = daily[
            ["OPPONENT", "ESPN_POSITION", "GAME_DATE",
             f"RATE_{window}"]
        ].rename(columns={
            "GAME_DATE": "HISTORY_DATE",
            f"RATE_{window}": f"ADJUSTMENT_{window}",
        })

        evaluation = pd.merge_asof(
            evaluation.sort_values("GAME_DATE"),
            lookup.sort_values("HISTORY_DATE"),
            left_on="GAME_DATE",
            right_on="HISTORY_DATE",
            by=["OPPONENT", "ESPN_POSITION"],
            direction="backward",
            allow_exact_matches=False,
        ).drop(columns=["HISTORY_DATE"])

        evaluation[f"ADJUSTMENT_{window}"] = (
            evaluation[f"ADJUSTMENT_{window}"].fillna(0)
        )

    evaluation["EXPECTED_MIN"] = (
        evaluation["EXPECTED_MIN"].fillna(0)
    )

    baseline = metrics(
        evaluation["ACTUAL"], evaluation["CATBOOST"]
    )

    print("MINUTES BASELINE:", minute_column)
    print("HISTORICAL PLAYER-GAMES:", len(history))
    print("EVALUATION ROWS:", len(evaluation))
    print(
        f"BASELINE: MAE={baseline[0]:.4f} "
        f"RMSE={baseline[1]:.4f} BIAS={baseline[2]:+.4f}"
    )

    for window in WINDOWS:
        print(f"\nWINDOW: {window} games")
        print("WEIGHT     MAE      RMSE      BIAS    DELTA_MAE")

        for weight in [0.25, 0.5, 0.75, 1.0]:
            adjusted = (
                evaluation["CATBOOST"]
                + weight
                * evaluation[f"ADJUSTMENT_{window}"]
                * evaluation["EXPECTED_MIN"]
            ).clip(lower=0)

            mae, rmse, bias = metrics(
                evaluation["ACTUAL"], adjusted
            )

            print(
                f"{weight:>5.2f}    {mae:>7.4f} "
                f"{rmse:>8.4f} {bias:>+8.4f} "
                f"{mae - baseline[0]:>+10.4f}"
            )

    print("\nFOLD RESULTS: 20 games, weight 0.25")
    adjusted = (
        evaluation["CATBOOST"]
        + 0.25
        * evaluation["ADJUSTMENT_20"]
        * evaluation["EXPECTED_MIN"]
    ).clip(lower=0)

    for fold, group in evaluation.groupby("FOLD"):
        original = metrics(
            group["ACTUAL"], group["CATBOOST"]
        )
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
