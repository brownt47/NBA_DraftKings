"""Pregame NBA offensive usage features."""

import numpy as np
import pandas as pd



def _historical_efficiency_slope(minutes, dk_points):
    """Estimate the slope of DK points/min versus minutes."""
    valid = (
        minutes.notna()
        & dk_points.notna()
        & (minutes > 0)
    )

    x = minutes[valid].astype(float)
    y = dk_points[valid].astype(float) / x

    if len(x) < 5 or x.nunique() < 2:
        return np.nan

    denominator = ((x - x.mean()) ** 2).sum()

    if denominator == 0:
        return np.nan

    return (
        ((x - x.mean()) * (y - y.mean())).sum()
        / denominator
    )


def add_historical_usage_features(players):
    """Add leakage-free historical offensive usage features."""
    players = players.copy()

    players["GAME_DATE"] = pd.to_datetime(players["GAME_DATE"])

    players = players.sort_values(
        ["PLAYER_ID", "GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)

    minutes = pd.to_numeric(players["MIN"], errors="coerce")
    attempts = pd.to_numeric(players["FGA"], errors="coerce")

    # Actual game-level rate, used only to construct history.
    players["_FGA_PER_36_ACTUAL"] = np.where(
        minutes > 0,
        36 * attempts / minutes,
        np.nan,
    )

    # Pregame prediction uses the previous five appearances.
    players["FGA_PER_36"] = (
        players.groupby("PLAYER_ID")["_FGA_PER_36_ACTUAL"]
        .transform(
            lambda s: s.shift(1).rolling(
                5, min_periods=1
            ).mean()
        )
    )



    # Compare recent offensive volume against the five-game baseline.
    players["_FGA_PER_36_AVG_3"] = (
        players.groupby("PLAYER_ID")["_FGA_PER_36_ACTUAL"]
        .transform(
            lambda s: s.shift(1).rolling(
                3, min_periods=3
            ).mean()
        )
    )

    players["FGA_PER_36_TREND"] = (
        players["_FGA_PER_36_AVG_3"] - players["FGA_PER_36"]
    )


    # Offensive involvement per 36 minutes.
    # FGA + 0.44 * FTA + turnovers.
    free_throws = pd.to_numeric(
        players["FTA"], errors="coerce"
    )
    turnovers = pd.to_numeric(
        players["TOV"], errors="coerce"
    )

    offensive_actions = (
        attempts + 0.44 * free_throws + turnovers
    )

    players["_USAGE_PROXY_ACTUAL"] = np.where(
        minutes > 0,
        36 * offensive_actions / minutes,
        np.nan,
    )

    players["PLAYER_USAGE_PROXY"] = (
        players.groupby("PLAYER_ID")["_USAGE_PROXY_ACTUAL"]
        .transform(
            lambda s: s.shift(1).rolling(
                5, min_periods=1
            ).mean()
        )
    )

    # Historical relationship between minutes and DK efficiency.
    # Use only the previous 10 appearances.
    players["_MIN_NUM"] = pd.to_numeric(
        players["MIN"], errors="coerce"
    )
    players["_DK_NUM"] = pd.to_numeric(
        players["DK_POINTS"], errors="coerce"
    )

    players["DK_PER_MIN_BY_MINUTES"] = np.nan

    for _, group in players.groupby("PLAYER_ID", sort=False):
        indices = group.index.to_list()

        for position, idx in enumerate(indices):
            if position < 5:
                continue

            prior_indices = indices[max(0, position - 10):position]

            prior = players.loc[
                prior_indices, ["_MIN_NUM", "_DK_NUM"]
            ]

            players.at[idx, "DK_PER_MIN_BY_MINUTES"] = (
                _historical_efficiency_slope(
                    prior["_MIN_NUM"],
                    prior["_DK_NUM"],
                )
            )


    return players.drop(
        columns=[
            "_FGA_PER_36_ACTUAL",
            "_FGA_PER_36_AVG_3",
            "_MIN_NUM",
            "_DK_NUM",
            "_USAGE_PROXY_ACTUAL",
        ]
    )


