"""Pregame NBA rotation and playing-time distribution features."""

import pandas as pd


def add_historical_rotation_features(players, teams):
    """Add historical team rotation concentration without data leakage."""
    players = players.copy()
    teams = teams.copy()

    players["GAME_DATE"] = pd.to_datetime(players["GAME_DATE"])
    teams["GAME_DATE"] = pd.to_datetime(teams["GAME_DATE"])

    players["MIN_NUM"] = pd.to_numeric(players["MIN"], errors="coerce")

    # Calculate concentration for each completed team-game.
    keys = ["GAME_ID", "TEAM"]

    total_minutes = players.groupby(keys)["MIN_NUM"].transform("sum")

    players["_MIN_SHARE"] = players["MIN_NUM"] / total_minutes
    players["_SHARE_SQUARED"] = players["_MIN_SHARE"] ** 2

    concentration = (
        players.groupby(keys, as_index=False)["_SHARE_SQUARED"]
        .sum(min_count=1)
        .rename(columns={"_SHARE_SQUARED": "_ACTUAL_ROTATION_HHI"})
    )

    teams = teams.merge(
        concentration,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    teams = teams.sort_values(
        ["TEAM", "GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)

    grouped = teams.groupby("TEAM")["_ACTUAL_ROTATION_HHI"]

    teams["TEAM_ROTATION_CONCENTRATION"] = grouped.shift(1)

    teams["TEAM_ROTATION_CONCENTRATION_AVG_5"] = (
        teams.groupby("TEAM")["_ACTUAL_ROTATION_HHI"]
        .transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        )
    )

    return players.drop(
        columns=["MIN_NUM", "_MIN_SHARE", "_SHARE_SQUARED"]
    ).merge(
        teams[
            keys
            + [
                "TEAM_ROTATION_CONCENTRATION",
                "TEAM_ROTATION_CONCENTRATION_AVG_5",
            ]
        ],
        on=keys,
        how="left",
        validate="many_to_one",
    )


def add_historical_player_minutes_share(players, teams):
    """Calculate pregame player minutes-share features."""
    players = players.copy()

    game_lengths = teams[
        ["GAME_ID", "TEAM", "GAME_MINUTES"]
    ].copy()

    players = players.merge(
        game_lengths,
        on=["GAME_ID", "TEAM"],
        how="left",
        validate="many_to_one",
    )

    players["GAME_DATE"] = pd.to_datetime(players["GAME_DATE"])
    players = players.sort_values(
        ["PLAYER_ID", "GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)

    minutes = pd.to_numeric(players["MIN"], errors="coerce")
    available = 5 * pd.to_numeric(
        players["GAME_MINUTES"], errors="coerce"
    )

    players["_MINUTES_SHARE"] = minutes / available

    grouped = players.groupby("PLAYER_ID")["_MINUTES_SHARE"]

    players["PLAYER_MINUTES_SHARE_AVG_5"] = grouped.transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )

    players["PLAYER_MINUTES_SHARE_STD_5"] = grouped.transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).std()
    )

    avg_3 = grouped.transform(
        lambda s: s.shift(1).rolling(3, min_periods=3).mean()
    )

    players["PLAYER_MINUTES_SHARE_TREND"] = (
        avg_3 - players["PLAYER_MINUTES_SHARE_AVG_5"]
    )

    return players.drop(columns=["GAME_MINUTES", "_MINUTES_SHARE"])
