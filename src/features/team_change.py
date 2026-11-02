"""Pregame features describing player team changes and current-team roles."""

import pandas as pd


def add_historical_team_change_features(players):
    """Calculate team-stint features without using current-game statistics."""
    players = players.copy()
    players["GAME_DATE"] = pd.to_datetime(players["GAME_DATE"])

    players = players.sort_values(
        ["PLAYER_ID", "GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)

    grouped = players.groupby("PLAYER_ID", sort=False)

    previous_team = grouped["TEAM"].shift(1)

    players["TEAM_CHANGE_FLAG"] = (
        previous_team.notna() & players["TEAM"].ne(previous_team)
    ).astype(int)

    players["_TEAM_STINT"] = (
        players.groupby("PLAYER_ID")["TEAM_CHANGE_FLAG"].cumsum()
    )

    stint = players.groupby(
        ["PLAYER_ID", "_TEAM_STINT"], sort=False
    )

    players["GAMES_WITH_CURRENT_TEAM"] = stint.cumcount()


    for source, target in (
        ("MIN", "MIN_AVG_CURRENT_TEAM"),
        ("DK_POINTS", "DK_AVG_CURRENT_TEAM"),
    ):
        values = pd.to_numeric(players[source], errors="coerce")

        groups = [
            players["PLAYER_ID"],
            players["_TEAM_STINT"],
        ]

        prior_total = (
            values.fillna(0).groupby(groups).cumsum()
            - values.fillna(0)
        )

        valid = values.notna().astype(int)

        prior_count = (
            valid.groupby(groups).cumsum()
            - valid
        )

        players[target] = prior_total / prior_count.replace(0, float("nan"))

    return players.drop(columns=["_TEAM_STINT"])
