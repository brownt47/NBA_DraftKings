
import pandas as pd


def add_historical_rotation_features(games):
    """
    Add historical team rotation features.

    All calculations use completed games before the
    current game to avoid target leakage.
    """
    games = games.copy()

    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games["MIN"] = pd.to_numeric(games["MIN"], errors="coerce")

    # Share of total team minutes in each completed game.
    # This is an intermediate calculation, not a pregame feature.
    games["GAME_MINUTES_SHARE"] = (
        games["MIN"]
        / games.groupby(["TEAM", "GAME_ID"])["MIN"]
        .transform("sum")
        .replace(0, float("nan"))
    )
  

    # Average share of team minutes across the player's
    # previous five appearances.
    games = games.sort_values(
        ["PLAYER_ID", "GAME_DATE", "GAME_ID"]
    )

    games["PLAYER_MINUTES_SHARE_AVG_5"] = (
        games.groupby("PLAYER_ID")["GAME_MINUTES_SHARE"]
        .transform(
            lambda s: s.shift(1).rolling(
                window=5, min_periods=1
            ).mean()
        )
    )


    # Variability in minutes share over the previous
    # five appearances.
    games["PLAYER_MINUTES_SHARE_STD_5"] = (
        games.groupby("PLAYER_ID")["GAME_MINUTES_SHARE"]
        .transform(
            lambda s: s.shift(1).rolling(
                window=5, min_periods=2
            ).std()
        )
    )


    # Recent minutes-share trend compared with
    # the player's previous five appearances.
    games["PLAYER_MINUTES_SHARE_AVG_3"] = (
        games.groupby("PLAYER_ID")["GAME_MINUTES_SHARE"]
        .transform(
            lambda s: s.shift(1).rolling(
                window=3, min_periods=1
            ).mean()
        )
    )

    games["PLAYER_MINUTES_SHARE_TREND"] = (
        games["PLAYER_MINUTES_SHARE_AVG_3"]
        - games["PLAYER_MINUTES_SHARE_AVG_5"]
    )


    # Total minutes played by the five highest-minute
    # players in each completed team game.
    team_game_load = (
        games.groupby(["TEAM", "GAME_ID", "GAME_DATE"])["MIN"]
        .apply(lambda s: s.nlargest(5).sum())
        .reset_index(name="TEAM_TOP_5_MIN_LOAD")
    )


    # Rotation concentration using the Herfindahl index.
    # Higher values indicate minutes concentrated
    # among fewer players.
    team_game_hhi = (
        games.groupby(["TEAM", "GAME_ID", "GAME_DATE"])["MIN"]
        .apply(
            lambda s: (
                ((s / s.sum()) ** 2).sum()
                if s.sum() > 0 else float("nan")
            )
        )
        .reset_index(name="TEAM_ROTATION_CONCENTRATION")
    )

    team_game_load = team_game_load.merge(
        team_game_hhi,
        on=["TEAM", "GAME_ID", "GAME_DATE"],
        how="left",
        validate="one_to_one",
    )


    # Sort games chronologically within each team.
    team_game_load = team_game_load.sort_values(
        ["TEAM", "GAME_DATE", "GAME_ID"]
    )

    # Only use the previous completed game.
    team_game_load["TEAM_TOP_5_MIN_LOAD"] = (
        team_game_load.groupby("TEAM")["TEAM_TOP_5_MIN_LOAD"]
        .shift(1)
    )


    # Average rotation concentration over the previous
    # five completed team games.
    team_game_load["TEAM_ROTATION_CONCENTRATION_LAST_5"] = (
        team_game_load.groupby("TEAM")[
            "TEAM_ROTATION_CONCENTRATION"
        ].transform(
            lambda s: s.shift(1).rolling(
                window=5, min_periods=1
            ).mean()
        )
    )


    team_game_load["TEAM_ROTATION_CONCENTRATION"] = (
        team_game_load.groupby("TEAM")[
            "TEAM_ROTATION_CONCENTRATION"
        ].shift(1)
    )


    # Apply the same pregame value to all players
    # participating in that team game.
    
    games = games.merge(
        team_game_load[
            [
                "TEAM",
                "GAME_ID",
                "TEAM_TOP_5_MIN_LOAD",
                "TEAM_ROTATION_CONCENTRATION",
                "TEAM_ROTATION_CONCENTRATION_LAST_5",
            ]
        ],
        on=["TEAM", "GAME_ID"],
        how="left",
        validate="many_to_one",
    )


    # Remove the current-game intermediate calculation.
    # Only historical features should reach the model.
    games = games.drop(columns=["GAME_MINUTES_SHARE"])

    return games
