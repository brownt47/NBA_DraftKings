import pandas as pd


def calculate_scoring_features(games):
    """
    Calculate season and recent DraftKings scoring statistics.

    Expects game history already filtered to dates
    before the selected slate.
    """

    stats = (
        games.groupby("PLAYER_ID")
        .agg(
            GAMES_PLAYED=("GAME_ID", "nunique"),
            AVG_DK_POINTS=("DK_POINTS", "mean"),
            AVG_PTS=("PTS", "mean"),
            AVG_REB=("REB", "mean"),
            AVG_AST=("AST", "mean"),
        )
        .reset_index()
        .rename(columns={"PLAYER_ID": "ESPN_PLAYER_ID"})
    )

    return stats

def calculate_rolling_scoring_features(games):
    """
    Calculate recent DraftKings scoring statistics
    using the last 3, 5, and 10 games.
    """

    recent_stats = []

    for player_id, player_games in games.groupby("PLAYER_ID"):
        player_games = player_games.sort_values(
            ["GAME_DATE", "GAME_ID"]
        )

        record = {"ESPN_PLAYER_ID": player_id}

        for window in [3, 5, 10]:
            recent = player_games.tail(window)

            record[f"DK_N_LAST_{window}"] = (
                recent["DK_POINTS"].count()
            )

            record[f"DK_AVG_LAST_{window}"] = (
                recent["DK_POINTS"].mean()
            )

            record[f"DK_STD_LAST_{window}"] = (
                recent["DK_POINTS"].std()
            )


        recent_stats.append(record)

    return pd.DataFrame(recent_stats)


def calculate_value_per_1000(players):
    """
    Calculate average DraftKings fantasy points
    per $1,000 of player salary.
    """

    players = players.copy()

    players["VALUE_PER_1000"] = (
        players["AVG_DK_POINTS"] /
        (players["Salary"] / 1000)
    )

    return players


def add_historical_scoring_features(games, windows=(3, 5, 7, 10)):
    """
    Calculate pregame scoring features for every historical
    player-game using only previously completed games.
    """
    games = games.copy()

    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])

    games = games.sort_values(
        ["PLAYER_ID", "GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)

    grouped = games.groupby("PLAYER_ID")

    # Number of games played before this game
    games["PRIOR_GAMES"] = grouped.cumcount()

    # Season-to-date averages
    for stat in ["DK_POINTS", "PTS", "REB", "AST"]:
        games[f"{stat}_AVG_SEASON"] = grouped[stat].transform(
            lambda s: s.shift(1).expanding().mean()
        )

    # Rolling statistics
    for window in windows:
        for stat in ["DK_POINTS", "PTS", "REB", "AST"]:
            games[f"{stat}_AVG_LAST_{window}"] = (
                grouped[stat].transform(
                    lambda s: s.shift(1).rolling(
                        window=window,
                        min_periods=1
                    ).mean()
                )
            )

        games[f"DK_STD_LAST_{window}"] = (
            grouped["DK_POINTS"].transform(
                lambda s: s.shift(1).rolling(
                    window=window,
                    min_periods=2
                ).std()
            )
        )

    return games
