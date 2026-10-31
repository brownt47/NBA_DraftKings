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