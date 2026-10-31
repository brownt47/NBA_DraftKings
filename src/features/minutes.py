import pandas as pd


def calculate_minutes_features(games):
    """
    Calculate playing-time and per-minute production features.

    Expects valid played-game records already filtered
    to dates before the selected slate.
    """

    records = []

    for player_id, player_games in games.groupby("PLAYER_ID"):
        player_games = player_games.sort_values(
            ["GAME_DATE", "GAME_ID"]
        )

        minutes = player_games["MIN"]
        total_minutes = minutes.sum()

        record = {
            "ESPN_PLAYER_ID": player_id,
            "MIN_AVG_SEASON": minutes.mean(),
            "DK_PER_MIN": (
                player_games["DK_POINTS"].sum() / total_minutes
                if total_minutes > 0 else float("nan")
            ),
            "MIN_TREND_3": (
                minutes.tail(3).mean() - minutes.mean()
            ),
            "MIN_STD_LAST_5": minutes.tail(5).std(),
        }

        for window in [3, 5, 10]:
            recent = minutes.tail(window)
            record[f"MIN_AVG_LAST_{window}"] = recent.mean()
            record[f"MIN_N_LAST_{window}"] = recent.count()

        records.append(record)

    return pd.DataFrame(records)