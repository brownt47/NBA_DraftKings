import pandas as pd
from src.features.config import ROLLING_WINDOWS

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

        for window in ROLLING_WINDOWS:
            recent = minutes.tail(window)
            record[f"MIN_AVG_LAST_{window}"] = recent.mean()
            record[f"MIN_N_LAST_{window}"] = recent.count()

        records.append(record)

    return pd.DataFrame(records)


def add_historical_minutes_features(games, windows=ROLLING_WINDOWS):
    """Calculate historical pregame minutes and productivity features."""
    games = games.copy()

    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games["MIN"] = pd.to_numeric(games["MIN"], errors="coerce")

    games = games.sort_values(
        ["PLAYER_ID", "GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)

    grouped = games.groupby("PLAYER_ID")

    # Season-to-date minutes average
    games["MIN_AVG_SEASON"] = grouped["MIN"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    # Season-to-date DK points per minute
    prior_dk = grouped["DK_POINTS"].transform(
        lambda s: s.shift(1).expanding().sum()
    )

    prior_minutes = grouped["MIN"].transform(
        lambda s: s.shift(1).expanding().sum()
    )

    games["DK_PER_MIN"] = (
        prior_dk / prior_minutes.replace(0, float("nan"))
    )

    for window in windows:
        games[f"MIN_AVG_LAST_{window}"] = grouped["MIN"].transform(
            lambda s: s.shift(1).rolling(
                window=window, min_periods=1
            ).mean()
        )

        games[f"MIN_STD_LAST_{window}"] = grouped["MIN"].transform(
            lambda s: s.shift(1).rolling(
                window=window, min_periods=2
            ).std()
        )

    # Recent playing time compared with season average
    games["MIN_TREND_3"] = (
        games["MIN_AVG_LAST_3"] - games["MIN_AVG_SEASON"]
    )


    # Minutes played in the player's most recent appearance.
    games["LAST_GAME_MIN"] = (
        games.groupby("PLAYER_ID")["MIN"].shift(1)
    )


    return games
