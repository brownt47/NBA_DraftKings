import pandas as pd
from src.features.config import ROLLING_WINDOWS

def calculate_team_features(games):
    """Calculate season and recent team performance metrics."""

    games = games.copy()
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])

    metrics = {
        "PACE": "PACE_EST",
        "OFF_RATING": "OFF_RATING_EST",
        "DEF_RATING": "DEF_RATING_EST",
    }

    records = []

    for team, group in games.groupby("TEAM"):
        group = group.sort_values(["GAME_DATE", "GAME_ID"])

        record = {
            "TEAM": team,
            "TEAM_GAMES_PLAYED": len(group),
        }

        for name, column in metrics.items():
            record[f"{name}_SEASON"] = group[column].mean()

            for window in ROLLING_WINDOWS:
                record[f"{name}_LAST_{window}"] = (
                    group[column].tail(window).mean()
                )

        records.append(record)

    return pd.DataFrame(records)


def add_historical_team_features(games, windows=ROLLING_WINDOWS):
    """Calculate pregame team performance features."""
    games = games.copy()

    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])

    games = games.sort_values(
        ["TEAM", "GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)

    grouped = games.groupby("TEAM")

    games["TEAM_PRIOR_GAMES"] = grouped.cumcount()

    metrics = {
        "PACE": "PACE_EST",
        "OFF_RATING": "OFF_RATING_EST",
        "DEF_RATING": "DEF_RATING_EST",
    }

    for name, column in metrics.items():
        games[f"{name}_SEASON"] = grouped[column].transform(
            lambda s: s.shift(1).expanding().mean()
        )

        for window in windows:
            games[f"{name}_LAST_{window}"] = grouped[column].transform(
                lambda s: s.shift(1).rolling(
                    window=window,
                    min_periods=1
                ).mean()
            )

    return games
