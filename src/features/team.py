
import pandas as pd


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

            for window in (3, 5, 7):
                record[f"{name}_LAST_{window}"] = (
                    group[column].tail(window).mean()
                )

        records.append(record)

    return pd.DataFrame(records)
