import pandas as pd
from src.features.config import ROLLING_WINDOWS

def minutes_stability_features(minutes):
    """Calculate stability metrics from completed player-games."""

    s = pd.to_numeric(minutes, errors="coerce")

    avg5 = s.tail(5).mean()
    avg7 = s.tail(7).mean()
    avg10 = s.tail(10).mean()

    last5 = s.tail(5)
    changes = last5.diff().dropna()

    return {
        "MIN_RATIO_5_10": (
            avg5 / avg10 if avg10 > 0 else float("nan")
        ),
        "MIN_RATIO_5_7": (
            avg5 / avg7 if avg7 > 0 else float("nan")
        ),
        "MIN_WHIPSAW_5": (
            changes.abs().mean()
            if len(last5) == 5 and last5.notna().all() else float("nan")
        ),
        "MIN_DIRECTION_CHANGES_5": (
            int(
                (
                    (changes * changes.shift(1)) < 0
                ).sum()
            )
            if len(last5) == 5 and last5.notna().all() else float("nan")
        ),
        "MIN_MIN_LAST_5": (
            last5.min() if len(last5) == 5 and last5.notna().all() else float("nan")
        ),
    }


def count_minutes_direction_changes(values):
    changes = pd.Series(values).diff().dropna()
    return int(((changes * changes.shift(1)) < 0).sum())


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

        record.update(minutes_stability_features(minutes))

        previous_three = minutes.iloc[-6:-3]
        recent_three = minutes.tail(3)

        record["MIN_TREND_RECENT_3"] = (
            recent_three.mean() - previous_three.mean()
            if len(minutes.dropna()) >= 6
            else float("nan")
        )

        record["MIN_P90_LAST_10"] = (
            minutes.tail(10).quantile(0.90)
            if minutes.tail(10).count() >= 3
            else float("nan")
        )

        ceiling = record["MIN_P90_LAST_10"]

        record["MIN_HEADROOM_LAST_10"] = (
            ceiling - recent_three.mean()
            if pd.notna(ceiling)
            else float("nan")
        )

        record["MIN_CEILING_RATIO_LAST_10"] = (
            recent_three.mean() / ceiling
            if pd.notna(ceiling) and ceiling > 0
            else float("nan")
        )


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

    # Ceiling-aware minutes features.
    # All calculations use previous games only.

    games["MIN_TREND_RECENT_3"] = grouped["MIN"].transform(
        lambda s: (
            s.shift(1).rolling(3, min_periods=3).mean()
            - s.shift(4).rolling(3, min_periods=3).mean()
        )
    )

    games["MIN_P90_LAST_10"] = grouped["MIN"].transform(
        lambda s: s.shift(1).rolling(
            window=10, min_periods=3
        ).quantile(0.90)
    )

    games["MIN_HEADROOM_LAST_10"] = (
        games["MIN_P90_LAST_10"]
        - games["MIN_AVG_LAST_3"]
    )

    games["MIN_CEILING_RATIO_LAST_10"] = (
        games["MIN_AVG_LAST_3"]
        / games["MIN_P90_LAST_10"].replace(
            0, float("nan")
        )
    )



    # Minutes played in the player's most recent appearance.
    games["LAST_GAME_MIN"] = (
        games.groupby("PLAYER_ID")["MIN"].shift(1)
    )

    # Pregame minutes stability features.
    # Each rolling window excludes the current game.

    for window in (5, 7, 10):
        if f"MIN_AVG_LAST_{window}" not in games.columns:
            games[f"MIN_AVG_LAST_{window}"] = grouped["MIN"].transform(
                lambda s: s.shift(1).rolling(
                    window, min_periods=1
                ).mean()
            )

    games["MIN_RATIO_5_10"] = (
        games["MIN_AVG_LAST_5"]
        / games["MIN_AVG_LAST_10"].replace(0, float("nan"))
    )

    games["MIN_RATIO_5_7"] = (
        games["MIN_AVG_LAST_5"]
        / games["MIN_AVG_LAST_7"].replace(0, float("nan"))
    )

    games["MIN_MIN_LAST_5"] = grouped["MIN"].transform(
        lambda s: s.shift(1).rolling(
            5, min_periods=5
        ).min()
    )

    games["MIN_WHIPSAW_5"] = grouped["MIN"].transform(
        lambda s: s.shift(1).rolling(
            5, min_periods=5
        ).apply(
            lambda x: abs(pd.Series(x).diff()).iloc[1:].mean(),
            raw=False,
        )
    )

    games["MIN_DIRECTION_CHANGES_5"] = grouped["MIN"].transform(
        lambda s: s.shift(1).rolling(
            5, min_periods=5
        ).apply(
            count_minutes_direction_changes,
            raw=False,
        )
    )

    return games
