"""Build pregame feature rows for a DraftKings NBA slate."""

import pandas as pd


def build_slate_team_rows(slate_games, team_columns):
    """Create future team-game records without box-score statistics."""

    records = []

    for game in slate_games:
        for home_away, team, opponent in (
            ("home", game["HOME_TEAM"], game["AWAY_TEAM"]),
            ("away", game["AWAY_TEAM"], game["HOME_TEAM"]),
        ):
            records.append({
                "GAME_ID": game["GAME_ID"],
                "GAME_DATE": game["GAME_DATE"],
                "TEAM": team,
                "OPPONENT": opponent,
                "HOME_AWAY": home_away,
            })

    rows = pd.DataFrame(records)

    # All unknown box-score fields remain missing.
    return rows.reindex(columns=team_columns)

def build_slate_player_rows(mapped_slate, player_columns):
    """Create future player-game records without game statistics."""

    required = [
        "ESPN_PLAYER_ID",
        "ESPN_PLAYER_NAME",
        "ESPN_TEAM",
        "ESPN_GAME_ID",
        "GAME_DATE",
        "MATCHUP",
    ]

    missing = set(required) - set(mapped_slate.columns)

    if missing:
        raise ValueError(
            f"Missing mapped slate columns: {sorted(missing)}"
        )

    rows = pd.DataFrame({
        "GAME_ID": mapped_slate["ESPN_GAME_ID"].to_numpy(),
        "GAME_DATE": mapped_slate["GAME_DATE"].to_numpy(),
        "PLAYER_ID": mapped_slate["ESPN_PLAYER_ID"].to_numpy(),
        "PLAYER_NAME": mapped_slate["ESPN_PLAYER_NAME"].to_numpy(),
        "TEAM": mapped_slate["ESPN_TEAM"].to_numpy(),
        "MATCHUP": mapped_slate["MATCHUP"].to_numpy(),
    })

    if rows.duplicated(["GAME_ID", "PLAYER_ID"]).any():
        raise ValueError("Duplicate slate player-game records.")

    return rows.reindex(columns=player_columns)