
import pandas as pd


def load_player_games(file_path, slate_date):
    """
    Load historical ESPN player games.

    Convert statistical columns to numeric, exclude
    records without valid minutes, and retain only
    games played before the selected slate date.
    """

    games = pd.read_csv(
        file_path,
        dtype={"PLAYER_ID": str}
    )

    # Ensure statistical columns are numeric
    numeric_columns = [
        "MIN", "PTS", "REB", "AST", "DK_POINTS"
    ]

    for column in numeric_columns:
        games[column] = pd.to_numeric(
            games[column],
            errors="coerce"
        )

    # Exclude records without valid playing time
    games = games[games["MIN"].notna()].copy()

    # Convert dates and sort chronologically
    games["GAME_DATE"] = pd.to_datetime(
        games["GAME_DATE"],
        errors="coerce"
    )

    games = games.sort_values(
        ["PLAYER_ID", "GAME_DATE", "GAME_ID"]
    )

    # Only use games played before the slate
    games = games[
        games["GAME_DATE"] < slate_date
    ].copy()

    return games


def load_team_games(file_path, slate_date):
    """
    Load historical NBA team games.

    Convert game dates and exclude games played
    on or after the selected slate date.
    """

    team_games = pd.read_csv(
        file_path,
        dtype={"GAME_ID": str}
    )

    team_games["GAME_DATE"] = pd.to_datetime(
        team_games["GAME_DATE"]
    )

    # Only use games played before the slate
    team_games = team_games[
        team_games["GAME_DATE"] < slate_date
    ].copy()

    return team_games


def load_slate_data(dk_file, mapping_file):
    """
    Load DraftKings salaries and ESPN player mappings.

    Preserve player IDs as strings to ensure
    consistent merging.
    """

    dk = pd.read_csv(
        dk_file,
        dtype={"ID": str}
    )

    mapping = pd.read_csv(
        mapping_file,
        dtype={
            "DK_ID": str,
            "ESPN_PLAYER_ID": str
        }
    )

    return dk, mapping
