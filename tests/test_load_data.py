
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------
# IMPORT SETUP
# Allow tests to import modules from src.
# ---------------------------------------------------------

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from data_processing.load_data import (
    load_player_games,
    load_team_games,
    load_slate_data,
)


# ---------------------------------------------------------
# PLAYER GAME LOADING TEST
# Verify numeric conversion, missing minutes,
# chronological sorting, and slate-date filtering.
# ---------------------------------------------------------

def test_load_player_games(tmp_path):

    games = pd.DataFrame({
        "PLAYER_ID": ["101", "101", "101", "102", "102"],
        "GAME_ID": ["G3", "G1", "G2", "G4", "G5"],
        "GAME_DATE": [
            "2026-10-31",
            "2026-10-28",
            "2026-10-29",
            "2026-10-27",
            "2026-10-30",
        ],
        "MIN": ["30", "25", "DNP", "20", "35"],
        "PTS": ["20", "15", "10", "12", "25"],
        "REB": ["8", "5", "4", "6", "10"],
        "AST": ["4", "3", "2", "5", "7"],
        "DK_POINTS": ["40", "30", "20", "28", "50"],
    })

    file_path = tmp_path / "player_games.csv"
    games.to_csv(file_path, index=False)

    slate_date = pd.Timestamp("2026-10-31")

    result = load_player_games(file_path, slate_date)

    # G3 is on slate date and G2 has invalid minutes
    assert len(result) == 3
    assert set(result["GAME_ID"]) == {"G1", "G4", "G5"}

    # IDs remain strings
    assert result["PLAYER_ID"].dtype == object

    # Numeric conversion
    assert pd.api.types.is_numeric_dtype(result["MIN"])
    assert pd.api.types.is_numeric_dtype(result["DK_POINTS"])

    # Historical games only
    assert (result["GAME_DATE"] < slate_date).all()

    # Chronological sorting within each player
    player_102 = result[result["PLAYER_ID"] == "102"]
    assert player_102["GAME_ID"].tolist() == ["G4", "G5"]


# ---------------------------------------------------------
# TEAM GAME LOADING TEST
# Verify historical date filtering and game ID preservation.
# ---------------------------------------------------------

def test_load_team_games(tmp_path):

    games = pd.DataFrame({
        "GAME_ID": ["001", "002", "003", "004"],
        "GAME_DATE": [
            "2026-10-28",
            "2026-10-30",
            "2026-10-31",
            "2026-11-01",
        ],
        "TEAM": ["ATL", "BOS", "ATL", "BOS"],
        "PACE_EST": [100, 105, 102, 108],
    })

    file_path = tmp_path / "team_games.csv"
    games.to_csv(file_path, index=False)

    slate_date = pd.Timestamp("2026-10-31")

    result = load_team_games(file_path, slate_date)

    # Only games before the slate date
    assert len(result) == 2
    assert result["GAME_ID"].tolist() == ["001", "002"]

    # Preserve leading zeros in game IDs
    assert result["GAME_ID"].dtype == object

    # Dates converted to datetime
    assert pd.api.types.is_datetime64_any_dtype(
        result["GAME_DATE"]
    )

    # No games on or after slate date
    assert (result["GAME_DATE"] < slate_date).all()


# ---------------------------------------------------------
# SLATE DATA LOADING TEST
# Verify DraftKings and ESPN IDs remain strings.
# ---------------------------------------------------------

def test_load_slate_data(tmp_path):

    dk = pd.DataFrame({
        "ID": ["00101", "00202"],
        "Name": ["Player A", "Player B"],
        "Salary": [10000, 7500],
    })

    mapping = pd.DataFrame({
        "DK_ID": ["00101", "00202"],
        "ESPN_PLAYER_ID": ["000123", "000456"],
    })

    dk_file = tmp_path / "salaries.csv"
    mapping_file = tmp_path / "mapping.csv"

    dk.to_csv(dk_file, index=False)
    mapping.to_csv(mapping_file, index=False)

    result_dk, result_mapping = load_slate_data(
        dk_file,
        mapping_file
    )

    # DraftKings IDs remain strings
    assert result_dk["ID"].tolist() == [
        "00101", "00202"
    ]

    # Mapping IDs retain leading zeros
    assert result_mapping["DK_ID"].tolist() == [
        "00101", "00202"
    ]

    assert result_mapping["ESPN_PLAYER_ID"].tolist() == [
        "000123", "000456"
    ]

    # Salary remains numeric
    assert pd.api.types.is_numeric_dtype(
        result_dk["Salary"]
    )
